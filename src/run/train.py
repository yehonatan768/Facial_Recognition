from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from src.utils.read_config import load_config_files, resolve_project_root
from src.utils.logger import build_logger
from src.utils.seed import seed_everything
from src.utils.io import save_json
from src.data.load_pairs import parse_pairs_file
from src.data.graph_split import split_by_components
from src.data.datasets import build_dataloaders, Pair
from src.model.cnn_embedder import ConvEmbeddingConfig
from src.model.head import SimilarityHeadConfig
from src.model.siamese import SiameseConfig, SiameseNetwork
from src.model.init import init_weights_like_reference
from src.training.optim import LayerHyper, build_optimizer, step_schedule
from src.training.metrics import MetricsConfig
from src.training.loop import train_one_epoch, eval_one_epoch
from src.training.early_stop import EarlyStopper
from src.evaluation.plot.plots import plot_all_metrics


def _as_layer_hypers(d: Dict[str, Any]) -> Dict[str, LayerHyper]:
    out: Dict[str, LayerHyper] = {}
    for k, v in d.items():
        out[k] = LayerHyper(lr=float(v["lr"]), momentum_target=float(v["momentum_target"]), weight_decay=float(v["weight_decay"]))
    return out


def _timestamp() -> str:
    import datetime
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=None, help="Optional project root to resolve relative paths")
    ap.add_argument("--configs", nargs="+", default=[
        "src/config/paths.yaml",
        "src/config/transform.yaml",
        "src/config/model.yaml",
        "src/config/optim.yaml",
        "src/config/train.yaml",
        "src/config/metrics.yaml",
        "src/config/logging.yaml",
        "src/config/experiment.yaml",
    ])
    args = ap.parse_args()

    cfg_paths = [Path(p) for p in args.configs]
    cfg = load_config_files(cfg_paths)

    project_root = resolve_project_root(cfg, cli_project_root=args.project_root)
    paths = cfg["paths"]

    outputs_dir = (project_root / paths.get("outputs_dir", "outputs")).resolve()
    run_dir = outputs_dir / cfg.get("experiment", {}).get("name", "run") / _timestamp()
    run_dir.mkdir(parents=True, exist_ok=True)

    log_file = run_dir / "run.log" if cfg.get("logging", {}).get("log_to_file", True) else None
    logger = build_logger("train", level=cfg.get("logging", {}).get("level", "INFO"), log_file=log_file)

    seed = int(cfg.get("train", {}).get("seed", 42))
    seed_everything(seed)

    device_str = str(cfg.get("train", {}).get("device", "cuda"))
    device = torch.device(device_str if (device_str == "cpu" or torch.cuda.is_available()) else "cpu")
    logger.info("Device: %s", device)

    # -------------------------
    # Data
    # -------------------------
    images_root = (project_root / paths["images_root"]).resolve()
    pairs_train = (project_root / paths["pairs_train"]).resolve()
    pairs_val = str(paths.get("pairs_val", "")).strip()
    pairs_val_path = (project_root / pairs_val).resolve() if pairs_val else None

    train_pairs = parse_pairs_file(pairs_train, images_root=images_root)
    logger.info("Loaded train pairs: %d", len(train_pairs))

    if pairs_val_path is not None and pairs_val_path.exists():
        val_pairs = parse_pairs_file(pairs_val_path, images_root=images_root)
        logger.info("Loaded val pairs: %d", len(val_pairs))
    else:
        sp = cfg.get("train", {}).get("split", {})
        split = split_by_components(
            pairs=train_pairs,
            val_ratio=float(sp.get("val_ratio", 0.2)),
            target_pos_frac=float(sp.get("target_pos_frac", 0.5)),
            min_val_identities=int(sp.get("min_val_identities", 30)),
            seed=int(sp.get("seed", seed)),
            logger=logger,
        )
        train_pairs, val_pairs = split.train_pairs, split.val_pairs
        logger.info("Split train=%d val=%d", len(train_pairs), len(val_pairs))

    train_loader, val_loader = build_dataloaders(cfg, train_pairs=train_pairs, val_pairs=val_pairs)

    # -------------------------
    # Model
    # -------------------------
    mcfg = cfg["model"]
    enc = mcfg["encoder"]
    in_ch = int(mcfg["input"]["channels"])

    enc_cfg = ConvEmbeddingConfig(
        in_channels=in_ch,
        conv1_out=int(enc["conv1_out"]),
        conv1_kernel=int(enc["conv1_kernel"]),
        conv2_out=int(enc["conv2_out"]),
        conv2_kernel=int(enc["conv2_kernel"]),
        conv3_out=int(enc["conv3_out"]),
        conv3_kernel=int(enc["conv3_kernel"]),
        conv4_out=int(enc["conv4_out"]),
        conv4_kernel=int(enc["conv4_kernel"]),
        pool_kernel=int(enc["pool_kernel"]),
        pool_stride=int(enc["pool_stride"]),
        fc_out=int(enc["fc_out"]),
    )
    head_cfg = SimilarityHeadConfig(use_bias=bool(mcfg["head"].get("use_bias", False)))
    model = SiameseNetwork(SiameseConfig(encoder=enc_cfg, head=head_cfg)).to(device)
    init_weights_like_reference(model)

    # Warm-up forward to initialize lazy FC
    with torch.no_grad():
        x1, x2, _ = next(iter(val_loader))
        model(x1.to(device), x2.to(device))

    init_weights_like_reference(model)  # re-init after FC materializes

    # -------------------------
    # Optim / schedule
    # -------------------------
    ocfg = cfg["optim"]
    layerwise = _as_layer_hypers(ocfg["layerwise"])
    opt_state = build_optimizer(
        model=model,
        layerwise=layerwise,
        lr_decay=float(ocfg.get("lr_decay", 0.99)),
        momentum_start=float(ocfg.get("momentum_start", 0.5)),
        momentum_ramp_epochs=int(ocfg.get("momentum_ramp_epochs", 50)),
    )

    # -------------------------
    # Metrics + early stop
    # -------------------------
    m = cfg.get("metrics", {})
    metrics_cfg = MetricsConfig(threshold=float(m.get("threshold", 0.5)), compute_auc=bool(m.get("compute_auc", False)))

    es_cfg = cfg.get("train", {}).get("early_stop", {})
    early = EarlyStopper(
        monitor=str(es_cfg.get("monitor", "val_accuracy")),
        mode=str(es_cfg.get("mode", "max")),
        patience=int(es_cfg.get("patience", 20)),
        min_delta=float(es_cfg.get("min_delta", 0.0)),
    ) if bool(es_cfg.get("enabled", True)) else None

    # -------------------------
    # Train loop
    # -------------------------
    epochs = int(cfg.get("train", {}).get("epochs", 200))
    history: Dict[str, List[float]] = {"epoch": []}

    best_metric = float("-inf")
    best_epoch = -1
    best_path = run_dir / "best.pt"

    ckpt_metric = str(cfg.get("train", {}).get("checkpoints", {}).get("metric", "val_accuracy"))

    for epoch in range(epochs):
        step_schedule(opt_state, epoch)
        opt = opt_state.optimizer

        tr_loss, tr_m = train_one_epoch(model, train_loader, device, opt, metrics_cfg)
        va_loss, va_m = eval_one_epoch(model, val_loader, device, metrics_cfg)

        # log
        logger.info(
            "Epoch %03d/%03d | train_loss=%.6f train_acc=%.4f | val_loss=%.6f val_acc=%.4f",
            epoch + 1, epochs, tr_loss, tr_m["accuracy"], va_loss, va_m["accuracy"],
        )

        history["epoch"].append(epoch + 1)
        history.setdefault("train_loss", []).append(tr_loss)
        history.setdefault("val_loss", []).append(va_loss)
        for k, v in tr_m.items():
            history.setdefault(f"train_{k}", []).append(v)
        for k, v in va_m.items():
            history.setdefault(f"val_{k}", []).append(v)

        current = float(va_m.get(ckpt_metric.replace("val_", ""), va_m.get("accuracy", 0.0))) if ckpt_metric.startswith("val_") else float(va_m.get(ckpt_metric, va_m.get("accuracy", 0.0)))
        if current > best_metric:
            best_metric = current
            best_epoch = epoch + 1
            torch.save({"model": model.state_dict(), "epoch": best_epoch, "metric": best_metric}, best_path)

        if early is not None:
            # Map monitor like "val_accuracy" into va_m["accuracy"]
            mon = early.monitor
            val_key = mon.replace("val_", "")
            val_value = va_m.get(val_key, None) if mon.startswith("val_") else va_m.get(mon, None)
            if val_value is None:
                raise KeyError(f"EarlyStop monitor '{mon}' not found in validation metrics")
            if early.update(float(val_value), epoch + 1):
                logger.info("Early stopping at epoch %d (best epoch=%d best=%s=%.6f)", epoch + 1, early.best_epoch, early.monitor, float(early.best_value))
                break

    # -------------------------
    # Save artifacts
    # -------------------------
    save_json(run_dir / "history.json", history)

    merged_cfg_path = run_dir / "config_merged.yaml"
    import yaml
    merged_cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    results = {
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "checkpoint_metric": ckpt_metric,
        "best_checkpoint": str(best_path),
    }
    save_json(run_dir / "results.json", results)

    plot_dir = run_dir / "plots"
    plot_all_metrics(history, plot_dir)

    logger.info("Saved run to: %s", run_dir)
    logger.info("Best epoch=%d | %s=%.6f", best_epoch, ckpt_metric, best_metric)


if __name__ == "__main__":
    main()
