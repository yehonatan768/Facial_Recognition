from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch

from src.utils.read_config import load_config_files, resolve_project_root
from src.utils.logger import build_logger
from src.utils.seed import seed_everything
from src.utils.io import save_json

from src.data.load_pairs import parse_pairs_file
from src.data.graph_split import split_by_components
from src.data.datasets import build_dataloaders, Pair
from src.data.transforms import build_transform  # NEW: train/eval transform split

from src.model.cnn_embedder import ConvEmbeddingConfig, ConvEmbeddingNet
from src.model.head import SimilarityHeadConfig, WeightedL1Head
from src.model.siamese import SiameseNet
from src.model.init import init_weights_like_reference

from src.training.optim import build_optimizer, step_schedule
from src.training.loop import train_one_epoch, eval_one_epoch
from src.training.early_stop import EarlyStopper

from src.evaluation.plots import plot_loss_curves


def _config_paths(config_dir: Path) -> List[Path]:
    names = [
        "paths.yaml",
        "model.yaml",
        "optim.yaml",
        "train.yaml",
        "transform.yaml",
        "logging.yaml",
        "experiment.yaml",
    ]
    return [config_dir / n for n in names if (config_dir / n).exists()]


def _infer_hw_from_cfg(cfg: dict) -> tuple[int, int]:
    """
    Try to infer the model input H,W from config.
    Falls back to 105x105 (common for LFW Siamese setups) if not found.
    """
    tcfg = cfg.get("transform", {}) or {}
    for k in ("size", "image_size", "input_size", "resize"):
        v = tcfg.get(k, None)
        if isinstance(v, int) and v > 0:
            return int(v), int(v)
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return int(v[0]), int(v[1])
    return 105, 105


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_dir", type=str, default="src/config")
    ap.add_argument("--project_root", type=str, default=None)
    ap.add_argument("--outdir", type=str, default="outputs")
    args = ap.parse_args()

    config_dir = Path(args.config_dir).resolve()
    cfg = load_config_files(_config_paths(config_dir))

    project_root = resolve_project_root(cfg, args.project_root)
    out_root = Path(args.outdir).expanduser().resolve()

    exp_name = str(cfg.get("experiment", {}).get("name", "run"))
    run_dir = out_root / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = build_logger(
        "train",
        level=str(cfg.get("logging", {}).get("level", "INFO")),
        log_file=run_dir / "train.log",
    )

    seed = int(cfg.get("train", {}).get("seed", 42))
    deterministic = bool(cfg.get("train", {}).get("deterministic", False))
    seed_everything(seed, deterministic=deterministic)

    device_str = str(cfg.get("train", {}).get("device", "cuda"))
    device = torch.device(device_str if device_str == "cpu" or torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # -------------------------
    # Data: TRAIN + (VAL split from TRAIN unless paths.pairs_val is set)
    # -------------------------
    paths = cfg.get("paths", {})
    images_root = (project_root / str(paths["images_root"])).resolve()
    pairs_train = (project_root / str(paths["pairs_train"])).resolve()
    pairs_val_str = str(paths.get("pairs_val", "")).strip()

    train_pairs: List[Pair] = parse_pairs_file(pairs_train, images_root=images_root)

    if pairs_val_str:
        val_pairs = parse_pairs_file((project_root / pairs_val_str).resolve(), images_root=images_root)
        logger.info("Validation source: pairs_val file=%s", (project_root / pairs_val_str).resolve())
    else:
        split_cfg = cfg.get("split", {})
        res = split_by_components(
            train_pairs,
            val_ratio=float(split_cfg.get("val_ratio", 0.2)),
            target_pos_frac=float(split_cfg.get("target_pos_frac", 0.5)),
            min_val_identities=int(split_cfg.get("min_val_identities", 30)),
            seed=seed,
            logger=logger,
        )
        val_pairs = res.val_pairs
        train_pairs = res.train_pairs
        logger.info("Validation source: split_from_train (train_txt=%s)", pairs_train)

    # -------------------------
    # Transforms: train vs eval (deterministic eval)
    # -------------------------
    train_transform = build_transform(cfg, train=True)
    eval_transform = build_transform(cfg, train=False)

    # -------------------------
    # DataLoaders (must accept the two transforms)
    # -------------------------
    train_loader, val_loader = build_dataloaders(
        cfg,
        train_pairs,
        val_pairs,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )

    logger.info("Loaded train pairs: %d", len(train_pairs))
    logger.info("Loaded val pairs: %d", len(val_pairs))

    # -------------------------
    # Model (logits)
    # -------------------------
    enc_cfg = ConvEmbeddingConfig(**cfg["model"]["encoder"])
    _ = SimilarityHeadConfig(**cfg["model"]["head"])  # kept for config completeness

    encoder = ConvEmbeddingNet(enc_cfg)
    head = WeightedL1Head(dim=int(enc_cfg.fc_out))
    model = SiameseNet(encoder=encoder, head=head).to(device)

    # -------------------------
    # Lazy-FC materialization + init (NO DataLoader read)
    # -------------------------
    init_weights_like_reference(model)

    H, W = _infer_hw_from_cfg(cfg)
    C = int(enc_cfg.in_channels)
    with torch.no_grad():
        dummy1 = torch.zeros((1, C, H, W), device=device, dtype=torch.float32)
        dummy2 = torch.zeros((1, C, H, W), device=device, dtype=torch.float32)
        _ = model(dummy1, dummy2)

    init_weights_like_reference(model)

    # -------------------------
    # Optim
    # -------------------------
    optim_cfg = cfg.get("optim", {})
    state = build_optimizer(
        model=model,
        layerwise=optim_cfg["layerwise"],
        lr_decay=float(optim_cfg.get("lr_decay", 0.99)),
        momentum_start=float(optim_cfg.get("momentum_start", 0.5)),
        momentum_ramp_epochs=int(optim_cfg.get("momentum_ramp_epochs", 20)),
    )

    # -------------------------
    # Early stop (monitor/mode from config)
    # -------------------------
    es_cfg = cfg.get("train", {}).get("early_stop", {}) or {}
    monitor = str(es_cfg.get("monitor", "val_loss"))
    mode = str(es_cfg.get("mode", "min"))
    patience = int(es_cfg.get("patience", 20))
    min_delta = float(es_cfg.get("min_delta", 0.0))

    early = EarlyStopper(
        monitor=monitor,
        mode=mode,
        patience=patience,
        min_delta=min_delta,
    )

    # -------------------------
    # Best checkpoint ALWAYS by val_accuracy
    # -------------------------
    best_path = run_dir / "best.pt"
    best_val_acc = float("-inf")
    best_epoch = 0

    max_epochs = int(cfg.get("train", {}).get("max_epochs", 200))

    history: Dict[str, List[float]] = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "lr": [],
        "momentum": [],
    }

    for epoch in range(max_epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, state.optimizer, device)
        va_loss, va_acc = eval_one_epoch(model, val_loader, device)

        # step schedule once per epoch (updates lr and momentum in param groups)
        step_schedule(state, epoch)

        lr0 = float(state.optimizer.param_groups[0]["lr"])
        m0 = float(state.optimizer.param_groups[0].get("momentum", 0.0))

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(float(tr_loss))
        history["val_loss"].append(float(va_loss))
        history["train_accuracy"].append(float(tr_acc))
        history["val_accuracy"].append(float(va_acc))
        history["lr"].append(lr0)
        history["momentum"].append(m0)

        logger.info(
            "Epoch %03d/%03d | loss: train=%.6f val=%.6f | acc: train=%.4f val=%.4f | lr=%.8f m=%.4f",
            epoch + 1,
            max_epochs,
            tr_loss,
            va_loss,
            tr_acc,
            va_acc,
            lr0,
            m0,
        )

        # Best checkpoint always by highest val_accuracy
        if float(va_acc) > best_val_acc:
            best_val_acc = float(va_acc)
            best_epoch = epoch + 1
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": best_epoch,
                    "val_accuracy": best_val_acc,
                    "val_loss": float(va_loss),
                },
                best_path,
            )

        # Early stop should use the configured monitor
        monitor_map = {
            "train_loss": float(tr_loss),
            "val_loss": float(va_loss),
            "train_accuracy": float(tr_acc),
            "val_accuracy": float(va_acc),
        }
        if monitor not in monitor_map:
            raise ValueError(
                f"Unsupported early_stop.monitor={monitor!r}. "
                f"Choose one of: {sorted(monitor_map.keys())}"
            )

        if early.update(monitor_map[monitor], epoch + 1):
            logger.info(
                "Early stopping at epoch %d (best epoch=%d best %s=%.6f)",
                epoch + 1,
                early.best_epoch,
                monitor,
                float(early.best_value),
            )
            break

    save_json(run_dir / "history.json", history)

    try:
        import yaml

        (run_dir / "config_merged.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    except Exception:
        pass

    save_json(
        run_dir / "results.json",
        {"best_epoch": best_epoch, "best_val_accuracy": best_val_acc},
    )

    plot_loss_curves(history, run_dir / "plots")
    logger.info("Done. Best epoch=%d | best_val_accuracy=%.6f", best_epoch, best_val_acc)


if __name__ == "__main__":
    main()
