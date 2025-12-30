from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

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
from src.training.optim import build_optimizer_and_scheduler
from src.training.loop import train_one_epoch, eval_one_epoch
from src.training.early_stop import EarlyStopper, EarlyStopConfig
from src.evaluation.plots import plot_loss_curves


def _config_paths(config_dir: Path) -> List[Path]:
    # Load these files in order (later ones can override earlier ones)
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

    log_file = run_dir / "train.log"
    logger = build_logger("train", level=str(cfg.get("logging", {}).get("level", "INFO")), log_file=log_file)

    seed = int(cfg.get("train", {}).get("seed", 42))
    deterministic = bool(cfg.get("train", {}).get("deterministic", False))
    seed_everything(seed, deterministic=deterministic)

    device_str = str(cfg.get("train", {}).get("device", "cuda"))
    device = torch.device(device_str if (device_str == "cpu" or torch.cuda.is_available()) else "cpu")
    logger.info("Device: %s", device)

    # -------------------------
    # Data
    # -------------------------
    paths = cfg.get("paths", {})
    images_root = (project_root / str(paths["images_root"])).resolve()
    pairs_train = (project_root / str(paths["pairs_train"])).resolve()
    pairs_val_str = str(paths.get("pairs_val", "")).strip()

    train_pairs: List[Pair] = parse_pairs_file(pairs_train, images_root=images_root)
    if pairs_val_str:
        val_pairs = parse_pairs_file((project_root / pairs_val_str).resolve(), images_root=images_root)
    else:
        split_cfg = cfg.get("split", {})
        val_ratio = float(split_cfg.get("val_ratio", 0.2))
        target_pos_frac = float(split_cfg.get("target_pos_frac", 0.5))
        min_val_identities = int(split_cfg.get("min_val_identities", 30))
        val_pairs, train_pairs = split_by_components(
            train_pairs,
            val_ratio=val_ratio,
            target_pos_frac=target_pos_frac,
            min_val_identities=min_val_identities,
            seed=seed,
        )

    train_loader, val_loader = build_dataloaders(cfg, train_pairs, val_pairs)

    logger.info("Loaded train pairs: %d", len(train_pairs))
    logger.info("Loaded val pairs: %d", len(val_pairs))

    # -------------------------
    # Model
    # -------------------------
    mcfg = cfg.get("model", {})
    enc = mcfg.get("encoder", {})
    enc_cfg = ConvEmbeddingConfig(
        in_channels=int(enc.get("in_channels", 1)),
        conv_filters=[int(v) for v in enc.get("conv_filters", [64, 128, 256, 256])],
        conv_kernel=int(enc.get("conv_kernel", 10)),
        conv_stride=int(enc.get("conv_stride", 1)),
        pool_kernel=int(enc.get("pool_kernel", 2)),
        pool_stride=int(enc.get("pool_stride", 2)),
        fc_out=int(enc.get("fc_out", 4096)),
    )
    head_cfg = SimilarityHeadConfig(use_bias=bool(mcfg.get("head", {}).get("use_bias", False)))

    model = SiameseNetwork(SiameseConfig(encoder=enc_cfg, head=head_cfg)).to(device)

    # Initialize exactly like the reference, including after the lazy FC materializes
    init_weights_like_reference(model)
    with torch.no_grad():
        x1, x2, _ = next(iter(val_loader))
        _ = model(x1.to(device), x2.to(device))
    init_weights_like_reference(model)

    # -------------------------
    # Optim
    # -------------------------
    optim_cfg = cfg.get("optim", {})
    opt_bundle = build_optimizer_and_scheduler(model, optim_cfg)

    # -------------------------
    # Early stop (BCE only)
    # -------------------------
    early = None
    es_cfg = cfg.get("train", {}).get("early_stop", {})
    if bool(es_cfg.get("enabled", True)):
        early = EarlyStopper(
            EarlyStopConfig(
                monitor="val_loss",
                mode="min",
                patience=int(es_cfg.get("patience", 20)),
                min_delta=float(es_cfg.get("min_delta", 0.0)),
            )
        )

    # Save best model by val_loss
    best_path = run_dir / "best.pt"
    best_val = float("inf")
    best_epoch = 0

    max_epochs = int(cfg.get("train", {}).get("max_epochs", 200))

    history: Dict[str, List[float]] = {"epoch": [], "train_loss": [], "val_loss": []}

    for epoch in range(max_epochs):
        tr_loss = train_one_epoch(model, train_loader, device, opt_bundle.optimizer)
        va_loss = eval_one_epoch(model, val_loader, device)

        # step scheduler once per epoch
        opt_bundle.scheduler.step()

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(float(tr_loss))
        history["val_loss"].append(float(va_loss))

        logger.info(
            "Epoch %03d/%03d | train_loss=%.6f | val_loss=%.6f",
            epoch + 1,
            max_epochs,
            tr_loss,
            va_loss,
        )

        # best checkpoint by val_loss
        if va_loss < best_val:
            best_val = float(va_loss)
            best_epoch = epoch + 1
            torch.save({"model": model.state_dict(), "epoch": best_epoch, "val_loss": best_val}, best_path)

        if early is not None and early.update(float(va_loss), epoch + 1):
            logger.info(
                "Early stopping at epoch %d (best epoch=%d best_val_loss=%.6f)",
                epoch + 1,
                early.best_epoch,
                float(early.best_value),
            )
            break

    # -------------------------
    # Save artifacts
    # -------------------------
    save_json(run_dir / "history.json", history)

    # merged config snapshot
    try:
        import yaml
        (run_dir / "config_merged.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    except Exception:
        pass

    results = {"best_epoch": best_epoch, "best_val_loss": best_val}
    save_json(run_dir / "results.json", results)

    # plots
    plots_dir = run_dir / "plots"
    plot_loss_curves(history, plots_dir)

    logger.info("Done. Best epoch=%d | best_val_loss=%.6f", best_epoch, best_val)


if __name__ == "__main__":
    main()
