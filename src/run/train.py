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
from src.model.cnn_embedder import ConvEmbeddingConfig
from src.model.head import SimilarityHeadConfig
from src.model.siamese import SiameseConfig, SiameseNet
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
        val_pairs, train_pairs = split_by_components(
            train_pairs,
            val_ratio=float(split_cfg.get("val_ratio", 0.2)),
            target_pos_frac=float(split_cfg.get("target_pos_frac", 0.5)),
            min_val_identities=int(split_cfg.get("min_val_identities", 30)),
            seed=seed,
        )

    train_loader, val_loader = build_dataloaders(cfg, train_pairs, val_pairs)
    logger.info("Loaded train pairs: %d", len(train_pairs))
    logger.info("Loaded val pairs: %d", len(val_pairs))

    # -------------------------
    # Model (logits output)
    # -------------------------
    enc_cfg = ConvEmbeddingConfig(**cfg["model"]["encoder"])
    head_cfg = SimilarityHeadConfig(**cfg["model"]["head"])

    # SiameseNet expects encoder/head MODULES (your siamese.py)
    from src.model.cnn_embedder import ConvEmbeddingNet
    from src.model.head import WeightedL1Head

    encoder = ConvEmbeddingNet(enc_cfg)
    head = WeightedL1Head(dim=int(enc_cfg.fc_out))
    model = SiameseNet(encoder=encoder, head=head).to(device)

    # Lazy-FC materialization + init (use TRAIN batch, not val)
    init_weights_like_reference(model)
    with torch.no_grad():
        x1, x2, _ = next(iter(train_loader))
        _ = model(x1.to(device), x2.to(device))
    init_weights_like_reference(model)

    # -------------------------
    # Optim (SGD + exp lr decay + linear momentum ramp)
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
    # Early stop on val_loss
    # -------------------------
    es_cfg = cfg.get("train", {}).get("early_stop", {})
    early = EarlyStopper(
        monitor="val_loss",
        mode="min",
        patience=int(es_cfg.get("patience", 20)),
        min_delta=float(es_cfg.get("min_delta", 0.0)),
    )

    best_path = run_dir / "best.pt"
    best_val = float("inf")
    best_epoch = 0

    max_epochs = int(cfg.get("train", {}).get("max_epochs", 200))
    history: Dict[str, List[float]] = {"epoch": [], "train_loss": [], "val_loss": []}

    for epoch in range(max_epochs):
        # IMPORTANT: your loop.py signature is (model, loader, optimizer, device)
        tr_loss = train_one_epoch(model, train_loader, state.optimizer, device)
        va_loss = eval_one_epoch(model, val_loader, device)

        # schedule step (epoch index)
        step_schedule(state, epoch)

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
        if float(va_loss) < best_val:
            best_val = float(va_loss)
            best_epoch = epoch + 1
            torch.save({"model": model.state_dict(), "epoch": best_epoch, "val_loss": best_val}, best_path)

        if early.update(float(va_loss), epoch + 1):
            logger.info(
                "Early stopping at epoch %d (best epoch=%d best_val_loss=%.6f)",
                epoch + 1,
                early.best_epoch,
                float(early.best_value),
            )
            break

    save_json(run_dir / "history.json", history)
    save_json(run_dir / "results.json", {"best_epoch": best_epoch, "best_val_loss": best_val})

    plot_loss_curves(history, run_dir / "plots")
    logger.info("Done. Best epoch=%d | best_val_loss=%.6f", best_epoch, best_val)


if __name__ == "__main__":
    main()
