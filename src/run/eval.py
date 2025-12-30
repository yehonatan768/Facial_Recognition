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
from src.data.datasets import PairPathDataset, Pair
from src.data.transforms import build_transform
from src.model.cnn_embedder import ConvEmbeddingConfig
from src.model.head import SimilarityHeadConfig
from src.model.siamese import SiameseConfig, SiameseNetwork
from src.model.init import init_weights_like_reference
from src.training.optim import build_optimizer_and_scheduler
from src.training.loop import train_one_epoch, eval_one_epoch


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
    run_dir = out_root / exp_name / "eval"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_file = run_dir / "eval.log"
    logger = build_logger("eval", level=str(cfg.get("logging", {}).get("level", "INFO")), log_file=log_file)

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
    pairs_test = (project_root / str(paths["pairs_test"])).resolve()

    train_pairs: List[Pair] = parse_pairs_file(pairs_train, images_root=images_root)
    test_pairs: List[Pair] = parse_pairs_file(pairs_test, images_root=images_root)

    transform = build_transform(cfg)
    bs = int(cfg.get("optim", {}).get("batch_size", 128))

    train_ds = PairPathDataset(train_pairs, transform=transform)
    test_ds = PairPathDataset(test_pairs, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)

    logger.info("Loaded train pairs: %d", len(train_pairs))
    logger.info("Loaded test pairs:  %d", len(test_pairs))

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

    init_weights_like_reference(model)
    with torch.no_grad():
        x1, x2, _ = next(iter(test_loader))
        _ = model(x1.to(device), x2.to(device))
    init_weights_like_reference(model)

    # -------------------------
    # Train on all train pairs then evaluate on test (BCE only)
    # -------------------------
    optim_cfg = cfg.get("optim", {})
    opt_bundle = build_optimizer_and_scheduler(model, optim_cfg)
    epochs = int(cfg.get("train", {}).get("max_epochs", 200))

    for epoch in range(epochs):
        tr_loss = train_one_epoch(model, train_loader, device, opt_bundle.optimizer)
        opt_bundle.scheduler.step()
        logger.info("Epoch %03d/%03d | train_loss=%.6f", epoch + 1, epochs, tr_loss)

    test_loss = eval_one_epoch(model, test_loader, device)
    logger.info("Test BCE loss: %.6f", test_loss)

    save_json(run_dir / "results.json", {"test_loss": float(test_loss)})


if __name__ == "__main__":
    main()
