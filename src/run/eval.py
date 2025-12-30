from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch

from src.utils.read_config import load_config_files, resolve_project_root
from src.utils.logger import build_logger
from src.utils.seed import seed_everything
from src.utils.io import save_json
from src.data.load_pairs import parse_pairs_file
from src.data.datasets import PairPathDataset
from src.data.transforms import build_transform
from src.model.cnn_embedder import ConvEmbeddingConfig
from src.model.head import SimilarityHeadConfig
from src.model.siamese import SiameseConfig, SiameseNetwork
from src.model.init import init_weights_like_reference
from src.training.optim import LayerHyper, build_optimizer, step_schedule
from src.training.metrics import MetricsConfig, compute_binary_metrics
from torch.utils.data import DataLoader
import torch.nn as nn


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
    ap.add_argument("--project-root", default=None)
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

    cfg = load_config_files([Path(p) for p in args.configs])
    project_root = resolve_project_root(cfg, cli_project_root=args.project_root)

    paths = cfg["paths"]
    outputs_dir = (project_root / paths.get("outputs_dir", "outputs")).resolve()
    run_dir = outputs_dir / (cfg.get("experiment", {}).get("name", "eval") + "_full") / _timestamp()
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = build_logger("eval", level=cfg.get("logging", {}).get("level", "INFO"), log_file=run_dir / "run.log")

    seed = int(cfg.get("train", {}).get("seed", 42))
    seed_everything(seed)

    device_str = str(cfg.get("train", {}).get("device", "cuda"))
    device = torch.device(device_str if (device_str == "cpu" or torch.cuda.is_available()) else "cpu")
    logger.info("Device: %s", device)

    images_root = (project_root / paths["images_root"]).resolve()
    pairs_train = (project_root / paths["pairs_train"]).resolve()
    pairs_val = str(paths.get("pairs_val", "")).strip()
    pairs_val_path = (project_root / pairs_val).resolve() if pairs_val else None
    pairs_test = (project_root / paths["pairs_test"]).resolve()

    train_pairs = parse_pairs_file(pairs_train, images_root=images_root)
    if pairs_val_path is not None and pairs_val_path.exists():
        val_pairs = parse_pairs_file(pairs_val_path, images_root=images_root)
    else:
        val_pairs = []

    all_pairs = train_pairs + val_pairs
    test_pairs = parse_pairs_file(pairs_test, images_root=images_root)

    transform = build_transform(cfg)
    bs = int(cfg.get("optim", {}).get("batch_size", 128))
    train_loader = DataLoader(PairPathDataset(all_pairs, transform=transform), batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(PairPathDataset(test_pairs, transform=transform), batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)

    # Model
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

    # Warm-up forward for lazy FC, then re-init
    with torch.no_grad():
        x1, x2, _ = next(iter(test_loader))
        model(x1.to(device), x2.to(device))
    init_weights_like_reference(model)

    # Optim
    ocfg = cfg["optim"]
    opt_state = build_optimizer(
        model=model,
        layerwise=_as_layer_hypers(ocfg["layerwise"]),
        lr_decay=float(ocfg.get("lr_decay", 0.99)),
        momentum_start=float(ocfg.get("momentum_start", 0.5)),
        momentum_ramp_epochs=int(ocfg.get("momentum_ramp_epochs", 50)),
    )
    opt = opt_state.optimizer
    loss_fn = nn.BCELoss()

    # Metrics cfg
    m = cfg.get("metrics", {})
    metrics_cfg = MetricsConfig(threshold=float(m.get("threshold", 0.5)), compute_auc=bool(m.get("compute_auc", False)))

    # Train on all data
    epochs = int(cfg.get("train", {}).get("epochs", 200))
    for epoch in range(epochs):
        step_schedule(opt_state, epoch)
        model.train(True)
        for x1, x2, y in train_loader:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float().view(-1, 1)
            opt.zero_grad(set_to_none=True)
            p = model(x1, x2)
            loss = loss_fn(p, y)
            loss.backward()
            opt.step()

    # Evaluate on test
    model.train(False)
    all_p = []
    all_y = []
    total_loss = 0.0
    with torch.no_grad():
        for x1, x2, y in test_loader:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float().view(-1, 1)
            p = model(x1, x2)
            loss = loss_fn(p, y)
            total_loss += float(loss.cpu()) * int(y.shape[0])
            all_p.append(p.cpu())
            all_y.append(y.cpu())

    probs = torch.cat(all_p, dim=0)
    targets = torch.cat(all_y, dim=0)
    metrics = compute_binary_metrics(probs=probs, targets=targets, cfg=metrics_cfg)
    avg_loss = total_loss / max(1, int(targets.shape[0]))
    metrics["loss"] = float(avg_loss)

    torch.save({"model": model.state_dict()}, run_dir / "model_final.pt")
    save_json(run_dir / "test_metrics.json", metrics)
    (run_dir / "config_merged.yaml").write_text(__import__("yaml").safe_dump(cfg, sort_keys=False), encoding="utf-8")

    logger.info("Test loss=%.6f | acc=%.4f f1=%.4f", metrics["loss"], metrics["accuracy"], metrics["f1"])
    logger.info("Saved eval run to: %s", run_dir)


if __name__ == "__main__":
    main()
