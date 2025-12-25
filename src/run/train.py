from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from src.utils.read_config import read_model_config
from src.utils.logger import setup_logger

from src.data.load_data import load_train_test_pairs
from src.data.graph_split import split_by_components
from src.data.datasets import build_pair_loaders

from src.models.siamese import PaperSiameseModel
from src.models.init import init_weights_like_paper
from src.training.optim import build_optimizer_and_scheduler
from src.training.loop import train_one_epoch, eval_one_epoch
from src.training.early_stop import EarlyStopping


def _project_root() -> Path:
    # src/run/train.py -> parents[2] == project root
    return Path(__file__).resolve().parents[2]


def _default_config_path() -> Path:
    return Path("src/config/config.yaml")


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursive dict merge: override wins."""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _resolve_paths(cfg: Dict[str, Any], args) -> Dict[str, Path]:
    """
    Resolve required paths either from cfg["paths"] or CLI.
    """
    paths = cfg.get("paths", {})
    images_root = Path(args.images_root or paths.get("images_root", "images"))
    pairs_train = Path(args.pairs_train or paths.get("pairs_train", "assets/pairsDevTrain.txt"))
    pairs_test = Path(args.pairs_test or paths.get("pairs_test", "assets/pairsDevTest.txt"))

    # Interpret relative paths as project-root-relative
    if not images_root.is_absolute():
        images_root = (_project_root() / images_root).resolve()
    if not pairs_train.is_absolute():
        pairs_train = (_project_root() / pairs_train).resolve()
    if not pairs_test.is_absolute():
        pairs_test = (_project_root() / pairs_test).resolve()

    return {"images_root": images_root, "pairs_train": pairs_train, "pairs_test": pairs_test}


def _save_ckpt(
    path: Path,
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    epoch: int,
    best: float,
    cfg: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "best_score": float(best),
            "model_state": model.state_dict(),
            "optim_state": optim.state_dict(),
            "cfg": cfg,
        },
        path,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="", help="Optional. If omitted, uses src/config/config.yaml")
    ap.add_argument("--defaults", type=str, default="", help="Optional defaults YAML (paper defaults).")
    ap.add_argument("--workdir", type=str, default="outputs", help="Where to write checkpoints/logs.")
    ap.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|cuda:0")
    ap.add_argument("--resume", type=str, default="", help="Optional checkpoint to resume.")
    ap.add_argument("--images-root", type=str, default="", help="Override cfg.paths.images_root")
    ap.add_argument("--pairs-train", type=str, default="", help="Override cfg.paths.pairs_train")
    ap.add_argument("--pairs-test", type=str, default="", help="Override cfg.paths.pairs_test")
    args = ap.parse_args()

    cfg_path = Path(args.config) if args.config else _default_config_path()
    cfg_user = read_model_config(cfg_path)

    cfg = cfg_user
    if args.defaults:
        cfg_defaults = read_model_config(args.defaults)
        cfg = _deep_merge(cfg_defaults, cfg_user)

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(name="train", log_file=workdir / "train.log")

    device = _get_device(args.device)
    logger.info(f"Device: {device}")

    seed = int(cfg.get("train", {}).get("seed", 0))
    _seed_everything(seed)

    paths = _resolve_paths(cfg, args)
    images_root = paths["images_root"]
    pairs_train_path = paths["pairs_train"]
    pairs_test_path = paths["pairs_test"]

    ext = str(cfg.get("data", {}).get("image_ext", cfg.get("data", {}).get("ext", ".jpg")))
    strict_exists = bool(cfg.get("data", {}).get("strict_exists", True))

    loaded = load_train_test_pairs(
        train_pairs_path=pairs_train_path,
        test_pairs_path=pairs_test_path,
        images_root=images_root,
        ext=ext,
        logger=logger,
        strict_exists=strict_exists,
    )

    train_pairs_all = loaded["train_pairs"]
    test_pairs = loaded["test_pairs"]

    if len(train_pairs_all) == 0:
        raise RuntimeError(
            "No training pairs were loaded (all pairs filtered out as missing).\n"
            f"images_root={images_root}\n"
            f"pairs_train={pairs_train_path}\n"
            f"pairs_test={pairs_test_path}\n"
            f"ext={ext!r} strict_exists={strict_exists}\n"
        )

    # Split train into train/val without identity leakage (graph split)
    split_cfg = cfg.get("split", {})
    split_res = split_by_components(
        pairs=train_pairs_all,
        val_ratio=float(split_cfg.get("val_ratio", 0.5)),
        min_val_identities=int(split_cfg.get("min_val_identities", 150)),
        seed=int(split_cfg.get("seed", seed)),
    )
    train_pairs = split_res.train_pairs
    val_pairs = split_res.val_pairs

    # DataLoaders
    loaders = build_pair_loaders(cfg=cfg, train_pairs=train_pairs, val_pairs=val_pairs, test_pairs=test_pairs)
    train_loader = loaders.train_loader
    val_loader = loaders.val_loader

    # Model
    model_cfg = cfg.get("model", {})
    model = PaperSiameseModel(
        in_channels=int(model_cfg.get("in_channels", 1)),
        enforce_105=bool(model_cfg.get("enforce_105", True)),
        embedding_dim=int(model_cfg.get("embedding_dim", 4096)),
    ).to(device)

    # Paper init (your existing implementation)
    init_weights_like_paper(model)

    # Optim + scheduler
    optim_bundle = build_optimizer_and_scheduler(model=model, cfg=cfg)
    optimizer = optim_bundle.optimizer
    scheduler = optim_bundle.scheduler

    # Resume if provided
    start_epoch = 1
    best_score = float("inf")  # minimize by default if monitoring val_loss
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state"], strict=True)
        optimizer.load_state_dict(ckpt["optim_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_score = float(ckpt.get("best_score", best_score))
        logger.info(f"Resumed from {args.resume} (start_epoch={start_epoch}, best_score={best_score:.6f})")

    # Early stopping
    es_cfg = cfg.get("early_stop", {}) or {}
    es_enabled = bool(es_cfg.get("enabled", True))
    patience = int(es_cfg.get("patience", 20))
    min_delta = float(es_cfg.get("min_delta", 0.0))
    monitor = str(es_cfg.get("monitor", "val_loss"))  # val_loss | val_acc
    mode = str(es_cfg.get("mode", "min"))  # min | max
    maximize = mode.lower() == "max"

    early = EarlyStopping(patience=patience, min_delta=min_delta, maximize=maximize)

    # Initialize best_score consistent with mode
    if maximize:
        best_score = -float("inf") if best_score == float("inf") else best_score
    else:
        best_score = float("inf") if best_score == -float("inf") else best_score

    epochs = int(cfg.get("train", {}).get("epochs", 200))
    metrics_path = workdir / "metrics.jsonl"
    ckpt_dir = workdir / "checkpoints"
    ckpt_best = ckpt_dir / "best.pt"
    ckpt_last = ckpt_dir / "last.pt"

    logger.info(f"Train pairs: {len(train_pairs)} | Val pairs: {len(val_pairs)} | Test pairs: {len(test_pairs) if test_pairs else 0}")
    logger.info(f"epochs={epochs} batch_size={cfg.get('train', {}).get('batch_size', '??')} monitor={monitor} mode={mode} early_stop={es_enabled}")

    for epoch in range(start_epoch, epochs + 1):
        # Paper schedule is epoch-based; apply before the epoch’s updates.
        sched_info: Dict[str, float] = {}
        if scheduler is not None:
            sched_info = scheduler.step(epoch - 1)

        tr_stats = train_one_epoch(model=model, loader=train_loader, device=device, optimizer=optimizer)
        va_stats, _, _ = eval_one_epoch(model=model, loader=val_loader, device=device)

        # Scalar lr/momentum (no lists)
        lr = float(sched_info.get("lr", optimizer.param_groups[0].get("lr", 0.0)))
        mom = float(sched_info.get("momentum", optimizer.param_groups[0].get("momentum", 0.0)))

        # Select early-stop / best checkpoint score
        if monitor == "val_acc":
            score = float(va_stats.acc)
        else:
            score = float(va_stats.loss)  # default

        logger.info(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"train_loss={tr_stats.loss:.6f} train_acc={tr_stats.acc:.4f} | "
            f"val_loss={va_stats.loss:.6f} val_acc={va_stats.acc:.4f} | "
            f"lr={lr:.8f} momentum={mom:.3f}"
        )

        # Save metrics for plotting
        row = {
            "epoch": int(epoch),
            "train_loss": float(tr_stats.loss),
            "train_acc": float(tr_stats.acc),
            "val_loss": float(va_stats.loss),
            "val_acc": float(va_stats.acc),
            "lr": float(lr),
            "momentum": float(mom),
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        # Save last
        _save_ckpt(ckpt_last, model, optimizer, epoch, best_score, cfg)

        # Best model selection
        improved = (score > best_score) if maximize else (score < best_score)
        if improved:
            best_score = score
            _save_ckpt(ckpt_best, model, optimizer, epoch, best_score, cfg)

        # Early stop
        if es_enabled and early.update(epoch=epoch, score=score).should_stop:
            logger.info(f"Early stop at epoch {epoch} (best_score={best_score:.6f})")
            break

    logger.info(f"Training complete. Best checkpoint: {ckpt_best}")


if __name__ == "__main__":
    main()
