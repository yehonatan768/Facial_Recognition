from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Optional

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

from src.evaluation.verification import find_best_threshold
from src.evaluation.one_shot import evaluate_one_shot


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
    images_root = Path(args.images_root or paths.get("images_root", ""))
    pairs_train = Path(args.pairs_train or paths.get("pairs_train", ""))
    pairs_test = Path(args.pairs_test or paths.get("pairs_test", ""))

    missing = []
    if not str(images_root):
        missing.append("images_root")
    if not str(pairs_train):
        missing.append("pairs_train")
    if not str(pairs_test):
        missing.append("pairs_test")
    if missing:
        raise ValueError(
            f"Missing required paths: {missing}. Provide in config under paths: "
            f"or via CLI flags --images-root/--pairs-train/--pairs-test."
        )

    return {
        "images_root": images_root,
        "pairs_train": pairs_train,
        "pairs_test": pairs_test,
    }


def _save_ckpt(path: Path, model: torch.nn.Module, optim: torch.optim.Optimizer, epoch: int, best: float, cfg: Dict[str, Any]) -> None:
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
    ap.add_argument("--config", type=str, required=True, help="Your run config YAML (can be minimal).")
    ap.add_argument("--defaults", type=str, default="", help="Optional defaults YAML (paper defaults).")
    ap.add_argument("--workdir", type=str, default="outputs", help="Where to write checkpoints/logs.")
    ap.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|cuda:0")
    ap.add_argument("--resume", type=str, default="", help="Optional checkpoint to resume.")
    ap.add_argument("--images-root", type=str, default="", help="Override cfg.paths.images_root")
    ap.add_argument("--pairs-train", type=str, default="", help="Override cfg.paths.pairs_train")
    ap.add_argument("--pairs-test", type=str, default="", help="Override cfg.paths.pairs_test")
    args = ap.parse_args()

    # Load cfg (defaults first, then config overrides)
    cfg_user = read_model_config(args.config)
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

    # Load paper-style pairs (train/test) using your loader
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

    # Split train into train/val without identity leakage (your graph split)
    split_cfg = cfg.get("split", {})
    split_res = split_by_components(
        pairs=train_pairs_all,
        val_ratio=float(split_cfg.get("val_ratio", 0.2)),
        min_val_identities=int(split_cfg.get("min_val_identities", 20)),
        seed=int(split_cfg.get("seed", seed)),
    )
    train_pairs = split_res.train_pairs
    val_pairs = split_res.val_pairs

    # DataLoaders (uses PaperImageTransform through your datasets.py)
    loaders = build_pair_loaders(cfg=cfg, train_pairs=train_pairs, val_pairs=val_pairs, test_pairs=test_pairs)
    train_loader = loaders.train_loader
    val_loader = loaders.val_loader
    test_loader = loaders.test_loader  # optional

    # Model
    model_cfg = cfg.get("model", {})
    model = PaperSiameseModel(
        in_channels=int(model_cfg.get("in_channels", 1)),
        enforce_105=bool(model_cfg.get("enforce_105", True)),
        embedding_dim=int(model_cfg.get("embedding_dim", 4096)),
    ).to(device)

    # Paper initialization (conv + fc + alpha as fc)
    init_weights_like_paper(model)

    # Optim + scheduler (paper lr decay + momentum ramp, potentially layer-wise)
    optim_bundle = build_optimizer_and_scheduler(model=model, cfg=cfg)
    optimizer = optim_bundle.optimizer
    scheduler = optim_bundle.scheduler

    # Resume if provided
    start_epoch = 1
    best_score = float("inf")  # we minimize one-shot error
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state"], strict=True)
        optimizer.load_state_dict(ckpt["optim_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_score = float(ckpt.get("best_score", best_score))
        logger.info(f"Resumed from {args.resume} (start_epoch={start_epoch}, best_score={best_score:.6f})")

    # Early stopping: paper uses patience 20 on one-shot validation error
    es_cfg = cfg.get("early_stop", {})
    patience = int(es_cfg.get("patience", 20))
    early = EarlyStopping(patience=patience, min_delta=float(es_cfg.get("min_delta", 0.0)), maximize=False)

    epochs = int(cfg.get("train", {}).get("epochs", 200))
    metrics_path = workdir / "metrics.jsonl"
    ckpt_dir = workdir / "checkpoints"
    ckpt_best = ckpt_dir / "best.pt"
    ckpt_last = ckpt_dir / "last.pt"

    # One-shot settings
    os_cfg = cfg.get("one_shot", {})
    oneshot_enabled = bool(os_cfg.get("enabled", True))
    n_way = int(os_cfg.get("n_way", 20))
    val_trials = int(os_cfg.get("val_trials", 320))
    seed_base = int(os_cfg.get("seed", seed))

    logger.info(f"Train pairs: {len(train_pairs)} | Val pairs: {len(val_pairs)} | Test pairs: {len(test_pairs) if test_pairs else 0}")
    logger.info(f"epochs={epochs} batch_size={cfg.get('train', {}).get('batch_size', '??')} oneshot_enabled={oneshot_enabled}")

    for epoch in range(start_epoch, epochs + 1):
        # Scheduler is epoch-based (paper)
        if scheduler is not None:
            scheduler.step(epoch - 1)

        tr_stats = train_one_epoch(model=model, loader=train_loader, device=device, optimizer=optimizer)
        va_stats, va_probs, va_labels = eval_one_epoch(model=model, loader=val_loader, device=device)

        # Verification best threshold on VAL
        thr, ver_acc = find_best_threshold(va_probs, va_labels)

        # One-shot validation (paper monitors error)
        oneshot_acc = float("nan")
        oneshot_err = float("nan")
        if oneshot_enabled:
            os = evaluate_one_shot(
                model=model,
                cfg=cfg,
                images_root=images_root,
                device=device,
                n_way=n_way,
                n_trials=val_trials,
                seed=seed_base + epoch,  # change per epoch => new random tasks
                ext=ext,
            )
            oneshot_acc = float(os.accuracy)
            oneshot_err = 1.0 - oneshot_acc

        # Log line (your preferred fields)
        lr0 = float(optimizer.param_groups[0]["lr"])
        m0 = float(optimizer.param_groups[0].get("momentum", 0.0))
        logger.info(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"train_acc={tr_stats.acc:.4f} train_loss={tr_stats.loss:.6f} | "
            f"val_acc={va_stats.acc:.4f} val_loss={va_stats.loss:.6f} | "
            f"thr={thr:.3f} oneshot_acc={oneshot_acc:.4f} oneshot_err={oneshot_err:.4f} | "
            f"lr0={lr0:.8f m0={m0:.3f}}".replace("m0=", "| m0=")  # keep formatting stable
        )

        row = {
            "epoch": epoch,
            "train_loss": float(tr_stats.loss),
            "train_acc": float(tr_stats.acc),
            "val_loss": float(va_stats.loss),
            "val_acc": float(va_stats.acc),
            "verif_thr": float(thr),
            "verif_acc_best": float(ver_acc),
            "oneshot_acc": float(oneshot_acc),
            "oneshot_err": float(oneshot_err),
            "lr0": lr0,
            "m0": m0,
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        # Save last
        _save_ckpt(ckpt_last, model, optimizer, epoch, best_score, cfg)

        # Paper: select best epoch by lowest one-shot validation error
        score = oneshot_err if oneshot_enabled else float(va_stats.loss)
        improved = score < best_score
        if improved:
            best_score = score
            _save_ckpt(ckpt_best, model, optimizer, epoch, best_score, cfg)

        # Early stop
        if early.update(epoch=epoch, score=score).should_stop:
            logger.info(f"Early stop at epoch {epoch} (best_score={best_score:.6f})")
            break

    logger.info(f"Training complete. Best checkpoint: {ckpt_best}")


if __name__ == "__main__":
    main()
