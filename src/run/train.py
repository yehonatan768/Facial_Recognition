from __future__ import annotations

import argparse
import json
import os
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
from src.training.loop import train_one_epoch, run_val_and_dump
from src.training.early_stop import EarlyStopping

# Helps fragmentation in some CUDA environments (safe to keep)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["ORT_LOGGING_LEVEL"] = "4"  # 0=verbose ... 4=fatal
os.environ["ORT_LOG_SEVERITY_LEVEL"] = "4"
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"

def _require(cfg: Dict[str, Any], path: str) -> Any:
    cur: Any = cfg
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError(f"Missing required config key: {path}")
        cur = cur[k]
    return cur


def _require_str(cfg: Dict[str, Any], path: str) -> str:
    v = _require(cfg, path)
    if not isinstance(v, str) or not v:
        raise TypeError(f"Config key {path} must be a non-empty string, got {v!r}")
    return v


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_config_path() -> Path:
    return (_project_root() / "src" / "config" / "config.yaml").resolve()


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
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

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _resolve_paths(cfg: Dict[str, Any], args) -> Dict[str, Path]:
    # Optional CLI overrides can stay if you want, but no implicit defaults.
    # If you want *only* config.yaml and no overrides at all, remove args usage here.
    images_root_s = args.images_root or _require_str(cfg, "paths.images_root")
    pairs_train_s = args.pairs_train or _require_str(cfg, "paths.pairs_train")
    pairs_test_s = args.pairs_test or _require_str(cfg, "paths.pairs_test")

    images_root = Path(images_root_s)
    pairs_train = Path(pairs_train_s)
    pairs_test = Path(pairs_test_s)

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
    best_monitor_score: float,
    best_val_acc: float,
    cfg: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "best_monitor_score": float(best_monitor_score),
            "best_val_acc": float(best_val_acc),
            "model_state": model.state_dict(),
            "optim_state": optim.state_dict(),
            "cfg": cfg,
        },
        path,
    )


def _get_monitor_score(monitor: str, val_loss: float, val_acc: float) -> float:
    if monitor == "val_acc":
        return float(val_acc)
    if monitor == "val_loss":
        return float(val_loss)
    raise ValueError(f"Unsupported early_stop.monitor={monitor!r}. Use 'val_loss' or 'val_acc'.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="", help="Optional. If omitted, uses src/config/config.yaml")
    ap.add_argument("--defaults", type=str, default="", help="Optional defaults YAML to merge under config.yaml")
    ap.add_argument("--workdir", type=str, default=str(_project_root() / "outputs"), help="Where to write checkpoints/logs.")
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

    seed = int(_require(cfg, "train.seed"))
    _seed_everything(seed)

    # Reset metrics file on fresh run
    metrics_path = workdir / "metrics.jsonl"
    if not args.resume and metrics_path.exists():
        metrics_path.unlink()
        logger.info(f"Cleared previous metrics file: {metrics_path}")

    paths = _resolve_paths(cfg, args)
    images_root = paths["images_root"]
    pairs_train_path = paths["pairs_train"]
    pairs_test_path = paths["pairs_test"]

    ext = str(_require(cfg, "data.image_ext"))
    strict_exists = bool(_require(cfg, "data.strict_exists"))

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

    split_res = split_by_components(
        pairs=train_pairs_all,
        val_ratio=float(_require(cfg, "split.val_ratio")),
        target_pos_frac=float(_require(cfg, "split.target_pos_frac")),
        min_val_identities=int(_require(cfg, "split.min_val_identities")),
        seed=int(_require(cfg, "split.seed")),
        logger=logger,
    )

    train_pairs = split_res.train_pairs
    val_pairs = split_res.val_pairs

    loaders = build_pair_loaders(cfg=cfg, train_pairs=train_pairs, val_pairs=val_pairs, test_pairs=test_pairs)
    train_loader = loaders.train_loader
    val_loader = loaders.val_loader

    model_cfg = cfg.get("model", {}) or {}
    model = PaperSiameseModel(
        in_channels=int(model_cfg.get("in_channels", 1)),
        enforce_105=bool(model_cfg.get("enforce_105", True)),
        embedding_dim=int(model_cfg.get("embedding_dim", 4096)),
    ).to(device)

    init_weights_like_paper(model)

    optim_bundle = build_optimizer_and_scheduler(model=model, cfg=cfg)
    optimizer = optim_bundle.optimizer
    scheduler = optim_bundle.scheduler

    start_epoch = 1

    es_cfg = cfg.get("early_stop", {}) or {}
    monitor = str(es_cfg.get("monitor", "val_acc"))
    mode = str(es_cfg.get("mode", "max"))
    maximize = mode.lower() == "max"
    best_monitor_score = -float("inf") if maximize else float("inf")
    best_val_acc = -float("inf")  # ALWAYS maximize accuracy for best checkpoint

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state"], strict=True)
        optimizer.load_state_dict(ckpt["optim_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_monitor_score = float(ckpt.get("best_monitor_score", best_monitor_score))
        best_val_acc = float(ckpt.get("best_val_acc", best_val_acc))

        logger.info(
            f"Resumed from {args.resume} (start_epoch={start_epoch}, "
            f"best_monitor_score={best_monitor_score:.6f}, best_val_acc={best_val_acc:.6f})"
        )

    es_enabled = bool(es_cfg.get("enabled", True))
    patience = int(es_cfg.get("patience", 20))
    min_delta = float(es_cfg.get("min_delta", 0.0))
    early = EarlyStopping(patience=patience, min_delta=min_delta, maximize=maximize)

    epochs = int(_require(cfg, "train.epochs"))
    dump_every = int(_require(cfg, "train.dump_every"))

    ckpt_dir = workdir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_best = ckpt_dir / "best.pt"
    ckpt_last = ckpt_dir / "last.pt"

    logger.info(
        f"Train pairs: {len(train_pairs)} | Val pairs: {len(val_pairs)} | Test pairs: {len(test_pairs) if test_pairs else 0}"
    )
    logger.info(
        f"epochs={epochs} batch_size={cfg.get('train', {}).get('batch_size', '??')} monitor={monitor} mode={mode} early_stop={es_enabled}"
    )

    for epoch in range(start_epoch, epochs + 1):
        sched_info: Dict[str, float] = {}
        if scheduler is not None:
            sched_info = scheduler.step(epoch - 1)

        tr_stats = train_one_epoch(model=model, loader=train_loader, device=device, optimizer=optimizer)

        do_dump = (epoch == start_epoch) or (dump_every > 0 and (epoch % dump_every == 0))

        va_stats, dump_path = run_val_and_dump(
            model=model,
            loader=val_loader,
            device=device,
            epoch=epoch,
            workdir=workdir,
            dump=do_dump,
        )

        lr = float(sched_info.get("lr", optimizer.param_groups[0].get("lr", 0.0)))
        mom = float(sched_info.get("momentum", optimizer.param_groups[0].get("momentum", 0.0)))

        monitor_score = _get_monitor_score(
            monitor,
            val_loss=float(va_stats.loss),
            val_acc=float(va_stats.acc),
        )

        logger.info(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"train_loss={tr_stats.loss:.6f} val_loss={va_stats.loss:.6f} | "
            f"train_acc={tr_stats.acc:.4f} val_acc={va_stats.acc:.4f} | "
            f"lr={lr:.8f} momentum={mom:.3f}"
            + (f" | val_dump={dump_path}" if dump_path is not None else "")
        )

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

        # Always save last
        _save_ckpt(ckpt_last, model, optimizer, epoch, best_monitor_score, best_val_acc, cfg)

        # Save BEST checkpoint by val_acc only
        cur_val_acc = float(va_stats.acc)
        if cur_val_acc > best_val_acc:
            best_val_acc = cur_val_acc
            _save_ckpt(ckpt_best, model, optimizer, epoch, best_monitor_score, best_val_acc, cfg)

        # Update best_monitor_score for logging (early stop uses monitor_score)
        improved_monitor = (monitor_score > best_monitor_score) if maximize else (monitor_score < best_monitor_score)
        if improved_monitor:
            best_monitor_score = monitor_score

        if es_enabled and early.update(epoch=epoch, score=monitor_score).should_stop:
            logger.info(
                f"Early stop at epoch {epoch} "
                f"(best_monitor_score={best_monitor_score:.6f}, best_val_acc={best_val_acc:.6f})"
            )
            break

        logger.info(f"Training complete. Best checkpoint (by val_acc): {ckpt_best}")

if __name__ == "__main__":
    main()
