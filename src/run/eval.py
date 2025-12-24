from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch

from src.utils.read_config import read_model_config
from src.utils.logger import setup_logger

from src.data.load_data import load_train_test_pairs
from src.data.datasets import build_pair_loaders

from src.models.siamese import PaperSiameseModel
from src.training.loop import eval_one_epoch
from src.evaluation.verification import find_best_threshold
from src.evaluation.one_shot import evaluate_one_shot


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _resolve_paths(cfg: Dict[str, Any], args) -> Dict[str, Path]:
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
    return {"images_root": images_root, "pairs_train": pairs_train, "pairs_test": pairs_test}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--defaults", type=str, default="")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out", type=str, default="outputs/eval.json")
    ap.add_argument("--images-root", type=str, default="")
    ap.add_argument("--pairs-train", type=str, default="")
    ap.add_argument("--pairs-test", type=str, default="")
    args = ap.parse_args()

    cfg_user = read_model_config(args.config)
    cfg = cfg_user
    if args.defaults:
        cfg_defaults = read_model_config(args.defaults)
        cfg = _deep_merge(cfg_defaults, cfg_user)

    device = _get_device(args.device)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(name="eval", log_file=out_path.parent / "eval.log")
    logger.info(f"Device: {device}")

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
    test_pairs = loaded["test_pairs"]

    # Build only test loader (train/val not needed here)
    # We can pass dummy empty train/val lists because build_pair_loaders expects them.
    loaders = build_pair_loaders(cfg=cfg, train_pairs=[], val_pairs=[], test_pairs=test_pairs)
    test_loader = loaders.test_loader
    if test_loader is None:
        raise RuntimeError("Failed to build test loader.")

    # Model
    model_cfg = cfg.get("model", {})
    model = PaperSiameseModel(
        in_channels=int(model_cfg.get("in_channels", 1)),
        enforce_105=bool(model_cfg.get("enforce_105", True)),
        embedding_dim=int(model_cfg.get("embedding_dim", 4096)),
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    # Verification evaluation on test pairs
    te_stats, te_probs, te_labels = eval_one_epoch(model=model, loader=test_loader, device=device)
    res = find_best_threshold(te_probs, te_labels)
    if isinstance(res, tuple) and len(res) == 2:
        thr, acc_best = float(res[0]), float(res[1])
    else:
        thr = float(getattr(res, "thr_best", getattr(res, "thr", getattr(res, "threshold", 0.5))))
        acc_best = float(getattr(res, "acc_best", getattr(res, "acc", getattr(res, "accuracy", 0.0))))

    # One-shot test evaluation (paper uses 400)
    os_cfg = cfg.get("one_shot", {})
    oneshot_enabled = bool(os_cfg.get("enabled", True))
    n_way = int(os_cfg.get("n_way", 20))
    test_trials = int(os_cfg.get("test_trials", 400))
    seed = int(os_cfg.get("seed", 0))

    oneshot = None
    if oneshot_enabled:
        os = evaluate_one_shot(
            model=model,
            cfg=cfg,
            images_root=images_root,
            device=device,
            n_way=n_way,
            n_trials=test_trials,
            seed=seed,
            ext=ext,
        )
        oneshot = {"accuracy": float(os.accuracy), "n_trials": int(os.n_trials), "n_way": int(os.n_way)}

    result = {
        "checkpoint": str(Path(args.ckpt).resolve()),
        "device": str(device),
        "verification": {
            "loss": float(te_stats.loss),
            "acc@0.5": float(te_stats.acc),
            "acc_best": float(acc_best),
            "thr_best": float(thr),
            "n_pairs": int(te_probs.numel()),
        },
        "one_shot": oneshot,
    }

    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info(f"Saved evaluation to: {out_path}")
    logger.info(f"Verification acc_best={acc_best:.4f} thr_best={thr:.3f}")
    if oneshot is not None:
        logger.info(f"One-shot acc={oneshot['accuracy']:.4f} (n_way={oneshot['n_way']} trials={oneshot['n_trials']})")


if __name__ == "__main__":
    main()
