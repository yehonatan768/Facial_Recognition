from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from src.utils.read_config import load_config_files, resolve_project_root
from src.utils.logger import build_logger
from src.utils.seed import seed_everything
from src.utils.io import save_json

from src.data.load_pairs import parse_pairs_file
from src.data.datasets import PairPathDataset, Pair
from src.data.transforms import build_transform

from src.model.cnn_embedder import ConvEmbeddingConfig, ConvEmbeddingNet
from src.model.head import SimilarityHeadConfig, WeightedL1Head
from src.model.siamese import SiameseNet
from src.model.init import init_weights_like_reference

from src.training.loop import eval_one_epoch


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


def _infer_hw_from_cfg(cfg: dict) -> Tuple[int, int]:
    tf = cfg.get("transform", {}) or {}
    size = int(tf.get("size", 105))
    return size, size


def _pick_checkpoint(run_dir: Path, ckpt_arg: Optional[str]) -> Path:
    """
    Priority:
      1) --ckpt if provided
      2) run_dir/final.pt if exists
      3) run_dir/best.pt if exists
    """
    if ckpt_arg:
        p = Path(ckpt_arg).expanduser()
        if not p.is_absolute():
            p = (run_dir / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p

    final_p = run_dir / "final.pt"
    if final_p.exists():
        return final_p

    best_p = run_dir / "best.pt"
    if best_p.exists():
        return best_p

    raise FileNotFoundError(f"No checkpoint found in {run_dir} (expected final.pt or best.pt).")


def _load_state_dict(ckpt_path: Path) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    if isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
        # Might be a raw state_dict
        return ckpt
    raise ValueError(f"Unsupported checkpoint format at {ckpt_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a saved model checkpoint on the test set.")
    ap.add_argument("--config_dir", type=str, default="src/config")
    ap.add_argument("--project_root", type=str, default=None)
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="If provided, overrides experiment.name for selecting outputs/<exp_name>/.",
    )
    ap.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Checkpoint to evaluate (absolute path or relative to outputs/<exp_name>/).",
    )
    ap.add_argument("--batch_size", type=int, default=None, help="Override batch size for evaluation.")
    args = ap.parse_args()

    config_dir = Path(args.config_dir).resolve()
    cfg = load_config_files(_config_paths(config_dir))

    project_root = resolve_project_root(cfg, args.project_root)
    out_root = Path(args.outdir).expanduser().resolve()

    exp_name = args.exp_name or str(cfg.get("experiment", {}).get("name", "run"))
    run_dir = out_root / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = build_logger(cfg, run_dir)

    # Seed
    seed = int(cfg.get("train", {}).get("seed", 42))
    deterministic = bool(cfg.get("train", {}).get("deterministic", False))
    seed_everything(seed, deterministic=deterministic)

    # Device
    device_str = str(cfg.get("train", {}).get("device", "cuda"))
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # Paths
    paths = cfg["paths"]
    images_root = (project_root / str(paths["images_root"])).resolve()
    pairs_test = (project_root / str(paths["pairs_test"])).resolve()

    test_pairs: List[Pair] = parse_pairs_file(pairs_test, images_root=images_root)

    # Transform (eval)
    eval_transform = build_transform(cfg, train=False)

    # DataLoader
    optim_cfg = cfg.get("optim", {}) or {}
    bs = int(args.batch_size) if args.batch_size is not None else int(optim_cfg.get("batch_size", 128))
    test_ds = PairPathDataset(test_pairs, transform=eval_transform)
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=int((cfg.get("data", {}) or {}).get("num_workers", 0)),
        pin_memory=bool((cfg.get("data", {}) or {}).get("pin_memory", False)),
    )

    logger.info("Project root: %s", project_root)
    logger.info("Run dir:      %s", run_dir)
    logger.info("Device:       %s", device)
    logger.info("Loaded test pairs: %d", len(test_pairs))

    # Build model
    mcfg = cfg.get("model", {}) or {}
    enc = mcfg.get("encoder", {}) or {}
    head_cfg_dict = mcfg.get("head", {}) or {}

    enc_cfg = ConvEmbeddingConfig(**enc)
    _ = SimilarityHeadConfig(**head_cfg_dict)  # config validation (if used)

    encoder = ConvEmbeddingNet(enc_cfg)
    head = WeightedL1Head(dim=int(enc_cfg.fc_out))
    model = SiameseNet(encoder=encoder, head=head).to(device)

    # Materialize lazy FC layers so strict load works
    init_weights_like_reference(model)

    H, W = _infer_hw_from_cfg(cfg)
    C = int(enc_cfg.in_channels)
    with torch.no_grad():
        dummy1 = torch.zeros((1, C, H, W), device=device, dtype=torch.float32)
        dummy2 = torch.zeros((1, C, H, W), device=device, dtype=torch.float32)
        _ = model(dummy1, dummy2)

    ckpt_path = _pick_checkpoint(run_dir, args.ckpt)
    state_dict = _load_state_dict(ckpt_path)
    model.load_state_dict(state_dict, strict=True)

    logger.info("Evaluating checkpoint: %s", ckpt_path)

    # Evaluate
    test_loss, test_acc = eval_one_epoch(model=model, loader=test_loader, device=device)
    logger.info("TEST | loss=%.6f | acc=%.4f", float(test_loss), float(test_acc))

    save_json(
        run_dir / "eval_results.json",
        {"checkpoint": str(ckpt_path), "test_loss": float(test_loss), "test_accuracy": float(test_acc)},
    )


if __name__ == "__main__":
    main()
