from __future__ import annotations

from typing import Any, Dict

from torchvision import transforms

from src.preprocess.background_remove import BackgroundRemover, BgRemoveConfig
from src.preprocess.paper_transforms import PaperImageTransform
from src.preprocess.preprocess import CenterCropMinSide


def _require(cfg: Dict[str, Any], path: str) -> Any:
    cur: Any = cfg
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError(f"Missing required config key: {path}")
        cur = cur[k]
    return cur


def build_transform_from_config(cfg: Dict[str, Any], train: bool) -> transforms.Compose:
    """
    pipeline.mode = "paper":
      PaperImageTransform

    pipeline.mode = "advanced":
      CenterCropMinSide -> (optional BackgroundRemover) -> Grayscale -> Resize -> ToTensor -> Normalize

    NOTE: Advanced currently does background removal only (no face mask).
    """
    mode = str(_require(cfg, "pipeline.mode")).strip().lower()

    mean = float(_require(cfg, "transform.mean"))
    std = float(_require(cfg, "transform.std"))
    input_size = int(_require(cfg, "transform.input_size"))

    if mode == "paper":
        return PaperImageTransform.from_config(cfg=cfg, train=train).t

    if mode != "advanced":
        raise ValueError(f"Invalid pipeline.mode='{mode}'. Expected 'paper' or 'advanced'.")

    # Crop first (helps segmentation focus on the subject/face)
    pre_crop_ratio = float(_require(cfg, "advanced.pre_crop_ratio"))
    crop = CenterCropMinSide(ratio=pre_crop_ratio)

    # Build tail (always applied)
    tail = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean], std=[std]),
        ]
    )

    # Compose ops in order
    ops = [crop]

    # Only create + append BackgroundRemover when enabled
    bg_enabled = bool(_require(cfg, "advanced.background_remover.enabled"))
    if bg_enabled:
        bg_cfg = BgRemoveConfig(
            enabled=True,
            backend=str(_require(cfg, "advanced.background_remover.backend")),
            device=str(_require(cfg, "advanced.background_remover.device")),
            threshold=float(_require(cfg, "advanced.background_remover.threshold")),
            min_fg_fraction=float(_require(cfg, "advanced.background_remover.min_fg_fraction")),
            fg_black_threshold=int(_require(cfg, "advanced.background_remover.fg_black_threshold")),
            min_gray_variance=float(_require(cfg, "advanced.background_remover.min_gray_variance")),
        )
        ops.append(BackgroundRemover(bg_cfg))

    ops.append(tail)
    return transforms.Compose(ops)
