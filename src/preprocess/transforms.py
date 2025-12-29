from __future__ import annotations

from typing import Any, Dict

from torchvision import transforms

from src.preprocess.background_remove import BackgroundRemover, RembgConfig
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
      CenterCropMinSide -> rembg -> Grayscale -> Resize -> ToTensor -> Normalize

    NOTE: Advanced currently does background removal only (no face mask).
          BackgroundRemover has a safety fallback if rembg removes almost everything.
    """
    mode = str(_require(cfg, "pipeline.mode")).strip().lower()

    mean = float(_require(cfg, "transform.mean"))
    std = float(_require(cfg, "transform.std"))
    input_size = int(_require(cfg, "transform.input_size"))

    if mode == "paper":
        return PaperImageTransform.from_config(cfg=cfg, train=train).t

    if mode != "advanced":
        raise ValueError(f"Invalid pipeline.mode='{mode}'. Expected 'paper' or 'advanced'.")

    # Crop first (helps rembg not delete the face)
    pre_crop_ratio = float(_require(cfg, "advanced.pre_crop_ratio"))
    crop = CenterCropMinSide(ratio=pre_crop_ratio)

    # rembg config
    bg_enabled = bool(_require(cfg, "advanced.background_remover.enabled"))

    bg_cfg = RembgConfig(
        enabled=bg_enabled,
        model=str(_require(cfg, "advanced.background_remover.model")),
        alpha_matting=bool(_require(cfg, "advanced.background_remover.alpha_matting")),
        alpha_matting_foreground_threshold=int(
            _require(cfg, "advanced.background_remover.alpha_matting_foreground_threshold")
        ),
        alpha_matting_background_threshold=int(
            _require(cfg, "advanced.background_remover.alpha_matting_background_threshold")
        ),
        alpha_matting_erode_size=int(
            _require(cfg, "advanced.background_remover.alpha_matting_erode_size")
        ),
        # Fail-safe thresholds:
        # 1) fg fraction threshold
        min_fg_fraction=float(_require(cfg, "advanced.background_remover.min_fg_fraction")),
        fg_black_threshold=int(_require(cfg, "advanced.background_remover.fg_black_threshold")),
        # 2) variance threshold (NEW)
        min_gray_variance=float(_require(cfg, "advanced.background_remover.min_gray_variance")),
    )

    remover = BackgroundRemover(bg_cfg)

    # After rembg: format for model
    tail = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean], std=[std]),
        ]
    )

    ops = [crop]
    if bg_enabled:
        ops.append(remover)
    ops.append(tail)

    return transforms.Compose(ops)
