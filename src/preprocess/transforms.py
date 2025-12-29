from __future__ import annotations

from typing import Any, Dict

from torchvision import transforms

from src.preprocess.background_remove import BackgroundRemover, RembgConfig
from src.preprocess.paper_transforms import PaperImageTransform
from src.preprocess.preprocess import FaceMaskConfig, build_face_mask_transform, CenterCropMinSide


def _require(cfg: Dict[str, Any], path: str) -> Any:
    cur: Any = cfg
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError(f"Missing required config key: {path}")
        cur = cur[k]
    return cur


def build_transform_from_config(cfg: Dict[str, Any], train: bool) -> transforms.Compose:
    """
    Two pipelines controlled by config:

      pipeline.mode = "paper":
         PaperImageTransform

      pipeline.mode = "advanced":
         (optional) pre-crop -> (optional) rembg -> grayscale/resize/tensor ->
         (optional) ellipse mask -> normalize
    """
    mode = str(_require(cfg, "pipeline.mode")).strip().lower()

    mean = float(_require(cfg, "transform.mean"))
    std = float(_require(cfg, "transform.std"))

    if mode == "paper":
        return PaperImageTransform.from_config(cfg=cfg, train=train).t

    if mode != "advanced":
        raise ValueError(f"Invalid pipeline.mode='{mode}'. Expected 'paper' or 'advanced'.")

    # --- advanced: face mask config (also provides pre-crop ratio) ---
    face_cfg = FaceMaskConfig(
        enabled=bool(_require(cfg, "advanced.face_mask.enabled")),
        size=int(_require(cfg, "advanced.face_mask.size")),
        pre_crop_ratio=float(_require(cfg, "advanced.face_mask.pre_crop_ratio")),
        center=tuple(_require(cfg, "advanced.face_mask.center")),
        axes=tuple(_require(cfg, "advanced.face_mask.axes")),
        edge_softness=float(_require(cfg, "advanced.face_mask.edge_softness")),
        mask_power=float(_require(cfg, "advanced.face_mask.mask_power")),
        jitter_center=float(_require(cfg, "advanced.face_mask.jitter_center")),
        jitter_axes=float(_require(cfg, "advanced.face_mask.jitter_axes")),
        randomize_outside=bool(_require(cfg, "advanced.face_mask.randomize_outside")),
        outside_noise_std=float(_require(cfg, "advanced.face_mask.outside_noise_std")),
        outside_fill=float(_require(cfg, "advanced.face_mask.outside_fill")),
    )

    # --- advanced: background remover config ---
    bg_enabled = bool(_require(cfg, "advanced.background_remover.enabled"))
    bg_model = str(_require(cfg, "advanced.background_remover.model"))

    bg_cfg = RembgConfig(
        enabled=bg_enabled,
        model=bg_model,
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
    )

    ops = []

    # KEY FIX: pre-crop first so face occupies most of the image
    # (this drastically reduces "rembg removed the whole face" failures)
    ops.append(CenterCropMinSide(ratio=face_cfg.pre_crop_ratio))

    # Optional rembg
    if bg_enabled:
        ops.append(BackgroundRemover(bg_cfg))

    # Then the rest (grayscale/resize/tensor/(optional mask)/normalize)
    ops.append(
        build_face_mask_transform(
            train=train,
            cfg=face_cfg,
            normalize_mean=mean,
            normalize_std=std,
            apply_crop=False,  # we already cropped above
        )
    )

    return transforms.Compose(ops)
