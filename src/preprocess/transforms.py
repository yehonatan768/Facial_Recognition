from __future__ import annotations

from typing import Any, Dict

from torchvision import transforms

from src.preprocess.background_remove import BackgroundRemover, RembgConfig
from src.preprocess.paper_transforms import PaperImageTransform
from src.preprocess.preprocess import FaceMaskConfig, build_face_mask_transform


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
         PaperImageTransform (config-only)

      pipeline.mode = "advanced":
         (optional) BackgroundRemover(rembg) -> face mask pipeline
    """
    mode = str(_require(cfg, "pipeline.mode")).strip().lower()

    mean = float(_require(cfg, "transform.mean"))
    std = float(_require(cfg, "transform.std"))

    if mode == "paper":
        return PaperImageTransform.from_config(cfg=cfg, train=train).t

    if mode == "advanced":
        # rembg config
        bg_enabled = bool(_require(cfg, "advanced.background_remover.enabled"))
        bg_model = str(_require(cfg, "advanced.background_remover.model"))

        bg = BackgroundRemover(RembgConfig(enabled=bg_enabled, model=bg_model))

        # face mask config
        fm = cfg["advanced"]["face_mask"]  # will KeyError if missing (desired)
        face_cfg = FaceMaskConfig(
            enabled=bool(_require(cfg, "advanced.face_mask.enabled")),

            size=int(_require(cfg, "advanced.face_mask.size")),
            pre_crop_ratio=float(_require(cfg, "advanced.face_mask.pre_crop_ratio")),

            center=(_require(cfg, "advanced.face_mask.center")),
            axes=(_require(cfg, "advanced.face_mask.axes")),
            edge_softness=float(_require(cfg, "advanced.face_mask.edge_softness")),
            mask_power=float(_require(cfg, "advanced.face_mask.mask_power")),

            jitter_center=float(_require(cfg, "advanced.face_mask.jitter_center")),
            jitter_axes=float(_require(cfg, "advanced.face_mask.jitter_axes")),

            randomize_outside=bool(_require(cfg, "advanced.face_mask.randomize_outside")),
            outside_noise_std=float(_require(cfg, "advanced.face_mask.outside_noise_std")),
            outside_fill=float(_require(cfg, "advanced.face_mask.outside_fill")),
        )
        face_t = build_face_mask_transform(
            train=train,
            cfg=face_cfg,
            normalize_mean=mean,
            normalize_std=std,
        )

        if bg_enabled:
            return transforms.Compose([bg, face_t])
        return face_t

    raise ValueError(f"Invalid pipeline.mode='{mode}'. Expected 'paper' or 'advanced'.")
