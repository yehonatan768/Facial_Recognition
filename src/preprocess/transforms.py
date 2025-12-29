from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from torchvision import transforms

from src.preprocess.preprocess import FaceFocusConfig, build_focus_face_transform
from src.preprocess.paper_transforms import PaperImageTransform  # your existing class

# NEW: background removal (PIL -> PIL)
from src.preprocess.background_remove import BackgroundRemover


def _get(cfg: Dict[str, Any], path: str, default: Any) -> Any:
    cur: Any = cfg
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def build_transform_from_config(cfg: Dict[str, Any], train: bool) -> transforms.Compose:
    """
    Single entrypoint used by datasets.

    Behavior:
      - Default: paper transform (grayscale->resize->tensor->normalize, optional jitter)
      - If data.face_focus.enabled: optionally remove background first (PIL),
        then apply focus-face pipeline (crop+mask) and normalize with same mean/std.
    """
    # Mean/std should match training (paper: 0.5/0.5)
    mean = float(_get(cfg, "transform.mean", 0.5))
    std = float(_get(cfg, "transform.std", 0.5))

    # paper base params
    input_size = int(_get(cfg, "transform.input_size", 105))

    # toggle face focus
    focus_enabled = bool(_get(cfg, "data.face_focus.enabled", False))

    if not focus_enabled:
        # Use existing paper transform class
        return PaperImageTransform(
            train=train,
            input_size=input_size,
            mean=mean,
            std=std,
        ).t

    # NEW: toggle background removal (default True when face_focus is enabled)
    remove_bg = bool(_get(cfg, "data.face_focus.remove_background", True))

    # Face focus config
    ff_cfg_dict = _get(cfg, "data.face_focus", {}) or {}
    ff_cfg = FaceFocusConfig(
        size=int(ff_cfg_dict.get("size", input_size)),
        pre_crop_ratio=float(ff_cfg_dict.get("pre_crop_ratio", 0.90)),
        center=tuple(ff_cfg_dict.get("center", (0.0, 0.0))),
        axes=tuple(ff_cfg_dict.get("axes", (0.75, 0.90))),
        edge_softness=float(ff_cfg_dict.get("edge_softness", 0.08)),
        mask_power=float(ff_cfg_dict.get("mask_power", 2.0)),
        randomize_background=bool(ff_cfg_dict.get("randomize_background", train)),
        background_noise_std=float(ff_cfg_dict.get("background_noise_std", 0.08)),
        jitter_center=float(ff_cfg_dict.get("jitter_center", 0.03)),
        jitter_axes=float(ff_cfg_dict.get("jitter_axes", 0.05)),
        background_fill=float(ff_cfg_dict.get("background_fill", 0.5)),
    )

    # Build your proven focus-face pipeline (grayscale/resize/tensor/mask/normalize)
    focus_t = build_focus_face_transform(
        train=train,
        cfg=ff_cfg,
        normalize_mean=mean,
        normalize_std=std,
    )

    # NEW: ensure background removal happens BEFORE crop/grayscale/resize/etc.
    if remove_bg:
        return transforms.Compose([
            BackgroundRemover(),
            focus_t,
        ])

    return focus_t
