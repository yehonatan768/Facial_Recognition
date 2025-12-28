from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from torchvision import transforms

from src.preprocess import FaceFocusConfig, build_focus_face_transform
from src.paper_transforms import PaperImageTransform  # your existing class


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
      - If data.face_focus.enabled: use focus-face pipeline (crop+mask) AND normalize same mean/std
        (Optionally still allow paper jitter before ToTensor if you want it.)
    """
    # Mean/std should match whatever you use for training (paper: 0.5/0.5)
    mean = float(_get(cfg, "transform.mean", 0.5))
    std = float(_get(cfg, "transform.std", 0.5))

    # paper base params
    input_size = int(_get(cfg, "transform.input_size", 105))

    # toggle face focus
    focus_enabled = bool(_get(cfg, "data.face_focus.enabled", False))

    if not focus_enabled:
        # Use existing paper transform class
        # Note: your PaperImageTransform currently uses its own defaults and config keys;
        # keeping it simple here and using constructor directly.
        return PaperImageTransform(
            train=train,
            input_size=input_size,
            mean=mean,
            std=std,
        ).t

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

    # Use your proven correct focus-face pipeline (includes grayscale/resize/tensor/mask/normalize)
    return build_focus_face_transform(
        train=train,
        cfg=ff_cfg,
        normalize_mean=mean,
        normalize_std=std,
    )
