from __future__ import annotations

from typing import Any, Dict

from torchvision import transforms

from src.preprocess.preprocess import CenterCropMinSide
from src.preprocess.face_mask import EllipseFaceMask
from src.preprocess.paper_transforms import PaperImageTransform

# Filters are implemented and configured ONLY in filters.py
from src.preprocess.filters import build_equalize_from_cfg, build_edge_from_cfg


def _as_float_list(v, n: int) -> list[float]:
    """
    Accepts scalar or list/tuple. Returns list[float] length n.
    """
    if isinstance(v, (int, float)):
        return [float(v)] * n
    if isinstance(v, (list, tuple)):
        if len(v) == 1:
            return [float(v[0])] * n
        if len(v) != n:
            raise ValueError(f"Expected list of length {n} (or 1), got {len(v)}: {v}")
        return [float(x) for x in v]
    raise TypeError(f"mean/std must be float or list/tuple, got: {type(v)}")


def _require(cfg: Dict[str, Any], path: str) -> Any:
    cur: Any = cfg
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError(f"Missing required config key: {path}")
        cur = cur[k]
    return cur


def build_transform_from_config(cfg: Dict[str, Any], train: bool) -> transforms.Compose:
    """
    - paper: paper-faithful baseline (paper_transforms.py), unchanged.
    - advanced: crop -> filters -> face_mask -> grayscale/resize/tensor -> normalize
    """
    mode = str(_require(cfg, "pipeline.mode")).strip().lower()
    input_size = int(_require(cfg, "transform.input_size"))
    # Decide channels from model config (default 1)
    in_channels = int(((cfg.get("model", {}) or {}).get("in_channels", 1)))

    mean_raw = _require(cfg, "transform.mean")
    std_raw = _require(cfg, "transform.std")

    mean = _as_float_list(mean_raw, in_channels)
    std = _as_float_list(std_raw, in_channels)

    if mode == "paper":
        return PaperImageTransform.from_config(cfg=cfg, train=train).t

    if mode != "advanced":
        raise ValueError(f"Invalid pipeline.mode={mode!r} (expected 'paper' or 'advanced')")

    ops: list = []

    # 1) deterministic crop
    ops.append(CenterCropMinSide(ratio=float(_require(cfg, "advanced.pre_crop_ratio"))))

    # 2) deterministic filters (defined ONLY in filters.py)
    ops.append(build_equalize_from_cfg(cfg))
    ops.append(build_edge_from_cfg(cfg))

    # 3) face mask (deterministic spatial prior)
    ops.append(
        EllipseFaceMask(
            center=tuple(_require(cfg, "advanced.face_mask.center")),
            axes=tuple(_require(cfg, "advanced.face_mask.axes")),
            edge_softness=float(_require(cfg, "advanced.face_mask.edge_softness")),
            power=float(_require(cfg, "advanced.face_mask.power")),
        )
    )

    # 3.5) train-only augmentations (applied on PIL image before tensor conversion)
    if train:
        aug = (cfg.get("augment", {}) or {})

        # Horizontal flip
        hf = (aug.get("hflip", {}) or {})
        if bool(hf.get("enabled", False)):
            ops.append(transforms.RandomHorizontalFlip(p=float(hf.get("p", 0.5))))

        # Small rotations (helps pose variation)
        rot = (aug.get("rotation", {}) or {})
        if bool(rot.get("enabled", False)):
            ops.append(transforms.RandomRotation(degrees=float(rot.get("degrees", 5.0))))

        # Mild blur
        bl = (aug.get("blur", {}) or {})
        if bool(bl.get("enabled", False)):
            ops.append(
                transforms.RandomApply(
                    [
                        transforms.GaussianBlur(
                            kernel_size=int(bl.get("kernel_size", 3)),
                            sigma=tuple(bl.get("sigma", [0.3, 1.0])),
                        )
                    ],
                    p=float(bl.get("p", 0.15)),
                )
            )

    # 4) final format (applied once)
    ops.extend(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean], std=[std]),
        ]
    )

    # Tensor-level train-only aug (after normalize)
    if train:
        re = (cfg.get("augment", {}) or {}).get("random_erasing", {}) or {}
        if bool(re.get("enabled", False)):
            ops.append(
                transforms.RandomErasing(
                    p=float(re.get("p", 0.1)),
                    scale=tuple(re.get("scale", [0.02, 0.08])),
                    ratio=tuple(re.get("ratio", [0.3, 3.3])),
                    value=float(re.get("value", 0.0)),
                )
            )

    return transforms.Compose(ops)
