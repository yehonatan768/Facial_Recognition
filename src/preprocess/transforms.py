from __future__ import annotations

from typing import Any, Dict, List

from torchvision import transforms

from src.preprocess.preprocess import CenterCropMinSide
from src.preprocess.face_mask import EllipseFaceMask
from src.preprocess.filters import build_equalize_from_cfg, build_edge_from_cfg


def _require(cfg: Dict[str, Any], path: str) -> Any:
    cur: Any = cfg
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError(f"Missing required config key: {path}")
        cur = cur[k]
    return cur


def _as_float_list(v: Any, n: int) -> List[float]:
    """
    Robustly parse a mean/std value from config.

    Accepts:
      - scalar (int/float) -> repeats n times
      - list/tuple of floats of length n
      - nested single list/tuple (e.g. [[...]]), common YAML mistake -> flattened
      - list/tuple length 1 -> repeats n times

    Returns: list[float] of length n.
    """
    # scalar
    if isinstance(v, (int, float)):
        return [float(v) for _ in range(n)]

    if isinstance(v, (list, tuple)):
        # Flatten one level of nesting, e.g. [[0.5,0.5,0.5]]
        if len(v) == 1 and isinstance(v[0], (list, tuple)):
            v = v[0]

        if len(v) == 1 and isinstance(v[0], (int, float)):
            return [float(v[0]) for _ in range(n)]

        if len(v) != n:
            raise ValueError(f"Expected list/tuple of length {n}, got length {len(v)}: {v!r}")

        out: List[float] = []
        for item in v:
            if isinstance(item, (list, tuple)):
                raise TypeError(f"Nested list/tuple values are not supported: {v!r}")
            out.append(float(item))
        return out

    raise TypeError(f"Expected scalar or list/tuple for mean/std, got {type(v).__name__}: {v!r}")


def build_transform_from_config(cfg: Dict[str, Any], train: bool) -> transforms.Compose:
    mode = str(_require(cfg, "pipeline.mode")).strip().lower()

    input_size = int(_require(cfg, "transform.input_size"))
    in_channels = int(_require(cfg, "model.in_channels"))

    # mean/std may be scalar or list; normalize expects per-channel lists
    mean_raw = _require(cfg, "transform.mean")
    std_raw = _require(cfg, "transform.std")
    mean = _as_float_list(mean_raw, n=in_channels)
    std = _as_float_list(std_raw, n=in_channels)

    ops: List[Any] = []

    # Base crop/resize (shared)
    ops.append(CenterCropMinSide())
    ops.append(transforms.Resize((input_size, input_size)))

    # Optional: advanced face mask (your custom pipeline)
    if mode == "advanced":
        adv = cfg.get("advanced", {}) or {}
        pre_crop_ratio = float(adv.get("pre_crop_ratio", 1.0))
        if 0.0 < pre_crop_ratio < 1.0:
            ops.append(transforms.CenterCrop(int(round(input_size * pre_crop_ratio))))
            ops.append(transforms.Resize((input_size, input_size)))

        fm = (adv.get("face_mask", {}) or {})
        ops.append(
            EllipseFaceMask(
                center=tuple(fm.get("center", [0.0, 0.0])),
                axes=tuple(fm.get("axes", [0.75, 0.90])),
                edge_softness=float(fm.get("edge_softness", 0.12)),
                power=float(fm.get("power", 1.0)),
            )
        )

    # Deterministic filters (train/val/test)
    eq = build_equalize_from_cfg(cfg)
    if eq is not None:
        ops.append(eq)

    ed = build_edge_from_cfg(cfg)
    if ed is not None:
        ops.append(ed)

    # Train-only augmentations
    if train:
        aug = cfg.get("augment", {}) or {}

        hf = aug.get("hflip", {}) or {}
        if bool(hf.get("enabled", False)):
            ops.append(transforms.RandomHorizontalFlip(p=float(hf.get("p", 0.5))))

        rot = aug.get("rotation", {}) or {}
        if bool(rot.get("enabled", False)):
            ops.append(transforms.RandomRotation(degrees=float(rot.get("degrees", 5.0))))

        blur = aug.get("blur", {}) or {}
        if bool(blur.get("enabled", False)):
            ops.append(
                transforms.RandomApply(
                    [
                        transforms.GaussianBlur(
                            kernel_size=int(blur.get("kernel_size", 3)),
                            sigma=tuple(blur.get("sigma", [0.3, 1.0])),
                        )
                    ],
                    p=float(blur.get("p", 0.15)),
                )
            )

        re = aug.get("random_erasing", {}) or {}
        if bool(re.get("enabled", False)):
            # RandomErasing is tensor-based, so we append after ToTensor below
            pass

    # Enforce channel mode to match the model (filters may change modes).
    if in_channels == 1:
        ops.append(transforms.Grayscale(num_output_channels=1))
    else:
        ops.append(transforms.Lambda(lambda im: im.convert("RGB")))

    # Tensor + normalize
    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(mean=mean, std=std))

    # Random erasing (train-only, tensor op)
    if train:
        re = (cfg.get("augment", {}) or {}).get("random_erasing", {}) or {}
        if bool(re.get("enabled", False)):
            ops.append(
                transforms.RandomErasing(
                    p=float(re.get("p", 0.10)),
                    scale=tuple(re.get("scale", [0.02, 0.08])),
                    ratio=tuple(re.get("ratio", [0.3, 3.3])),
                    value=float(re.get("value", 0.0)),
                )
            )

    return transforms.Compose(ops)
