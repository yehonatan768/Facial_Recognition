from __future__ import annotations

from typing import Any, Dict

from torchvision import transforms

from src.preprocess.face_mask import EllipseFaceMask
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
    Modular preprocessing pipeline.
    Easy to extend by appending steps.
    """
    mode = str(_require(cfg, "pipeline.mode")).strip().lower()

    mean = float(_require(cfg, "transform.mean"))
    std = float(_require(cfg, "transform.std"))
    input_size = int(_require(cfg, "transform.input_size"))

    if mode == "paper":
        return PaperImageTransform.from_config(cfg=cfg, train=train).t

    if mode != "advanced":
        raise ValueError(f"Invalid pipeline.mode='{mode}'")

    ops = []

    # ---- 1. Center crop ----
    ops.append(
        CenterCropMinSide(
            ratio=float(_require(cfg, "advanced.pre_crop_ratio"))
        )
    )

    # ---- 2. Face mask (RESTORED) ----
    ops.append(
        EllipseFaceMask(
            center=tuple(_require(cfg, "advanced.face_mask.center")),
            axes=tuple(_require(cfg, "advanced.face_mask.axes")),
            edge_softness=float(_require(cfg, "advanced.face_mask.edge_softness")),
            power=float(_require(cfg, "advanced.face_mask.power")),
        )
    )

    # ---- 3. Final tensor steps ----
    ops.extend(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean], std=[std]),
        ]
    )

    return transforms.Compose(ops)
