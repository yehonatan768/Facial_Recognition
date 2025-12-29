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


def _add_shared_augs_pil(ops: list, cfg: Dict[str, Any], train: bool) -> None:
    """
    Augmentations that operate on PIL images.
    Applied only when train=True.
    """
    if not train:
        return

    # Horizontal flip
    if bool(_require(cfg, "augment.hflip.enabled")):
        p = float(_require(cfg, "augment.hflip.p"))
        if p > 0:
            ops.append(transforms.RandomHorizontalFlip(p=p))

    # Light rotation
    if bool(_require(cfg, "augment.rotation.enabled")):
        deg = float(_require(cfg, "augment.rotation.degrees"))
        if deg > 0:
            ops.append(
                transforms.RandomRotation(
                    degrees=deg,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    fill=0,
                )
            )

    # Very light blur (RandomApply over GaussianBlur)
    if bool(_require(cfg, "augment.blur.enabled")):
        p = float(_require(cfg, "augment.blur.p"))
        if p > 0:
            k = int(_require(cfg, "augment.blur.kernel_size"))
            sigma = _require(cfg, "augment.blur.sigma")
            if not isinstance(sigma, (list, tuple)) or len(sigma) != 2:
                raise ValueError("augment.blur.sigma must be a list/tuple of length 2, e.g. [0.3, 1.0]")
            ops.append(
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=k, sigma=(float(sigma[0]), float(sigma[1])))],
                    p=p,
                )
            )


def _add_shared_augs_tensor(ops: list, cfg: Dict[str, Any], train: bool) -> None:
    """
    Augmentations that operate on tensors.
    Applied only when train=True.
    """
    if not train:
        return

    if bool(_require(cfg, "augment.random_erasing.enabled")):
        p = float(_require(cfg, "augment.random_erasing.p"))
        if p > 0:
            scale = _require(cfg, "augment.random_erasing.scale")
            ratio = _require(cfg, "augment.random_erasing.ratio")
            if (not isinstance(scale, (list, tuple)) or len(scale) != 2 or
                not isinstance(ratio, (list, tuple)) or len(ratio) != 2):
                raise ValueError("augment.random_erasing.scale and ratio must be list/tuple length 2")
            value = float(_require(cfg, "augment.random_erasing.value"))
            ops.append(
                transforms.RandomErasing(
                    p=p,
                    scale=(float(scale[0]), float(scale[1])),
                    ratio=(float(ratio[0]), float(ratio[1])),
                    value=value,
                )
            )


def build_transform_from_config(cfg: Dict[str, Any], train: bool) -> transforms.Compose:
    """
    Modular preprocessing pipeline.
    Easy to extend by appending steps.
    """
    mode = str(_require(cfg, "pipeline.mode")).strip().lower()

    mean = float(_require(cfg, "transform.mean"))
    std = float(_require(cfg, "transform.std"))
    input_size = int(_require(cfg, "transform.input_size"))

    # Paper mode delegates to PaperImageTransform (which also supports shared augs).
    if mode == "paper":
        return PaperImageTransform.from_config(cfg=cfg, train=train).t

    if mode != "advanced":
        raise ValueError(f"Invalid pipeline.mode='{mode}'")

    ops: list = []

    # ---- 1. Center crop ----
    ops.append(CenterCropMinSide(ratio=float(_require(cfg, "advanced.pre_crop_ratio"))))

    # ---- 2. Shared PIL augmentations (train only) ----
    _add_shared_augs_pil(ops=ops, cfg=cfg, train=train)

    # ---- 3. Face mask ----
    ops.append(
        EllipseFaceMask(
            center=tuple(_require(cfg, "advanced.face_mask.center")),
            axes=tuple(_require(cfg, "advanced.face_mask.axes")),
            edge_softness=float(_require(cfg, "advanced.face_mask.edge_softness")),
            power=float(_require(cfg, "advanced.face_mask.power")),
        )
    )

    # ---- 4. Final tensor steps ----
    ops.extend(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((input_size, input_size)),
        ]
    )

    # Blur can also be applied after resize; we already apply it in _add_shared_augs_pil().
    ops.append(transforms.ToTensor())

    # Tensor augmentations (train only): RandomErasing
    _add_shared_augs_tensor(ops=ops, cfg=cfg, train=train)

    ops.append(transforms.Normalize(mean=[mean], std=[std]))

    return transforms.Compose(ops)
