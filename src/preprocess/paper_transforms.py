from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

import torch
from PIL import Image
from torchvision import transforms


def _require(cfg: Dict[str, Any], path: str) -> Any:
    cur: Any = cfg
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError(f"Missing required config key: {path}")
        cur = cur[k]
    return cur


@dataclass
class PaperImageTransform:
    """
    Paper pipeline (extended with optional, config-driven augmentations):

      Grayscale -> Resize -> (optional geom/photometric aug) -> (optional jitter) -> ToTensor
      -> (optional RandomErasing) -> Normalize
    """

    train: bool
    input_size: int
    mean: float
    std: float

    # Paper jitter (original behavior)
    enable_jitter: bool
    jitter_brightness: float
    jitter_contrast: float
    jitter_saturation: float
    jitter_hue: float

    # Shared augmentations (config-driven)
    hflip_enabled: bool
    hflip_p: float

    rotation_enabled: bool
    rotation_degrees: float

    blur_enabled: bool
    blur_p: float
    blur_kernel_size: int
    blur_sigma_min: float
    blur_sigma_max: float

    random_erasing_enabled: bool
    random_erasing_p: float
    random_erasing_scale_min: float
    random_erasing_scale_max: float
    random_erasing_ratio_min: float
    random_erasing_ratio_max: float
    random_erasing_value: float

    def __post_init__(self) -> None:
        ops: list[transforms.Transform] = [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((self.input_size, self.input_size)),
        ]

        # --- Geometric aug (train only) ---
        if self.train and self.hflip_enabled and self.hflip_p > 0:
            ops.append(transforms.RandomHorizontalFlip(p=float(self.hflip_p)))

        if self.train and self.rotation_enabled and float(self.rotation_degrees) > 0:
            ops.append(
                transforms.RandomRotation(
                    degrees=float(self.rotation_degrees),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    fill=0,
                )
            )

        # --- Paper photometric aug (train only) ---
        if self.train and self.enable_jitter:
            ops.append(
                transforms.ColorJitter(
                    brightness=self.jitter_brightness,
                    contrast=self.jitter_contrast,
                    saturation=self.jitter_saturation,
                    hue=self.jitter_hue,
                )
            )

        # --- Very light blur (train only) ---
        if self.train and self.blur_enabled and self.blur_p > 0:
            ops.append(
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=int(self.blur_kernel_size),
                                             sigma=(float(self.blur_sigma_min), float(self.blur_sigma_max)))],
                    p=float(self.blur_p),
                )
            )

        # --- Tensor + normalize ---
        ops.append(transforms.ToTensor())

        # RandomErasing works on tensors (train only), so apply AFTER ToTensor and BEFORE Normalize.
        if self.train and self.random_erasing_enabled and self.random_erasing_p > 0:
            ops.append(
                transforms.RandomErasing(
                    p=float(self.random_erasing_p),
                    scale=(float(self.random_erasing_scale_min), float(self.random_erasing_scale_max)),
                    ratio=(float(self.random_erasing_ratio_min), float(self.random_erasing_ratio_max)),
                    value=float(self.random_erasing_value),
                )
            )

        ops.append(transforms.Normalize(mean=[self.mean], std=[self.std]))

        self.t = transforms.Compose(ops)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.t(img)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], train: bool) -> "PaperImageTransform":
        # blur.sigma is stored as [min, max]
        blur_sigma = _require(cfg, "augment.blur.sigma")
        if not isinstance(blur_sigma, (list, tuple)) or len(blur_sigma) != 2:
            raise ValueError("augment.blur.sigma must be a list/tuple of length 2, e.g. [0.3, 1.0]")

        er_scale = _require(cfg, "augment.random_erasing.scale")
        er_ratio = _require(cfg, "augment.random_erasing.ratio")
        if (not isinstance(er_scale, (list, tuple)) or len(er_scale) != 2 or
            not isinstance(er_ratio, (list, tuple)) or len(er_ratio) != 2):
            raise ValueError("augment.random_erasing.scale and ratio must be list/tuple length 2")

        return cls(
            train=train,
            input_size=int(_require(cfg, "transform.input_size")),
            mean=float(_require(cfg, "transform.mean")),
            std=float(_require(cfg, "transform.std")),

            enable_jitter=bool(_require(cfg, "paper.enable_jitter")),
            jitter_brightness=float(_require(cfg, "paper.jitter.brightness")),
            jitter_contrast=float(_require(cfg, "paper.jitter.contrast")),
            jitter_saturation=float(_require(cfg, "paper.jitter.saturation")),
            jitter_hue=float(_require(cfg, "paper.jitter.hue")),

            hflip_enabled=bool(_require(cfg, "augment.hflip.enabled")),
            hflip_p=float(_require(cfg, "augment.hflip.p")),

            rotation_enabled=bool(_require(cfg, "augment.rotation.enabled")),
            rotation_degrees=float(_require(cfg, "augment.rotation.degrees")),

            blur_enabled=bool(_require(cfg, "augment.blur.enabled")),
            blur_p=float(_require(cfg, "augment.blur.p")),
            blur_kernel_size=int(_require(cfg, "augment.blur.kernel_size")),
            blur_sigma_min=float(blur_sigma[0]),
            blur_sigma_max=float(blur_sigma[1]),

            random_erasing_enabled=bool(_require(cfg, "augment.random_erasing.enabled")),
            random_erasing_p=float(_require(cfg, "augment.random_erasing.p")),
            random_erasing_scale_min=float(er_scale[0]),
            random_erasing_scale_max=float(er_scale[1]),
            random_erasing_ratio_min=float(er_ratio[0]),
            random_erasing_ratio_max=float(er_ratio[1]),
            random_erasing_value=float(_require(cfg, "augment.random_erasing.value")),
        )
