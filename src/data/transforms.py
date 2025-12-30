from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from torchvision import transforms
from torchvision.transforms import functional as TF


@dataclass(frozen=True)
class ImageTransformConfig:
    size: int = 105
    to_grayscale: bool = True
    normalize: bool = False
    mean: float = 0.5
    std: float = 0.5


def _pil_grayscale_jitter(
    img,
    brightness: float,
    contrast: float,
    gamma: float,
):
    """
    Apply mild photometric jitter on a PIL image for grayscale pipelines.
    brightness/contrast are symmetric ranges around 1.0 (e.g., 0.10 -> [0.90, 1.10])
    gamma is symmetric range around 1.0 as well.
    """
    import random

    if brightness > 0:
        b = 1.0 + random.uniform(-brightness, brightness)
        img = TF.adjust_brightness(img, b)
    if contrast > 0:
        c = 1.0 + random.uniform(-contrast, contrast)
        img = TF.adjust_contrast(img, c)
    if gamma > 0:
        g = 1.0 + random.uniform(-gamma, gamma)
        # TF.adjust_gamma requires gamma > 0
        g = max(0.05, g)
        img = TF.adjust_gamma(img, g)

    return img


def build_transform(cfg: dict, train: bool = False) -> Callable:
    """
    Transform builder with an explicit train/eval split.

    Eval (train=False): deterministic preprocessing (matches prior behavior). :contentReference[oaicite:1]{index=1}
    Train (train=True): adds mild augmentations to improve robustness while keeping eval deterministic.
    """
    c = cfg.get("transform", {}) if isinstance(cfg, dict) else {}

    size = int(c.get("size", 105))
    to_gray = bool(c.get("to_grayscale", True))
    normalize = bool(c.get("normalize", False))
    mean = float(c.get("mean", 0.5))
    std = float(c.get("std", 0.5))

    # Augmentation sub-config (optional)
    aug = c.get("augment", {}) if isinstance(c, dict) else {}

    # Master switch: by default, augmentation is ON for train, OFF otherwise
    aug_enabled = bool(aug.get("enabled", True)) if train else False

    # Spatial augmentation (gentle)
    # RandomResizedCrop is the most effective "centering robustness" augmentation.
    rrc_enabled = bool(aug.get("random_resized_crop", True))
    rrc_scale_min = float(aug.get("rrc_scale_min", 0.85))
    rrc_scale_max = float(aug.get("rrc_scale_max", 1.00))
    rrc_ratio_min = float(aug.get("rrc_ratio_min", 0.95))
    rrc_ratio_max = float(aug.get("rrc_ratio_max", 1.05))

    # Small affine jitter
    affine_enabled = bool(aug.get("random_affine", True))
    affine_degrees = float(aug.get("affine_degrees", 5.0))
    affine_translate = float(aug.get("affine_translate", 0.03))  # fraction of image
    affine_scale_min = float(aug.get("affine_scale_min", 0.98))
    affine_scale_max = float(aug.get("affine_scale_max", 1.02))

    # Horizontal flip (usually safe for faces; disable if you suspect left/right artifacts)
    hflip_p = float(aug.get("hflip_p", 0.5))

    # Photometric augmentation
    # For RGB: ColorJitter; for grayscale: custom brightness/contrast/gamma jitter.
    cj_enabled = bool(aug.get("color_jitter", True))
    cj_brightness = float(aug.get("cj_brightness", 0.10))
    cj_contrast = float(aug.get("cj_contrast", 0.10))
    cj_saturation = float(aug.get("cj_saturation", 0.05))  # only meaningful for RGB
    cj_hue = float(aug.get("cj_hue", 0.02))                # only meaningful for RGB
    gray_gamma = float(aug.get("gray_gamma", 0.10))        # grayscale-only helper

    # Optional slight blur
    blur_p = float(aug.get("blur_p", 0.15))
    blur_kernel = int(aug.get("blur_kernel", 3))
    blur_sigma_min = float(aug.get("blur_sigma_min", 0.1))
    blur_sigma_max = float(aug.get("blur_sigma_max", 1.0))

    ops = []

    # ----- Train-time augmentation (PIL-space) -----
    if train and aug_enabled:
        # Spatial: RandomResizedCrop OR deterministic resize
        if rrc_enabled:
            ops.append(
                transforms.RandomResizedCrop(
                    size=(size, size),
                    scale=(rrc_scale_min, rrc_scale_max),
                    ratio=(rrc_ratio_min, rrc_ratio_max),
                )
            )
        else:
            ops.append(transforms.Resize((size, size)))

        if affine_enabled:
            ops.append(
                transforms.RandomAffine(
                    degrees=affine_degrees,
                    translate=(affine_translate, affine_translate),
                    scale=(affine_scale_min, affine_scale_max),
                    shear=None,
                )
            )

        if hflip_p > 0:
            ops.append(transforms.RandomHorizontalFlip(p=hflip_p))

        # Photometric (before grayscale conversion if RGB, or after if grayscale-only)
        if cj_enabled:
            if to_gray:
                # Apply a grayscale-friendly jitter on PIL image
                ops.append(
                    transforms.Lambda(
                        lambda img: _pil_grayscale_jitter(
                            img,
                            brightness=cj_brightness,
                            contrast=cj_contrast,
                            gamma=gray_gamma,
                        )
                    )
                )
            else:
                ops.append(
                    transforms.ColorJitter(
                        brightness=cj_brightness,
                        contrast=cj_contrast,
                        saturation=cj_saturation,
                        hue=cj_hue,
                    )
                )

        # Optional blur
        if blur_p > 0:
            ops.append(
                transforms.RandomApply(
                    [
                        transforms.GaussianBlur(
                            kernel_size=blur_kernel,
                            sigma=(blur_sigma_min, blur_sigma_max),
                        )
                    ],
                    p=blur_p,
                )
            )

    else:
        # ----- Eval-time deterministic preprocessing -----
        ops.append(transforms.Resize((size, size)))

    # ----- Common steps -----
    if to_gray:
        ops.append(transforms.Grayscale(num_output_channels=1))

    ops.append(transforms.ToTensor())

    if normalize:
        if to_gray:
            ops.append(transforms.Normalize(mean=[mean], std=[std]))
        else:
            ops.append(transforms.Normalize(mean=[mean] * 3, std=[std] * 3))

    return transforms.Compose(ops)
