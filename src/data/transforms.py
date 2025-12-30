from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
from torchvision import transforms
from torchvision.transforms import functional as TF


@dataclass(frozen=True)
class ImageTransformConfig:
    size: int = 105
    to_grayscale: bool = True
    normalize: bool = False
    mean: float = 0.5
    std: float = 0.5


def build_transform(cfg: dict) -> Callable:
    """Minimal transform builder.

    Assumes images are already reasonably cropped/cleaned. This stage only:
      - optionally converts to grayscale
      - resizes to cfg.size x cfg.size
      - converts to torch tensor in [0,1]
      - optional normalization
    """
    c = cfg.get("transform", {}) if isinstance(cfg, dict) else {}
    size = int(c.get("size", 105))
    to_gray = bool(c.get("to_grayscale", True))
    normalize = bool(c.get("normalize", False))
    mean = float(c.get("mean", 0.5))
    std = float(c.get("std", 0.5))

    ops = []
    if to_gray:
        ops.append(transforms.Grayscale(num_output_channels=1))
    ops.append(transforms.Resize((size, size)))
    ops.append(transforms.ToTensor())
    if normalize:
        ops.append(transforms.Normalize(mean=[mean], std=[std]) if to_gray else transforms.Normalize(mean=[mean]*3, std=[std]*3))
    return transforms.Compose(ops)
