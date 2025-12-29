from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF


@dataclass(frozen=True)
class FaceMaskConfig:
    enabled: bool

    size: int
    pre_crop_ratio: float

    # Ellipse params in normalized coords [-1..1]
    center: Tuple[float, float]
    axes: Tuple[float, float]
    edge_softness: float
    mask_power: float

    # Train-time jitter
    jitter_center: float
    jitter_axes: float

    # Outside fill behavior
    randomize_outside: bool
    outside_noise_std: float
    outside_fill: float  # set to 0.0 for black


class CenterCropMinSide(torch.nn.Module):
    """
    Center-crop a PIL image to a square based on the shorter side.
    Optionally crop tighter with ratio < 1.0 to zoom in.
    """

    def __init__(self, ratio: float):
        super().__init__()
        self.ratio = float(ratio)

    def forward(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            raise TypeError(f"CenterCropMinSide expects PIL Image. Got {type(img)}")

        w, h = img.size
        side = min(w, h)
        side = int(round(side * self.ratio))
        side = max(1, min(side, w, h))

        left = (w - side) // 2
        top = (h - side) // 2
        return TF.crop(img, top=top, left=left, height=side, width=side)


class SoftEllipseMask(torch.nn.Module):
    """
    Apply a soft elliptical mask to a (C,H,W) tensor in [0,1].
    Keeps inside ellipse, sets outside to black (or configured fill).
    """

    def __init__(self, cfg: FaceMaskConfig, train: bool):
        super().__init__()
        self.cfg = cfg
        self.train = train

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected tensor (C,H,W), got shape={tuple(x.shape)}")

        c, h, w = x.shape
        device = x.device
        cfg = self.cfg

        # Jitter (train only)
        cx, cy = cfg.center
        ax, ay = cfg.axes

        if self.train:
            if cfg.jitter_center != 0.0:
                cx = cx + (2 * torch.rand((), device=device) - 1).item() * cfg.jitter_center
                cy = cy + (2 * torch.rand((), device=device) - 1).item() * cfg.jitter_center
            if cfg.jitter_axes != 0.0:
                ax = ax * (1.0 + (2 * torch.rand((), device=device) - 1).item() * cfg.jitter_axes)
                ay = ay * (1.0 + (2 * torch.rand((), device=device) - 1).item() * cfg.jitter_axes)

        # Coordinate grid in [-1,1]
        yy = torch.linspace(-1.0, 1.0, h, device=device).view(h, 1).expand(h, w)
        xx = torch.linspace(-1.0, 1.0, w, device=device).view(1, w).expand(h, w)

        dx = (xx - cx) / max(ax, 1e-6)
        dy = (yy - cy) / max(ay, 1e-6)
        r = torch.sqrt(dx * dx + dy * dy)

        sigma = max(cfg.edge_softness, 1e-6)
        mask = torch.exp(-torch.clamp(r - 1.0, min=0.0) / sigma)  # (H,W) in (0,1]

        # Make outside fall off faster
        if cfg.mask_power != 1.0:
            mask = mask.pow(float(cfg.mask_power))

        mask = mask.unsqueeze(0).expand(c, h, w)

        # Outside fill (black by config)
        if cfg.randomize_outside and self.train:
            noise = torch.randn_like(x) * float(cfg.outside_noise_std)
            bg = torch.clamp(float(cfg.outside_fill) + noise, 0.0, 1.0)
        else:
            bg = torch.full_like(x, float(cfg.outside_fill))

        out = x * mask + bg * (1.0 - mask)
        return out


def build_face_mask_transform(
    train: bool,
    cfg: FaceMaskConfig,
    normalize_mean: float,
    normalize_std: float,
) -> transforms.Compose:
    """
    If cfg.enabled = False:
      Grayscale -> Resize -> ToTensor -> Normalize

    If cfg.enabled = True:
      CenterCrop -> Grayscale -> Resize -> ToTensor -> EllipseMask -> Normalize
    """

    ops = []

    if cfg.enabled:
        ops.append(CenterCropMinSide(ratio=cfg.pre_crop_ratio))

    ops.extend([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((cfg.size, cfg.size)),
        transforms.ToTensor(),
    ])

    if cfg.enabled:
        ops.append(SoftEllipseMask(cfg=cfg, train=train))

    ops.append(transforms.Normalize(mean=[normalize_mean], std=[normalize_std]))

    return transforms.Compose(ops)

