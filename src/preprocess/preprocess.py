# src/preprocess.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF


@dataclass
class FaceFocusConfig:
    # Target size for the network
    size: int

    # Pre-crop: crop around center before resizing to 'size'
    # 1.0 = crop to the shorter side (square crop)
    # <1.0 = tighter crop (zooms in). Recommended: 0.85 - 0.95
    pre_crop_ratio: float

    # Ellipse mask parameters (in normalized coordinates [-1..1])
    center: Tuple[float, float]      # (cx, cy)
    axes: Tuple[float, float]     # (ax, ay)
    edge_softness: float                   # bigger = softer edge

    # Mask shaping: >1 makes the outside fall off faster (less background leakage)
    # Recommended: 1.5 - 3.0. Try 2.0.
    mask_power: float

    # If True, add noise outside mask instead of constant
    randomize_background: bool
    background_noise_std: float          # noise in [0,1] scale

    # Small random jitter of ellipse per image (train only)
    jitter_center: float                # +/- jitter in normalized coords
    jitter_axes: float                    # +/- relative jitter on axes

    # What value to use outside mask (if not randomizing)
    background_fill: float                 # mid-gray


class CenterCropMinSide(torch.nn.Module):
    """
    Center-crop a PIL image to a square based on the shorter side.
    Optionally crop tighter with ratio < 1.0 to zoom in.
    """
    def __init__(self, ratio: float = 1.0):
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
    Keeps inside ellipse, suppresses outside ellipse.
    """
    def __init__(self, cfg: FaceFocusConfig, train: bool):
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

        # sample jitter
        cx, cy = cfg.center
        ax, ay = cfg.axes

        if self.train:
            if cfg.jitter_center > 0:
                cx = cx + (2 * torch.rand((), device=device) - 1).item() * cfg.jitter_center
                cy = cy + (2 * torch.rand((), device=device) - 1).item() * cfg.jitter_center
            if cfg.jitter_axes > 0:
                ax = ax * (1.0 + (2 * torch.rand((), device=device) - 1).item() * cfg.jitter_axes)
                ay = ay * (1.0 + (2 * torch.rand((), device=device) - 1).item() * cfg.jitter_axes)

        # coordinate grid in [-1,1]
        yy = torch.linspace(-1.0, 1.0, h, device=device).view(h, 1).expand(h, w)
        xx = torch.linspace(-1.0, 1.0, w, device=device).view(1, w).expand(h, w)

        dx = (xx - cx) / max(ax, 1e-6)
        dy = (yy - cy) / max(ay, 1e-6)
        r = torch.sqrt(dx * dx + dy * dy)

        sigma = max(cfg.edge_softness, 1e-6)
        mask = torch.exp(-torch.clamp(r - 1.0, min=0.0) / sigma)  # (H,W) in (0,1]

        # Make outside fall off faster (reduces background leakage)
        mp = float(getattr(cfg, "mask_power", 1.0))
        if mp != 1.0:
            mask = mask.pow(mp)

        mask = mask.unsqueeze(0).expand(c, h, w)

        if cfg.randomize_background and self.train:
            noise = torch.randn_like(x) * cfg.background_noise_std
            bg = torch.clamp(cfg.background_fill + noise, 0.0, 1.0)
        else:
            bg = torch.full_like(x, cfg.background_fill)

        out = x * mask + bg * (1.0 - mask)
        return out


def build_focus_face_transform(
    train: bool,
    cfg: FaceFocusConfig,
    normalize_mean: float,
    normalize_std: float,
) -> transforms.Compose:
    """
    Correct order:
      CenterCrop (min-side, optional tighter) ->
      Grayscale ->
      Resize(cfg.size) ->
      ToTensor ([0,1]) ->
      Ellipse mask / background fill ->
      Normalize
    """
    ops = [
        CenterCropMinSide(ratio=getattr(cfg, "pre_crop_ratio", 1.0)),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((cfg.size, cfg.size)),
        transforms.ToTensor(),
        SoftEllipseMask(cfg=cfg, train=train),
        transforms.Normalize(mean=[normalize_mean], std=[normalize_std]),
    ]
    return transforms.Compose(ops)
