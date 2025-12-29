from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

import torch
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights


# COCO class index for "person"
_PERSON_CLASS = 15

# Cache model per process (important for DataLoader workers)
_MODEL: Optional[torch.nn.Module] = None
_MODEL_DEVICE: Optional[str] = None

# Use official weights + preprocessing
_WEIGHTS = DeepLabV3_ResNet50_Weights.DEFAULT
_PREPROCESS = _WEIGHTS.transforms()


@dataclass(frozen=True)
class BgRemoveConfig:
    """
    TorchVision DeepLabV3-based background remover (COCO person segmentation).

    backend: must be "torchvision_deeplabv3"

    device:
      - "cpu" is strongly recommended when used inside DataLoader workers.
      - "cuda" is only recommended if num_workers=0 (single-process) or for offline preprocessing.

    threshold:
      Person probability threshold in [0..1]. Higher => keeps less foreground.
      Typical range: 0.25 - 0.50

    Fail-safe:
      If output becomes too empty / low-variance => return original image.
    """
    enabled: bool
    backend: str
    device: str

    threshold: float

    min_fg_fraction: float
    fg_black_threshold: int
    min_gray_variance: float


def _get_model(device: str) -> torch.nn.Module:
    global _MODEL, _MODEL_DEVICE

    dev = str(device).strip().lower()
    if dev not in ("cpu", "cuda"):
        raise ValueError(f"BgRemoveConfig.device must be 'cpu' or 'cuda', got {device!r}")

    # If requested CUDA but unavailable, gracefully fall back to CPU
    if dev == "cuda" and not torch.cuda.is_available():
        dev = "cpu"

    if _MODEL is not None and _MODEL_DEVICE == dev:
        return _MODEL

    model = deeplabv3_resnet50(weights=_WEIGHTS)
    model.eval()
    model.to(torch.device(dev))

    _MODEL = model
    _MODEL_DEVICE = dev
    return model


class BackgroundRemover:
    def __init__(self, cfg: BgRemoveConfig):
        self.cfg = cfg

        raw = str(cfg.backend)

        # Robust normalization: keep only [a-z0-9_]
        backend = "".join(ch for ch in raw.strip().lower() if (ch.isalnum() or ch == "_"))

        # Allow a no-op backend
        if backend in ("none", "off", "disabled"):
            self.cfg = BgRemoveConfig(
                enabled=False,
                backend="none",
                device=str(cfg.device),
                threshold=float(cfg.threshold),
                min_fg_fraction=float(cfg.min_fg_fraction),
                fg_black_threshold=int(cfg.fg_black_threshold),
                min_gray_variance=float(cfg.min_gray_variance),
            )
            return

        # Accept deeplab backend aliases
        if backend in ("torchvision_deeplabv3", "deeplabv3"):
            # normalize stored backend (optional)
            self.cfg = BgRemoveConfig(
                enabled=bool(cfg.enabled),
                backend="torchvision_deeplabv3",
                device=str(cfg.device),
                threshold=float(cfg.threshold),
                min_fg_fraction=float(cfg.min_fg_fraction),
                fg_black_threshold=int(cfg.fg_black_threshold),
                min_gray_variance=float(cfg.min_gray_variance),
            )
            return

        raise ValueError(f"Unsupported background remover backend: {cfg.backend!r}")

    @torch.no_grad()
    def __call__(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            raise TypeError(f"BackgroundRemover expects PIL.Image, got {type(img)}")

        if not self.cfg.enabled:
            return img

        inp = img.convert("RGB")

        # Prepare input tensor using official weights transforms
        x = _PREPROCESS(inp).unsqueeze(0)  # (1,3,H,W)

        model = _get_model(self.cfg.device)
        device = next(model.parameters()).device
        x = x.to(device)

        # Forward
        out = model(x)["out"]  # (1,C,H,W)
        probs = torch.softmax(out, dim=1)[0, _PERSON_CLASS]  # (H,W)

        thr = float(self.cfg.threshold)
        mask = (probs > thr).to(torch.uint8).cpu().numpy()  # (H,W) {0,1}

        rgb = np.asarray(inp, dtype=np.uint8)  # (H,W,3)
        out_arr = np.zeros_like(rgb, dtype=np.uint8)
        m = mask.astype(bool)
        out_arr[m] = rgb[m]

        # ---- Fail-safe checks: avoid output being empty/black ----
        bthr = int(self.cfg.fg_black_threshold)
        non_black = (out_arr[..., 0] > bthr) | (out_arr[..., 1] > bthr) | (out_arr[..., 2] > bthr)
        fg_frac = float(non_black.mean())

        gray = (0.299 * out_arr[..., 0] + 0.587 * out_arr[..., 1] + 0.114 * out_arr[..., 2]).astype(np.float32)
        var = float(gray.var())

        if fg_frac < float(self.cfg.min_fg_fraction) or var < float(self.cfg.min_gray_variance):
            # Return original image (NOT masked)
            return inp

        return Image.fromarray(out_arr, mode="RGB")
