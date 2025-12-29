from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

import torch
from torchvision import transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights


@dataclass(frozen=True)
class BgRemoveConfig:
    """
    TorchVision DeepLabV3 person segmentation background removal.

    backend: must be "torchvision_deeplabv3"

    device:
      "cpu" recommended when using DataLoader workers.
      "cuda" only if you set num_workers=0 (or you precompute offline).
    """
    enabled: bool
    backend: str  # "torchvision_deeplabv3"
    device: str   # "cpu" or "cuda"

    # Segmentation threshold on person probability (0..1)
    threshold: float

    # Fail-safe fallback (prevents "all black" outputs)
    min_fg_fraction: float
    fg_black_threshold: int
    min_gray_variance: float


# COCO "person" class index in common segmentation models
_PERSON_CLASS = 15

# Global cached model per process
_MODEL: Optional[torch.nn.Module] = None
_MODEL_DEVICE: Optional[str] = None


def _get_model(device: str) -> torch.nn.Module:
    global _MODEL, _MODEL_DEVICE
    device = str(device).lower().strip()
    if device not in ("cpu", "cuda"):
        raise ValueError(f"Invalid device for background remover: {device!r} (use 'cpu' or 'cuda')")

    if _MODEL is not None and _MODEL_DEVICE == device:
        return _MODEL

    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = deeplabv3_resnet50(weights=weights)
    model.eval()
    model.to(torch.device(device))

    _MODEL = model
    _MODEL_DEVICE = device
    return model


# Preprocess as TorchVision weights expect
_weights = DeepLabV3_ResNet50_Weights.DEFAULT
_preprocess = _weights.transforms()


class BackgroundRemover:
    """
    Background removal using TorchVision DeepLabV3 person segmentation.

    Output:
      - RGB with background black.
      - Fail-safe fallback to original image if output is too empty / low-variance.
    """

    def __init__(self, cfg: BgRemoveConfig):
        self.cfg = cfg
        if str(cfg.backend).strip().lower() != "torchvision_deeplabv3":
            raise ValueError(f"Unsupported background remover backend: {cfg.backend!r}")

    @torch.no_grad()
    def __call__(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            raise TypeError(f"BackgroundRemover expects PIL.Image, got {type(img)}")

        if not self.cfg.enabled:
            return img

        inp = img.convert("RGB")
        device = str(self.cfg.device).lower().strip()

        # TorchVision preprocess returns a tensor (C,H,W) normalized for the model
        x = _preprocess(inp).unsqueeze(0)  # (1,3,H,W)

        model = _get_model(device=device)
        x = x.to(torch.device(device))

        out = model(x)["out"]  # (1, num_classes, H, W)
        probs = torch.softmax(out, dim=1)[0, _PERSON_CLASS]  # (H,W)

        mask = (probs > float(self.cfg.threshold)).to(torch.uint8).cpu().numpy()  # (H,W) {0,1}

        rgb = np.asarray(inp, dtype=np.uint8)  # (H,W,3)
        out_arr = np.zeros_like(rgb, dtype=np.uint8)
        out_arr[mask.astype(bool)] = rgb[mask.astype(bool)]

        out_rgb = Image.fromarray(out_arr, mode="RGB")

        # ---- Fail-safe checks: prevent "all black" faces ----
        arr = out_arr
        bthr = int(self.cfg.fg_black_threshold)

        non_black = (arr[..., 0] > bthr) | (arr[..., 1] > bthr) | (arr[..., 2] > bthr)
        fg_frac = float(non_black.mean())

        gray = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.float32)
        var = float(gray.var())

        if fg_frac < float(self.cfg.min_fg_fraction) or var < float(self.cfg.min_gray_variance):
            return inp

        return out_rgb
