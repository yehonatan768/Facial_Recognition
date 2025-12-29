from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

import mediapipe as mp


@dataclass(frozen=True)
class MpSegConfig:
    """
    MediaPipe Selfie Segmentation config.

    model_selection:
      0 = general model (default)
      1 = landscape model (often cleaner on wider shots)
    """
    enabled: bool
    backend: str  # must be "mediapipe_selfie"

    model_selection: int
    threshold: float

    # Safety fallback thresholds (same idea as before)
    min_fg_fraction: float
    fg_black_threshold: int
    min_gray_variance: float


class BackgroundRemover:
    """
    MediaPipe-based background removal with safety fallback:
    If segmentation removes almost everything, return the original image.
    """

    def __init__(self, cfg: MpSegConfig):
        self.cfg = cfg
        if str(cfg.backend).strip().lower() != "mediapipe_selfie":
            raise ValueError(f"Unsupported background remover backend: {cfg.backend!r}")

        self._mp = mp.solutions.selfie_segmentation
        self._segmenter = self._mp.SelfieSegmentation(model_selection=int(cfg.model_selection))

    def __call__(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            raise TypeError(f"BackgroundRemover expects PIL.Image, got {type(img)}")

        if not self.cfg.enabled:
            return img

        inp = img.convert("RGB")
        rgb = np.asarray(inp, dtype=np.uint8)  # (H,W,3), RGB

        # MediaPipe expects RGB ndarray
        res = self._segmenter.process(rgb)
        mask = getattr(res, "segmentation_mask", None)

        # If mediapipe fails, do not destroy the sample
        if mask is None:
            return inp

        # Hard threshold (background -> black)
        thr = float(self.cfg.threshold)
        fg = (mask > thr)  # (H,W) boolean

        out = np.zeros_like(rgb, dtype=np.uint8)
        out[fg] = rgb[fg]
        out_rgb = Image.fromarray(out, mode="RGB")

        # ---- Safety checks: avoid "all black" samples ----
        arr = np.asarray(out_rgb, dtype=np.uint8)
        bthr = int(self.cfg.fg_black_threshold)

        non_black = (arr[..., 0] > bthr) | (arr[..., 1] > bthr) | (arr[..., 2] > bthr)
        fg_frac = float(non_black.mean())

        gray = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.float32)
        var = float(gray.var())

        if fg_frac < float(self.cfg.min_fg_fraction) or var < float(self.cfg.min_gray_variance):
            return inp

        return out_rgb
