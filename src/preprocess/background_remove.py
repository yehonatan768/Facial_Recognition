from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from rembg import new_session, remove


@dataclass(frozen=True)
class RembgConfig:
    enabled: bool
    model: str

    alpha_matting: bool
    alpha_matting_foreground_threshold: int
    alpha_matting_background_threshold: int
    alpha_matting_erode_size: int

    # NEW: safety fallback
    min_fg_fraction: float          # e.g. 0.02 (2%)
    fg_black_threshold: int         # e.g. 12 (0..255)


class BackgroundRemover:
    """
    rembg-based background removal with a safety fallback:
    if rembg removes almost everything, return the original image.
    """

    def __init__(self, cfg: RembgConfig):
        self.cfg = cfg
        self._session = new_session(cfg.model)

    def __call__(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            raise TypeError(f"BackgroundRemover expects PIL.Image, got {type(img)}")

        if not self.cfg.enabled:
            return img

        inp = img.convert("RGB")

        out = remove(
            inp,
            session=self._session,
            alpha_matting=self.cfg.alpha_matting,
            alpha_matting_foreground_threshold=self.cfg.alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=self.cfg.alpha_matting_background_threshold,
            alpha_matting_erode_size=self.cfg.alpha_matting_erode_size,
        )

        # Force deterministic RGB with black background
        if out.mode == "RGBA":
            bg = Image.new("RGBA", out.size, (0, 0, 0, 255))
            out_rgb = Image.alpha_composite(bg, out).convert("RGB")
        else:
            out_rgb = out.convert("RGB")

        # ---- Safety check: did we delete almost everything? ----
        arr = np.asarray(out_rgb, dtype=np.uint8)  # (H,W,3)
        # Foreground = pixels that are not near-black
        thr = int(self.cfg.fg_black_threshold)
        fg = (arr[..., 0] > thr) | (arr[..., 1] > thr) | (arr[..., 2] > thr)
        fg_frac = float(fg.mean())

        if fg_frac < float(self.cfg.min_fg_fraction):
            # Fallback: return original input (do not lose the face)
            return inp

        return out_rgb
