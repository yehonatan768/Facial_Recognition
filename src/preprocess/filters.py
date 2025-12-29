from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from PIL import Image, ImageFilter, ImageOps
import torch


EqualizeMethod = Literal["autocontrast", "equalize", "clahe"]


@dataclass
class EqualizeConfig:
    enabled: bool = False
    method: EqualizeMethod = "autocontrast"

    # autocontrast params
    autocontrast_cutoff: float = 0.0  # percent 0..100
    autocontrast_ignore: Optional[int] = None  # e.g., 0 to ignore black

    # clahe params (requires cv2)
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8


class ContrastEqualize(torch.nn.Module):
    """
    Deterministic contrast normalization on PIL images.

    - autocontrast: robust contrast stretch (recommended default)
    - equalize: global histogram equalization
    - clahe: adaptive hist-eq (requires opencv-python)
    """

    def __init__(self, cfg: EqualizeConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            raise TypeError("ContrastEqualize expects PIL.Image")

        if not self.cfg.enabled:
            return img

        # Work in grayscale for stable behavior
        in_mode = img.mode
        g = img.convert("L")

        if self.cfg.method == "autocontrast":
            out = ImageOps.autocontrast(
                g,
                cutoff=float(self.cfg.autocontrast_cutoff),
                ignore=self.cfg.autocontrast_ignore,
            )

        elif self.cfg.method == "equalize":
            out = ImageOps.equalize(g)

        elif self.cfg.method == "clahe":
            try:
                import cv2  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "filters.equalize.method='clahe' requires OpenCV (cv2). "
                    "Install opencv-python or switch to 'autocontrast'/'equalize'."
                ) from e

            arr = np.asarray(g, dtype=np.uint8)
            tgs = int(self.cfg.clahe_tile_grid_size)
            clahe = cv2.createCLAHE(
                clipLimit=float(self.cfg.clahe_clip_limit),
                tileGridSize=(tgs, tgs),
            )
            arr2 = clahe.apply(arr)
            out = Image.fromarray(arr2, mode="L")

        else:
            raise ValueError(f"Unknown equalize method: {self.cfg.method!r}")

        # Keep compatibility with downstream steps that may expect RGB-like images
        if in_mode in ("RGB", "RGBA"):
            return out.convert("RGB")
        return out


@dataclass
class EdgeConfig:
    enabled: bool = False

    # currently supported edge method
    method: Literal["find_edges"] = "find_edges"

    # how to apply edges
    mode: Literal["mix", "edges_only"] = "mix"

    # when mode=mix: 0 => original, 1 => edges
    alpha: float = 0.35

    # optional smoothing to reduce speckle
    blur_radius: float = 0.0


class EdgeEmphasis(torch.nn.Module):
    """
    Deterministic edge emphasis on PIL images.

    Note: This is a representation change (not augmentation). If enabled, it should be
    applied consistently to train/val/test.
    """

    def __init__(self, cfg: EdgeConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            raise TypeError("EdgeEmphasis expects PIL.Image")

        if not self.cfg.enabled:
            return img

        g = img.convert("L")

        if self.cfg.method == "find_edges":
            edges = g.filter(ImageFilter.FIND_EDGES)
        else:
            raise ValueError(f"Unknown edge method: {self.cfg.method!r}")

        if float(self.cfg.blur_radius) > 0:
            edges = edges.filter(ImageFilter.GaussianBlur(radius=float(self.cfg.blur_radius)))

        if self.cfg.mode == "edges_only":
            out = edges
        elif self.cfg.mode == "mix":
            a = max(0.0, min(1.0, float(self.cfg.alpha)))
            edges2 = ImageOps.autocontrast(edges)
            out = Image.blend(g, edges2, alpha=a)
        else:
            raise ValueError(f"Unknown edge mode: {self.cfg.mode!r}")

        # Return RGB for compatibility with face_mask (if used before grayscale)
        return out.convert("RGB")
