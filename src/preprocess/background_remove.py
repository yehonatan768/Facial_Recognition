from __future__ import annotations
import cv2
import numpy as np
from PIL import Image


class BackgroundRemover:
    """
    Lightweight foreground extraction using GrabCut-like heuristics.
    Deterministic, fast, no ML dependency.
    """

    def __init__(self, blur_ksize: int = 5):
        self.blur_ksize = blur_ksize

    def __call__(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            raise TypeError("BackgroundRemover expects PIL.Image")

        # Convert to OpenCV BGR
        img_np = np.array(img.convert("RGB"))
        h, w, _ = img_np.shape

        # Initial mask: assume center is foreground
        mask = np.zeros((h, w), np.uint8)
        rect = (
            int(w * 0.1),
            int(h * 0.1),
            int(w * 0.8),
            int(h * 0.8),
        )

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        cv2.grabCut(
            img_np,
            mask,
            rect,
            bgdModel,
            fgdModel,
            iterCount=3,
            mode=cv2.GC_INIT_WITH_RECT,
        )

        # Foreground mask
        fg_mask = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
            1,
            0,
        ).astype("uint8")

        # Smooth edges
        fg_mask = cv2.GaussianBlur(
            fg_mask * 255,
            (self.blur_ksize, self.blur_ksize),
            0,
        ) / 255.0

        # Apply mask
        out = img_np * fg_mask[..., None]
        out = out.astype(np.uint8)

        return Image.fromarray(out)
