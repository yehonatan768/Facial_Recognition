from __future__ import annotations

import torch
from PIL import Image
from torchvision.transforms import functional as TF


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
