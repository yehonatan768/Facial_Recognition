from __future__ import annotations

import numpy as np
from PIL import Image
import torch


class EllipseFaceMask(torch.nn.Module):
    """
    Soft elliptical face mask.
    Keeps center (face) and fades out background.
    """

    def __init__(
        self,
        center=(0.0, 0.0),     # normalized [-1,1]
        axes=(0.75, 0.90),     # ellipse radii
        edge_softness=0.08,    # softness of edge
        power=1.5,             # >1 => stronger background suppression
    ):
        super().__init__()
        self.center = center
        self.axes = axes
        self.edge_softness = edge_softness
        self.power = power

    def forward(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            raise TypeError("EllipseFaceMask expects PIL.Image")

        w, h = img.size
        yy, xx = np.mgrid[0:h, 0:w]

        # Normalize to [-1,1]
        x = (xx / (w - 1)) * 2 - 1 - self.center[0]
        y = (yy / (h - 1)) * 2 - 1 - self.center[1]

        rx, ry = self.axes
        r = (x / rx) ** 2 + (y / ry) ** 2

        # Soft mask
        mask = np.exp(-((r - 1.0).clip(min=0) / self.edge_softness))
        mask = mask ** self.power
        mask = np.clip(mask, 0.0, 1.0)

        arr = np.asarray(img).astype(np.float32)
        arr *= mask[..., None]

        return Image.fromarray(arr.astype(np.uint8))
