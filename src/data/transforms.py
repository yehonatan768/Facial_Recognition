from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

import torch
from PIL import Image
from torchvision.transforms import functional as TF


def _maybe(p: float) -> bool:
    return random.random() < p


@dataclass
class PaperImageTransform:
    """
    Paper-style preprocessing + augmentation:
      - Convert to grayscale (1-channel)
      - Resize to model input_size (e.g., 105x105)
      - Convert to tensor in [0,1]
      - If train: apply stochastic affine distortions where each component
        is included with probability p (paper: 0.5).
    """

    input_size: int
    augment_enabled: bool
    # paper params
    rotation_deg: Tuple[float, float]
    shear_x: Tuple[float, float]
    shear_y: Tuple[float, float]
    scale_x: Tuple[float, float]
    scale_y: Tuple[float, float]
    translate_x_px: Tuple[float, float]
    translate_y_px: Tuple[float, float]
    component_apply_prob: float

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], train: bool) -> "PaperImageTransform":
        size = int(cfg["model"]["input_size"])
        aug = cfg["augment"]
        return cls(
            input_size=size,
            augment_enabled=bool(aug["enabled"]) and train,
            rotation_deg=(float(aug["rotation_deg"][0]), float(aug["rotation_deg"][1])),
            shear_x=(float(aug["shear_x"][0]), float(aug["shear_x"][1])),
            shear_y=(float(aug["shear_y"][0]), float(aug["shear_y"][1])),
            scale_x=(float(aug["scale_x"][0]), float(aug["scale_x"][1])),
            scale_y=(float(aug["scale_y"][0]), float(aug["scale_y"][1])),
            translate_x_px=(float(aug["translate_x_px"][0]), float(aug["translate_x_px"][1])),
            translate_y_px=(float(aug["translate_y_px"][0]), float(aug["translate_y_px"][1])),
            component_apply_prob=float(aug["component_apply_prob"]),
        )

    def _preprocess(self, img: Image.Image) -> Image.Image:
        # Paper uses single-channel input (Omniglot is binary/single channel).
        img = img.convert("L")
        if img.size != (self.input_size, self.input_size):
            img = img.resize((self.input_size, self.input_size))
        return img

    def _apply_paper_affine(self, img: Image.Image) -> Image.Image:
        """
        Paper: T = (theta, rho_x, rho_y, s_x, s_y, t_x, t_y),
        with each component included with probability p.

        torchvision's affine has one scale and a 2D shear; we approximate the
        paper's anisotropic scaling by composing two affines.
        """
        p = self.component_apply_prob

        # Defaults = identity
        angle = 0.0
        shx = 0.0
        shy = 0.0
        scx = 1.0
        scy = 1.0
        tx = 0.0
        ty = 0.0

        if _maybe(p):
            angle = random.uniform(*self.rotation_deg)
        if _maybe(p):
            shx = random.uniform(*self.shear_x)
        if _maybe(p):
            shy = random.uniform(*self.shear_y)
        if _maybe(p):
            scx = random.uniform(*self.scale_x)
        if _maybe(p):
            scy = random.uniform(*self.scale_y)
        if _maybe(p):
            tx = random.uniform(*self.translate_x_px)
        if _maybe(p):
            ty = random.uniform(*self.translate_y_px)

        # First affine: rotation + translation + x-scale + x-shear
        img = TF.affine(
            img,
            angle=angle,
            translate=[int(round(tx)), int(round(ty))],
            scale=scx,
            shear=[shx, 0.0],
        )
        # Second affine: y-scale + y-shear (compose)
        img = TF.affine(
            img,
            angle=0.0,
            translate=[0, 0],
            scale=scy,
            shear=[0.0, shy],
        )
        return img

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img = self._preprocess(img)

        if self.augment_enabled:
            img = self._apply_paper_affine(img)

        # To tensor (1,H,W) in [0,1]
        return TF.to_tensor(img)
