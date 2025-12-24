from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

import torch
from PIL import Image
import math
import torch.nn.functional as F


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

    def _apply_paper_affine_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply ONE affine warp built from paper parameters:
          theta, rho_x, rho_y, s_x, s_y, t_x, t_y
        each included with probability p.

        x: (1,H,W) float tensor in [0,1]
        returns: (1,H,W)
        """
        p = self.component_apply_prob

        # identity defaults
        theta = 0.0
        rho_x = 0.0
        rho_y = 0.0
        s_x = 1.0
        s_y = 1.0
        t_x = 0.0
        t_y = 0.0

        if _maybe(p):
            theta = random.uniform(*self.rotation_deg)  # degrees
        if _maybe(p):
            rho_x = random.uniform(*self.shear_x)
        if _maybe(p):
            rho_y = random.uniform(*self.shear_y)
        if _maybe(p):
            s_x = random.uniform(*self.scale_x)
        if _maybe(p):
            s_y = random.uniform(*self.scale_y)
        if _maybe(p):
            t_x = random.uniform(*self.translate_x)  # pixels
        if _maybe(p):
            t_y = random.uniform(*self.translate_y)  # pixels

        # Build affine matrix in pixel coords: A = T * R * Sh * S
        # Sh = [[1, rho_x],
        #       [rho_y, 1]]
        H = x.shape[-2]
        W = x.shape[-1]

        th = math.radians(theta)
        c = math.cos(th)
        s = math.sin(th)

        # 2x2 matrices
        S = torch.tensor([[s_x, 0.0],
                          [0.0, s_y]], dtype=torch.float32)
        Sh = torch.tensor([[1.0, rho_x],
                           [rho_y, 1.0]], dtype=torch.float32)
        R = torch.tensor([[c, -s],
                          [s, c]], dtype=torch.float32)

        A2 = R @ Sh @ S  # 2x2
        t = torch.tensor([t_x, t_y], dtype=torch.float32)  # pixels

        # Convert pixel-space affine to normalized-space affine for affine_grid.
        # Use homogeneous 3x3:
        # x_pix = N_pix @ x_norm, where N_pix maps [-1,1] -> [0,W-1] and [0,H-1]
        N_pix = torch.tensor([
            [(W - 1) / 2.0, 0.0, (W - 1) / 2.0],
            [0.0, (H - 1) / 2.0, (H - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ], dtype=torch.float32)

        N_norm = torch.inverse(N_pix)

        A_pix = torch.eye(3, dtype=torch.float32)
        A_pix[0:2, 0:2] = A2
        A_pix[0:2, 2] = t

        A_norm = N_norm @ A_pix @ N_pix
        theta_norm = A_norm[0:2, :]  # 2x3

        # Warp once
        x_b = x.unsqueeze(0)  # (B=1,C=1,H,W)
        theta_b = theta_norm.unsqueeze(0)  # (1,2,3)
        grid = F.affine_grid(theta_b, size=x_b.shape, align_corners=True)
        y = F.grid_sample(
            x_b, grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return y.squeeze(0)

