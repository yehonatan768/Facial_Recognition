from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Any, Dict

import torch
from torchvision import transforms
from PIL import Image


@dataclass
class PaperImageTransform:
    """
    Implements preprocessing/augmentation pipeline described in the paper.

    - Convert to grayscale
    - Resize to 105x105
    - (Train only) small random transforms (flip / jitter) if you want them
    - Convert to tensor in [0,1]
    - Normalize with fixed mean/std
    """

    train: bool
    input_size: int = 105   # paper default
    mean: float = 0.5
    std: float = 0.5
    jitter_brightness: float = 0.3
    jitter_contrast: float = 0.3
    jitter_saturation: float = 0.3
    jitter_hue: float = 0.02

    def __post_init__(self) -> None:
        ops = []

        # paper is grayscale
        ops.append(transforms.Grayscale(num_output_channels=1))
        ops.append(transforms.Resize((self.input_size, self.input_size)))

        if self.train:
            # augmentation (you can disable if you want *strict* paper behaviour)
            ops.append(
                transforms.ColorJitter(
                    brightness=self.jitter_brightness,
                    contrast=self.jitter_contrast,
                    saturation=self.jitter_saturation,
                    hue=self.jitter_hue,
                )
            )

        ops.append(transforms.ToTensor())
        ops.append(transforms.Normalize(mean=[self.mean], std=[self.std]))

        self.t = transforms.Compose(ops)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.t(img)

    # ------------------ NEW VERSION ------------------ #
    @classmethod
    def from_config(cls, cfg: Dict[str, Any], train: bool) -> "PaperImageTransform":
        """
        Safe config-based constructor:
        - If cfg["model"]["input_size"] (etc.) exists -> use it.
        - Otherwise fall back to paper defaults (105, 0.5, 0.5 etc.).
        This way an empty config.yaml still works.
        """

        def _get(d, key, default):
            if isinstance(d, dict):
                return d.get(key, default)
            # in case cfg is a SimpleNamespace or similar
            return getattr(d, key, default)

        model_cfg = _get(cfg, "model", {}) or {}

        input_size = int(
            model_cfg.get("input_size", 105)
        )  # paper: 105x105
        mean = float(model_cfg.get("mean", 0.5))
        std = float(model_cfg.get("std", 0.5))

        jitter_brightness = float(model_cfg.get("jitter_brightness", 0.3))
        jitter_contrast = float(model_cfg.get("jitter_contrast", 0.3))
        jitter_saturation = float(model_cfg.get("jitter_saturation", 0.3))
        jitter_hue = float(model_cfg.get("jitter_hue", 0.02))

        return cls(
            train=train,
            input_size=input_size,
            mean=mean,
            std=std,
            jitter_brightness=jitter_brightness,
            jitter_contrast=jitter_contrast,
            jitter_saturation=jitter_saturation,
            jitter_hue=jitter_hue,
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

