from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNEncoder(nn.Module):
    """
    Paper-style conv stack, but configurable:
      - input_size (H=W)
      - embedding_dim (FC output dim)
      - optional L2-normalization on embedding (recommended for verification)
      - optional dropout before FC

    If enforce_input_size is True, checks that input matches input_size.
    """

    def __init__(
        self,
        in_channels: int = 1,
        input_size: int = 105,
        embedding_dim: int = 4096,
        enforce_input_size: bool = True,
        activation: str = "leaky_relu",      # "relu" | "leaky_relu"
        embed_activation: str = "none",       # "sigmoid" | "none"
        l2_normalize: bool = True,            # <<< NEW
        l2_eps: float = 1e-12,                # <<< NEW
        dropout_p: float = 0.0,               # <<< NEW (0.2-0.4 often helps)
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.input_size = int(input_size)
        self.embedding_dim = int(embedding_dim)
        self.enforce_input_size = bool(enforce_input_size)

        self.embed_activation = str(embed_activation)
        self.l2_normalize = bool(l2_normalize)
        self.l2_eps = float(l2_eps)
        self.dropout_p = float(dropout_p)

        if activation == "relu":
            Act = lambda: nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            Act = lambda: nn.LeakyReLU(negative_slope=0.01, inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation!r}")

        # ---- conv stack (unchanged) ----
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=10, stride=1, padding=0),
            Act(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=0),
            Act(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=4, stride=1, padding=0),
            Act(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=0),
            Act(),
        )

        # ---- build FC dynamically based on input_size ----
        flatten_dim, (c, h, w) = self._infer_flatten_dim(self.input_size)
        self._feature_shape = (c, h, w)

        layers = [nn.Flatten()]
        if self.dropout_p > 0.0:
            layers.append(nn.Dropout(p=self.dropout_p))

        layers.append(nn.Linear(flatten_dim, self.embedding_dim))

        if self.embed_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif self.embed_activation == "none":
            pass
        else:
            raise ValueError(f"Unsupported embed_activation: {self.embed_activation!r}")

        self.fc = nn.Sequential(*layers)

    def _infer_flatten_dim(self, input_size: int) -> Tuple[int, Tuple[int, int, int]]:
        with torch.no_grad():
            x = torch.zeros(1, self.in_channels, input_size, input_size)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            c, h, w = x.shape[1], x.shape[2], x.shape[3]
            return int(c * h * w), (int(c), int(h), int(w))

    def _check_input(self, x: torch.Tensor) -> None:
        if x.dim() != 4:
            raise ValueError(f"Expected (B,C,H,W), got {tuple(x.shape)}")
        _, c, h, w = x.shape
        if c != self.in_channels:
            raise ValueError(f"Expected C={self.in_channels}, got C={c}")
        if self.enforce_input_size and (h != self.input_size or w != self.input_size):
            raise ValueError(
                f"Expected {self.input_size}x{self.input_size}, got {h}x{w}. "
                f"Either resize in transforms or set enforce_input_size=False."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc(x)  # (B, embedding_dim)

        # ---- NEW: L2-normalize embeddings (recommended for verification) ----
        if self.l2_normalize:
            x = F.normalize(x, p=2, dim=1, eps=self.l2_eps)

        return x
