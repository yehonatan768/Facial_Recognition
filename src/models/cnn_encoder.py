from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    """
    Koch et al. (2015) Siamese CNN encoder (Figure 4) for Omniglot.

    Architecture (paper):
      Input: 1 x 105 x 105
      Conv(64, 10x10, stride=1, valid) + ReLU -> 64 x 96 x 96
      MaxPool(2x2, stride=2)                -> 64 x 48 x 48

      Conv(128, 7x7, stride=1, valid) + ReLU -> 128 x 42 x 42
      MaxPool(2x2, stride=2)                 -> 128 x 21 x 21

      Conv(128, 4x4, stride=1, valid) + ReLU -> 128 x 18 x 18
      MaxPool(2x2, stride=2)                 -> 128 x 9 x 9

      Conv(256, 4x4, stride=1, valid) + ReLU -> 256 x 6 x 6

      Flatten: 256*6*6 = 9216
      FC: 9216 -> 4096
      Sigmoid

    Output: embedding h in R^4096 (used by the Siamese head)

    Source: Figure 4 + Section 3.1 model description. :contentReference[oaicite:1]{index=1}
    """

    def __init__(self, in_channels: int = 1, enforce_105: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.enforce_105 = enforce_105

        # valid conv => padding=0, stride=1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=10, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        # Paper fixed input size 105x105 => 256x6x6 before FC
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.Sigmoid(),
        )

    def _check_input(self, x: torch.Tensor) -> None:
        if x.dim() != 4:
            raise ValueError(f"Expected (B,C,H,W), got {tuple(x.shape)}")
        b, c, h, w = x.shape
        if c != self.in_channels:
            raise ValueError(f"Expected C={self.in_channels}, got C={c}")
        if self.enforce_105 and (h != 105 or w != 105):
            raise ValueError(
                f"PaperCNNEncoder expects 105x105 inputs (paper). Got {h}x{w}. "
                f"Either resize in transforms or set enforce_105=False and adjust FC accordingly."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc(x)  # (B, 4096)
        return x


def paper_feature_map_shapes() -> Tuple[Tuple[int, int, int], ...]:
    """
    Returns the expected spatial shapes after each stage for 105x105 input.
    Useful for debugging / confirming you match the paper.
    """
    return (
        (64, 48, 48),   # after conv1+pool
        (128, 21, 21),  # after conv2+pool
        (128, 9, 9),    # after conv3+pool
        (256, 6, 6),    # after conv4
    )


if __name__ == "__main__":
    # Sanity check: confirm exact dimensions for paper input
    net = CNNEncoder(in_channels=1, enforce_105=True)
    x = torch.randn(2, 1, 105, 105)
    h = net(x)
    print("embedding:", h.shape)  # expected: torch.Size([2, 4096])
