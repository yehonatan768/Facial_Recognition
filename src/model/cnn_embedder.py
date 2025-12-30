from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ConvEmbeddingConfig:
    in_channels: int = 1
    conv1_out: int = 64
    conv1_kernel: int = 10
    conv2_out: int = 128
    conv2_kernel: int = 7
    conv3_out: int = 128
    conv3_kernel: int = 4
    conv4_out: int = 256
    conv4_kernel: int = 4
    pool_kernel: int = 2
    pool_stride: int = 2
    fc_out: int = 4096


class ConvEmbeddingNet(nn.Module):
    """Twin-network embedding extractor.

    Architecture (as used in the original one-shot Siamese network):
      Conv -> ReLU -> MaxPool
      Conv -> ReLU -> MaxPool
      Conv -> ReLU -> MaxPool
      Conv -> ReLU
      Flatten
      FC -> Sigmoid  (embedding vector)
    """

    def __init__(self, cfg: ConvEmbeddingConfig):
        super().__init__()
        self.cfg = cfg

        self.conv1 = nn.Conv2d(cfg.in_channels, cfg.conv1_out, kernel_size=cfg.conv1_kernel, stride=1, padding=0)
        self.conv2 = nn.Conv2d(cfg.conv1_out, cfg.conv2_out, kernel_size=cfg.conv2_kernel, stride=1, padding=0)
        self.conv3 = nn.Conv2d(cfg.conv2_out, cfg.conv3_out, kernel_size=cfg.conv3_kernel, stride=1, padding=0)
        self.conv4 = nn.Conv2d(cfg.conv3_out, cfg.conv4_out, kernel_size=cfg.conv4_kernel, stride=1, padding=0)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=cfg.pool_kernel, stride=cfg.pool_stride)

        # FC input dimension depends on input size; we infer on first forward.
        self._fc_in: int | None = None
        self.fc = nn.Linear(1, cfg.fc_out)  # placeholder; reset lazily
        self.sigmoid = nn.Sigmoid()

    def _ensure_fc(self, x: torch.Tensor) -> None:
        if self._fc_in is not None:
            return
        self._fc_in = int(x.shape[1])

        # Create the FC layer on the same device/dtype as x
        self.fc = nn.Linear(self._fc_in, self.cfg.fc_out).to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.relu(self.conv4(x))
        x = torch.flatten(x, start_dim=1)
        self._ensure_fc(x)
        x = self.sigmoid(self.fc(x))
        return x
