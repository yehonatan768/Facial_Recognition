from __future__ import annotations

from dataclasses import dataclass

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

    # Activation
    use_leaky_relu: bool = True
    leaky_slope: float = 0.10

    # BatchNorm control
    use_batchnorm: bool = True
    bn_on_conv4: bool = False   # safer default


class ConvEmbeddingNet(nn.Module):
    """Twin-network embedding extractor.

    Paper structure (Figure 4):
      Conv -> ReLU -> MaxPool
      Conv -> ReLU -> MaxPool
      Conv -> ReLU -> MaxPool
      Conv -> ReLU
      Flatten
      FC -> Sigmoid

    This version:
      - preserves the structure
      - optionally replaces ReLU with LeakyReLU
      - optionally adds BatchNorm after conv layers
    """

    def __init__(self, cfg: ConvEmbeddingConfig):
        super().__init__()
        self.cfg = cfg

        # Convs
        self.conv1 = nn.Conv2d(cfg.in_channels, cfg.conv1_out, cfg.conv1_kernel)
        self.conv2 = nn.Conv2d(cfg.conv1_out, cfg.conv2_out, cfg.conv2_kernel)
        self.conv3 = nn.Conv2d(cfg.conv2_out, cfg.conv3_out, cfg.conv3_kernel)
        self.conv4 = nn.Conv2d(cfg.conv3_out, cfg.conv4_out, cfg.conv4_kernel)

        # BatchNorm (optional)
        self.bn1 = nn.BatchNorm2d(cfg.conv1_out) if cfg.use_batchnorm else nn.Identity()
        self.bn2 = nn.BatchNorm2d(cfg.conv2_out) if cfg.use_batchnorm else nn.Identity()
        self.bn3 = nn.BatchNorm2d(cfg.conv3_out) if cfg.use_batchnorm else nn.Identity()
        self.bn4 = (
            nn.BatchNorm2d(cfg.conv4_out)
            if (cfg.use_batchnorm and cfg.bn_on_conv4)
            else nn.Identity()
        )

        # Activation
        if cfg.use_leaky_relu:
            self.act = nn.LeakyReLU(cfg.leaky_slope, inplace=True)
        else:
            self.act = nn.ReLU(inplace=True)

        self.pool = nn.MaxPool2d(cfg.pool_kernel, cfg.pool_stride)

        # FC (lazy)
        self._fc_in: int | None = None
        self.fc = nn.Linear(1, cfg.fc_out)  # placeholder
        self.sigmoid = nn.Sigmoid()

    def _ensure_fc(self, x: torch.Tensor) -> None:
        if self._fc_in is not None:
            return
        self._fc_in = int(x.shape[1])
        self.fc = nn.Linear(self._fc_in, self.cfg.fc_out).to(x.device, x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.act(self.bn1(self.conv1(x))))
        x = self.pool(self.act(self.bn2(self.conv2(x))))
        x = self.pool(self.act(self.bn3(self.conv3(x))))
        x = self.act(self.bn4(self.conv4(x)))
        x = torch.flatten(x, 1)
        self._ensure_fc(x)
        x = self.sigmoid(self.fc(x))
        return x
