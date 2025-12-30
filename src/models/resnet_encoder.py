from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet18Encoder(nn.Module):
    """ResNet18-based embedding encoder.

    Rationale:
    - Your dataset is small; the paper Omniglot CNN can underfit/overfit depending on preprocessing.
    - A pretrained ResNet18 backbone (ImageNet) usually gives a substantial jump in separability,
      often enough to push pair-accuracy beyond 0.80 when combined with sensible augmentation
      and threshold selection.

    Notes:
    - Expects grayscale inputs shaped (B,1,H,W). If you provide RGB, set in_channels=3.
    - Outputs an embedding of size embedding_dim.
    - Optional L2 normalization at the end is recommended for metric-style heads.
    """

    def __init__(
        self,
        in_channels: int = 1,
        embedding_dim: int = 512,
        pretrained: bool = True,
        l2_normalize: bool = True,
        l2_eps: float = 1e-12,
        dropout_p: float = 0.0,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        try:
            from torchvision.models import resnet18, ResNet18_Weights
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "torchvision is required for ResNet18Encoder. Install torchvision and retry."
            ) from e

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        m = resnet18(weights=weights)

        # Adapt first conv to match in_channels
        if in_channels != 3:
            old = m.conv1
            new = nn.Conv2d(
                in_channels,
                old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                bias=old.bias is not None,
            )
            with torch.no_grad():
                if weights is not None:
                    # old.weight: (64,3,7,7). Convert to 1ch by mean.
                    if in_channels == 1:
                        new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
                    else:
                        # For arbitrary channels, tile/trim the mean weights.
                        w = old.weight.mean(dim=1, keepdim=True)
                        new.weight.copy_(w.repeat(1, in_channels, 1, 1)[:, :in_channels])
                else:
                    nn.init.kaiming_normal_(new.weight, mode="fan_out", nonlinearity="relu")
            m.conv1 = new

        # Replace classifier head with projection to embedding_dim
        backbone_dim = m.fc.in_features
        m.fc = nn.Identity()
        self.backbone = m

        self.proj = nn.Linear(backbone_dim, int(embedding_dim), bias=True)
        self.dropout = nn.Dropout(p=float(dropout_p)) if float(dropout_p) > 0.0 else nn.Identity()

        self.l2_normalize = bool(l2_normalize)
        self.l2_eps = float(l2_eps)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Enable/disable gradient updates for the ResNet backbone."""
        trainable = bool(trainable)
        for p in self.backbone.parameters():
            p.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected (B,C,H,W), got {tuple(x.shape)}")
        z = self.backbone(x)          # (B, backbone_dim)
        z = self.dropout(z)
        h = self.proj(z)              # (B, embedding_dim)

        if self.l2_normalize:
            h = F.normalize(h, p=2, dim=1, eps=self.l2_eps)
        return h

    @property
    def embedding_dim(self) -> int:
        return int(self.proj.out_features)


if __name__ == "__main__":
    enc = ResNet18Encoder(in_channels=1, embedding_dim=256, pretrained=False)
    x = torch.randn(2, 1, 160, 160)
    y = enc(x)
    print(y.shape)
