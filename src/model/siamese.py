from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.model.cnn_embedder import ConvEmbeddingConfig, ConvEmbeddingNet
from src.model.head import SimilarityHeadConfig, WeightedL1Head


@dataclass(frozen=True)
class SiameseConfig:
    encoder: ConvEmbeddingConfig
    head: SimilarityHeadConfig


class SiameseNetwork(nn.Module):
    """End-to-end siamese verification model."""

    def __init__(self, cfg: SiameseConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = ConvEmbeddingNet(cfg.encoder)
        self.head = WeightedL1Head(embed_dim=cfg.encoder.fc_out, cfg=cfg.head)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        return self.head(h1, h2)
