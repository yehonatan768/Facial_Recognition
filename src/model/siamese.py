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


class SiameseNet(nn.Module):
    def __init__(self, encoder: nn.Module, head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        logits = self.head(h1, h2)
        return logits  # logits ONLY
