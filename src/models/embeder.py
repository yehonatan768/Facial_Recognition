from __future__ import annotations

from typing import Dict, Any

import torch
import torch.nn as nn


class WeightedL1Embedder(nn.Module):
    """
    The paper's comparison/energy function:
        s = sum_j alpha_j * |h1_j - h2_j|
    where alpha_j are learnable parameters.
    """

    def __init__(self, embedding_dim: int, alpha_init: float = 1.0) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.full((embedding_dim,), float(alpha_init)))

    def forward(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        # (B, D)
        d = torch.abs(h1 - h2)
        s = torch.sum(self.alpha * d, dim=1, keepdim=True)  # (B, 1)
        return s


class SigmoidDecider(nn.Module):
    """
    Paper: a single sigmoidal output unit fed by the induced metric.
    """

    def __init__(self) -> None:
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, score: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(score)


class FullSiameseModel(nn.Module):
    """
    Convenience wrapper:
      x1 -> siamese_cnn -> h1
      x2 -> siamese_cnn -> h2
      (h1,h2) -> embedder -> score
      score -> decider -> p
    """

    def __init__(
        self,
        siamese_cnn: nn.Module,
        embedder: WeightedL1Embedder,
        decider: SigmoidDecider,
    ) -> None:
        super().__init__()
        self.siamese_cnn = siamese_cnn
        self.embedder = embedder
        self.decider = decider

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        h1 = self.siamese_cnn(x1)
        h2 = self.siamese_cnn(x2)
        score = self.embedder(h1, h2)
        p = self.decider(score)
        return p
