from __future__ import annotations

import math
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def inv_softplus(y: float) -> float:
    """
    Inverse softplus for y > 0.
    Ensures: softplus(inv_softplus(y)) == y
    """
    return math.log(math.exp(y) - 1.0)


class WeightedL1Embedder(nn.Module):
    """
    Paper's comparison/energy function:
        s = sum_j alpha_j * |h1_j - h2_j|
    We additionally constrain alpha_j >= 0 to keep this a proper weighted distance.
    """

    def __init__(self, embedding_dim: int, alpha_init: float = 1.0) -> None:
        super().__init__()
        # Store raw params and map through softplus => alpha > 0.
        # Initialize so that effective alpha starts at alpha_init (paper-friendly).
        raw_init = inv_softplus(float(alpha_init))
        self.alpha_raw = nn.Parameter(torch.full((embedding_dim,), float(raw_init)))

    def forward(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        d = torch.abs(h1 - h2)  # (B, D)
        alpha = F.softplus(self.alpha_raw)  # (D,) strictly positive
        s = torch.sum(alpha * d, dim=1, keepdim=True)  # (B, 1)
        return s


class LogitDecider(nn.Module):
    """
    Produces logits instead of probabilities for numerical stability:
      logits = bias - score
    """
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, score: torch.Tensor) -> torch.Tensor:
        return self.bias - score


class FullSiameseModel(nn.Module):
    """
    Convenience wrapper:
      x1 -> siamese_cnn -> h1
      x2 -> siamese_cnn -> h2
      (h1,h2) -> embedder -> score
      score -> decider -> logits
    """

    def __init__(
        self,
        siamese_cnn: nn.Module,
        embedder: WeightedL1Embedder,
        decider: LogitDecider,
    ) -> None:
        super().__init__()
        self.siamese_cnn = siamese_cnn
        self.embedder = embedder
        self.decider = decider

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        h1 = self.siamese_cnn(x1)
        h2 = self.siamese_cnn(x2)
        score = self.embedder(h1, h2)
        logits = self.decider(score)
        return logits
