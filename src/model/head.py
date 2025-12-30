from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class SimilarityHeadConfig:
    use_bias: bool = False


class WeightedL1Head(nn.Module):
    """Similarity head: p = sigmoid(sum_j alpha_j * |h1_j - h2_j|).

    This corresponds to a learned weighting of the component-wise L1 distance,
    followed by a sigmoid.
    """

    def __init__(self, embed_dim: int, cfg: SimilarityHeadConfig):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.cfg = cfg
        self.alpha = nn.Parameter(torch.zeros(self.embed_dim))
        self.bias = nn.Parameter(torch.zeros(())) if cfg.use_bias else None
        self.sigmoid = nn.Sigmoid()

    def forward(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        d = torch.abs(h1 - h2)  # [B, D]
        s = (d * self.alpha).sum(dim=1)  # [B]
        if self.bias is not None:
            s = s + self.bias
        return self.sigmoid(s).unsqueeze(1)  # [B, 1]
