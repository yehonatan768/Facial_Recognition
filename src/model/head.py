from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class SimilarityHeadConfig:
    use_bias: bool = False


class WeightedL1Head(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))

    def forward(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(h1 - h2)
        logits = (diff * self.alpha).sum(dim=1, keepdim=True)
        return logits
