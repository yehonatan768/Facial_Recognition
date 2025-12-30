from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedL1Head(nn.Module):
    """
    Paper join layer (Koch et al., 2015), but with an optional bias term.

      d = |h1 - h2|
      logit = a^T d + b
      p = sigmoid(logit)

    Notes:
    - Exposing `logits()` lets training use BCEWithLogitsLoss (more stable).
    - The bias improves calibration around the fixed 0.5 threshold.
    """

    def __init__(self, embedding_dim: int, bias: bool = True):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.alpha = nn.Linear(self.embedding_dim, 1, bias=bool(bias))

    def logits(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        d = torch.abs(h1 - h2)
        return self.alpha(d).squeeze(-1)  # (B,)

    def forward(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(h1, h2))  # (B,)


class CosineHead(nn.Module):
    """
    A strong baseline for verification with L2-normalized embeddings:

      cos = cosine_similarity(h1, h2) in [-1, 1]
      logit = scale * cos + bias
      p = sigmoid(logit)

    This tends to be better behaved with fixed thresholding than Weighted-L1
    when embeddings are L2-normalized.
    """

    def __init__(self, init_scale: float = 10.0, bias: bool = True):
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(float(init_scale)).log())
        self.bias = nn.Parameter(torch.zeros(())) if bias else None

    def logits(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        # h1/h2 may already be normalized, but normalize again defensively.
        h1n = F.normalize(h1, p=2, dim=1)
        h2n = F.normalize(h2, p=2, dim=1)
        cos = (h1n * h2n).sum(dim=1)  # (B,)
        scale = self.log_scale.exp().clamp(1e-3, 1e3)
        logit = scale * cos
        if self.bias is not None:
            logit = logit + self.bias
        return logit

    def forward(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(h1, h2))
