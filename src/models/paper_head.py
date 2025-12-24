from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class PaperWeightedL1Head(nn.Module):
    """
    Koch et al. (2015) Siamese join layer (Section 3.1):

      d_j = |h1_j - h2_j|
      s   = sum_j alpha_j * d_j
      p   = sigmoid(s)

    Implemented as:
      d = abs(h1 - h2)              # (B, D)
      s = Linear(d) with bias=False # (B, 1), weights are alpha
      p = sigmoid(s)                # (B, 1)

    Notes:
    - The paper describes the alpha_j as learnable parameters weighting each
      component-wise distance term. :contentReference[oaicite:1]{index=1}
    - No extra MLP; no second sigmoid; the sigmoid is applied once at the end.
    """

    def __init__(self, embedding_dim: int = 4096):
        super().__init__()
        self.embedding_dim = int(embedding_dim)

        # weights correspond to alpha_j, bias not described in the paper -> keep bias=False
        self.alpha = nn.Linear(self.embedding_dim, 1, bias=False)


    def forward(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        """
        Args:
          h1, h2: (B, D) embeddings from the shared CNN encoder (D=4096 in paper)

        Returns:
          p_same: (B, 1) probability that the pair is from the same class
        """
        if h1.shape != h2.shape:
            raise ValueError(f"h1 and h2 must have the same shape. Got {tuple(h1.shape)} vs {tuple(h2.shape)}")
        if h1.dim() != 2:
            raise ValueError(f"h1/h2 must be 2D (B,D). Got {tuple(h1.shape)}")
        if h1.size(1) != self.embedding_dim:
            raise ValueError(
                f"Expected embedding_dim={self.embedding_dim}, got D={h1.size(1)}. "
                f"Make sure your encoder outputs the paper embedding size."
            )

        d = torch.abs(h1 - h2)       # (B, D)
        s = self.alpha(d)            # (B, 1)
        p = torch.sigmoid(s)         # (B, 1)
        return p

    def raw_score(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        """
        Returns the pre-sigmoid score s = sum_j alpha_j * |h1_j - h2_j|.
        Useful for thresholding/analysis.
        """
        if h1.shape != h2.shape:
            raise ValueError(f"h1 and h2 must have the same shape. Got {tuple(h1.shape)} vs {tuple(h2.shape)}")
        d = torch.abs(h1 - h2)
        return self.alpha(d)


if __name__ == "__main__":
    head = PaperWeightedL1Head(embedding_dim=4096)
    h1 = torch.randn(4, 4096)
    h2 = torch.randn(4, 4096)
    p = head(h1, h2)
    print(p.shape)  # expected: torch.Size([4, 1])
