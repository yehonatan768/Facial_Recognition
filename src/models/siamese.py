from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from src.models.cnn_encoder import CNNEncoder
from src.models.head import WeightedL1Head


class PaperSiameseModel(nn.Module):
    """
    Full paper-style Siamese model:

      x1 -> shared CNNEncoder -> h1 (B,4096)
      x2 -> shared CNNEncoder -> h2 (B,4096)
      p_same = WeightedL1Head(h1, h2) -> (B,1)

    Returns:
      p_same, h1, h2

    This matches the paper’s description:
    - twin networks with tied weights
    - component-wise weighted L1 distance + sigmoid output :contentReference[oaicite:1]{index=1}
    """

    def __init__(self, in_channels: int = 1, enforce_105: bool = True, embedding_dim: int = 4096):
        super().__init__()
        if embedding_dim != 4096:
            raise ValueError("Paper model uses embedding_dim=4096. If you change it, it's not paper-exact.")

        self.encoder = CNNEncoder(in_channels=in_channels, enforce_105=enforce_105)
        self.head = WeightedL1Head(embedding_dim=embedding_dim)

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a single image through the shared encoder.
        """
        return self.encoder(x)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for a pair.

        Returns:
          p_same: (B,1)
          h1: (B,4096)
          h2: (B,4096)
        """
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        p_same = self.head(h1, h2)
        return p_same, h1, h2

    @torch.no_grad()
    def score(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Convenience method: returns p_same only.
        """
        p_same, _, _ = self.forward(x1, x2)
        return p_same
