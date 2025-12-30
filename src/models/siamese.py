from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from src.models.cnn_encoder import CNNEncoder
from src.models.head import WeightedL1Head


class SiameseModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        input_size: int = 105,
        enforce_input_size: bool = True,
        embedding_dim: int = 4096,
        embed_activation: str = "sigmoid",  # keep "sigmoid" for paper, allow "none"
    ):
        super().__init__()

        self.encoder = CNNEncoder(
            in_channels=in_channels,
            input_size=input_size,
            embedding_dim=embedding_dim,
            enforce_input_size=enforce_input_size,
            embed_activation=embed_activation,
        )
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
