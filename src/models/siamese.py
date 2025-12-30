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

        # Encoder knobs (forwarded to CNNEncoder)
        activation: str = "leaky_relu",
        embed_activation: str = "none",
        l2_normalize: bool = True,
        l2_eps: float = 1e-12,
        dropout_p: float = 0.0,
    ):
        super().__init__()

        self.encoder = CNNEncoder(
            in_channels=in_channels,
            input_size=input_size,
            embedding_dim=embedding_dim,
            enforce_input_size=enforce_input_size,
            activation=activation,
            embed_activation=embed_activation,
            l2_normalize=l2_normalize,
            l2_eps=l2_eps,
            dropout_p=dropout_p,
        )
        self.head = WeightedL1Head(embedding_dim=embedding_dim)

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        p_same = self.head(h1, h2)
        return p_same, h1, h2

    @torch.no_grad()
    def score(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        p_same, _, _ = self.forward(x1, x2)
        return p_same
