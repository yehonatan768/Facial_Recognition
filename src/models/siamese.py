from __future__ import annotations

from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn

from src.models.cnn_encoder import CNNEncoder
from src.models.resnet_encoder import ResNet18Encoder
from src.models.head import WeightedL1Head, CosineHead


class SiameseModel(nn.Module):
    """
    Wrapper that combines an encoder (Paper CNN or ResNet18) with a verification head.

    Outputs:
      - forward(x1, x2) -> p_same in [0,1]
      - logits(x1, x2) -> raw logit (for BCEWithLogitsLoss)
    """

    def __init__(
        self,
        encoder: str = "paper_cnn",  # paper_cnn | resnet18
        in_channels: int = 1,
        input_size: int = 105,
        embedding_dim: int = 4096,
        enforce_input_size: bool = True,
        activation: str = "leaky_relu",
        embed_activation: str = "none",
        l2_normalize: bool = False,
        l2_eps: float = 1e-12,
        dropout_p: float = 0.0,
        # ResNet options
        resnet_pretrained: bool = True,
        resnet_freeze_backbone: bool = False,
        # Head options
        head_type: str = "weighted_l1",  # weighted_l1 | cosine
        head_bias: bool = True,
        cosine_init_scale: float = 10.0,
    ):
        super().__init__()
        self.encoder_name = str(encoder).strip().lower()
        self.l2_normalize = bool(l2_normalize)
        self.l2_eps = float(l2_eps)

        if self.encoder_name == "paper_cnn":
            self.encoder = CNNEncoder(
                in_channels=int(in_channels),
                input_size=int(input_size),
                embedding_dim=int(embedding_dim),
                enforce_input_size=bool(enforce_input_size),
                activation=str(activation),
                embed_activation=str(embed_activation),
            )
        elif self.encoder_name == "resnet18":
            self.encoder = ResNet18Encoder(
                in_channels=int(in_channels),
                embedding_dim=int(embedding_dim),
                pretrained=bool(resnet_pretrained),
                freeze_backbone=bool(resnet_freeze_backbone),
                dropout_p=float(dropout_p),
            )
        else:
            raise ValueError(f"Unsupported encoder: {encoder!r}. Use 'paper_cnn' or 'resnet18'.")

        # ---- head ----
        head_type = str(head_type).strip().lower()
        if head_type == "cosine":
            self.head = CosineHead(init_scale=float(cosine_init_scale), bias=bool(head_bias))
        elif head_type == "weighted_l1":
            self.head = WeightedL1Head(embedding_dim=int(embedding_dim), bias=bool(head_bias))
        else:
            raise ValueError(f"Unsupported head_type: {head_type!r}. Use 'weighted_l1' or 'cosine'.")

    @torch.no_grad()
    def freeze_backbone(self, freeze: bool = True) -> None:
        """
        If the encoder supports it (ResNet18Encoder), toggles backbone trainability.
        """
        if hasattr(self.encoder, "freeze_backbone") and callable(getattr(self.encoder, "freeze_backbone")):
            self.encoder.freeze_backbone(bool(freeze))

    def _maybe_l2(self, h: torch.Tensor) -> torch.Tensor:
        if not self.l2_normalize:
            return h
        return torch.nn.functional.normalize(h, p=2, dim=1, eps=self.l2_eps)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self._maybe_l2(h)

    def logits(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        h1 = self.embed(x1)
        h2 = self.embed(x2)
        if hasattr(self.head, "logits"):
            return self.head.logits(h1, h2)
        # fallback
        p = self.head(h1, h2).clamp(1e-6, 1 - 1e-6)
        return torch.log(p) - torch.log1p(-p)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(x1, x2))
