from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.models.siamese import SiameseModel


def _init_conv_like_paper(m: nn.Conv2d) -> None:
    # Paper uses N(0, 0.01) for conv weights and bias=0.5
    nn.init.normal_(m.weight, mean=0.0, std=0.01)
    if m.bias is not None:
        nn.init.constant_(m.bias, 0.5)


def _init_linear_like_paper(m: nn.Linear) -> None:
    # Paper uses N(0, 0.2) for FC weights and bias=0.5
    nn.init.normal_(m.weight, mean=0.0, std=0.2)
    if m.bias is not None:
        nn.init.constant_(m.bias, 0.5)


def init_model_weights(model: SiameseModel, cfg: Optional[Dict[str, Any]] = None) -> None:
    """
    Weight init policy:

    - If encoder == paper_cnn:
        initialize conv/fc exactly like the paper, and initialize the head.
    - If encoder == resnet18 and pretrained == True:
        DO NOT overwrite pretrained backbone weights.
        Initialize ONLY:
          * backbone.fc (the projection layer we replaced)
          * backbone.conv1 if you replaced it (in_channels != 3)
          * verification head parameters (e.g. alpha / cosine scale+bias)

    - If encoder == resnet18 and pretrained == False:
        you may initialize the full encoder (rarely useful); we still only init
        the replaced layers + head by default (safer).

    This prevents the most common "accuracy stuck at ~0.5" failure mode:
    accidentally re-initializing a pretrained encoder and then freezing it.
    """
    enc_name = getattr(model, "encoder_name", "").lower()

    if enc_name == "paper_cnn":
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                _init_conv_like_paper(m)
            elif isinstance(m, nn.Linear):
                _init_linear_like_paper(m)

        # Head init: make sure it's not left at default init
        if hasattr(model, "head") and isinstance(getattr(model.head, "alpha", None), nn.Linear):
            _init_linear_like_paper(model.head.alpha)
        return

    # ---- resnet18 path ----
    # initialize only replaced parts + head
    enc = getattr(model, "encoder", None)
    backbone = getattr(enc, "backbone", None)

    if isinstance(backbone, nn.Module):
        # conv1 may have been replaced when in_channels != 3
        if hasattr(backbone, "conv1") and isinstance(backbone.conv1, nn.Conv2d):
            # If conv1 was replaced, it will have in_channels != 3
            if getattr(backbone.conv1, "in_channels", 3) != 3:
                nn.init.kaiming_normal_(backbone.conv1.weight, mode="fan_out", nonlinearity="relu")
                if backbone.conv1.bias is not None:
                    nn.init.zeros_(backbone.conv1.bias)

        # fc is always replaced to match embedding_dim
        if hasattr(backbone, "fc") and isinstance(backbone.fc, nn.Linear):
            nn.init.normal_(backbone.fc.weight, mean=0.0, std=0.02)
            if backbone.fc.bias is not None:
                nn.init.zeros_(backbone.fc.bias)

    # Head init
    if hasattr(model, "head"):
        # WeightedL1Head
        if isinstance(getattr(model.head, "alpha", None), nn.Linear):
            nn.init.normal_(model.head.alpha.weight, mean=0.0, std=0.02)
            if model.head.alpha.bias is not None:
                nn.init.zeros_(model.head.alpha.bias)
        # CosineHead
        if hasattr(model.head, "log_scale") and isinstance(model.head.log_scale, torch.nn.Parameter):
            with torch.no_grad():
                # keep whatever init_scale was passed, but clamp to sane range
                model.head.log_scale.data = model.head.log_scale.data.clamp(torch.tensor(-2.0), torch.tensor(6.0))
        if hasattr(model.head, "bias") and isinstance(getattr(model.head, "bias", None), torch.nn.Parameter):
            with torch.no_grad():
                model.head.bias.data.zero_()
