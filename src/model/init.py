from __future__ import annotations

import torch
import torch.nn as nn


def init_weights_like_reference(model: nn.Module) -> None:
    """Weight initialization matching the reference description.

    - Conv weights:  N(0, 1e-2)
    - Conv biases:   N(0.5, 1e-2)
    - FC weights:    N(0, 2e-1)
    - FC biases:     N(0.5, 1e-2)
    - Head alpha:    N(0, 2e-1)
    """

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0.0, std=1e-2)
            if m.bias is not None:
                nn.init.normal_(m.bias, mean=0.5, std=1e-2)
        elif isinstance(m, nn.Linear):
            # This covers the encoder FC (not the head, which is a Parameter)
            nn.init.normal_(m.weight, mean=0.0, std=2e-1)
            if m.bias is not None:
                nn.init.normal_(m.bias, mean=0.5, std=1e-2)

    # Head parameters
    if hasattr(model, "head") and hasattr(model.head, "alpha"):
        nn.init.normal_(model.head.alpha, mean=0.0, std=2e-1)
        if getattr(model.head, "bias", None) is not None:
            nn.init.normal_(model.head.bias, mean=0.5, std=1e-2)
