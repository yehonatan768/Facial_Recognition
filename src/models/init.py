from __future__ import annotations

import torch.nn as nn


def init_weights_like_paper(model: nn.Module) -> None:
    """
    Koch et al. (2015) initialization:

      Conv2d:
        W ~ N(0, 1e-2)
        b ~ N(0.5, 1e-2)

      Linear (fully-connected):
        W ~ N(0, 2e-1)
        b ~ N(0.5, 1e-2)

    This applies to ALL Linear layers, including the final weighted-L1 alpha layer,
    treating it as the paper's final fully-connected layer.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0.0, std=1e-2)
            if m.bias is not None:
                nn.init.normal_(m.bias, mean=0.5, std=1e-2)

        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=2e-1)
            if m.bias is not None:
                nn.init.normal_(m.bias, mean=0.5, std=1e-2)


def init_weights_like_paper_encoder_only(model: nn.Module) -> None:
    """
    Same as init_weights_like_paper(), but intended for cases where you do NOT want
    to apply the FC-style init to the head alpha layer.

    It will initialize:
      - all Conv2d layers (conv-style)
      - all Linear layers EXCEPT those with out_features == 1 and bias == False
        (the typical shape of the paper's alpha head)

    Use this if you want alpha weights to stay at their constructor init (often 1.0).
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0.0, std=1e-2)
            if m.bias is not None:
                nn.init.normal_(m.bias, mean=0.5, std=1e-2)

        elif isinstance(m, nn.Linear):
            is_alpha_like = (m.out_features == 1) and (m.bias is None)
            if is_alpha_like:
                continue
            nn.init.normal_(m.weight, mean=0.0, std=2e-1)
            if m.bias is not None:
                nn.init.normal_(m.bias, mean=0.5, std=1e-2)
