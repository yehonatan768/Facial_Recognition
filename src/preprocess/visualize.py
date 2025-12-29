from __future__ import annotations

import numpy as np
import torch


def denormalize_1ch(x: torch.Tensor, mean: float, std: float) -> np.ndarray:
    """
    x: (1,H,W) or (H,W) normalized tensor
    returns: (H,W) numpy array in [0,1] for visualization
    """
    if x.ndim == 3:
        x = x.squeeze(0)

    y = x.detach().cpu().numpy()
    y = y * std + mean
    return np.clip(y, 0.0, 1.0)
