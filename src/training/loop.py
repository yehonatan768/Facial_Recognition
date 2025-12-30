from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
) -> float:
    """One training epoch. Returns average BCE loss."""
    model.train()
    loss_fn = nn.BCELoss()

    total_loss = 0.0
    total_n = 0

    for x1, x2, y in loader:
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float().view(-1)

        p = model(x1, x2).view(-1)

        loss = loss_fn(p, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = int(y.numel())
        total_loss += float(loss.item()) * bs
        total_n += bs

    return total_loss / max(1, total_n)


@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """One evaluation epoch. Returns average BCE loss."""
    model.eval()
    loss_fn = nn.BCELoss()

    total_loss = 0.0
    total_n = 0

    for x1, x2, y in loader:
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float().view(-1)

        p = model(x1, x2).view(-1)
        loss = loss_fn(p, y)

        bs = int(y.numel())
        total_loss += float(loss.item()) * bs
        total_n += bs

    return total_loss / max(1, total_n)
