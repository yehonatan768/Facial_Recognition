from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class EpochStats:
    loss: float
    acc: float
    n: int


def _batch_accuracy(p: torch.Tensor, y: torch.Tensor, thr: float = 0.5) -> float:
    """
    p: (B,1) or (B,) probabilities in [0,1]
    y: (B,) labels in {0,1}
    """
    if p.dim() == 2 and p.size(1) == 1:
        p = p[:, 0]
    pred = (p >= thr).to(dtype=torch.float32)
    return float((pred == y).float().mean().item())


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: Optional[nn.Module] = None,
    grad_clip_norm: Optional[float] = None,
) -> EpochStats:
    """
    One epoch training on verification pairs.
    Assumes model(x1,x2) returns (p_same, h1, h2) and p_same is sigmoid output.
    Uses BCELoss by default.
    """
    model.train()
    if loss_fn is None:
        loss_fn = nn.BCELoss()

    total_loss = 0.0
    total_acc = 0.0
    total_n = 0

    for x1, x2, y in loader:
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)  # (B,)

        optimizer.zero_grad(set_to_none=True)

        p_same, _, _ = model(x1, x2)  # (B,1)
        p_same = p_same.clamp(1e-6, 1.0 - 1e-6)

        loss = loss_fn(p_same.view(-1), y.view(-1))
        loss.backward()

        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))

        optimizer.step()

        b = int(y.numel())
        total_loss += float(loss.item()) * b
        total_acc += _batch_accuracy(p_same.detach(), y.detach()) * b
        total_n += b

    return EpochStats(loss=total_loss / max(1, total_n), acc=total_acc / max(1, total_n), n=total_n)


@torch.no_grad()
def eval_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: Optional[nn.Module] = None,
) -> Tuple[EpochStats, torch.Tensor, torch.Tensor]:
    """
    Evaluate on pairs:
      returns (stats, probs, labels)
    probs: (N,) float tensor on CPU
    labels:(N,) float tensor on CPU
    """
    model.eval()
    if loss_fn is None:
        loss_fn = nn.BCELoss()

    probs_all = []
    y_all = []

    total_loss = 0.0
    total_acc = 0.0
    total_n = 0

    for x1, x2, y in loader:
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        p_same, _, _ = model(x1, x2)
        p_same = p_same.clamp(1e-6, 1.0 - 1e-6)

        loss = loss_fn(p_same.view(-1), y.view(-1))

        b = int(y.numel())
        total_loss += float(loss.item()) * b
        total_acc += _batch_accuracy(p_same, y) * b
        total_n += b

        probs_all.append(p_same.view(-1).detach().cpu())
        y_all.append(y.view(-1).detach().cpu())

    probs = torch.cat(probs_all, dim=0) if probs_all else torch.empty(0)
    labels = torch.cat(y_all, dim=0) if y_all else torch.empty(0)

    stats = EpochStats(loss=total_loss / max(1, total_n), acc=total_acc / max(1, total_n), n=total_n)
    return stats, probs, labels
