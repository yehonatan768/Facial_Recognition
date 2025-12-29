from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd


BatchType = Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Dict[str, Any]]


@dataclass
class EpochStats:
    loss: float
    acc: float
    n: int


def _batch_accuracy(p: torch.Tensor, y: torch.Tensor, thr: float = 0.5) -> float:
    """
    p: (B,1) or (B,) probabilities in [0,1]
    y: (B,1) or (B,) labels in {0,1}
    """
    p = p.view(-1)
    y = y.view(-1)
    pred = (p >= thr).to(dtype=torch.long)
    tgt = (y >= 0.5).to(dtype=torch.long)
    return float((pred == tgt).float().mean().item())


def _unpack_batch(batch: BatchType) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any, Any]:
    """
    Supports:
      - tuple: (x1, x2, y)
      - dict: {"x1":..., "x2":..., "y":..., ...}
    Returns: x1, x2, y, meta1, meta2 (meta values may be None).
    """
    if isinstance(batch, (tuple, list)):
        if len(batch) != 3:
            raise ValueError(f"Expected (x1,x2,y) batch tuple, got len={len(batch)}")
        x1, x2, y = batch
        return x1, x2, y, None, None

    if isinstance(batch, dict):
        x1 = batch["x1"]
        x2 = batch["x2"]
        y = batch["y"]
        return x1, x2, y, None, None

    raise TypeError(f"Unsupported batch type: {type(batch)}")


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: nn.Module | None = None,
    grad_clip_norm: float | None = None,
) -> EpochStats:
    """
    Trains for one epoch.

    Works with loaders yielding:
      - (x1, x2, y) tuples
      - {"x1","x2","y", ...} dicts
    """
    model.train()

    # SUM so we can scale to exact full-dataset mean gradient (paper-style accumulation)
    if loss_fn is None:
        loss_fn = nn.BCELoss(reduction="sum")

    total_examples = len(loader.dataset)

    optimizer.zero_grad(set_to_none=True)

    total_loss_sum = 0.0
    total_acc_sum = 0.0
    total_n = 0

    for batch in loader:
        x1, x2, y, _, _ = _unpack_batch(batch)

        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)

        # FIX: BCE expects float targets (and it matches val codepath behavior)
        y = y.to(device, non_blocking=True).float()

        p_same, _, _ = model(x1, x2)
        p_same = p_same.view(-1).clamp(1e-6, 1.0 - 1e-6)

        loss_sum = loss_fn(p_same, y.view(-1))
        loss = loss_sum / float(total_examples)  # scaled mean over full dataset
        loss.backward()

        b = int(y.numel())
        total_loss_sum += float(loss_sum.item())
        total_acc_sum += _batch_accuracy(p_same.detach(), y.view(-1).detach()) * b
        total_n += b

    if grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))

    optimizer.step()

    return EpochStats(
        loss=total_loss_sum / max(1, total_n),
        acc=total_acc_sum / max(1, total_n),
        n=total_n,
    )


@torch.no_grad()
def run_val_and_dump(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    dump_path: Optional[Path] = None,
) -> EpochStats:
    """
    Validation pass.

    Computes:
      - mean BCE loss (per sample)
      - accuracy at threshold 0.5
    Optionally dumps per-pair predictions to CSV for inspection.
    """
    model.eval()

    total_loss_sum = 0.0
    total_acc_sum = 0.0
    total_n = 0

    rows: List[Dict[str, Any]] = []

    for batch in loader:
        x1, x2, y, _, _ = _unpack_batch(batch)

        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float().view(-1)

        p_same, _, _ = model(x1, x2)
        p_same = p_same.view(-1).clamp(1e-6, 1.0 - 1e-6)

        # per-sample loss
        losses = F.binary_cross_entropy(p_same, y, reduction="none")
        loss_sum = losses.sum()

        b = int(y.numel())
        total_loss_sum += float(loss_sum.item())
        total_acc_sum += _batch_accuracy(p_same, y) * b
        total_n += b

        if dump_path is not None:
            # Dump simple per-sample stats (extend if you have paths/ids in batch dict)
            for i in range(b):
                rows.append(
                    {
                        "p_same": float(p_same[i].item()),
                        "y": float(y[i].item()),
                        "loss": float(losses[i].item()),
                    }
                )

    if dump_path is not None:
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(dump_path, index=False)

    return EpochStats(
        loss=total_loss_sum / max(1, total_n),
        acc=total_acc_sum / max(1, total_n),
        n=total_n,
    )
