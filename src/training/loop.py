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
    best_thr_acc: float
    best_thr: float
    n: int


def _batch_accuracy_from_probs(p: torch.Tensor, y: torch.Tensor, thr: float = 0.5) -> float:
    """
    p: (B,1) or (B,) probabilities in [0,1]
    y: (B,1) or (B,) labels in {0,1}
    """
    p = p.view(-1)
    y = y.view(-1)
    pred = (p >= thr).to(dtype=torch.long)
    tgt = (y >= 0.5).to(dtype=torch.long)
    return float((pred == tgt).float().mean().item())


@torch.no_grad()
def _best_threshold_acc(probs: torch.Tensor, y: torch.Tensor, num_thr: int = 401) -> Tuple[float, float]:
    """Returns (best_acc, best_thr) by sweeping thresholds on [0,1]."""
    probs = probs.view(-1)
    y = y.view(-1)
    # In case caller passes logits
    if probs.min() < 0.0 or probs.max() > 1.0:
        probs = probs.sigmoid()

    # thresholds include 0.0 and 1.0
    thrs = torch.linspace(0.0, 1.0, int(num_thr), device=probs.device)
    # (T,B)
    pred = (probs.unsqueeze(0) >= thrs.unsqueeze(1)).to(torch.int64)
    tgt = (y.unsqueeze(0) >= 0.5).to(torch.int64)
    acc = (pred == tgt).float().mean(dim=1)  # (T,)
    best_i = int(torch.argmax(acc).item())
    return float(acc[best_i].item()), float(thrs[best_i].item())


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

    # Use logits + BCEWithLogitsLoss for numerical stability.
    if loss_fn is None:
        loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

    total_loss_sum = 0.0
    total_acc_sum = 0.0
    total_n = 0

    # for best-threshold accuracy (epoch-level)
    all_probs = []
    all_y = []

    for batch in loader:
        x1, x2, y, _, _ = _unpack_batch(batch)

        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)

        # FIX: BCE expects float targets (and it matches val codepath behavior)
        y = y.to(device, non_blocking=True).float()

        # Prefer logits for training if available
        if hasattr(model, "head") and hasattr(model.head, "logits"):
            _, h1, h2 = model(x1, x2)
            logits = model.head.logits(h1, h2).view(-1)
        else:
            # fallback: model returns probabilities
            p_same, _, _ = model(x1, x2)
            logits = torch.logit(p_same.view(-1).clamp(1e-6, 1.0 - 1e-6))

        loss = loss_fn(logits, y.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        b = int(y.numel())
        probs = logits.detach().sigmoid()
        total_loss_sum += float(loss.item()) * b
        total_acc_sum += _batch_accuracy_from_probs(probs, y.view(-1).detach()) * b
        total_n += b

        all_probs.append(probs.cpu())
        all_y.append(y.view(-1).detach().cpu())

        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))

        optimizer.step()

    probs_epoch = torch.cat(all_probs, dim=0) if len(all_probs) else torch.empty(0)
    y_epoch = torch.cat(all_y, dim=0) if len(all_y) else torch.empty(0)
    best_thr_acc, best_thr = _best_threshold_acc(probs_epoch, y_epoch) if total_n > 0 else (0.0, 0.5)

    return EpochStats(
        loss=total_loss_sum / max(1, total_n),
        acc=total_acc_sum / max(1, total_n),
        best_thr_acc=best_thr_acc,
        best_thr=best_thr,
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

    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    total_loss_sum = 0.0
    total_acc_sum = 0.0
    total_n = 0
    all_probs = []
    all_y = []

    rows: List[Dict[str, Any]] = []

    for batch in loader:
        x1, x2, y, _, _ = _unpack_batch(batch)

        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float().view(-1)

        if hasattr(model, "head") and hasattr(model.head, "logits"):
            _, h1, h2 = model(x1, x2)
            logits = model.head.logits(h1, h2).view(-1)
        else:
            p_same, _, _ = model(x1, x2)
            logits = torch.logit(p_same.view(-1).clamp(1e-6, 1.0 - 1e-6))

        probs = logits.sigmoid()
        losses = loss_fn(logits, y)
        loss_sum = losses.sum()

        b = int(y.numel())
        total_loss_sum += float(loss_sum.item())
        total_acc_sum += _batch_accuracy_from_probs(probs, y) * b
        total_n += b

        all_probs.append(probs.cpu())
        all_y.append(y.detach().cpu())

        if dump_path is not None:
            # Dump simple per-sample stats (extend if you have paths/ids in batch dict)
            for i in range(b):
                rows.append(
                    {
                        "p_same": float(probs[i].item()),
                        "y": float(y[i].item()),
                        "loss": float(losses[i].item()),
                    }
                )

    if dump_path is not None:
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(dump_path, index=False)

    probs_epoch = torch.cat(all_probs, dim=0) if len(all_probs) else torch.empty(0)
    y_epoch = torch.cat(all_y, dim=0) if len(all_y) else torch.empty(0)
    best_thr_acc, best_thr = _best_threshold_acc(probs_epoch, y_epoch) if total_n > 0 else (0.0, 0.5)

    return EpochStats(
        loss=total_loss_sum / max(1, total_n),
        acc=total_acc_sum / max(1, total_n),
        best_thr_acc=best_thr_acc,
        best_thr=best_thr,
        n=total_n,
    )
