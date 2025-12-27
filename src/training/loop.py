# src/training/loop.py
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
    y: (B,) labels in {0,1}
    """
    if p.dim() == 2 and p.size(1) == 1:
        p = p[:, 0]
    pred = (p >= thr).to(dtype=torch.float32)
    return float((pred == y).float().mean().item())


def _unpack_batch(
    batch: BatchType,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[List[str]], Optional[List[str]]]:
    """
    Supports BOTH batch formats:
      A) tuple: (x1, x2, y)
      B) dict: {"x1":..., "x2":..., "y":..., "path1":..., "path2":...}

    Returns: (x1, x2, y, path1_list_or_None, path2_list_or_None)
    """
    if isinstance(batch, dict):
        x1 = batch["x1"]
        x2 = batch["x2"]
        y = batch["y"]
        p1 = batch.get("path1", None)
        p2 = batch.get("path2", None)

        if p1 is not None and not isinstance(p1, list):
            p1 = list(p1)
        if p2 is not None and not isinstance(p2, list):
            p2 = list(p2)

        return x1, x2, y, p1, p2

    x1, x2, y = batch
    return x1, x2, y, None, None


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
        y = y.to(device, non_blocking=True)

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
    epoch: int,
    workdir: str | Path,
    dump: bool = True,
    thr: float = 0.5,
) -> Tuple[EpochStats, Optional[Path]]:
    """
    Validation epoch + optional per-sample dump for error analysis.

    Writes (when dump=True):
      workdir/val_predictions/val_epoch_{epoch:03d}.csv

    This avoids overwriting the same file each epoch (Windows IO stalls)
    and enables per-epoch error analysis.
    """
    model.eval()

    total_loss_sum = 0.0
    total_acc_sum = 0.0
    total_n = 0

    rows: List[Dict[str, Any]] = []

    for batch in loader:
        x1, x2, y, p1, p2 = _unpack_batch(batch)

        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float().view(-1)

        p_same, _, _ = model(x1, x2)
        p_same = p_same.view(-1).clamp(1e-6, 1.0 - 1e-6)

        loss_per = F.binary_cross_entropy(p_same, y, reduction="none")  # [B]
        b = int(y.numel())

        total_loss_sum += float(loss_per.sum().item())
        total_acc_sum += _batch_accuracy(p_same, y, thr=thr) * b
        total_n += b

        if dump:
            ps = p_same.detach().cpu().numpy()
            ys = y.detach().cpu().numpy()
            ls = loss_per.detach().cpu().numpy()

            for i in range(b):
                rows.append(
                    {
                        "epoch": int(epoch),
                        "y_true": int(ys[i]),
                        "p_same": float(ps[i]),
                        "loss": float(ls[i]),
                        "path1": str(p1[i]) if p1 is not None else None,
                        "path2": str(p2[i]) if p2 is not None else None,
                    }
                )

    stats = EpochStats(
        loss=total_loss_sum / max(1, total_n),
        acc=total_acc_sum / max(1, total_n),
        n=total_n,
    )

    out_path: Optional[Path] = None
    if dump:
        workdir = Path(workdir)
        out_dir = workdir / "val_predictions"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"val_epoch_{epoch:03d}.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)

    return stats, out_path


@torch.no_grad()
def eval_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: Optional[nn.Module] = None,
    thr: float = 0.5,
) -> Tuple[EpochStats, torch.Tensor, torch.Tensor]:
    """
    Evaluate on pairs:
      returns (stats, probs, labels)

    probs:  (N,) float tensor on CPU
    labels: (N,) float tensor on CPU
    """
    model.eval()
    if loss_fn is None:
        loss_fn = nn.BCELoss(reduction="mean")

    probs_all: List[torch.Tensor] = []
    y_all: List[torch.Tensor] = []

    total_loss = 0.0
    total_acc = 0.0
    total_n = 0

    for batch in loader:
        x1, x2, y, _, _ = _unpack_batch(batch)

        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float().view(-1)

        p_same, _, _ = model(x1, x2)
        p_same = p_same.view(-1).clamp(1e-6, 1.0 - 1e-6)

        loss = loss_fn(p_same, y)

        b = int(y.numel())
        total_loss += float(loss.item()) * b
        total_acc += _batch_accuracy(p_same, y, thr=thr) * b
        total_n += b

        probs_all.append(p_same.detach().cpu())
        y_all.append(y.detach().cpu())

    probs = torch.cat(probs_all, dim=0) if probs_all else torch.empty(0)
    labels = torch.cat(y_all, dim=0) if y_all else torch.empty(0)

    stats = EpochStats(
        loss=total_loss / max(1, total_n),
        acc=total_acc / max(1, total_n),
        n=total_n,
    )
    return stats, probs, labels
