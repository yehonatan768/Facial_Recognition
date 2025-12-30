from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.metrics import MetricsConfig, compute_binary_metrics


@dataclass(frozen=True)
class LoopConfig:
    metrics: MetricsConfig


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    metrics_cfg: MetricsConfig,
) -> Tuple[float, Dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    all_p = []
    all_y = []

    for x1, x2, y in loader:
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).float().view(-1, 1)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        p = model(x1, x2)  # [B,1]
        loss = loss_fn(p, y)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach().cpu()) * int(y.shape[0])
        all_p.append(p.detach().cpu())
        all_y.append(y.detach().cpu())

    probs = torch.cat(all_p, dim=0)
    targets = torch.cat(all_y, dim=0)
    metrics = compute_binary_metrics(probs=probs, targets=targets, cfg=metrics_cfg)
    avg_loss = total_loss / max(1, int(targets.shape[0]))

    return avg_loss, metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    metrics_cfg: MetricsConfig,
) -> Tuple[float, Dict[str, float]]:
    loss_fn = nn.BCELoss()
    return _run_epoch(model, loader, device, loss_fn, optimizer, metrics_cfg)


@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    metrics_cfg: MetricsConfig,
) -> Tuple[float, Dict[str, float]]:
    loss_fn = nn.BCELoss()
    return _run_epoch(model, loader, device, loss_fn, None, metrics_cfg)
