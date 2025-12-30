from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
) -> Tuple[float, float]:
    """
    Returns:
      (avg_loss, avg_accuracy)
    Uses logits + BCEWithLogitsLoss for numerical stability.
    Accuracy uses threshold 0.5 on sigmoid(logits).
    """
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_correct = 0
    total_n = 0

    for x1, x2, y in loader:
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device).float().view(-1, 1)

        optimizer.zero_grad(set_to_none=True)

        logits = model(x1, x2)
        loss = loss_fn(logits, y)

        loss.backward()
        optimizer.step()

        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        total_n += bs

        # Accuracy: sigmoid(logits) >= 0.5
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).to(y.dtype)
        total_correct += int((preds == y).sum().item())

    avg_loss = total_loss / max(1, total_n)
    avg_acc = total_correct / max(1, total_n)
    return avg_loss, float(avg_acc)


@torch.no_grad()
def eval_one_epoch(
    model,
    loader,
    device,
) -> Tuple[float, float]:
    """
    Returns:
      (avg_loss, avg_accuracy)
    """
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_correct = 0
    total_n = 0

    for x1, x2, y in loader:
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device).float().view(-1, 1)

        logits = model(x1, x2)
        loss = loss_fn(logits, y)

        bs = y.size(0)
        total_loss += float(loss.item()) * bs
        total_n += bs

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).to(y.dtype)
        total_correct += int((preds == y).sum().item())

    avg_loss = total_loss / max(1, total_n)
    avg_acc = total_correct / max(1, total_n)
    return avg_loss, float(avg_acc)
