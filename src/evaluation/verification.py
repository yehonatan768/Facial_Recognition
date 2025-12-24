from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class VerificationResult:
    best_thr: float
    best_acc: float
    acc_at_05: float


def _accuracy_at_threshold(probs: torch.Tensor, labels: torch.Tensor, thr: float) -> float:
    pred = (probs >= thr).to(dtype=torch.float32)
    return float((pred == labels).float().mean().item())


def find_best_threshold(probs: torch.Tensor, labels: torch.Tensor) -> VerificationResult:
    """
    Finds the threshold that maximizes verification accuracy.

    probs:  (N,) float tensor in [0,1]
    labels: (N,) float tensor in {0,1}
    """
    if probs.numel() == 0:
        return VerificationResult(best_thr=0.5, best_acc=0.0, acc_at_05=0.0)

    probs = probs.view(-1).float()
    labels = labels.view(-1).float()

    # Candidate thresholds: midpoints between sorted unique probabilities + endpoints
    uniq = torch.unique(probs)
    uniq, _ = torch.sort(uniq)

    if uniq.numel() == 1:
        thr = float(uniq.item())
        acc = _accuracy_at_threshold(probs, labels, thr)
        return VerificationResult(best_thr=thr, best_acc=acc, acc_at_05=_accuracy_at_threshold(probs, labels, 0.5))

    mids = (uniq[:-1] + uniq[1:]) / 2.0
    candidates = torch.cat([torch.tensor([0.0]), mids, torch.tensor([1.0])]).to(probs.device)

    # Vectorized accuracy computation:
    # preds: (T, N)
    preds = (probs.unsqueeze(0) >= candidates.unsqueeze(1)).to(dtype=torch.float32)
    accs = (preds == labels.unsqueeze(0)).to(dtype=torch.float32).mean(dim=1)

    best_idx = int(torch.argmax(accs).item())
    best_thr = float(candidates[best_idx].item())
    best_acc = float(accs[best_idx].item())
    acc05 = _accuracy_at_threshold(probs, labels, 0.5)

    return VerificationResult(best_thr=best_thr, best_acc=best_acc, acc_at_05=acc05)
