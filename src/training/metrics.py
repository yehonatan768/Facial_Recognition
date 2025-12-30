from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class MetricsConfig:
    threshold: float = 0.5
    compute_auc: bool = False


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den != 0 else 0.0


def _auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # Simple ROC AUC via ranking (equivalent to Mann–Whitney U statistic)
    y_true = y_true.astype(np.int32)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0
    ranks = y_score.argsort().argsort().astype(np.float64) + 1.0  # 1..N
    sum_ranks_pos = float(ranks[pos].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def compute_binary_metrics(
    probs: torch.Tensor,
    targets: torch.Tensor,
    cfg: MetricsConfig,
) -> Dict[str, float]:
    """Metrics for verification (pair classification)."""
    p = probs.detach().view(-1).cpu().float()
    y = targets.detach().view(-1).cpu().float()

    thr = float(cfg.threshold)
    yhat = (p >= thr).float()

    tp = float(((yhat == 1) & (y == 1)).sum())
    tn = float(((yhat == 0) & (y == 0)).sum())
    fp = float(((yhat == 1) & (y == 0)).sum())
    fn = float(((yhat == 0) & (y == 1)).sum())

    acc = _safe_div(tp + tn, tp + tn + fp + fn)
    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * prec * rec, prec + rec)

    out = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }

    if cfg.compute_auc:
        out["auc"] = _auc_roc(y_true=y.numpy(), y_score=p.numpy())

    return out
