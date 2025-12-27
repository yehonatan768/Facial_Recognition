from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def load_metrics_jsonl(path: Path) -> Dict[str, List[float]]:
    """
    Loads metrics.jsonl into column lists.

    Expected keys:
      - epoch
      - train_loss, train_acc
      - val_loss, val_acc
      - lr, momentum
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"metrics.jsonl not found: {path}")

    cols: Dict[str, List] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for k, v in row.items():
                cols.setdefault(k, []).append(v)
    return cols


def plot_train_val_curves(metrics_jsonl: Path) -> None:
    import matplotlib.pyplot as plt

    cols = load_metrics_jsonl(metrics_jsonl)
    epoch = cols.get("epoch")
    if epoch is None or len(epoch) == 0:
        raise ValueError("metrics.jsonl missing 'epoch' column or empty")

    # --- FIX: keep only the last run if metrics.jsonl contains multiple runs ---
    # Find the last index where epoch resets (epoch decreases).
    start = 0
    for i in range(1, len(epoch)):
        if epoch[i] < epoch[i - 1]:
            start = i

    def _slice(k: str):
        return cols[k][start:] if k in cols else None

    e = epoch[start:]

    if _slice("train_loss") is not None and _slice("val_loss") is not None:
        plt.figure()
        plt.plot(e, _slice("train_loss"), label="train_loss")
        plt.plot(e, _slice("val_loss"), label="val_loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.title("Train vs Val Loss")
        plt.show()

    if _slice("train_acc") is not None and _slice("val_acc") is not None:
        plt.figure()
        plt.plot(e, _slice("train_acc"), label="train_acc")
        plt.plot(e, _slice("val_acc"), label="val_acc")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.legend()
        plt.title("Train vs Val Accuracy")
        plt.show()

