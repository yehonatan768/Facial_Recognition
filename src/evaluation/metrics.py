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
    """
    Plots:
      - train_loss vs val_loss
      - train_acc  vs val_acc
    """
    import matplotlib.pyplot as plt  # local import

    cols = load_metrics_jsonl(metrics_jsonl)

    epoch = cols.get("epoch")
    if epoch is None:
        raise ValueError("metrics.jsonl missing 'epoch' column")

    # Loss plot
    if "train_loss" in cols and "val_loss" in cols:
        plt.figure()
        plt.plot(epoch, cols["train_loss"], label="train_loss")
        plt.plot(epoch, cols["val_loss"], label="val_loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.title("Loss vs Epoch")
        plt.show()

    # Accuracy plot
    if "train_acc" in cols and "val_acc" in cols:
        plt.figure()
        plt.plot(epoch, cols["train_acc"], label="train_acc")
        plt.plot(epoch, cols["val_acc"], label="val_acc")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.legend()
        plt.title("Accuracy vs Epoch")
        plt.show()
