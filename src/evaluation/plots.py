from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def plot_loss_curves(history: Dict[str, List[float]], out_dir: Path) -> None:
    """Save loss curves (train_loss vs val_loss)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = history.get("epoch", [])
    tr = history.get("train_loss", [])
    va = history.get("val_loss", [])
    if not epochs or len(tr) != len(epochs) or len(va) != len(epochs):
        return

    plt.figure(figsize=(9, 6))
    plt.plot(epochs, tr, label="train_loss")
    plt.plot(epochs, va, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("BCE Loss over epochs")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_base.png")
    plt.close()
