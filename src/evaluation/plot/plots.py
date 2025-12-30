from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def plot_metric(epochs: List[int], values: List[float], title: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, values)
    plt.xlabel("Epoch")
    plt.ylabel(title)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_all_metrics(history: Dict[str, List[float]], out_dir: Path) -> None:
    epochs = history.get("epoch", [])
    for k, v in history.items():
        if k == "epoch":
            continue
        plot_metric(epochs, v, k, out_dir / f"{k}.png")

    # Combined plot (all metrics except loss)
    plt.figure(figsize=(9, 6))
    for k, v in history.items():
        if k == "epoch" or "loss" in k:
            continue
        plt.plot(epochs, v, label=k)
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("Metrics over epochs")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "metrics_all.png")
    plt.close()

    # Loss plot (train vs val if available)
    if "train_loss" in history and "val_loss" in history:
        plt.figure(figsize=(9, 6))
        plt.plot(epochs, history["train_loss"], label="train_loss")
        plt.plot(epochs, history["val_loss"], label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss over epochs")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "loss.png")
        plt.close()
