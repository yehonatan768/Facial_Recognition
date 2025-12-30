from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class EarlyStopper:
    monitor: str
    mode: str = "max"  # "max" or "min"
    patience: int = 20
    min_delta: float = 0.0

    best_value: Optional[float] = None
    best_epoch: int = -1
    bad_epochs: int = 0

    def update(self, value: float, epoch: int) -> bool:
        """Return True if should stop."""
        value = float(value)

        if self.best_value is None:
            self.best_value = value
            self.best_epoch = epoch
            self.bad_epochs = 0
            return False

        improved = False
        if self.mode == "max":
            improved = value > (self.best_value + self.min_delta)
        elif self.mode == "min":
            improved = value < (self.best_value - self.min_delta)
        else:
            raise ValueError("mode must be 'max' or 'min'")

        if improved:
            self.best_value = value
            self.best_epoch = epoch
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        return self.bad_epochs >= self.patience
