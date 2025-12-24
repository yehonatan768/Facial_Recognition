from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class EarlyStopState:
    best_score: float
    best_epoch: int
    bad_epochs: int
    should_stop: bool


class EarlyStopping:
    """
    Early stopping with patience.

    Use case (paper-like):
      - monitor one-shot validation accuracy (maximize)
      - stop when it does not improve for `patience` epochs (paper uses 20)

    Call `update(epoch, score)` once per epoch after evaluation.
    """

    def __init__(self, patience: int = 20, min_delta: float = 0.0, maximize: bool = True):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.maximize = bool(maximize)

        self.best_score: Optional[float] = None
        self.best_epoch: int = -1
        self.bad_epochs: int = 0
        self.should_stop: bool = False

    def update(self, epoch: int, score: float) -> EarlyStopState:
        """
        epoch: current epoch index (0-based or 1-based; just be consistent)
        score: metric value; higher is better if maximize=True else lower is better
        """
        score = float(score)

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = int(epoch)
            self.bad_epochs = 0
            self.should_stop = False
            return self.state()

        improved = False
        if self.maximize:
            improved = score > (self.best_score + self.min_delta)
        else:
            improved = score < (self.best_score - self.min_delta)

        if improved:
            self.best_score = score
            self.best_epoch = int(epoch)
            self.bad_epochs = 0
            self.should_stop = False
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.should_stop = True

        return self.state()

    def state(self) -> EarlyStopState:
        return EarlyStopState(
            best_score=float(self.best_score) if self.best_score is not None else float("nan"),
            best_epoch=int(self.best_epoch),
            bad_epochs=int(self.bad_epochs),
            should_stop=bool(self.should_stop),
        )
