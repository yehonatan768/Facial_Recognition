from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union, Sequence

import torch


@dataclass
class OptimBundle:
    optimizer: torch.optim.Optimizer
    scheduler: "PaperLrMomentumScheduler"


def _as_float_list(x: Union[float, Sequence[float]], n: int) -> list[float]:
    if isinstance(x, (list, tuple)):
        if len(x) != n:
            raise ValueError(f"Expected momentum_final list of length {n}, got {x}")
        return [float(v) for v in x]
    return [float(x) for _ in range(n)]


class PaperLrMomentumScheduler:
    """
    Paper-style scheduler:
      lr(epoch) = base_lr * (lr_decay ** epoch)
      momentum ramps linearly from momentum_start -> momentum_final over momentum_ramp_epochs
    Supports per-param-group momentum_final via list/tuple.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr_decay: float = 0.99,
        momentum_start: float = 0.5,
        momentum_final: Union[float, Sequence[float]] = 0.9,
        momentum_ramp_epochs: int = 200,
        epoch0_lr: Optional[float] = None,
    ):
        self.optimizer = optimizer
        self.lr_decay = float(lr_decay)
        self.momentum_start = float(momentum_start)
        self.momentum_ramp_epochs = max(1, int(momentum_ramp_epochs))

        # Base lrs are whatever the optimizer currently has for each group
        self.base_lrs = [float(g["lr"]) for g in self.optimizer.param_groups]
        if epoch0_lr is not None:
            for g in self.optimizer.param_groups:
                g["lr"] = float(epoch0_lr)
            self.base_lrs = [float(epoch0_lr) for _ in self.base_lrs]

        self.momentum_finals = _as_float_list(momentum_final, n=len(self.optimizer.param_groups))

    def _t(self, epoch: int) -> float:
        # clamp epoch into [0, ramp_epochs]
        e = min(max(int(epoch), 0), self.momentum_ramp_epochs)
        return e / float(self.momentum_ramp_epochs)

    def step(self, epoch: int) -> Dict[str, float]:
        """
        Apply lr decay and momentum ramp for a given epoch (0-indexed).
        Returns a small dict for convenience logging.
        """
        t = self._t(epoch)

        # Update per group
        last_mom: Optional[float] = None
        for i, g in enumerate(self.optimizer.param_groups):
            g["lr"] = self.base_lrs[i] * (self.lr_decay ** int(epoch))

            if "momentum" in g:
                mom_i = self.momentum_start + t * (self.momentum_finals[i] - self.momentum_start)
                g["momentum"] = float(mom_i)
                last_mom = float(mom_i)

        # If optimizer has no momentum field (unlikely for SGD here), still compute a value.
        if last_mom is None:
            last_mom = float(self.momentum_start + t * (self.momentum_finals[0] - self.momentum_start))

        return {
            "lr": float(self.optimizer.param_groups[0]["lr"]),
            "momentum": float(last_mom),
        }


def build_optimizer_and_scheduler(model: torch.nn.Module, cfg: Dict[str, Any]) -> OptimBundle:
    """
    Paper schedule (Koch et al.):
      - SGD
      - LR decays exponentially by 1% per epoch:
            lr(epoch) = lr0 * (lr_decay ** epoch)  with lr_decay=0.99
      - Momentum ramps linearly from 0.5 to momentum_final over the schedule.
    """
    optim_cfg = cfg.get("optim", {}) or {}
    train_cfg = cfg.get("train", {}) or {}

    # Accept both keys (your config currently uses lr_base)
    base_lr = float(optim_cfg.get("lr", optim_cfg.get("lr_base", 1e-2)))
    lr_decay = float(optim_cfg.get("lr_decay", 0.99))

    momentum_start = float(optim_cfg.get("momentum_start", 0.5))
    momentum_final = float(optim_cfg.get("momentum_final", 0.9))

    # Ramp over the whole schedule by default (paper-style)
    ramp_epochs = int(
        optim_cfg.get(
            "momentum_ramp_epochs",
            train_cfg.get("epochs", 200),
        )
    )
    ramp_epochs = max(1, ramp_epochs)

    weight_decay = float(optim_cfg.get("weight_decay", 0.0))

    # Optional layer-wise overrides (keep your structure)
    lr_conv = float(optim_cfg.get("lr_conv", base_lr))
    lr_fc = float(optim_cfg.get("lr_fc", base_lr))
    lr_alpha = float(optim_cfg.get("lr_alpha", base_lr))

    wd_conv = float(optim_cfg.get("weight_decay_conv", weight_decay))
    wd_fc = float(optim_cfg.get("weight_decay_fc", weight_decay))
    wd_alpha = float(optim_cfg.get("weight_decay_alpha", weight_decay))

    # Optional per-group momentum finals (paper allows per-layer μ_j)
    mu_conv = float(optim_cfg.get("momentum_final_conv", momentum_final))
    mu_fc = float(optim_cfg.get("momentum_final_fc", momentum_final))
    mu_alpha = float(optim_cfg.get("momentum_final_alpha", momentum_final))

    # ---- Collect params by name ----
    conv_params, fc_params, alpha_params = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("encoder.conv"):
            conv_params.append(p)
        elif name.startswith("encoder.fc"):
            fc_params.append(p)
        elif name.startswith("head.alpha"):
            alpha_params.append(p)
        else:
            fc_params.append(p)

    param_groups = [
        {"params": conv_params, "lr": lr_conv, "momentum": momentum_start, "weight_decay": wd_conv},
        {"params": fc_params, "lr": lr_fc, "momentum": momentum_start, "weight_decay": wd_fc},
        {"params": alpha_params, "lr": lr_alpha, "momentum": momentum_start, "weight_decay": wd_alpha},
    ]
    # Drop empty groups
    param_groups = [g for g in param_groups if len(g["params"]) > 0]

    optimizer = torch.optim.SGD(param_groups)

    # Momentum final values per group (PaperLrMomentumScheduler supports list/tuple)
    momentum_finals = []
    # Match group ordering above (conv, fc, alpha) but only for groups that exist
    for g in param_groups:
        if g["params"] is conv_params:
            momentum_finals.append(mu_conv)
        elif g["params"] is alpha_params:
            momentum_finals.append(mu_alpha)
        else:
            momentum_finals.append(mu_fc)

    scheduler = PaperLrMomentumScheduler(
        optimizer=optimizer,
        lr_decay=lr_decay,
        momentum_start=momentum_start,
        momentum_final=momentum_finals,
        momentum_ramp_epochs=ramp_epochs,
        epoch0_lr=None,  # keep optimizer group's lr as the base
    )
    return OptimBundle(optimizer=optimizer, scheduler=scheduler)
