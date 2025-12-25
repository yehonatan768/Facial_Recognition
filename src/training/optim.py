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
    optim_cfg = cfg.get("optim", {}) or {}
    train_cfg = cfg.get("train", {}) or {}

    # ---- Paper-ish defaults (strict-ish) ----
    # Backward-compatible: accept lr_decay OR lr_decay_gamma
    lr_decay = float(optim_cfg.get("lr_decay", optim_cfg.get("lr_decay_gamma", 0.99)))
    momentum_start = float(optim_cfg.get("momentum_start", 0.5))
    max_epochs = int(train_cfg.get("epochs", 200))
    momentum_ramp_epochs = int(optim_cfg.get("momentum_ramp_epochs", max_epochs))

    # Layer-wise hyperparams (allow override; default to single lr)
    base_lr = float(optim_cfg.get("lr", 1e-2))
    lr_conv = float(optim_cfg.get("lr_conv", base_lr))
    lr_fc = float(optim_cfg.get("lr_fc", base_lr))
    lr_alpha = float(optim_cfg.get("lr_alpha", base_lr))

    base_wd = float(optim_cfg.get("weight_decay", 0.0))
    wd_conv = float(optim_cfg.get("weight_decay_conv", base_wd))
    wd_fc = float(optim_cfg.get("weight_decay_fc", base_wd))
    wd_alpha = float(optim_cfg.get("weight_decay_alpha", base_wd))

    base_mu_final = float(optim_cfg.get("momentum_final", 0.9))
    mu_conv = float(optim_cfg.get("momentum_final_conv", base_mu_final))
    mu_fc = float(optim_cfg.get("momentum_final_fc", base_mu_final))
    mu_alpha = float(optim_cfg.get("momentum_final_alpha", base_mu_final))

    # ---- Collect params by name (PaperSiameseModel: encoder.*, head.alpha.*) ----
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
            # fall back: treat as FC-like
            fc_params.append(p)

    param_groups = [
        {
            "params": conv_params,
            "lr": lr_conv,
            "momentum": momentum_start,
            "weight_decay": wd_conv,
            "momentum_final": mu_conv,
        },
        {
            "params": fc_params,
            "lr": lr_fc,
            "momentum": momentum_start,
            "weight_decay": wd_fc,
            "momentum_final": mu_fc,
        },
        {
            "params": alpha_params,
            "lr": lr_alpha,
            "momentum": momentum_start,
            "weight_decay": wd_alpha,
            "momentum_final": mu_alpha,
        },
    ]

    # Remove empty groups to avoid optimizer errors
    param_groups = [g for g in param_groups if len(g["params"]) > 0]

    # IMPORTANT: keep "momentum_final" in param groups (used to build per-group finals list)
    optimizer = torch.optim.SGD(param_groups)

    # FIX: scheduler must use per-group finals, not a single global value
    momentum_finals = [
        float(g.get("momentum_final", base_mu_final)) for g in optimizer.param_groups
    ]

    scheduler = PaperLrMomentumScheduler(
        optimizer=optimizer,
        lr_decay=lr_decay,
        momentum_start=momentum_start,
        momentum_final=momentum_finals,  # <-- per-group finals
        momentum_ramp_epochs=momentum_ramp_epochs,
    )
    return OptimBundle(optimizer=optimizer, scheduler=scheduler)
