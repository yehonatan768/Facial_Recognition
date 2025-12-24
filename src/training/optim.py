from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union, Sequence

import torch


@dataclass
class OptimBundle:
    optimizer: torch.optim.Optimizer
    scheduler: "PaperLrMomentumScheduler"


class PaperLrMomentumScheduler:
    """
    Paper-style schedule:
      - LR decays multiplicatively each epoch: lr *= lr_decay (paper uses 0.99)
      - Momentum ramps linearly from momentum_start (paper uses 0.5) to momentum_final

    This class mutates optimizer.param_groups in-place.
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
        if isinstance(momentum_final, (list, tuple)):
            self.momentum_final = [float(x) for x in momentum_final]
        else:
            self.momentum_final = None
            self.momentum_final_scalar = float(momentum_final)
        self.momentum_ramp_epochs = max(1, int(momentum_ramp_epochs))

        # Record baseline LRs per param group (so decay is consistent)
        self.base_lrs = []
        for g in self.optimizer.param_groups:
            lr = float(g["lr"])
            self.base_lrs.append(lr)
        if epoch0_lr is not None:
            # override all group lrs at epoch 0
            for g in self.optimizer.param_groups:
                g["lr"] = float(epoch0_lr)
            self.base_lrs = [float(epoch0_lr) for _ in self.base_lrs]

    def _momentum_at(self, epoch: int) -> float:
        # Linear ramp from start to final
        t = min(max(epoch, 0), self.momentum_ramp_epochs) / float(self.momentum_ramp_epochs)
        return self.momentum_start + t * (self.momentum_final - self.momentum_start)

    def step(self, epoch: int) -> Dict[str, float]:
        """
        Call once per epoch (typically at the start of the epoch).

        Applies:
          lr_group_i = base_lr_group_i * (lr_decay ** epoch)
          momentum = linear_ramp(epoch)

        Returns a dict with current lr (group0) and momentum.
        """
        mom = self._momentum_at(epoch)

        for i, g in enumerate(self.optimizer.param_groups):
            g["lr"] = self.base_lrs[i] * (self.lr_decay ** epoch)

            if "momentum" in g:
                if self.momentum_final is not None:
                    mom_final_i = self.momentum_final[i]
                else:
                    mom_final_i = self.momentum_final_scalar

                t = min(max(epoch, 0), self.momentum_ramp_epochs) / float(self.momentum_ramp_epochs)
                mom = self.momentum_start + t * (mom_final_i - self.momentum_start)
                g["momentum"] = mom

        return {"lr": float(self.optimizer.param_groups[0]["lr"]), "momentum": mom}


def build_optimizer_and_scheduler(model: torch.nn.Module, cfg: Dict[str, Any]) -> OptimBundle:
    optim_cfg = cfg.get("optim", {})
    train_cfg = cfg.get("train", {})

    # Paper defaults (strict)
    lr_decay = float(optim_cfg.get("lr_decay", 0.99))
    momentum_start = float(optim_cfg.get("momentum_start", 0.5))
    max_epochs = int(train_cfg.get("epochs", 200))
    momentum_ramp_epochs = int(optim_cfg.get("momentum_ramp_epochs", max_epochs))

    # Layer-wise hyperparams (allow override; default to paper-ish single values)
    lr_conv = float(optim_cfg.get("lr_conv", optim_cfg.get("lr", 1e-2)))
    lr_fc   = float(optim_cfg.get("lr_fc",   optim_cfg.get("lr", 1e-2)))
    lr_alpha= float(optim_cfg.get("lr_alpha",optim_cfg.get("lr", 1e-2)))

    wd_conv = float(optim_cfg.get("weight_decay_conv", optim_cfg.get("weight_decay", 0.0)))
    wd_fc   = float(optim_cfg.get("weight_decay_fc",   optim_cfg.get("weight_decay", 0.0)))
    wd_alpha= float(optim_cfg.get("weight_decay_alpha",optim_cfg.get("weight_decay", 0.0)))

    mu_conv = float(optim_cfg.get("momentum_final_conv", optim_cfg.get("momentum_final", 0.9)))
    mu_fc   = float(optim_cfg.get("momentum_final_fc",   optim_cfg.get("momentum_final", 0.9)))
    mu_alpha= float(optim_cfg.get("momentum_final_alpha",optim_cfg.get("momentum_final", 0.9)))

    # Collect params by name (PaperSiameseModel: encoder.*, head.alpha.*)
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
        {"params": conv_params, "lr": lr_conv, "momentum": momentum_start, "weight_decay": wd_conv, "momentum_final": mu_conv},
        {"params": fc_params,   "lr": lr_fc,   "momentum": momentum_start, "weight_decay": wd_fc,   "momentum_final": mu_fc},
        {"params": alpha_params,"lr": lr_alpha,"momentum": momentum_start, "weight_decay": wd_alpha,"momentum_final": mu_alpha},
    ]

    # Remove empty groups to avoid optimizer errors
    param_groups = [g for g in param_groups if len(g["params"]) > 0]

    optimizer = torch.optim.SGD(param_groups)

    scheduler = PaperLrMomentumScheduler(
        optimizer=optimizer,
        lr_decay=lr_decay,
        momentum_start=momentum_start,
        momentum_final=[g.get("momentum_final", mu_fc) for g in optimizer.param_groups],
        momentum_ramp_epochs=momentum_ramp_epochs,
    )
    return OptimBundle(optimizer=optimizer, scheduler=scheduler)
