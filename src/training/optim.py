from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Mapping

import torch
import torch.nn as nn


@dataclass(frozen=True)
class LayerHyper:
    lr: float
    momentum_target: float
    weight_decay: float


def _as_layer_hyper(h: Any) -> LayerHyper:
    """
    Accept either:
      - LayerHyper
      - dict-like: {"lr": ..., "momentum_target": ..., "weight_decay": ...}
    """
    if isinstance(h, LayerHyper):
        return h
    if isinstance(h, Mapping):
        # allow a few common aliases
        lr = float(h.get("lr"))
        mt = h.get("momentum_target", h.get("momentum", h.get("momentum_final")))
        if mt is None:
            raise KeyError("layerwise.*.momentum_target (or momentum/momentum_final) is missing")
        wd = float(h.get("weight_decay", 0.0))
        return LayerHyper(lr=lr, momentum_target=float(mt), weight_decay=wd)

    raise TypeError(f"Unsupported hyperparam type: {type(h)} (expected LayerHyper or dict)")



@dataclass
class OptimState:
    optimizer: torch.optim.Optimizer
    lr_decay: float
    momentum_start: float
    momentum_ramp_epochs: int


def _named_params(module: nn.Module) -> List[Tuple[str, nn.Parameter]]:
    return [(n, p) for n, p in module.named_parameters() if p.requires_grad]


def build_optimizer(
    model: nn.Module,
    layerwise: Dict[str, LayerHyper],
    lr_decay: float,
    momentum_start: float,
    momentum_ramp_epochs: int,
) -> OptimState:
    """SGD optimizer with layer-wise (lr, momentum_target, weight_decay)."""

    # Expect these submodules to exist.
    enc = getattr(model, "encoder", None)
    head = getattr(model, "head", None)
    if enc is None or head is None:
        raise ValueError("Model must have .encoder and .head")

    groups = []

    def add_group(name: str, params: list[nn.Parameter], h: Any) -> None:
        h = _as_layer_hyper(h)

        groups.append(
            {
                "name": name,
                "params": params,
                "lr": float(h.lr),
                "momentum": float(momentum_start),
                "_momentum_target": float(h.momentum_target),
                "weight_decay": float(h.weight_decay),
            }
        )

    add_group("conv1", list(enc.conv1.parameters()))
    add_group("conv2", list(enc.conv2.parameters()))
    add_group("conv3", list(enc.conv3.parameters()))
    add_group("conv4", list(enc.conv4.parameters()))
    add_group("fc", list(enc.fc.parameters()))
    add_group("head", [head.alpha] + ([head.bias] if getattr(head, "bias", None) is not None else []))

    opt = torch.optim.SGD(groups)

    return OptimState(
        optimizer=opt,
        lr_decay=float(lr_decay),
        momentum_start=float(momentum_start),
        momentum_ramp_epochs=int(momentum_ramp_epochs),
    )


def step_schedule(state: OptimState, epoch: int) -> None:
    """Apply exponential LR decay and linear momentum ramp, per epoch."""
    opt = state.optimizer

    # LR: eta_j^(T) = lr_decay * eta_j^(T-1)
    if epoch > 0:
        for g in opt.param_groups:
            g["lr"] = float(g["lr"]) * state.lr_decay

    # Momentum: start at momentum_start, ramp linearly toward momentum_target
    ramp = max(1, state.momentum_ramp_epochs)
    t = min(epoch, ramp) / float(ramp)
    for g in opt.param_groups:
        target = float(g.get("_momentum_target", g.get("momentum", 0.0)))
        g["momentum"] = state.momentum_start + t * (target - state.momentum_start)
