# src/training/optim.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn


@dataclass
class LayerHyper:
    lr: float
    momentum_target: float
    weight_decay: float = 0.0


@dataclass
class OptimState:
    optimizer: torch.optim.Optimizer
    lr_decay: float
    momentum_start: float
    momentum_ramp_epochs: int


def _as_layer_hyper(h: Any) -> LayerHyper:
    """
    Accept either:
      - LayerHyper
      - dict-like: {"lr": ..., "momentum_target": ..., "weight_decay": ...}
    """
    if isinstance(h, LayerHyper):
        return h

    if isinstance(h, Mapping):
        if "lr" not in h:
            raise KeyError("layerwise.*.lr is missing")
        lr = float(h.get("lr"))

        mt = h.get("momentum_target", h.get("momentum", h.get("momentum_final")))
        if mt is None:
            raise KeyError("layerwise.*.momentum_target (or momentum/momentum_final) is missing")

        wd = float(h.get("weight_decay", 0.0))
        return LayerHyper(lr=lr, momentum_target=float(mt), weight_decay=wd)

    raise TypeError(f"Unsupported hyperparam type: {type(h)} (expected LayerHyper or dict)")


def build_optimizer(
    model: nn.Module,
    layerwise: dict,
    lr_decay: float = 0.99,
    momentum_start: float = 0.5,
    momentum_ramp_epochs: int = 20,
) -> OptimState:
    """
    Build an SGD optimizer with per-layer param groups and schedule state.
    layerwise: dict mapping group names -> LayerHyper or dict {lr, momentum_target, weight_decay}
    Expected keys: conv1, conv2, conv3, conv4, fc, head
    """
    # ---- validate keys early ----
    required = ["conv1", "conv2", "conv3", "conv4", "fc", "head"]
    missing = [k for k in required if k not in layerwise]
    if missing:
        raise KeyError(f"optim.layerwise missing keys: {missing}. Present: {sorted(layerwise.keys())}")

    enc = model.encoder
    head = model.head

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

    # ---- param groups ----
    add_group("conv1", list(enc.conv1.parameters()), layerwise["conv1"])
    add_group("conv2", list(enc.conv2.parameters()), layerwise["conv2"])
    add_group("conv3", list(enc.conv3.parameters()), layerwise["conv3"])
    add_group("conv4", list(enc.conv4.parameters()), layerwise["conv4"])

    # fc might be lazily created; if not created yet, just skip it for now.
    # It will be created after first forward; you should rebuild the optimizer then if you want FC separate.
    if getattr(enc, "fc", None) is not None:
        add_group("fc", list(enc.fc.parameters()), layerwise["fc"])

    # Head parameters: alpha always, bias optional
    head_params: list[nn.Parameter] = [head.alpha]
    if getattr(head, "bias", None) is not None:
        head_params.append(head.bias)
    add_group("head", head_params, layerwise["head"])

    optimizer = torch.optim.SGD(groups)

    return OptimState(
        optimizer=optimizer,
        lr_decay=float(lr_decay),
        momentum_start=float(momentum_start),
        momentum_ramp_epochs=int(momentum_ramp_epochs),
    )


def step_schedule(state: OptimState, epoch_idx: int) -> None:
    """
    Exponential LR decay each epoch and linear momentum ramp.
    Call once per epoch.
    """
    # Exponential LR decay: lr = base_lr * (lr_decay ** epoch_idx)
    for g in state.optimizer.param_groups:
        base_lr = float(g["lr"])  # note: this is the "current" lr slot; we treat it as base at epoch 0
        # To avoid compounding error, store initial lr once
        if "_lr0" not in g:
            g["_lr0"] = base_lr
        g["lr"] = float(g["_lr0"]) * (state.lr_decay ** int(epoch_idx))

    # Linear momentum ramp: from momentum_start to momentum_target over ramp epochs
    t = 1.0
    if state.momentum_ramp_epochs > 0:
        t = min(1.0, max(0.0, (epoch_idx + 1) / float(state.momentum_ramp_epochs)))

    for g in state.optimizer.param_groups:
        m0 = float(state.momentum_start)
        m1 = float(g.get("_momentum_target", m0))
        g["momentum"] = (1.0 - t) * m0 + t * m1
