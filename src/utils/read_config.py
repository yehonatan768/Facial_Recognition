from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_setdefault(d: Dict[str, Any], path: str, value: Any) -> None:
    cur = d
    keys = path.split(".")
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur.setdefault(keys[-1], value)


def _apply_paper_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Paper-critical defaults (architecture fixed in code)
    _deep_setdefault(cfg, "train.epochs", 200)
    _deep_setdefault(cfg, "train.batch_size", 128)
    _deep_setdefault(cfg, "train.seed", 0)

    _deep_setdefault(cfg, "optim.lr_decay", 0.99)
    _deep_setdefault(cfg, "optim.momentum_start", 0.5)
    _deep_setdefault(cfg, "optim.momentum_final", 0.7)  # your config.yaml uses 0.7
    _deep_setdefault(cfg, "optim.momentum_ramp_epochs", 150)

    # Early stopping (paper uses patience=20; we now monitor val_loss by default)
    _deep_setdefault(cfg, "early_stop.enabled", True)
    _deep_setdefault(cfg, "early_stop.patience", 20)
    _deep_setdefault(cfg, "early_stop.min_delta", 0.0)
    _deep_setdefault(cfg, "early_stop.monitor", "val_loss")  # val_loss | val_acc
    _deep_setdefault(cfg, "early_stop.mode", "min")          # min | max

    # Transform defaults (paper ranges)
    _deep_setdefault(cfg, "transform.input_size", 105)
    _deep_setdefault(cfg, "transform.component_apply_prob", 0.5)
    _deep_setdefault(cfg, "transform.rotation_deg", [-10.0, 10.0])
    _deep_setdefault(cfg, "transform.shear_x", [-0.3, 0.3])
    _deep_setdefault(cfg, "transform.shear_y", [-0.3, 0.3])
    _deep_setdefault(cfg, "transform.scale_x", [0.8, 1.2])
    _deep_setdefault(cfg, "transform.scale_y", [0.8, 1.2])
    _deep_setdefault(cfg, "transform.translate_x", [-2.0, 2.0])
    _deep_setdefault(cfg, "transform.translate_y", [-2.0, 2.0])

    return cfg


def _assert_strict(cfg: Dict[str, Any]) -> None:
    # Keep only invariants that still exist in your simplified pipeline
    assert cfg["train"]["epochs"] == 200
    assert cfg["train"]["batch_size"] == 128
    assert abs(cfg["optim"]["lr_decay"] - 0.99) < 1e-12
    assert abs(cfg["optim"]["momentum_start"] - 0.5) < 1e-12
    assert cfg["early_stop"]["patience"] == 20
    assert cfg["transform"]["input_size"] == 105


def read_model_config(path: str | Path, strict_paper: bool = True) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")

    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError("Config YAML must parse to a dict at the top level.")

    cfg = _apply_paper_defaults(cfg)

    if strict_paper:
        _assert_strict(cfg)

    return cfg
