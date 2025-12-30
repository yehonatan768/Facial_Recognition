from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml


def _deep_merge(a: Dict[str, Any], b: Mapping[str, Any]) -> Dict[str, Any]:
    """Deep-merge b into a (mutating a) and return a."""
    for k, v in b.items():
        if isinstance(v, Mapping) and isinstance(a.get(k), dict):
            _deep_merge(a[k], v)
        else:
            a[k] = v
    return a


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    if not isinstance(obj, dict):
        raise ValueError(f"Expected YAML mapping at {path}, got {type(obj)}")
    return obj


def load_config_files(paths: Iterable[Path]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    for p in paths:
        _deep_merge(cfg, load_yaml(p))
    return cfg


def resolve_project_root(cfg: Dict[str, Any], cli_project_root: str | None = None) -> Path:
    pr = (cli_project_root or cfg.get("paths", {}).get("project_root") or "").strip()
    if pr:
        return Path(pr).expanduser().resolve()
    # Default: working directory
    return Path.cwd().resolve()
