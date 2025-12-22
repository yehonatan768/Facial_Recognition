from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def read_model_config(path: str | Path) -> Dict[str, Any]:
    """
    Loads YAML config into a plain dict.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")

    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Config YAML must parse to a dict at the top level.")
    return cfg
