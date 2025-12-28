from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

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
    # ---- training defaults ----
    _deep_setdefault(cfg, "train.epochs", 200)
    _deep_setdefault(cfg, "train.batch_size", 128)
    _deep_setdefault(cfg, "train.seed", 0)

    # ---- optimizer defaults (paper-like, not enforced) ----
    _deep_setdefault(cfg, "optim.lr", 0.01)
    _deep_setdefault(cfg, "optim.lr_decay_gamma", 0.99)
    _deep_setdefault(cfg, "optim.momentum_start", 0.5)
    _deep_setdefault(cfg, "optim.momentum_final", 0.7)
    _deep_setdefault(cfg, "optim.momentum_ramp_epochs", 150)

    # ---- early stopping defaults ----
    _deep_setdefault(cfg, "early_stop.enabled", True)
    _deep_setdefault(cfg, "early_stop.patience", 20)
    _deep_setdefault(cfg, "early_stop.min_delta", 0.0)
    _deep_setdefault(cfg, "early_stop.monitor", "val_acc")  # val_loss | val_acc
    _deep_setdefault(cfg, "early_stop.mode", "max")          # min | max

    # ---- data split defaults ----
    _deep_setdefault(cfg, "split.val_ratio", 0.14)
    _deep_setdefault(cfg, "split.target_pos_frac", 0.5)
    _deep_setdefault(cfg, "split.min_val_identities", 150)
    _deep_setdefault(cfg, "split.seed", cfg.get("train", {}).get("seed", 0))

    # ---- transforms ----
    _deep_setdefault(cfg, "transform.input_size", 105)

    # ---- runtime defaults (mirror what your loaders expect) ----
    _deep_setdefault(cfg, "runtime.num_workers", 0)
    _deep_setdefault(cfg, "runtime.pin_memory", True)

    return cfg


def _find_project_root(start: Path) -> Path:
    """
    Heuristic: walk upwards and pick the first directory that looks like the repo root.
    """
    start = start.resolve()
    candidates = [start, *start.parents]

    for p in candidates:
        has_src = (p / "src").is_dir()
        has_assets = (p / "assets").is_dir()
        has_images = (p / "images").is_dir()
        has_git = (p / ".git").exists()

        # Strong signals: src + (assets or images) or .git
        if has_src and ((has_assets and has_images) or has_git):
            return p

        # Accept src-only as fallback (still better than CWD)
        if has_src:
            return p

    # Last resort
    return start


def _coerce_default_paths(cfg: Dict[str, Any], project_root: Path) -> Tuple[Dict[str, Any], bool]:
    """
    Ensures cfg['paths'] contains portable (relative) paths.
    If updated, returns (cfg, True).
    """
    updated = False
    paths = cfg.get("paths")
    if not isinstance(paths, dict):
        cfg["paths"] = {}
        paths = cfg["paths"]
        updated = True

    # Defaults (relative to project root)
    defaults = {
        "images_root": "images",
        "pairs_train": "assets/pairsDevTrain.txt",
        "pairs_test": "assets/pairsDevTest.txt",
    }

    # If user supplied bare filenames (like pairsDevTrain.txt), auto-fix to assets/...
    # This matches your loader expectations and is what you want for portability.
    for key, default_rel in defaults.items():
        cur = paths.get(key)

        if not cur:
            paths[key] = default_rel
            updated = True
            continue

        # Normalize common mistake: "pairsDevTrain.txt" should be "assets/pairsDevTrain.txt"
        cur_p = Path(str(cur))
        if not cur_p.is_absolute() and cur_p.parent == Path(".") and key in ("pairs_train", "pairs_test"):
            candidate = project_root / "assets" / cur_p.name
            if candidate.exists():
                paths[key] = f"assets/{cur_p.name}"
                updated = True

    return cfg, updated


def _write_yaml(path: Path, cfg: Dict[str, Any]) -> None:
    # Write stable, human-readable YAML
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    path.write_text(text, encoding="utf-8")


def read_model_config(path: str | Path, write_back_paths: bool = True) -> Dict[str, Any]:
    """
    Load YAML config with safe defaults, and ensure portable paths exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")

    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError("Config YAML must parse to a dict at the top level.")


    project_root = _find_project_root(p.parent)
    update = False

    # Helpful to keep around (not required, but useful for debugging/logging)
    cfg.setdefault("paths", {})
    cfg["paths"].setdefault("project_root", str(project_root))

    return cfg
