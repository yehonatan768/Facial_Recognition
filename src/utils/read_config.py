from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_set(d: Dict[str, Any], path: str, value: Any) -> bool:
    """
    Set d[path] = value if missing.
    Returns True if updated.
    """
    cur = d
    keys = path.split(".")
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    if keys[-1] not in cur:
        cur[keys[-1]] = value
        return True
    return False


def _find_project_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start, *start.parents]:
        if (p / "src").is_dir():
            return p
    return start


def _write_yaml(path: Path, cfg: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    path.write_text(text, encoding="utf-8")


def read_model_config(path: str | Path, write_back: bool = True) -> Dict[str, Any]:
    """
    Loads config.yaml, injects missing defaults, and optionally writes back.

    Background remover is intentionally REMOVED.
    Face mask is the only advanced spatial prior.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")

    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError("Config YAML must parse to a dict at the top level.")

    updated = False
    project_root = _find_project_root(p.parent)

    # -------------------------
    # Paths
    # -------------------------
    updated |= _deep_set(cfg, "paths.project_root", str(project_root))
    updated |= _deep_set(cfg, "paths.images_root", "images")
    updated |= _deep_set(cfg, "paths.pairs_train", "assets/pairsDevTrain.txt")
    updated |= _deep_set(cfg, "paths.pairs_test", "assets/pairsDevTest.txt")

    # -------------------------
    # Data
    # -------------------------
    updated |= _deep_set(cfg, "data.image_ext", ".jpg")
    updated |= _deep_set(cfg, "data.strict_exists", True)

    # -------------------------
    # Transform (shared)
    # -------------------------
    updated |= _deep_set(cfg, "transform.input_size", 105)
    updated |= _deep_set(cfg, "transform.mean", 0.5)
    updated |= _deep_set(cfg, "transform.std", 0.5)

    # -------------------------
    # Pipeline mode
    # -------------------------
    updated |= _deep_set(cfg, "pipeline.mode", "paper")  # "paper" | "advanced"

    # -------------------------
    # Paper pipeline defaults
    # -------------------------
    updated |= _deep_set(cfg, "paper.enable_jitter", True)
    updated |= _deep_set(cfg, "paper.jitter.brightness", 0.3)
    updated |= _deep_set(cfg, "paper.jitter.contrast", 0.3)
    updated |= _deep_set(cfg, "paper.jitter.saturation", 0.3)
    updated |= _deep_set(cfg, "paper.jitter.hue", 0.02)


    # -------------------------
    # Shared augmentations (optional)
    # -------------------------
    updated |= _deep_set(cfg, "augment.hflip.enabled", True)
    updated |= _deep_set(cfg, "augment.hflip.p", 0.5)

    updated |= _deep_set(cfg, "augment.rotation.enabled", True)
    updated |= _deep_set(cfg, "augment.rotation.degrees", 5.0)

    updated |= _deep_set(cfg, "augment.blur.enabled", True)
    updated |= _deep_set(cfg, "augment.blur.p", 0.15)
    updated |= _deep_set(cfg, "augment.blur.kernel_size", 3)
    updated |= _deep_set(cfg, "augment.blur.sigma", [0.3, 1.0])

    updated |= _deep_set(cfg, "augment.random_erasing.enabled", False)
    updated |= _deep_set(cfg, "augment.random_erasing.p", 0.10)
    updated |= _deep_set(cfg, "augment.random_erasing.scale", [0.02, 0.08])
    updated |= _deep_set(cfg, "augment.random_erasing.ratio", [0.3, 3.3])
    updated |= _deep_set(cfg, "augment.random_erasing.value", 0.0)


    # -------------------------
    # Advanced pipeline (face mask)
    # -------------------------
    updated |= _deep_set(cfg, "advanced.pre_crop_ratio", 0.90)

    updated |= _deep_set(cfg, "advanced.face_mask.center", [0.0, 0.0])
    updated |= _deep_set(cfg, "advanced.face_mask.axes", [0.75, 0.90])
    updated |= _deep_set(cfg, "advanced.face_mask.edge_softness", 0.08)
    updated |= _deep_set(cfg, "advanced.face_mask.power", 1.5)

    # -------------------------
    # Write back if needed
    # -------------------------
    if write_back and updated:
        _write_yaml(p, cfg)

    return cfg
