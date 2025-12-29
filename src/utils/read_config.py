from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_set(d: Dict[str, Any], path: str, value: Any) -> bool:
    """
    Set d[path] = value if missing. Returns True if updated.
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

    # Paths
    updated |= _deep_set(cfg, "paths.project_root", str(project_root))
    updated |= _deep_set(cfg, "paths.images_root", "images")
    updated |= _deep_set(cfg, "paths.pairs_train", "assets/pairsDevTrain.txt")
    updated |= _deep_set(cfg, "paths.pairs_test", "assets/pairsDevTest.txt")

    # Data
    updated |= _deep_set(cfg, "data.image_ext", ".jpg")
    updated |= _deep_set(cfg, "data.strict_exists", True)

    # Transform
    updated |= _deep_set(cfg, "transform.input_size", 105)
    updated |= _deep_set(cfg, "transform.mean", 0.5)
    updated |= _deep_set(cfg, "transform.std", 0.5)

    # Pipeline mode
    updated |= _deep_set(cfg, "pipeline.mode", "paper")  # "paper" or "advanced"

    # Paper pipeline defaults (if used)
    updated |= _deep_set(cfg, "paper.enable_jitter", True)
    updated |= _deep_set(cfg, "paper.jitter.brightness", 0.3)
    updated |= _deep_set(cfg, "paper.jitter.contrast", 0.3)
    updated |= _deep_set(cfg, "paper.jitter.saturation", 0.3)
    updated |= _deep_set(cfg, "paper.jitter.hue", 0.02)

    # Advanced pipeline (DeepLabV3)
    updated |= _deep_set(cfg, "advanced.pre_crop_ratio", 0.90)

    updated |= _deep_set(cfg, "advanced.background_remover.enabled", True)
    updated |= _deep_set(cfg, "advanced.background_remover.backend", "torchvision_deeplabv3")

    # IMPORTANT: CPU is safest with DataLoader workers
    updated |= _deep_set(cfg, "advanced.background_remover.device", "cpu")

    # Person probability threshold
    updated |= _deep_set(cfg, "advanced.background_remover.threshold", 0.35)

    # Fail-safe thresholds
    updated |= _deep_set(cfg, "advanced.background_remover.min_fg_fraction", 0.02)
    updated |= _deep_set(cfg, "advanced.background_remover.fg_black_threshold", 12)
    updated |= _deep_set(cfg, "advanced.background_remover.min_gray_variance", 5.0)

    if write_back and updated:
        _write_yaml(p, cfg)

    return cfg
