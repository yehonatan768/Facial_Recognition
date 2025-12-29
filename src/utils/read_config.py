from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

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
    candidates = [start, *start.parents]
    for p in candidates:
        has_src = (p / "src").is_dir()
        has_assets = (p / "assets").is_dir()
        has_images = (p / "images").is_dir()
        has_git = (p / ".git").exists()
        if has_src and ((has_assets and has_images) or has_git):
            return p
        if has_src:
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

    project_root = _find_project_root(p.parent)

    updated = False

    # ---- Always store resolved project_root (for notebooks / portability) ----
    cfg.setdefault("paths", {})
    if not isinstance(cfg["paths"], dict):
        cfg["paths"] = {}
        updated = True

    if cfg["paths"].get("project_root") != str(project_root):
        cfg["paths"]["project_root"] = str(project_root)
        updated = True

    # ---- Ensure portable paths exist ----
    updated |= _deep_set(cfg, "paths.images_root", "images")
    updated |= _deep_set(cfg, "paths.pairs_train", "assets/pairsDevTrain.txt")
    updated |= _deep_set(cfg, "paths.pairs_test", "assets/pairsDevTest.txt")

    # ---- Data defaults required by loaders ----
    updated |= _deep_set(cfg, "data.image_ext", ".jpg")
    updated |= _deep_set(cfg, "data.strict_exists", True)

    # ---- Required transform keys ----
    updated |= _deep_set(cfg, "transform.input_size", 105)
    updated |= _deep_set(cfg, "transform.mean", 0.5)
    updated |= _deep_set(cfg, "transform.std", 0.5)

    # ---- Pipeline selection (paper vs advanced) ----
    updated |= _deep_set(cfg, "pipeline.mode", "paper")  # set explicitly in YAML later

    # ---- Paper pipeline keys (required by PaperImageTransform.from_config) ----
    updated |= _deep_set(cfg, "paper.enable_jitter", True)
    updated |= _deep_set(cfg, "paper.jitter.brightness", 0.3)
    updated |= _deep_set(cfg, "paper.jitter.contrast", 0.3)
    updated |= _deep_set(cfg, "paper.jitter.saturation", 0.3)
    updated |= _deep_set(cfg, "paper.jitter.hue", 0.02)

    # ---- Advanced pipeline keys (required by build_transform_from_config when mode=advanced) ----
    updated |= _deep_set(cfg, "advanced.pre_crop_ratio", 0.90)
    updated |= _deep_set(cfg, "advanced.background_remover.enabled", True)
    updated |= _deep_set(cfg, "advanced.background_remover.model", "isnet-general-use")
    updated |= _deep_set(cfg, "advanced.background_remover.alpha_matting", True)
    updated |= _deep_set(cfg, "advanced.background_remover.alpha_matting_foreground_threshold", 240)
    updated |= _deep_set(cfg, "advanced.background_remover.alpha_matting_background_threshold", 10)
    updated |= _deep_set(cfg, "advanced.background_remover.alpha_matting_erode_size", 10)
    updated |= _deep_set(cfg, "advanced.background_remover.min_fg_fraction", 0.02)
    updated |= _deep_set(cfg, "advanced.background_remover.fg_black_threshold", 12)

    if write_back and updated:
        _write_yaml(p, cfg)

    return cfg
