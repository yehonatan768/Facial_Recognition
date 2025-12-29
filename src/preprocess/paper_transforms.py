from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
from PIL import Image
from torchvision import transforms


def _require(cfg: Dict[str, Any], path: str) -> Any:
    cur: Any = cfg
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError(f"Missing required config key: {path}")
        cur = cur[k]
    return cur


@dataclass
class PaperImageTransform:
    """
    Paper-faithful baseline preprocessing.

    Goal: keep the baseline as close as possible to a generic,
    non-domain-specific pipeline (no face-specific priors, no semantic filtering).

    Pipeline:
      (train-only optional paper jitter) -> Grayscale -> Resize -> ToTensor -> Normalize

    Notes:
    - No filters (equalize/edges) here.
    - No extra augmentations here (flip/rotation/blur/erasing).
    - Any improvements belong in the non-paper pipeline (e.g., advanced mode).
    """

    train: bool
    input_size: int
    mean: float
    std: float

    # Optional "paper baseline" photometric jitter (keep only if you already used it as baseline)
    enable_jitter: bool
    jitter_brightness: float
    jitter_contrast: float
    jitter_saturation: float
    jitter_hue: float

    def __post_init__(self) -> None:
        ops: list = []

        # Paper-only (baseline) jitter: applied BEFORE grayscale.
        # If you want strictest baseline, set paper.enable_jitter: false in config.
        if self.train and self.enable_jitter:
            ops.append(
                transforms.ColorJitter(
                    brightness=float(self.jitter_brightness),
                    contrast=float(self.jitter_contrast),
                    saturation=float(self.jitter_saturation),
                    hue=float(self.jitter_hue),
                )
            )

        ops.extend(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[self.mean], std=[self.std]),
            ]
        )

        self.t = transforms.Compose(ops)

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.t(img)

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], train: bool) -> "PaperImageTransform":
        return cls(
            train=train,
            input_size=int(_require(cfg, "transform.input_size")),
            mean=float(_require(cfg, "transform.mean")),
            std=float(_require(cfg, "transform.std")),
            enable_jitter=bool(_require(cfg, "paper.enable_jitter")),
            jitter_brightness=float(_require(cfg, "paper.jitter.brightness")),
            jitter_contrast=float(_require(cfg, "paper.jitter.contrast")),
            jitter_saturation=float(_require(cfg, "paper.jitter.saturation")),
            jitter_hue=float(_require(cfg, "paper.jitter.hue")),
        )
