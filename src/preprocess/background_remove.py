from __future__ import annotations

from dataclasses import dataclass
from PIL import Image

from rembg import new_session, remove


@dataclass(frozen=True)
class RembgConfig:
    enabled: bool
    model: str

    # Quality controls (explicitly set in config)
    alpha_matting: bool
    alpha_matting_foreground_threshold: int
    alpha_matting_background_threshold: int
    alpha_matting_erode_size: int


class BackgroundRemover:
    """
    rembg-based background removal.
    Output is forced to RGB with black background for determinism.
    """

    def __init__(self, cfg: RembgConfig):
        self.cfg = cfg
        self._session = new_session(cfg.model)

    def __call__(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            raise TypeError(f"BackgroundRemover expects PIL.Image, got {type(img)}")

        if not self.cfg.enabled:
            return img

        out = remove(
            img,
            session=self._session,
            alpha_matting=self.cfg.alpha_matting,
            alpha_matting_foreground_threshold=self.cfg.alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=self.cfg.alpha_matting_background_threshold,
            alpha_matting_erode_size=self.cfg.alpha_matting_erode_size,
        )

        # rembg may return RGBA; force deterministic RGB with black fill.
        if out.mode == "RGBA":
            bg = Image.new("RGBA", out.size, (0, 0, 0, 255))
            out = Image.alpha_composite(bg, out).convert("RGB")
        else:
            out = out.convert("RGB")

        return out
