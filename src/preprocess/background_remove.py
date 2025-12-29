from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PIL import Image


@dataclass(frozen=True)
class RembgConfig:
    enabled: bool
    model: str  # e.g. "u2net" / "isnet-general-use" etc. (depends on rembg version)


class BackgroundRemover:
    """
    Background remover powered by the `rembg` library.
    Returns a PIL RGB image where background is removed (transparent or filled depending on rembg).
    We convert output to RGB with black background for determinism downstream.
    """

    def __init__(self, cfg: RembgConfig):
        self.cfg = cfg
        self._session = None  # lazy

    def _ensure_session(self):
        if self._session is not None:
            return
        try:
            from rembg import new_session  # type: ignore
        except Exception as e:
            raise ImportError(
                "rembg is required for the advanced pipeline. Install it with: pip install rembg"
            ) from e

        self._session = new_session(self.cfg.model)

    def __call__(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            raise TypeError(f"BackgroundRemover expects PIL.Image, got {type(img)}")

        if not self.cfg.enabled:
            return img

        self._ensure_session()
        try:
            from rembg import remove  # type: ignore
        except Exception as e:
            raise ImportError(
                "rembg is required for the advanced pipeline. Install it with: pip install rembg"
            ) from e

        out = remove(img, session=self._session)

        # rembg may return RGBA; force deterministic RGB with black fill.
        if out.mode == "RGBA":
            bg = Image.new("RGBA", out.size, (0, 0, 0, 255))
            out = Image.alpha_composite(bg, out).convert("RGB")
        else:
            out = out.convert("RGB")

        return out
