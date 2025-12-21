from pathlib import Path
from typing import Dict, List, Tuple, Optional

from PIL import Image

from src.utils.load_pairs import parse_pairs_file, split_pairs


def load_image_rgb(path: Path) -> Image.Image:
    """
    Loads an image as RGB PIL.Image.
    """
    img = Image.open(path)
    return img.convert("RGB")


def load_train_test_pairs(
    train_pairs_path: Path,
    test_pairs_path: Path,
    images_root: Path,
    ext: str = ".jpg",
    logger=None,
    strict_exists: bool = False,
) -> Dict[str, object]:
    """
    Loads train/test pairs into:
      {
        "train_pairs": List[(p1, p2, label)],
        "test_pairs":  List[(p1, p2, label)],
        "X_train":     List[(p1, p2)],
        "y_train":     List[int],
        "X_test":      List[(p1, p2)],
        "y_test":      List[int],
      }
    """
    train_pairs = parse_pairs_file(
        pairs_txt=train_pairs_path,
        images_root=images_root,
        ext=ext,
        logger=logger,
        strict_exists=strict_exists,
    )
    test_pairs = parse_pairs_file(
        pairs_txt=test_pairs_path,
        images_root=images_root,
        ext=ext,
        logger=logger,
        strict_exists=strict_exists,
    )

    X_train, y_train = split_pairs(train_pairs)
    X_test, y_test = split_pairs(test_pairs)

    if logger is not None:
        logger.info("Loaded train pairs: %d", len(train_pairs))
        logger.info("Loaded test pairs:  %d", len(test_pairs))
        logger.info("Train positives: %d | negatives: %d", sum(y_train), len(y_train) - sum(y_train))
        logger.info("Test positives:  %d | negatives: %d", sum(y_test), len(y_test) - sum(y_test))

    return {
        "train_pairs": train_pairs,
        "test_pairs": test_pairs,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }


def batch_load_pair_images(
    X: List[Tuple[Path, Path]],
    max_items: Optional[int] = None,
    logger=None,
) -> List[Tuple[Image.Image, Image.Image]]:
    """
    Loads PIL images for a list of (img1_path, img2_path).
    Use for quick EDA or debugging (not recommended for large-scale training loops).
    """
    out: List[Tuple[Image.Image, Image.Image]] = []
    n = len(X) if max_items is None else min(len(X), max_items)

    for i in range(n):
        p1, p2 = X[i]
        try:
            out.append((load_image_rgb(p1), load_image_rgb(p2)))
        except Exception as e:
            if logger is not None:
                logger.warning("Failed to load pair %d: %s | %s (%s)", i, p1, p2, str(e))

    return out
