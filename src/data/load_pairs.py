from pathlib import Path
from typing import List, Tuple, Optional


def lfw_image_path(images_root: Path, person: str, idx: int, ext: str = ".jpg") -> Path:
    """
    LFW-style path:
      images_root/person/person_0001.jpg
    """
    return images_root / person / f"{person}_{idx:04d}{ext}"


def parse_pairs_file(
    pairs_txt: Path,
    images_root: Path,
    ext: str = ".jpg",
    logger=None,
    strict_exists: bool = False,
) -> List[Tuple[Path, Path, int]]:
    """
    Parses pairsDevTrain.txt / pairsDevTest.txt.

    Supported line formats:
      - First line: integer N_pos (number of positive pairs)
      - Positive lines (3 columns):  name idx1 idx2   => label=1
      - Negative lines (4 columns):  name1 idx1 name2 idx2 => label=0
      - Some files include single-integer separator lines in the second part; ignored.

    Returns:
      List[(img1_path, img2_path, label)] with label 1=same, 0=different
    """
    text = pairs_txt.read_text(encoding="utf-8", errors="ignore")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    if not lines or not lines[0].isdigit():
        raise ValueError(f"Expected first line to be integer count in: {pairs_txt}")

    n_pos = int(lines[0])
    pairs: List[Tuple[Path, Path, int]] = []

    # Positives: next n_pos lines
    pos_lines = lines[1:1 + n_pos]
    if len(pos_lines) < n_pos:
        raise ValueError(f"{pairs_txt} ended early: expected {n_pos} positive lines, got {len(pos_lines)}")

    missing = 0

    for ln in pos_lines:
        parts = ln.split()
        if len(parts) != 3:
            raise ValueError(f"Invalid positive pair line (expected 3 columns): {ln}")

        name, a, b = parts
        p1 = lfw_image_path(images_root, name, int(a), ext=ext)
        p2 = lfw_image_path(images_root, name, int(b), ext=ext)

        if strict_exists and (not p1.exists() or not p2.exists()):
            missing += int(not p1.exists()) + int(not p2.exists())
            continue

        pairs.append((p1, p2, 1))

    # Remaining: negatives + optional separators
    for ln in lines[1 + n_pos:]:
        parts = ln.split()

        # Separator line like "1" / "4"
        if len(parts) == 1 and parts[0].isdigit():
            continue

        if len(parts) == 4:
            name1, a, name2, b = parts
            p1 = lfw_image_path(images_root, name1, int(a), ext=ext)
            p2 = lfw_image_path(images_root, name2, int(b), ext=ext)

            if strict_exists and (not p1.exists() or not p2.exists()):
                missing += int(not p1.exists()) + int(not p2.exists())
                continue

            pairs.append((p1, p2, 0))
            continue

        # Some variants may include extra 3-col positives in the tail; handle safely.
        if len(parts) == 3:
            name, a, b = parts
            p1 = lfw_image_path(images_root, name, int(a), ext=ext)
            p2 = lfw_image_path(images_root, name, int(b), ext=ext)

            if strict_exists and (not p1.exists() or not p2.exists()):
                missing += int(not p1.exists()) + int(not p2.exists())
                continue

            pairs.append((p1, p2, 1))
            continue

        raise ValueError(f"Unrecognized line format ({len(parts)} columns): {ln}")

    if logger is not None and missing > 0:
        logger.warning("Skipped %d missing image paths due to strict_exists=True.", missing)

    return pairs


def split_pairs(
    pairs: List[Tuple[Path, Path, int]]
) -> Tuple[List[Tuple[Path, Path]], List[int]]:
    """
    Convenience: returns (X, y)
      X = [(img1, img2), ...]
      y = [label, ...]
    """
    X = [(a, b) for (a, b, _) in pairs]
    y = [label for (_, _, label) in pairs]
    return X, y
