from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from src.data.load_data import load_image_rgb
from src.data.paper_transforms import PaperImageTransform


@dataclass
class OneShotResult:
    accuracy: float
    n_trials: int
    n_way: int


def build_class_to_images(images_root: Path, ext: str = ".jpg") -> Dict[str, List[Path]]:
    """
    Builds a mapping: class_name (folder) -> list of image paths.
    Compatible with your LFW-style folder layout used by load_pairs.py. :contentReference[oaicite:3]{index=3}
    """
    class_to_imgs: Dict[str, List[Path]] = {}
    for person_dir in images_root.iterdir():
        if not person_dir.is_dir():
            continue
        imgs = sorted([p for p in person_dir.iterdir() if p.is_file() and p.suffix.lower() == ext.lower()])
        if len(imgs) >= 2:  # need at least 2 images per class for a query+support
            class_to_imgs[person_dir.name] = imgs
    return class_to_imgs


def _sample_two_distinct(items: Sequence[Path], rng: random.Random) -> Tuple[Path, Path]:
    a = rng.choice(items)
    b = rng.choice(items)
    while b == a:
        b = rng.choice(items)
    return a, b


@torch.no_grad()
def evaluate_one_shot(
    model,  # expects model.score(x1,x2)->(B,1) or model(x1,x2)->(p,_,_)
    cfg: Dict,
    images_root: Path,
    device: torch.device,
    n_way: int = 20,
    n_trials: int = 400,
    class_whitelist: Optional[Sequence[str]] = None,
    seed: int = 0,
    ext: str = ".jpg",
) -> OneShotResult:
    """
    Paper-style one-shot evaluation on your foldered dataset.

    - Builds trials dynamically from image folders (class = folder name).
    - Uses eval transform (no augmentation).
    - For each trial:
        choose N classes
        choose a query class among them
        query image = class image A
        supports = one image per class (query-class uses different image B)
        predict argmax p_same(query, support)

    Returns accuracy over trials.
    """
    rng = random.Random(seed)

    class_to_imgs = build_class_to_images(images_root=images_root, ext=ext)
    if class_whitelist is not None:
        allowed = set(class_whitelist)
        class_to_imgs = {k: v for k, v in class_to_imgs.items() if k in allowed}

    classes = sorted(class_to_imgs.keys())
    if len(classes) < n_way:
        raise ValueError(f"Not enough classes for {n_way}-way one-shot. Have {len(classes)} classes with >=2 images.")

    # eval transform
    transform = PaperImageTransform.from_config(cfg, train=False)

    model.eval()

    correct = 0
    for _ in range(n_trials):
        trial_classes = rng.sample(classes, k=n_way)
        query_class = rng.choice(trial_classes)

        # query + query-class support
        q_path, q_support_path = _sample_two_distinct(class_to_imgs[query_class], rng)

        support_paths: List[Path] = []
        support_labels: List[int] = []  # 1 for true class support, 0 else

        for c in trial_classes:
            if c == query_class:
                support_paths.append(q_support_path)
                support_labels.append(1)
            else:
                # just pick one support image from that class
                support_paths.append(rng.choice(class_to_imgs[c]))
                support_labels.append(0)

        # Load and transform images
        q_img = transform(load_image_rgb(q_path))  # (1,H,W)
        supports = [transform(load_image_rgb(p)) for p in support_paths]

        # Batch into tensors
        q_batch = torch.stack([q_img] * n_way, dim=0).to(device)  # (N,1,H,W)
        s_batch = torch.stack(supports, dim=0).to(device)         # (N,1,H,W)

        # Score all comparisons at once (vectorized)
        if hasattr(model, "score"):
            p = model.score(q_batch, s_batch)  # (N,1)
        else:
            p, _, _ = model(q_batch, s_batch)

        p = p.view(-1)  # (N,)
        pred_idx = int(torch.argmax(p).item())

        if support_labels[pred_idx] == 1:
            correct += 1

    acc = correct / float(n_trials) if n_trials > 0 else 0.0
    return OneShotResult(accuracy=acc, n_trials=n_trials, n_way=n_way)
