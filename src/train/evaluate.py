from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple, Optional

import torch
import torch.nn as nn
from PIL import Image


# -----------------------------
# Verification (pairs) evaluation
# -----------------------------
@torch.no_grad()
def collect_pair_probs_and_labels(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      probs:  (N,) float tensor in [0,1]
      labels: (N,) float tensor in {0,1}
    Assumes loader yields (x1, x2, y) with y shape (B,1) or (B,)
    and model(x1,x2) returns sigmoid probability (B,1) or (B,).
    """
    model.eval()
    probs_all: List[torch.Tensor] = []
    y_all: List[torch.Tensor] = []

    for x1, x2, y in loader:
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device)

        p = model(x1, x2)
        p = p.view(-1).detach().cpu()
        y = y.view(-1).detach().cpu()

        probs_all.append(p)
        y_all.append(y)

    probs = torch.cat(probs_all, dim=0)
    labels = torch.cat(y_all, dim=0)
    return probs, labels


def accuracy_at_threshold(probs: torch.Tensor, labels: torch.Tensor, thr: float) -> float:
    pred = (probs >= thr).float()
    return float((pred == labels).float().mean().item())


def find_best_threshold(
    probs: torch.Tensor,
    labels: torch.Tensor,
    num_steps: int = 400,
) -> Tuple[float, float]:
    """
    Grid-search threshold in [0,1] and return (best_acc, best_thr).
    Paper reports "best checkpoint and threshold" for verification. :contentReference[oaicite:3]{index=3}
    """
    best_acc = -1.0
    best_thr = 0.5

    # include endpoints
    for i in range(num_steps + 1):
        thr = i / float(num_steps)
        acc = accuracy_at_threshold(probs, labels, thr)
        if acc > best_acc:
            best_acc = acc
            best_thr = thr

    return best_acc, best_thr


@torch.no_grad()
def evaluate_verification(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold_steps: int = 400,
) -> Dict[str, float]:
    probs, labels = collect_pair_probs_and_labels(model, loader, device)
    best_acc, best_thr = find_best_threshold(probs, labels, num_steps=threshold_steps)

    # also provide acc at 0.5 for quick sanity checks
    acc_05 = accuracy_at_threshold(probs, labels, 0.5)

    return {
        "val_verif_acc_best_thr": float(best_acc),
        "val_verif_thr_best": float(best_thr),
        "val_verif_acc_thr_0.5": float(acc_05),
    }


# -----------------------------
# One-shot (episodic) evaluation
# -----------------------------
@dataclass(frozen=True)
class OneShotTrial:
    """
    N-way 1-shot trial:
      support: list of (class_id, image_path) length = N
      query:   (true_class_id, image_path)
    """
    support: List[Tuple[str, Path]]
    query: Tuple[str, Path]


def _is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_identity_index(images_root: Path) -> Dict[str, List[Path]]:
    """
    Build {identity_name: [img_paths...]} from folder structure:
      images_root/<identity>/<identity>_0001.jpg ...
    Your dataset is organized this way already (pairs format).
    """
    out: Dict[str, List[Path]] = {}
    for d in images_root.iterdir():
        if not d.is_dir():
            continue
        imgs = [p for p in d.iterdir() if p.is_file() and _is_image_file(p)]
        if len(imgs) >= 2:
            out[d.name] = sorted(imgs)
    return out


def sample_one_shot_trials(
    index: Dict[str, List[Path]],
    num_way: int,
    num_trials: int,
    seed: int = 0,
) -> List[OneShotTrial]:
    """
    Generic N-way 1-shot sampling (not Omniglot drawer-based).
    Paper's Omniglot one-shot protocol is drawer-based; this is the closest analogue
    for your identity-folder face dataset.
    """
    rng = random.Random(seed)
    classes = [c for c, imgs in index.items() if len(imgs) >= 2]
    if len(classes) < num_way:
        raise ValueError(f"Not enough identities with >=2 images. Have {len(classes)}, need {num_way}.")

    trials: List[OneShotTrial] = []
    for _ in range(num_trials):
        chosen = rng.sample(classes, num_way)
        true_c = rng.choice(chosen)

        support: List[Tuple[str, Path]] = []
        for c in chosen:
            imgs = index[c]
            support_img = rng.choice(imgs)
            support.append((c, support_img))

        # query must be different image from the true class
        true_imgs = index[true_c]
        # avoid picking the exact same file as the support for that class
        support_true_path = next(p for (c, p) in support if c == true_c)
        candidates = [p for p in true_imgs if p != support_true_path]
        if not candidates:
            # extremely rare if class has exactly 1 image; we filtered >=2, but keep safe
            candidates = true_imgs
        query_img = rng.choice(candidates)

        trials.append(OneShotTrial(support=support, query=(true_c, query_img)))

    return trials


@torch.no_grad()
def evaluate_one_shot(
    model: nn.Module,
    device: torch.device,
    transform: Callable[[Image.Image], torch.Tensor],
    trials: Sequence[OneShotTrial],
) -> Dict[str, float]:
    """
    For each trial:
      - compute similarity prob p(query, support_i) for i=1..N
      - predict class of max prob
    Paper: choose class with maximum similarity score. :contentReference[oaicite:4]{index=4}
    """
    model.eval()
    correct = 0

    for t in trials:
        true_class, q_path = t.query
        q_img = transform(Image.open(q_path)).unsqueeze(0).to(device)

        best_score = None
        best_class = None

        for c, s_path in t.support:
            s_img = transform(Image.open(s_path)).unsqueeze(0).to(device)
            p = model(q_img, s_img).view(-1)[0].item()

            if best_score is None or p > best_score:
                best_score = p
                best_class = c

        if best_class == true_class:
            correct += 1

    acc = correct / max(len(trials), 1)
    err = 1.0 - acc
    return {
        "val_oneshot_acc": float(acc),
        "val_oneshot_err": float(err),
        "val_oneshot_trials": float(len(trials)),
    }


def choose_early_stop_metric(metrics: Dict[str, float], prefer_one_shot: bool) -> Tuple[str, float]:
    """
    Paper stops based on one-shot validation error (320 tasks). :contentReference[oaicite:5]{index=5}
    If prefer_one_shot=True and oneshot metrics exist -> minimize oneshot_err
    else -> minimize val_loss (handled in train.py).
    """
    if prefer_one_shot and "val_oneshot_err" in metrics:
        return "val_oneshot_err", metrics["val_oneshot_err"]
    return "val_loss", metrics.get("val_loss", float("inf"))
