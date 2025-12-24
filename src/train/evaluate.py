from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

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
    model_outputs_logits: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      probs:  (N,) float tensor in [0,1]
      labels: (N,) float tensor in {0,1}
    Assumes loader yields (x1, x2, y) with y shape (B,1) or (B,)
    If model_outputs_logits=True then model(x1,x2) returns logits (B,1)/(B,)
    and we apply sigmoid() here.
    """
    model.eval()
    probs_all: List[torch.Tensor] = []
    y_all: List[torch.Tensor] = []

    for x1, x2, y in loader:
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device)

        out = model(x1, x2).view(-1)
        if model_outputs_logits:
            out = torch.sigmoid(out)

        p = out.detach().cpu()
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
    """
    best_acc = -1.0
    best_thr = 0.5

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
    model_outputs_logits: bool = True,
) -> Dict[str, float]:
    probs, labels = collect_pair_probs_and_labels(model, loader, device, model_outputs_logits=model_outputs_logits)
    best_acc, best_thr = find_best_threshold(probs, labels, num_steps=threshold_steps)

    # also provide acc at 0.5 for quick sanity checks
    acc_05 = accuracy_at_threshold(probs, labels, 0.5)

    return {
        "verif_acc_best_thr": float(best_acc),
        "verif_thr_best": float(best_thr),
        "verif_acc_thr_0.5": float(acc_05),
    }


@torch.no_grad()
def evaluate_verification_fixed_threshold(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    thr: float = 0.5,
    model_outputs_logits: bool = True,
) -> float:
    probs, labels = collect_pair_probs_and_labels(model, loader, device, model_outputs_logits=model_outputs_logits)
    return accuracy_at_threshold(probs, labels, thr)


# -----------------------------
# One-shot (episodic) evaluation
# -----------------------------
@dataclass(frozen=True)
class OneShotTrial:
    support: List[Tuple[str, Path]]
    query: Tuple[str, Path]


def _is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_identity_index(images_root: Path) -> Dict[str, List[Path]]:
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

        true_imgs = index[true_c]
        support_true_path = next(p for (c, p) in support if c == true_c)
        candidates = [p for p in true_imgs if p != support_true_path]
        if not candidates:
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
    model_outputs_logits: bool = True,
) -> Dict[str, float]:
    model.eval()
    correct = 0

    for t in trials:
        true_class, q_path = t.query
        q_img = transform(Image.open(q_path)).unsqueeze(0).to(device)

        best_score = None
        best_class = None

        for c, s_path in t.support:
            s_img = transform(Image.open(s_path)).unsqueeze(0).to(device)
            out = model(q_img, s_img).view(-1)[0]
            if model_outputs_logits:
                p = torch.sigmoid(out).item()
            else:
                p = out.item()

            if best_score is None or p > best_score:
                best_score = p
                best_class = c

        if best_class == true_class:
            correct += 1

    acc = correct / max(len(trials), 1)
    err = 1.0 - acc
    return {
        "oneshot_acc": float(acc),
        "oneshot_err": float(err),
        "oneshot_trials": float(len(trials)),
    }
