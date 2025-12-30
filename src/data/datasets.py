from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import math
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

from src.data.load_data import load_image_rgb
from src.data.transforms import build_transform

Pair = Tuple[Path, Path, int]


class PairPathDataset(Dataset):
    """Dataset of image pairs on disk.

    Expected pair format: (img1_path, img2_path, label) where label in {0,1}.
    The provided transform must convert PIL.Image -> torch.Tensor.
    """

    def __init__(self, pairs: List[Pair], transform):
        self.pairs = pairs
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        p1, p2, y = self.pairs[idx]
        im1 = load_image_rgb(p1)
        im2 = load_image_rgb(p2)
        x1 = self.transform(im1)
        x2 = self.transform(im2)
        return x1, x2, torch.tensor(int(y), dtype=torch.long)


class BalancedPairBatchSampler(Sampler[List[int]]):
    """
    Batch sampler that yields half positive and half negative indices.

    IMPORTANT: This version does NOT shuffle.
    It iterates positives and negatives in fixed index order and cycles when needed.
    """

    def __init__(self, pairs: List[Pair], batch_size: int):
        if batch_size < 2:
            raise ValueError("batch_size must be >= 2")
        self.batch_size = int(batch_size)
        self.half = self.batch_size // 2
        self._n_pairs = len(pairs)

        self.pos_idx = [i for i, (_, _, y) in enumerate(pairs) if int(y) == 1]
        self.neg_idx = [i for i, (_, _, y) in enumerate(pairs) if int(y) == 0]
        if len(self.pos_idx) == 0 or len(self.neg_idx) == 0:
            raise ValueError("BalancedPairBatchSampler requires both positive and negative examples")

    def __iter__(self):
        # Fixed order, no shuffling
        pi = 0
        ni = 0

        # stable number of batches per epoch (roughly one pass over dataset)
        n_batches = len(self)
        for _ in range(n_batches):
            batch: List[int] = []

            for _ in range(self.half):
                if pi >= len(self.pos_idx):
                    pi = 0
                batch.append(int(self.pos_idx[pi]))
                pi += 1

            for _ in range(self.batch_size - self.half):
                if ni >= len(self.neg_idx):
                    ni = 0
                batch.append(int(self.neg_idx[ni]))
                ni += 1

            yield batch

    def __len__(self) -> int:
        return int(math.ceil(self._n_pairs / float(self.batch_size)))


def build_dataloaders(
    cfg: Dict[str, Any],
    train_pairs: List[Pair],
    val_pairs: List[Pair],
    *,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    transform = build_transform(cfg)

    bs = int(cfg.get("optim", {}).get("batch_size", 128))

    train_ds = PairPathDataset(train_pairs, transform=transform)
    val_ds = PairPathDataset(val_pairs, transform=transform)

    # NO shuffle sampler
    sampler = BalancedPairBatchSampler(train_pairs, batch_size=bs)

    train_loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,  # already no shuffle
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader
