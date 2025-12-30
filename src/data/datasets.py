from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from PIL import Image

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
    """Batch sampler that yields half positive and half negative indices."""

    def __init__(self, pairs: List[Pair], batch_size: int, seed: int = 42):
        if batch_size < 2:
            raise ValueError("batch_size must be >= 2")
        self.batch_size = int(batch_size)
        self.half = self.batch_size // 2

        self.pos_idx = [i for i, (_, _, y) in enumerate(pairs) if int(y) == 1]
        self.neg_idx = [i for i, (_, _, y) in enumerate(pairs) if int(y) == 0]
        if len(self.pos_idx) == 0 or len(self.neg_idx) == 0:
            raise ValueError("BalancedPairBatchSampler requires both positive and negative examples")

        self.seed = int(seed)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed)

        pos = torch.tensor(self.pos_idx)
        neg = torch.tensor(self.neg_idx)

        pos_perm = pos[torch.randperm(len(pos), generator=g)].tolist()
        neg_perm = neg[torch.randperm(len(neg), generator=g)].tolist()

        # Cycle if one class runs out
        pi = 0
        ni = 0
        while pi < len(pos_perm) or ni < len(neg_perm):
            batch = []
            for _ in range(self.half):
                if pi >= len(pos_perm):
                    pi = 0
                batch.append(int(pos_perm[pi])); pi += 1
            for _ in range(self.batch_size - self.half):
                if ni >= len(neg_perm):
                    ni = 0
                batch.append(int(neg_perm[ni])); ni += 1
            yield batch

    def __len__(self) -> int:
        # approximate: number of batches to cover the larger class once
        return max(len(self.pos_idx), len(self.neg_idx)) // self.half


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
    seed = int(cfg.get("train", {}).get("seed", 42))

    train_ds = PairPathDataset(train_pairs, transform=transform)
    val_ds = PairPathDataset(val_pairs, transform=transform)

    sampler = BalancedPairBatchSampler(train_pairs, batch_size=bs, seed=seed)

    train_loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader
