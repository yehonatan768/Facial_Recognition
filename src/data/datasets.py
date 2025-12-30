from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

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
    """Batch sampler that yields half positive and half negative indices.

    Deterministic-but-changing shuffle across epochs:
      - Call set_epoch(e) once per epoch (train loop)
      - Seed becomes seed + epoch
    """

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
        self.epoch = 0
        self._n_pairs = len(pairs)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        pos = torch.tensor(self.pos_idx)
        neg = torch.tensor(self.neg_idx)

        pos_perm = pos[torch.randperm(len(pos), generator=g)].tolist()
        neg_perm = neg[torch.randperm(len(neg), generator=g)].tolist()

        pi = 0
        ni = 0

        # Yield a fixed number of batches per epoch for stable training logs
        n_batches = len(self)
        for _ in range(n_batches):
            batch: List[int] = []

            for _ in range(self.half):
                if pi >= len(pos_perm):
                    pi = 0
                batch.append(int(pos_perm[pi]))
                pi += 1

            for _ in range(self.batch_size - self.half):
                if ni >= len(neg_perm):
                    ni = 0
                batch.append(int(neg_perm[ni]))
                ni += 1

            yield batch

    def __len__(self) -> int:
        # Stable batches per epoch: cover dataset roughly once
        return int(math.ceil(self._n_pairs / float(self.batch_size)))


def build_dataloaders(
    cfg: Dict[str, Any],
    train_pairs: List[Pair],
    val_pairs: List[Pair],
) -> Tuple[DataLoader, DataLoader]:
    transform = build_transform(cfg)

    bs = int(cfg.get("optim", {}).get("batch_size", 128))
    seed = int(cfg.get("train", {}).get("seed", 42))

    # DataLoader performance knobs (override from cfg if present)
    dcfg = cfg.get("data", {}) or {}
    num_workers = int(dcfg.get("num_workers", 2))
    pin_memory = bool(dcfg.get("pin_memory", True))
    persistent_workers = bool(dcfg.get("persistent_workers", num_workers > 0))
    prefetch_factor = int(dcfg.get("prefetch_factor", 2))  # only used when num_workers>0

    train_ds = PairPathDataset(train_pairs, transform=transform)
    val_ds = PairPathDataset(val_pairs, transform=transform)

    sampler = BalancedPairBatchSampler(train_pairs, batch_size=bs, seed=seed)

    train_loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    return train_loader, val_loader
