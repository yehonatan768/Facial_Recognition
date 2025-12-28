from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from src.data.load_data import load_image_rgb  # loads PIL RGB; transform converts to grayscale
from src.preprocess.transforms import build_transform_from_config


# Your standard pair type across the project: (img1_path, img2_path, label)
Pair = Tuple[Path, Path, int]


class PairsPathDataset(Dataset):
    """
    Dataset of image pairs given as paths.

    Each item returns:
      x1: FloatTensor (1,H,W) in [0,1]
      x2: FloatTensor (1,H,W) in [0,1]
      y:  FloatTensor scalar in {0.0, 1.0}

    Uses PaperImageTransform to do preprocessing/augmentation.
    """

    def __init__(
        self,
        pairs: List[Pair],
        transform: build_transform_from_config,
        strict_exists: bool = False,
    ):
        self.pairs = pairs
        self.transform = transform
        self.strict_exists = strict_exists

        if self.strict_exists:
            missing = 0
            for p1, p2, _ in self.pairs:
                if not p1.exists():
                    missing += 1
                if not p2.exists():
                    missing += 1
            if missing > 0:
                raise FileNotFoundError(
                    f"PairsPathDataset: strict_exists=True but found {missing} missing paths."
                )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        p1, p2, y = self.pairs[idx]

        # load PIL images
        img1: Image.Image = load_image_rgb(p1)
        img2: Image.Image = load_image_rgb(p2)

        # transform to tensors (PaperImageTransform converts to grayscale + resizes + aug if enabled)
        x1: torch.Tensor = self.transform(img1)
        x2: torch.Tensor = self.transform(img2)

        # BCE-friendly label type
        y_t = torch.tensor(float(y), dtype=torch.float32)

        return {
            "x1": x1,
            "x2": x2,
            "y": y_t,
            "path1": str(p1),
            "path2": str(p2),
        }


def build_transforms(cfg: Dict[str, Any]) -> Tuple[build_transform_from_config, build_transform_from_config]:
    """
    Returns (train_transform, eval_transform)
    using your PaperImageTransform.from_config().
    """
    train_t = build_transform_from_config.from_config(cfg, train=True)
    eval_t = build_transform_from_config.from_config(cfg, train=False)
    return train_t, eval_t


@dataclass
class LoaderBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    # Optional: include test_loader if you want to build it here too
    test_loader: Optional[DataLoader] = None


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def build_pair_loaders(
    cfg: Dict[str, Any],
    train_pairs: List[Pair],
    val_pairs: List[Pair],
    test_pairs: Optional[List[Pair]] = None,
) -> LoaderBundle:
    train_t, eval_t = build_transforms(cfg)

    num_workers = int(cfg.get("train", {}).get("num_workers", 0))
    pin_memory = bool(cfg.get("train", {}).get("pin_memory", torch.cuda.is_available()))

    train_ds = PairsPathDataset(train_pairs, transform=train_t, strict_exists=False)
    val_ds   = PairsPathDataset(val_pairs,   transform=eval_t,  strict_exists=False)

    # FULL-BATCH: one batch per epoch
    train_loader = make_loader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),  # e.g. 128
        shuffle=False,  # deterministic
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    val_loader = make_loader(
        val_ds,
        batch_size=int(cfg["train"]["batch_size"]),  # or separate val_batch_size
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = None
    if test_pairs is not None:
        test_ds = PairsPathDataset(test_pairs, transform=eval_t, strict_exists=False)
        test_loader = make_loader(
            test_ds,
            batch_size=len(test_ds),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    return LoaderBundle(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)
