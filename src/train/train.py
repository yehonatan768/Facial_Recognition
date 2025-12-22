from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from src.utils.read_model_config import read_model_config

# --- Graph-based leakage-free split (pairsDevTrain ONLY) ---
from src.data.load_pairs import parse_pairs_file, split_pairs
from src.data.graph_split import split_by_components

# --- Evaluation (paper-like) ---
from src.train.evaluate import (
    evaluate_verification,
    build_identity_index,
    sample_one_shot_trials,
    evaluate_one_shot,
)

from src.data.transforms import PaperImageTransform
from src.models.cnn_siamese import SiameseCNN, init_weights_like_paper
from src.models.embeder import WeightedL1Embedder, SigmoidDecider, FullSiameseModel


class PairsPathDataset(Dataset):
    """
    Dataset built from lists:
      X: List[(path1, path2)]
      y: List[int] in {0,1}
    Uses PaperImageTransform to do ALL preprocessing/augmentation.
    """

    def __init__(self, X: List[Tuple[Path, Path]], y: List[int], transform: PaperImageTransform) -> None:
        self.X = X
        self.y = y
        self.transform = transform

        if len(self.X) != len(self.y):
            raise ValueError("X and y must have the same length.")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int):
        p1, p2 = self.X[i]
        label = float(self.y[i])

        img1 = Image.open(p1)
        img2 = Image.open(p2)

        x1 = self.transform(img1)
        x2 = self.transform(img2)

        return x1, x2, torch.tensor([label], dtype=torch.float32)


def momentum_at_epoch(m0: float, m_final: float, epoch: int, max_epochs: int) -> float:
    if max_epochs <= 1:
        return m_final
    t = epoch / float(max_epochs - 1)
    return (1.0 - t) * m0 + t * m_final


def build_param_groups(full_model: FullSiameseModel, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    per_layer = cfg["train"]["per_layer"]

    conv_modules = [m for m in full_model.siamese_cnn.conv.modules() if isinstance(m, nn.Conv2d)]
    if len(conv_modules) != 4:
        raise RuntimeError(f"Expected 4 conv layers (Figure 4). Found {len(conv_modules)}")

    groups: List[Dict[str, Any]] = []

    # conv1..conv4
    for i, conv in enumerate(conv_modules, start=1):
        name = f"conv{i}"
        g = per_layer[name]
        groups.append(
            {
                "params": list(conv.parameters()),
                "lr": float(g["lr"]),
                "momentum_final": float(g["momentum_final"]),
                "weight_decay": float(g["l2"]),
                "group_name": name,
            }
        )

    # fc
    g_fc = per_layer["fc"]
    groups.append(
        {
            "params": list(full_model.siamese_cnn.fc.parameters()),
            "lr": float(g_fc["lr"]),
            "momentum_final": float(g_fc["momentum_final"]),
            "weight_decay": float(g_fc["l2"]),
            "group_name": "fc",
        }
    )

    # alpha
    g_a = per_layer["alpha"]
    groups.append(
        {
            "params": [full_model.embedder.alpha],
            "lr": float(g_a["lr"]),
            "momentum_final": float(g_a["momentum_final"]),
            "weight_decay": float(g_a["l2"]),
            "group_name": "alpha",
        }
    )

    return groups


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    n = 0

    for x1, x2, y in loader:
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device)  # (B,1)

        opt.zero_grad(set_to_none=True)
        p = model(x1, x2)  # (B,1) sigmoid
        loss = loss_fn(p, y)
        loss.backward()
        opt.step()

        total += float(loss.item()) * x1.size(0)
        n += x1.size(0)

    return total / max(n, 1)


@torch.no_grad()
def eval_loss(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> float:
    model.eval()
    total = 0.0
    n = 0
    for x1, x2, y in loader:
        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device)
        p = model(x1, x2)
        loss = loss_fn(p, y)
        total += float(loss.item()) * x1.size(0)
        n += x1.size(0)
    return total / max(n, 1)


def main(config_path: str = "src/config/config.yaml") -> None:
    cfg = read_model_config(config_path)

    # seed
    seed = int(cfg["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device(cfg["device"])

    # ------------------------------------------------------------------
    # Leakage-free train/val split FROM pairsDevTrain ONLY (graph components)
    # ------------------------------------------------------------------
    pairs_train_path = Path(cfg["paths"]["pairs_train"])
    images_root = Path(cfg["paths"]["images_root"])
    ext = cfg["paths"].get("ext", ".jpg")
    strict_exists = bool(cfg["paths"].get("strict_exists", False))

    # Parse all pairs from pairsDevTrain (LFW-style)
    pairs_pool = parse_pairs_file(
        pairs_txt=pairs_train_path,
        images_root=images_root,
        ext=ext,
        logger=None,
        strict_exists=strict_exists,
    )

    split_cfg = cfg.get("split", {})
    val_ratio = float(split_cfg.get("val_ratio", 0.2))
    min_val_ids = int(split_cfg.get("min_val_identities", 20))

    split_res = split_by_components(
        pairs=pairs_pool,
        val_ratio=val_ratio,
        min_val_identities=min_val_ids,
        seed=seed,
    )

    X_train, y_train = split_pairs(split_res.train_pairs)
    X_val, y_val = split_pairs(split_res.val_pairs)

    print(f"[Split] identities: train={len(split_res.train_ids)} val={len(split_res.val_ids)}")
    print(f"[Split] pairs:      train={len(X_train)} val={len(X_val)}")

    # ---------- Paper transforms ----------
    train_tf = PaperImageTransform.from_config(cfg, train=True)
    val_tf = PaperImageTransform.from_config(cfg, train=False)

    train_ds = PairsPathDataset(X_train, y_train, train_tf)
    val_ds = PairsPathDataset(X_val, y_val, val_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=2,
        drop_last=False,
        pin_memory=True,
    )

    # model parts
    siamese_cnn = SiameseCNN(cfg)
    init_weights_like_paper(siamese_cnn, cfg)

    embedder = WeightedL1Embedder(
        embedding_dim=int(cfg["model"]["cnn"]["fc_units"]),
        alpha_init=float(cfg["model"]["embedder"]["alpha_init"]),
    )
    decider = SigmoidDecider()
    full = FullSiameseModel(siamese_cnn=siamese_cnn, embedder=embedder, decider=decider).to(device)

    # loss (paper: sigmoid + cross-entropy) => BCELoss
    loss_fn = nn.BCELoss()

    # optimizer schedule
    schedule = cfg["train"]["schedule"]
    m_start = float(schedule["momentum_start"])
    gamma = float(schedule["lr_decay_gamma"])

    param_groups = build_param_groups(full, cfg)
    for g in param_groups:
        g["momentum"] = m_start  # start at 0.5 for all groups

    opt = torch.optim.SGD(param_groups)

    max_epochs = int(cfg["train"]["max_epochs"])
    patience = int(cfg["train"]["early_stop_patience"])

    # -------------------------
    # Evaluation setup (paper-like)
    # -------------------------
    eval_cfg = cfg.get("eval", {})
    verif_cfg = eval_cfg.get("verification", {})
    one_cfg = eval_cfg.get("one_shot", {})

    threshold_steps = int(verif_cfg.get("threshold_steps", 400))

    # Optional one-shot episodic validation (closest analogue to paper protocol)
    use_one_shot = bool(one_cfg.get("enabled", False))
    oneshot_trials = None

    if use_one_shot:
        # IMPORTANT: to avoid leakage, build index ONLY from validation identities
        # selected by the graph split. This ensures one-shot validation is on unseen identities.
        # We build the index by scanning folders, then filter by val IDs.
        full_index = build_identity_index(images_root)
        val_index = {k: v for k, v in full_index.items() if k in split_res.val_ids}

        oneshot_trials = sample_one_shot_trials(
            index=val_index,
            num_way=int(one_cfg.get("num_way", 20)),
            num_trials=int(one_cfg.get("num_trials", 320)),
            seed=int(one_cfg.get("seed", seed)),
        )

    # Best metric (paper early-stops on one-shot validation error if enabled)
    best_metric = float("inf")
    bad = 0

    for epoch in range(max_epochs):
        # update momentum for each group linearly up to momentum_final
        for g in opt.param_groups:
            m_final = float(g["momentum_final"])
            g["momentum"] = momentum_at_epoch(m_start, m_final, epoch, max_epochs)

        train_loss = train_one_epoch(full, train_loader, opt, loss_fn, device)
        val_loss = eval_loss(full, val_loader, loss_fn, device)

        # ---- evaluation (per epoch) ----
        verif_metrics = evaluate_verification(
            model=full,
            loader=val_loader,
            device=device,
            threshold_steps=threshold_steps,
        )

        oneshot_metrics = {}
        if oneshot_trials is not None:
            oneshot_metrics = evaluate_one_shot(
                model=full,
                device=device,
                transform=val_tf,  # IMPORTANT: no augmentation at eval
                trials=oneshot_trials,
            )

        # LR decay 0.99 per epoch (uniformly)
        for g in opt.param_groups:
            g["lr"] = float(g["lr"]) * gamma

        # ---- print like paper-style monitoring ----
        msg = (
            f"Epoch {epoch+1:03d}/{max_epochs} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
            f"verif_acc(best_thr)={verif_metrics['val_verif_acc_best_thr']:.4f} "
            f"thr={verif_metrics['val_verif_thr_best']:.3f} | "
        )

        if oneshot_metrics:
            msg += (
                f"oneshot_acc={oneshot_metrics['val_oneshot_acc']:.4f} "
                f"oneshot_err={oneshot_metrics['val_oneshot_err']:.4f} | "
            )

        msg += f"lr0={opt.param_groups[0]['lr']:.6g} | m0={opt.param_groups[0]['momentum']:.3f}"
        print(msg)

        # ---- early stopping metric ----
        # Paper: stop based on one-shot validation error (if enabled)
        if oneshot_metrics:
            current_metric = float(oneshot_metrics["val_oneshot_err"])  # minimize
        else:
            current_metric = float(val_loss)  # minimize

        if current_metric < best_metric - 1e-6:
            best_metric = current_metric
            bad = 0
            torch.save(full.state_dict(), "siamese_paper_best.pt")
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stop: no improvement for {patience} epochs.")
                break

    print(f"Best metric: {best_metric:.6f}")
    print("Saved checkpoint: siamese_paper_best.pt")


if __name__ == "__main__":
    main()
