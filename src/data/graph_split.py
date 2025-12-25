from __future__ import annotations

import random
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple


Pair = Tuple[Path, Path, int]  # (img1_path, img2_path, label)

def _ratio(pairs):
    if not pairs:
        return 0.0
    pos = sum(1 for _,_,y in pairs if y == 1)
    return pos / len(pairs)


def rebalance_pairs_to_ratio(
    pairs: List[Pair],
    target_pos_frac: float = 0.5,
    seed: int = 42,
) -> List[Pair]:
    """
    Downsample within a split to make positives/negatives closer to target_pos_frac.

    Keeps all pairs inside the split (no leakage risk), only removes excess pairs.
    If exact target is impossible, gets as close as possible without oversampling.
    """
    rng = random.Random(seed)

    pos = [p for p in pairs if p[2] == 1]
    neg = [p for p in pairs if p[2] == 0]

    if not pos or not neg:
        # Can't rebalance if one class is missing
        return pairs

    # We want: pos / (pos+neg) ~= target_pos_frac
    # We will downsample the majority class.
    n_pos = len(pos)
    n_neg = len(neg)

    # Desired counts given the limiting class
    # If pos is limiting:
    #   keep_pos = n_pos
    #   keep_neg = n_pos * (1-target)/target
    # If neg is limiting:
    #   keep_neg = n_neg
    #   keep_pos = n_neg * target/(1-target)
    if target_pos_frac <= 0.0:
        return neg
    if target_pos_frac >= 1.0:
        return pos

    # Compute maximum feasible counts without oversampling
    max_neg_given_pos = int(round(n_pos * (1.0 - target_pos_frac) / target_pos_frac))
    max_pos_given_neg = int(round(n_neg * target_pos_frac / (1.0 - target_pos_frac)))

    if max_neg_given_pos <= n_neg:
        # positives are the limiting side (or equal)
        keep_pos = n_pos
        keep_neg = max_neg_given_pos
    else:
        # negatives are limiting
        keep_neg = n_neg
        keep_pos = max_pos_given_neg

    keep_pos = max(1, min(keep_pos, n_pos))
    keep_neg = max(1, min(keep_neg, n_neg))

    rng.shuffle(pos)
    rng.shuffle(neg)

    balanced = pos[:keep_pos] + neg[:keep_neg]
    rng.shuffle(balanced)
    return balanced


@dataclass(frozen=True)
class GraphSplitResult:
    train_pairs: List[Pair]
    val_pairs: List[Pair]
    train_ids: Set[str]
    val_ids: Set[str]
    # Optional debug stats
    num_components: int
    target_val_identities: int
    achieved_val_identities: int


def _identity(p: Path) -> str:
    # images_root/<identity>/<identity>_0001.jpg
    return p.parent.name


def build_identity_graph(pairs: List[Pair]) -> Dict[str, Set[str]]:
    """
    Nodes: identities.
    Edges: negative pairs connect the two identities.
    """
    g: Dict[str, Set[str]] = defaultdict(set)
    for p1, p2, y in pairs:
        a = _identity(p1)
        b = _identity(p2)
        _ = g[a]
        _ = g[b]
        if y == 0 and a != b:
            g[a].add(b)
            g[b].add(a)
    return g


def connected_components(g: Dict[str, Set[str]]) -> List[Set[str]]:
    seen: Set[str] = set()
    comps: List[Set[str]] = []
    for start in g.keys():
        if start in seen:
            continue
        q = deque([start])
        seen.add(start)
        comp = {start}
        while q:
            u = q.popleft()
            for v in g[u]:
                if v not in seen:
                    seen.add(v)
                    comp.add(v)
                    q.append(v)
        comps.append(comp)
    return comps


def _choose_val_components_close_to_target(
    comps: List[Set[str]],
    target: int,
    seed: int = 42,
) -> Set[int]:
    """
    Choose a subset of components whose total size is as close as possible to `target`,
    without splitting any component (leakage-safe).

    Heuristic:
      1) Greedy "best-fit" by descending size.
      2) Local improvement by attempting single swaps (replace one selected comp with one unselected comp)
         to reduce absolute error to target.
    """
    import random
    rng = random.Random(seed)

    sizes = [len(c) for c in comps]
    idxs = list(range(len(comps)))

    # Shuffle first to break ties deterministically by seed, then sort by size descending
    rng.shuffle(idxs)
    idxs.sort(key=lambda i: sizes[i], reverse=True)

    selected: Set[int] = set()
    cur = 0

    # --- Greedy best-fit ---
    # For each component, decide if including it improves closeness to target.
    for i in idxs:
        s = sizes[i]
        # Always allow adding if we are below target and it moves us closer
        if abs((cur + s) - target) < abs(cur - target):
            selected.add(i)
            cur += s

    # If greedy picks nothing (rare), pick the single closest component
    if not selected and comps:
        best_i = min(range(len(comps)), key=lambda i: abs(sizes[i] - target))
        selected.add(best_i)
        cur = sizes[best_i]

    # --- Local improvement (single swap search) ---
    # Try to reduce abs(cur-target) by swapping one-in/one-out.
    improved = True
    while improved:
        improved = False
        best_err = abs(cur - target)

        selected_list = list(selected)
        unselected_list = [i for i in range(len(comps)) if i not in selected]

        # Limit search for speed if huge number of components
        # (still typically fine for LFW-sized data)
        max_checks = 20000
        checks = 0

        for i_out in selected_list:
            for i_in in unselected_list:
                checks += 1
                if checks > max_checks:
                    break

                new_cur = cur - sizes[i_out] + sizes[i_in]
                new_err = abs(new_cur - target)
                if new_err < best_err:
                    # Perform the improving swap
                    selected.remove(i_out)
                    selected.add(i_in)
                    cur = new_cur
                    improved = True
                    best_err = new_err
                    break
            if checks > max_checks or improved:
                break

    return selected


def split_by_components(
    pairs: List[Pair],
    val_ratio: float = 0.4,
    min_val_identities: int = 80,
    seed: int = 42,
) -> GraphSplitResult:
    """
    Leakage-free split:
      - compute connected components in identity graph
      - choose whole components for VAL to get as close as possible to target #identities
      - TRAIN = remaining identities
      - keep only pairs whose BOTH identities belong to the same split
    """
    g = build_identity_graph(pairs)
    comps = connected_components(g)

    all_ids = set(g.keys())
    target = max(int(round(len(all_ids) * val_ratio)), min_val_identities)
    target = min(target, len(all_ids))

    # Choose subset of components that best matches target
    selected_comp_idxs = _choose_val_components_close_to_target(comps=comps, target=target, seed=seed)

    val_ids: Set[str] = set()
    for i in selected_comp_idxs:
        val_ids |= comps[i]

    train_ids = all_ids - val_ids

    train_pairs: List[Pair] = []
    val_pairs: List[Pair] = []

    for p1, p2, y in pairs:
        a = _identity(p1)
        b = _identity(p2)

        if a in train_ids and b in train_ids:
            train_pairs.append((p1, p2, y))
        elif a in val_ids and b in val_ids:
            val_pairs.append((p1, p2, y))
        else:
            # Crossing pair -> drop to preserve leakage-free split
            continue


    # train_pairs = rebalance_pairs_to_ratio(train_pairs, target_pos_frac=0.5, seed=seed)
    # val_pairs = rebalance_pairs_to_ratio(val_pairs, target_pos_frac=0.5, seed=seed + 1)

    print(f"[Split] train pairs after balance: n={len(train_pairs)} pos_frac={_ratio(train_pairs):.3f}")
    print(f"[Split] val   pairs after balance: n={len(val_pairs)} pos_frac={_ratio(val_pairs):.3f}")


    return GraphSplitResult(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        train_ids=train_ids,
        val_ids=val_ids,
        num_components=len(comps),
        target_val_identities=target,
        achieved_val_identities=len(val_ids),
    )
