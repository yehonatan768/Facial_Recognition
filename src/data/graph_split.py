from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Set, Optional

from pathlib import Path


# pair type expected: (path1, path2, y) OR (path1, path2, y, ...)
# We only require:
#   - path1, path2 are Paths
#   - y is int/float in {0,1}
Pair = Tuple


@dataclass
class SplitResult:
    train_pairs: List[Pair]
    val_pairs: List[Pair]
    train_ids: Set[str]
    val_ids: Set[str]


def _label(p: Pair) -> int:
    return int(p[2])


def _id_from_path(p: Path) -> str:
    # identity is folder name
    return p.parent.name


def _pair_ids(pair: Pair) -> Tuple[str, str]:
    p1 = pair[0]
    p2 = pair[1]
    return _id_from_path(p1), _id_from_path(p2)


def _pos_frac(pairs: List[Pair]) -> float:
    if not pairs:
        return 0.0
    s = sum(_label(p) for p in pairs)
    return float(s) / float(len(pairs))


def _build_identity_graph(pairs: List[Pair]) -> Dict[str, Set[str]]:
    g: Dict[str, Set[str]] = {}
    for pr in pairs:
        a, b = _pair_ids(pr)
        if a not in g:
            g[a] = set()
        if b not in g:
            g[b] = set()
        # connect a<->b (even if a==b this is fine)
        g[a].add(b)
        g[b].add(a)
    return g


def _connected_components(g: Dict[str, Set[str]]) -> List[Set[str]]:
    seen: Set[str] = set()
    comps: List[Set[str]] = []

    for node in g.keys():
        if node in seen:
            continue
        stack = [node]
        comp: Set[str] = set()
        seen.add(node)
        while stack:
            x = stack.pop()
            comp.add(x)
            for nb in g.get(x, ()):
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        comps.append(comp)
    return comps


def _pairs_for_ids(pairs: List[Pair], ids: Set[str]) -> List[Pair]:
    out: List[Pair] = []
    for pr in pairs:
        a, b = _pair_ids(pr)
        if a in ids or b in ids:
            out.append(pr)
    return out


def split_by_components(
    pairs: List[Pair],
    val_ratio: float,
    target_pos_frac: float = 0.5,
    min_val_identities: int = 150,
    seed: int = 0,
    logger: Optional[object] = None,
) -> SplitResult:
    """
    Split by connected components of the identity graph (prevents identity leakage).

    IMPORTANT: This function does NOT drop/downsample any pairs.
    It only assigns whole identity components to train or val.

    It tries (best-effort) to:
      - achieve |val_pairs| ~= val_ratio * |pairs|
      - achieve pos_frac(val) ~= target_pos_frac
    """
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be between 0 and 1 (exclusive).")

    rng = random.Random(seed)

    g = _build_identity_graph(pairs)
    comps = _connected_components(g)

    # Precompute component stats
    comp_stats = []
    for comp in comps:
        comp_pairs = _pairs_for_ids(pairs, comp)
        comp_stats.append(
            {
                "ids": comp,
                "n_ids": len(comp),
                "pairs": comp_pairs,
                "n_pairs": len(comp_pairs),
                "pos_frac": _pos_frac(comp_pairs),
            }
        )

    # Shuffle for tie-breaking
    rng.shuffle(comp_stats)

    total_pairs = len(pairs)
    target_val_pairs = int(round(val_ratio * total_pairs))

    val_ids: Set[str] = set()
    val_pairs: List[Pair] = []

    # Greedy: add components that reduce objective:
    #   objective = |val_pairs - target_val_pairs| + lambda * |pos_frac(val) - target_pos_frac|
    # We use lambda scaled by target_val_pairs so it matters early.
    lam = max(1.0, 0.25 * float(target_val_pairs))

    remaining = comp_stats[:]
    while remaining:
        # stop if we already have enough identities AND close enough in size and can't improve much
        if len(val_ids) >= min_val_identities and len(val_pairs) >= target_val_pairs:
            break

        cur_n = len(val_pairs)
        cur_pf = _pos_frac(val_pairs)
        cur_obj = abs(cur_n - target_val_pairs) + lam * abs(cur_pf - target_pos_frac)

        best_i = None
        best_obj = cur_obj

        for i, st in enumerate(remaining):
            new_pairs = val_pairs + st["pairs"]
            new_n = len(new_pairs)
            new_pf = _pos_frac(new_pairs)
            obj = abs(new_n - target_val_pairs) + lam * abs(new_pf - target_pos_frac)

            # prefer improvements; if equal, prefer component that helps reach min_val_identities
            if obj < best_obj - 1e-12:
                best_obj = obj
                best_i = i

        if best_i is None:
            # no objective improvement; add something that helps reach min_val_identities first,
            # otherwise add the smallest component to avoid overshooting too much.
            if len(val_ids) < min_val_identities:
                best_i = max(range(len(remaining)), key=lambda j: remaining[j]["n_ids"])
            else:
                best_i = min(range(len(remaining)), key=lambda j: remaining[j]["n_pairs"])

        chosen = remaining.pop(best_i)
        val_ids |= set(chosen["ids"])
        val_pairs.extend(chosen["pairs"])

    # Train = all other pairs (by ids not in val_ids)
    # Note: Because pairs can reference two identities, we used "a in ids or b in ids" above.
    # For strict non-leakage, we must ensure train uses only identities outside val_ids.
    train_pairs: List[Pair] = []
    train_ids: Set[str] = set()

    for pr in pairs:
        a, b = _pair_ids(pr)
        if (a in val_ids) or (b in val_ids):
            continue
        train_pairs.append(pr)
        train_ids.add(a)
        train_ids.add(b)

    # Recompute actual val_ids from val_pairs (robustness)
    val_ids_final: Set[str] = set()
    for pr in val_pairs:
        a, b = _pair_ids(pr)
        val_ids_final.add(a)
        val_ids_final.add(b)

    if logger is None:
        def _log(msg: str) -> None:
            print(msg)
    else:
        def _log(msg: str) -> None:
            logger.info(msg)

    _log(
        f"[Split] train pairs (no-drop): n={len(train_pairs)} pos_frac={_pos_frac(train_pairs):.3f} "
        f"| val pairs (no-drop): n={len(val_pairs)} pos_frac={_pos_frac(val_pairs):.3f} "
        f"| target_val_pairs={target_val_pairs}"
    )
    _log(f"[Split] train_ids={len(train_ids)} | val_ids={len(val_ids_final)} | components={len(comps)}")

    return SplitResult(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        train_ids=train_ids,
        val_ids=val_ids_final,
    )
