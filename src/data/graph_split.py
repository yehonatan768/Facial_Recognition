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

    Strategy:
      1) Build identity graph over ALL pairs; connected components guarantee
         that no pair crosses components -> splitting components prevents leakage.
      2) Greedily choose whole components for VAL to match:
           - target number of val pairs
           - target number of positive val pairs
           - minimum number of val identities
      3) After split, if class balance is still poor, minimally downsample the
         majority class WITHIN each split (does not move identities across splits,
         so leakage is still prevented).

    Keeps as much data as possible:
      - Primary split does not drop pairs.
      - Post-balance only drops if necessary and only the minimum required.
    """
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be between 0 and 1 (exclusive).")

    rng = random.Random(seed)

    def _log(msg: str) -> None:
        if logger is None:
            print(msg)
        else:
            logger.info(msg)

    g = _build_identity_graph(pairs)
    comps = _connected_components(g)

    total_pairs = len(pairs)
    target_val_pairs = int(round(val_ratio * total_pairs))
    target_val_pairs = max(1, min(target_val_pairs, total_pairs - 1))

    target_val_pos = int(round(target_pos_frac * target_val_pairs))

    comp_stats = []
    for comp in comps:
        comp_pairs = _pairs_for_ids(pairs, comp)
        n_pairs = len(comp_pairs)
        n_pos = sum(_label(p) for p in comp_pairs)
        comp_stats.append(
            {
                "ids": comp,
                "n_ids": len(comp),
                "pairs": comp_pairs,
                "n_pairs": n_pairs,
                "n_pos": n_pos,
            }
        )

    rng.shuffle(comp_stats)


    wN = 1.0
    wP = 1.0
    wI = float(target_val_pairs) * 0.10

    val_ids: Set[str] = set()
    val_pairs: List[Pair] = []
    val_n = 0
    val_pos = 0

    remaining = comp_stats[:]

    def obj(n_val: int, pos_val: int, n_ids: int) -> float:
        id_pen = 0.0
        if n_ids < min_val_identities:
            id_pen = float(min_val_identities - n_ids)
        return wN * abs(n_val - target_val_pairs) + wP * abs(pos_val - target_val_pos) + wI * id_pen

    max_iters = len(remaining)
    for _ in range(max_iters):
        if not remaining:
            break

        if len(val_ids) >= min_val_identities and val_n >= target_val_pairs:
            break

        cur_obj = obj(val_n, val_pos, len(val_ids))

        best_i = None
        best_obj = cur_obj

        for i, st in enumerate(remaining):
            new_n = val_n + st["n_pairs"]
            new_pos = val_pos + st["n_pos"]
            new_ids = len(val_ids | set(st["ids"]))
            new_obj = obj(new_n, new_pos, new_ids)

            if new_obj < best_obj - 1e-12:
                best_obj = new_obj
                best_i = i

        if best_i is None:
            if len(val_ids) < min_val_identities:
                best_i = max(range(len(remaining)), key=lambda j: remaining[j]["n_ids"])
            else:
                best_i = min(range(len(remaining)), key=lambda j: remaining[j]["n_pairs"])

        chosen = remaining.pop(best_i)
        val_ids |= set(chosen["ids"])
        val_pairs.extend(chosen["pairs"])
        val_n += chosen["n_pairs"]
        val_pos += chosen["n_pos"]

    train_pairs: List[Pair] = []
    train_ids: Set[str] = set()
    for pr in pairs:
        a, b = _pair_ids(pr)
        if (a in val_ids) or (b in val_ids):
            continue
        train_pairs.append(pr)
        train_ids.add(a)
        train_ids.add(b)

    val_ids_final: Set[str] = set()
    for pr in val_pairs:
        a, b = _pair_ids(pr)
        val_ids_final.add(a)
        val_ids_final.add(b)

    # Hard leakage safety check
    if (train_ids & val_ids_final):
        overlap = sorted(list(train_ids & val_ids_final))[:10]
        raise RuntimeError(f"Identity leakage detected (train_ids ∩ val_ids != ∅). Example overlap: {overlap}")


    def _rebalance_min_drop(pairs_in: List[Pair], target_pf: float) -> List[Pair]:
        """
        Make pos_frac closer to target by dropping the minimum number of samples
        from the majority class. Never moves identities across splits.
        """
        if not pairs_in:
            return pairs_in

        pos = [p for p in pairs_in if _label(p) == 1]
        neg = [p for p in pairs_in if _label(p) == 0]

        if not pos or not neg:
            return pairs_in


        p = len(pos)
        n = len(neg)


        cur_pf = float(p) / float(p + n)
        if abs(cur_pf - target_pf) < 1e-6:
            return pairs_in

        rng.shuffle(pos)
        rng.shuffle(neg)

        if cur_pf < target_pf:
            k = int(round(p * (1.0 - target_pf) / max(1e-12, target_pf)))
            k = max(1, min(k, n))
            return pos + neg[:k]
        else:
            k = int(round(target_pf * n / max(1e-12, (1.0 - target_pf))))
            k = max(1, min(k, p))
            return pos[:k] + neg


    val_pf_before = _pos_frac(val_pairs)
    if abs(val_pf_before - target_pos_frac) >= 0.05:
        val_pairs_bal = _rebalance_min_drop(val_pairs, target_pos_frac)
        if len(val_pairs_bal) >= int(0.50 * len(val_pairs)):
            val_pairs = val_pairs_bal

    train_pf_before = _pos_frac(train_pairs)
    if abs(train_pf_before - target_pos_frac) >= 0.10:
        train_pairs_bal = _rebalance_min_drop(train_pairs, target_pos_frac)
        if len(train_pairs_bal) >= int(0.70 * len(train_pairs)):
            train_pairs = train_pairs_bal
            train_ids = set()
            for pr in train_pairs:
                a, b = _pair_ids(pr)
                train_ids.add(a)
                train_ids.add(b)

    _log(
        f"[Split] train pairs: n={len(train_pairs)} pos_frac={_pos_frac(train_pairs):.3f} "
        f"| val pairs: n={len(val_pairs)} pos_frac={_pos_frac(val_pairs):.3f} "
        f"| target_val_pairs={target_val_pairs} target_pos_frac={target_pos_frac:.2f}"
    )
    _log(f"[Split] train_ids={len(train_ids)} | val_ids={len(val_ids_final)} | components={len(comps)}")

    return SplitResult(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        train_ids=train_ids,
        val_ids=val_ids_final,
    )
