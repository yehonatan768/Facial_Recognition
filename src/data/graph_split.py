from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple


Pair = Tuple[Path, Path, int]  # (img1_path, img2_path, label)


@dataclass(frozen=True)
class GraphSplitResult:
    train_pairs: List[Pair]
    val_pairs: List[Pair]
    train_ids: Set[str]
    val_ids: Set[str]


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


def split_by_components(
    pairs: List[Pair],
    val_ratio: float = 0.2,
    min_val_identities: int = 20,
    seed: int = 42,
) -> GraphSplitResult:
    """
    Leakage-free split:
      - compute connected components in identity graph
      - assign whole components to VAL until reaching target #identities
      - TRAIN = remaining identities
      - keep only pairs whose BOTH identities belong to the same split
    """
    import random
    rng = random.Random(seed)

    g = build_identity_graph(pairs)
    comps = connected_components(g)
    rng.shuffle(comps)

    all_ids = set(g.keys())
    target = max(int(round(len(all_ids) * val_ratio)), min_val_identities)
    target = min(target, len(all_ids))

    val_ids: Set[str] = set()
    for comp in comps:
        if len(val_ids) >= target:
            break
        val_ids |= comp

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
            # Crossing pair (rare but possible) -> drop to preserve leakage-free split
            continue

    return GraphSplitResult(
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        train_ids=train_ids,
        val_ids=val_ids,
    )
