"""Cost model of tree sequences, in node-equivalents.

Used to cut trajectories into trees and to balance work across DP ranks. The weights are
empirical (tune per model and hardware).
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

TREE_ATTN_PAIR_COST = 9.15e-6   # one (query, key) pair of causal full attention
TREE_LEVEL_COST = 4200.0        # one serial GDN level (a round of varlen kernel calls)


def _lcp(a: np.ndarray, b: np.ndarray) -> int:
    m = min(len(a), len(b))
    diff = np.flatnonzero(a[:m] != b[:m])
    return int(diff[0]) if diff.size else m


def chain_cost(rows: Sequence[np.ndarray]) -> float:
    """Estimated cost of a trajectory's trainable rows packed as one tree: nodes +
    attention pairs + GDN levels (branch points as in ``index.batch.chain_branch_points``)."""
    nodes = pairs = 0.0
    levels = 1
    prev, prev_bp = None, 0
    for row in rows:
        n = len(row)
        if n == 0:
            continue
        bp = 0 if prev is None else min(_lcp(prev, row), n - 1)
        if bp == 0 or bp < prev_bp:
            bp = 0                     # chain restart
        nodes += n - bp
        pairs += (n * (n + 1) - bp * (bp + 1)) / 2
        levels += 1
        prev, prev_bp = row, bp
    return nodes + TREE_ATTN_PAIR_COST * pairs + TREE_LEVEL_COST * levels
