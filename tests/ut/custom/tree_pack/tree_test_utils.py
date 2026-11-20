"""Helpers shared by the tree-training tests (plain module: no GPU / Megatron imports, so
importing it never skips or fails the importing test module)."""
from __future__ import annotations

import random

import torch


def caterpillar(seed, n_rows, spine0, grow, thorn):
    """Random caterpillar rows: row k = growing spine + its own thorn.

    Returns ``(rows, bp, resp)``: token lists, branch points (LCP with the previous row)
    and response (thorn) lengths. ``grow`` / ``thorn`` are ``(lo, hi)`` length ranges.
    """
    rng = random.Random(seed)
    rows, bp, resp = [], [], []
    spine = [rng.randrange(10**6) for _ in range(spine0)]
    for k in range(n_rows):
        spine = spine + [rng.randrange(10**6) for _ in range(rng.randint(*grow))]
        th = [rng.randrange(10**6) for _ in range(rng.randint(*thorn))]
        rows.append(spine + th)
        resp.append(len(th))
        if k == 0:
            bp.append(0)
        else:
            m, i = min(len(rows[k]), len(rows[k - 1])), 0
            while i < m and rows[k][i] == rows[k - 1][i]:
                i += 1
            bp.append(i)
    return rows, bp, resp


def rel_err(a, b):
    """Max-abs error of ``a`` relative to the max-abs of ``b``."""
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-12)).item()


def cu_seqlens(lens, dev):
    """Cumulative sequence lengths ``[0, l0, l0+l1, ...]`` as a long tensor on ``dev``."""
    t = torch.zeros(len(lens) + 1, dtype=torch.long, device=dev)
    t[1:] = torch.tensor(lens, device=dev).cumsum(0)
    return t
