"""Unit tests for the tree packer (coda/backends/megatron/tree_pack.py).

CPU-only index math: reconstruction correctness, node de-dup, monotone positions,
response-node recovery, greedy capping. Uses both synthetic caterpillar chains with known
overlap and recorded row-length / branch-point shapes in fixtures/tree_rows_step1.jsonl.
"""
from __future__ import annotations

import json
import os
import random

import pytest

pytest.importorskip("megatron", reason="Megatron-Core is not installed")

from coda.backends.megatron.tree_pack import (  # noqa: E402
    build_tree_seq,
    plan_sequences,
    row_path_nodes,
)


def reconstruct_row(ts, r, row_tokens_of):
    """Row r's tokens rebuilt by walking its root path through the tree."""
    return [row_tokens_of(*ts.node_order[n]) for n in row_path_nodes(ts, r)]


def _make_chain(seed, n_rows):
    """Build a caterpillar: spine tokens shared, each row adds a thorn. Returns rows, bp, resp."""
    rng = random.Random(seed)
    spine = [rng.randrange(1000, 9000) for _ in range(rng.randint(20, 60))]
    rows, bp, resp = [], [], []
    cur_spine = spine[: rng.randint(5, len(spine))]
    for k in range(n_rows):
        # grow the shared spine a little each row (monotone bp), then append a unique thorn
        grow = rng.randint(0, 8)
        cur_spine = cur_spine + [rng.randrange(1000, 9000) for _ in range(grow)]
        thorn = [rng.randrange(20000, 30000) for _ in range(rng.randint(3, 15))]
        row = list(cur_spine) + thorn
        rows.append(row)
        resp.append(len(thorn))  # loss region = the thorn tail
        if k == 0:
            bp.append(0)
        else:
            # bp_k = LCP(row_k, row_{k-1})
            m = min(len(rows[k]), len(rows[k - 1]))
            i = 0
            while i < m and rows[k][i] == rows[k - 1][i]:
                i += 1
            bp.append(i)
    return rows, bp, resp


def _check_trajectory(rows, bp, resp, cap):
    row_lens = [len(r) for r in rows]
    seqs = plan_sequences(row_lens, bp, cap)
    # sequences tile the rows exactly
    assert seqs[0][0] == 0 and seqs[-1][1] == len(rows)
    assert all(seqs[i][1] == seqs[i + 1][0] for i in range(len(seqs) - 1))

    total_nodes = 0
    for (a, b) in seqs:
        ts = build_tree_seq(a, b, row_lens, bp, resp)
        total_nodes += ts.num_nodes
        # node budget respected (unless a single row exceeds cap)
        assert ts.num_nodes <= cap or (b - a == 1)
        # positions are the full-row coords; within each piece they are contiguous & increasing
        for p in ts.pieces:
            segpos = ts.positions[p.node_start:p.node_end]
            assert segpos == list(range(segpos[0], segpos[0] + len(segpos)))
            assert segpos[0] == p.depth_start
        # every stored node is unique (row, offset)
        assert len(set(ts.node_order)) == ts.num_nodes
        # reconstruction: each row rebuilt from the tree equals the original row
        def tok_of(row, off):
            return rows[row][off]
        for r in range(a, b):
            assert reconstruct_row(ts, r, tok_of) == rows[r], f"row {r} mismatch"
        # response nodes recover exactly the row's loss tail tokens, in order
        for r in range(a, b):
            got = [rows[ts.node_order[i][0]][ts.node_order[i][1]] for i in ts.row_nodes[r]]
            assert got == rows[r][row_lens[r] - resp[r]:], f"row {r} response mismatch"
    return seqs, total_nodes


class TestTreePackSynthetic:
    @pytest.mark.parametrize("seed", range(12))
    def test_reconstruction_and_dedup(self, seed):
        rows, bp, resp = _make_chain(seed, n_rows=random.Random(seed).randint(1, 9))
        cap = max(len(r) for r in rows) + random.Random(seed).randint(0, 200)
        seqs, total_nodes = _check_trajectory(rows, bp, resp, cap)
        # de-dup lower bound: packed nodes <= naive per-row full-length sum
        assert total_nodes <= sum(len(r) for r in rows)

    def test_single_sequence_dedup_matches_unique(self):
        rows, bp, resp = _make_chain(0, n_rows=6)
        row_lens = [len(r) for r in rows]
        ts = build_tree_seq(0, len(rows), row_lens, bp, resp)
        unique = row_lens[0] + sum(row_lens[k] - bp[k] for k in range(1, len(rows)))
        assert ts.num_nodes == unique

    def test_cap_forces_prefix_recompute(self):
        rows, bp, resp = _make_chain(3, n_rows=6)
        row_lens = [len(r) for r in rows]
        # tiny cap => most rows become their own sequence => prefix recompute kicks in.
        seqs = plan_sequences(row_lens, bp, cap=max(row_lens))
        assert len(seqs) >= 2  # cap smaller than the natural single-sequence packing
        _check_trajectory(rows, bp, resp, cap=max(row_lens))


class TestTreePackRealShapes:
    def test_real_step1_shapes(self):
        # Recorded row lengths / branch points of multi-turn trajectories (no token data).
        path = os.path.join(os.path.dirname(__file__), "fixtures", "tree_rows_step1.jsonl")
        trajs = [json.loads(l) for l in open(path)]
        saved = naive = 0
        for t in trajs[:40]:
            lens, bp = t["len"], t["prev"]  # prev = LCP with previous row (chain bp)
            bp = [0] + [bp[i] for i in range(1, len(lens))]
            # synthesize resp as a small tail (structure test only; real resp comes from dumps)
            resp = [max(1, l // 10) for l in lens]
            cap = max(lens) * 3
            seqs = plan_sequences(lens, bp, cap)
            tot = 0
            for (a, b) in seqs:
                ts = build_tree_seq(a, b, lens, bp, resp)
                tot += ts.num_nodes
                assert len(set(ts.node_order)) == ts.num_nodes
                for p in ts.pieces:
                    sp = ts.positions[p.node_start:p.node_end]
                    assert sp == list(range(sp[0], sp[0] + len(sp)))
            saved += tot
            naive += sum(lens)
        # real data is 55-77% redundant; packed should be well under naive
        assert saved < 0.6 * naive


# ---------------------------------------------------------------------------------------
# min-cost cut planning
# ---------------------------------------------------------------------------------------
import itertools  # noqa: E402

from coda.backends.megatron.tree_pack import (  # noqa: E402
    TreeCostWeights, group_cost, group_nodes, plan_sequences, plan_sequences_min_cost,
)


def _chain(n, seed):
    """Random caterpillar: row k = spine_k + thorn; bp_k ~ spine_{k-1} (non-decreasing)."""
    import random
    rng = random.Random(seed)
    lens, bp, spine = [], [0], rng.randint(200, 800)
    for k in range(n):
        prev_spine = spine
        spine += rng.randint(50, 600)
        lens.append(spine + rng.randint(20, 400))
        if k:
            bp.append(max(bp[-1], 1, min(prev_spine - rng.randint(0, 50), lens[k - 1], lens[k] - 1)))
    return lens, bp


def _total(lens, bp, cuts, w):
    return sum(group_cost(lens, bp, a, b, w) for a, b in cuts)


def _feasible(lens, bp, cuts, cap):
    pos = 0
    for a, b in cuts:
        assert a == pos and b > a
        assert b - a == 1 or group_nodes(lens, bp, a, b) <= cap
        pos = b
    assert pos == len(lens)


def test_min_cost_matches_brute_force():
    w = TreeCostWeights()
    for seed in range(40):
        n = 2 + seed % 7
        lens, bp = _chain(n, seed)
        cap = int(max(lens) * (1.2 + 0.3 * (seed % 4)))
        best = None
        for mask in range(1 << (n - 1)):
            cuts, a = [], 0
            for k in range(1, n):
                if mask >> (k - 1) & 1:
                    cuts.append((a, k)); a = k
            cuts.append((a, n))
            if all(b - a == 1 or group_nodes(lens, bp, a, b) <= cap for a, b in cuts):
                c = _total(lens, bp, cuts, w)
                best = c if best is None else min(best, c)
        got = plan_sequences_min_cost(lens, bp, cap, w)
        _feasible(lens, bp, got, cap)
        assert abs(_total(lens, bp, got, w) - best) < 1e-6 * best


def test_min_cost_never_worse_than_greedy():
    w = TreeCostWeights()
    for seed in range(60):
        lens, bp = _chain(3 + seed % 15, 100 + seed)
        cap = int(max(lens) * 1.5)
        g, d = plan_sequences(lens, bp, cap), plan_sequences_min_cost(lens, bp, cap, w)
        _feasible(lens, bp, d, cap)
        assert _total(lens, bp, d, w) <= _total(lens, bp, g, w) + 1e-9


def test_min_cost_cuts_before_a_long_shared_prefix():
    # rows 2..4 share a long prefix with each other. Greedy packs rows 0..2 together and
    # must restart at row 3, recomputing its ~9k prefix (20150 nodes); cutting right after
    # row 0 keeps rows 1..4 in one sequence (1000 + 10350 = 11350 nodes).
    lens = [1000, 1100, 9000, 9100, 9200]
    bp = [0, 100, 150, 8900, 9000]
    cap = 11000
    w = TreeCostWeights(attn_pair=0.0, level=0.0)      # pure node count
    g = plan_sequences(lens, bp, cap)
    d = plan_sequences_min_cost(lens, bp, cap, w)
    assert sum(group_nodes(lens, bp, a, b) for a, b in d) < sum(group_nodes(lens, bp, a, b) for a, b in g)
    assert g == [(0, 3), (3, 5)] and d == [(0, 1), (1, 5)]
    assert sum(group_nodes(lens, bp, a, b) for a, b in d) == 11350


def test_planners_respect_logit_row_cap():
    lens = [1000 + 300 * k for k in range(8)]
    bp = [0] + [1000 + 300 * (k - 1) for k in range(1, 8)]
    resp = [300] * 8
    for plan in (plan_sequences, plan_sequences_min_cost):
        free = plan(lens, bp, 10**9)
        capped = plan(lens, bp, 10**9, resp_lens=resp, lm_cap=650)
        assert len(free) == 1 and len(capped) == 4
        assert all(sum(resp[a:b]) <= 650 for a, b in capped)
        assert [r for a, b in capped for r in range(a, b)] == list(range(8))
    # a single row over the cap still forms its own sequence
    assert plan_sequences([500, 600], [0, 400], 10**9, resp_lens=[900, 900], lm_cap=100) \
        == [(0, 1), (1, 2)]



def _min_cost_reference(row_lens, bp, cap, w, resp_lens=None, lm_cap=None):
    """Direct O(n^3) DP calling group_cost for every candidate group."""
    n = len(row_lens)
    best, arg = [0.0] + [float("inf")] * n, [0] * (n + 1)
    for j in range(1, n + 1):
        lm = 0
        for i in range(j - 1, -1, -1):
            if lm_cap is not None:
                lm += resp_lens[i]
            if j - i > 1 and (group_nodes(row_lens, bp, i, j) > cap
                              or (lm_cap is not None and lm > lm_cap)):
                break
            c = best[i] + group_cost(row_lens, bp, i, j, w)
            if c < best[j]:
                best[j], arg[j] = c, i
    out, j = [], n
    while j > 0:
        out.append((arg[j], j))
        j = arg[j]
    return out[::-1]


def test_min_cost_matches_reference_dp():
    import random
    for seed in range(200):
        rng = random.Random(seed)
        lens, bp = _chain(1 + seed % 40, 7000 + seed)
        w = TreeCostWeights(*rng.choice([(9.15e-6, 4200.0), (0.0, 0.0), (1e-3, 1.0)]))
        cap = int(max(lens) * rng.uniform(1.0, 4.0))
        resp = [rng.randint(1, 500) for _ in lens]
        lm_cap = rng.choice([None, 600, 2000])
        assert plan_sequences_min_cost(lens, bp, cap, w, resp, lm_cap) == \
            _min_cost_reference(lens, bp, cap, w, resp, lm_cap)
