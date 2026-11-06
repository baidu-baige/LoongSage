"""CPU tests for tree_batch: node tokens and R3 routing selection + mismatch statistics."""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("megatron", reason="Megatron-Core is not installed")

from coda.backends.megatron.tree_batch import gather_nodes, tree_routed_experts  # noqa: E402
from coda.backends.megatron.tree_pack import build_tree_seq, plan_sequences, row_path_nodes  # noqa: E402

from .tree_test_utils import caterpillar as _caterpillar  # noqa: E402

LAYERS, TOPK, E = 6, 8, 256


def _setup(seed, cap=10**9):
    rows, bp, resp = _caterpillar(seed, n_rows=6, spine0=300, grow=(50, 150), thorn=(30, 90))
    lens = [len(r) for r in rows]
    seqs = [build_tree_seq(a, b, lens, bp, resp) for a, b in plan_sequences(lens, bp, cap)]
    return rows, seqs


def _paths(seqs):
    off, out = 0, {}
    for ts in seqs:
        for r in range(ts.row_begin, ts.row_end):
            out[r] = [n + off for n in row_path_nodes(ts, r)]
        off += ts.num_nodes
    return out


def test_gather_nodes_tokens_reconstruct_rows():
    for cap in (10**9, 900):   # one sequence, and several (prefix recomputed per sequence)
        rows, seqs = _setup(0, cap)
        node_tok = gather_nodes(seqs, [torch.tensor(r) for r in rows])
        for r, p in _paths(seqs).items():
            assert node_tok[torch.tensor(p)].tolist() == rows[r]


def _consistent_routing(seed, seqs, rows):
    g = torch.Generator().manual_seed(seed)
    routed = [torch.stack([torch.randperm(E, generator=g)[:TOPK] for _ in range(len(r) * LAYERS)])
              .view(len(r), LAYERS, TOPK).to(torch.uint8) for r in rows]
    node = gather_nodes(seqs, routed)
    for r, p in _paths(seqs).items():       # make every row agree with the emitting row
        routed[r] = node[torch.tensor(p)].clone()
    return routed


def test_r3_consistent_routing_has_no_mismatch_and_order_is_ignored():
    rows, seqs = _setup(1)
    routed = _consistent_routing(1, seqs, rows)
    # permuting the top-k slots of shared tokens is not a routing change
    routed[3] = routed[3].flip(-1)
    node, st = tree_routed_experts(seqs, routed)
    assert node.shape == (sum(ts.num_nodes for ts in seqs), LAYERS, TOPK)
    assert st["tree/r3_shared_tokens"] > 0
    assert st["tree/r3_mismatch_tokens"] == 0 and st["tree/r3_mismatch_layer_ratio"] == 0.0


def _brute_force_stats(seqs, routed):
    """Reference: last row containing a node provides it; count others that differ."""
    paths = _paths(seqs)
    provider, uses = {}, {}
    for r, p in paths.items():
        for n in p:
            provider[n] = r
            uses[n] = uses.get(n, 0) + 1
    shared = tok = lay = 0
    for r, p in paths.items():
        for d, n in enumerate(p):
            if uses[n] < 2 or provider[n] == r:
                continue
            q = provider[n]
            dq = paths[q].index(n)
            diff = (routed[r][d].sort(-1).values != routed[q][dq].sort(-1).values).any(-1)
            shared += 1
            tok += int(diff.any())
            lay += int(diff.sum())
    return shared, tok, lay


def test_r3_mismatch_counts_exact():
    rows, seqs = _setup(2)
    routed = _consistent_routing(2, seqs, rows)
    paths = _paths(seqs)
    owner = [r for ts in seqs for r, _ in ts.node_order]
    for r in (0, 2, 4):     # change one layer of a few shared tokens in some rows
        cand = [d for d, n in enumerate(paths[r]) if owner[n] != r or r == 0][:600]
        for d in cand[::200]:
            e = routed[r][d, 1]
            routed[r][d, 1, 0] = next(x for x in range(E) if x not in e.tolist())
    node, st = tree_routed_experts(seqs, routed)
    shared, tok, lay = _brute_force_stats(seqs, routed)
    assert st["tree/r3_shared_tokens"] == shared and st["tree/r3_mismatch_tokens"] == tok > 0
    assert abs(st["tree/r3_mismatch_layer_ratio"] - lay / (shared * LAYERS)) < 1e-12
    # the kept routing is the last row's on every node
    for r, p in paths.items():
        later = [q for q in paths if q > r and set(p) & set(paths[q])]
        if not later:
            assert torch.equal(node[torch.tensor(p)], routed[r])


def test_r3_root_only_disagreement_costs_one_row():
    """Rollout pattern: rows >= 1 copy their parent's routing, the root row differs on the
    shared prefix. Keeping the last row's routing leaves only the root row mismatched."""
    rows, seqs = _setup(3)
    routed = _consistent_routing(3, seqs, rows)
    paths = _paths(seqs)
    ts = seqs[0]
    bp1 = len(set(paths[ts.row_begin]) & set(paths[ts.row_begin + 1]))
    root = routed[ts.row_begin][:bp1]
    root[:, :, 0] = ((root[:, :, 0].long() + 1) % E).to(torch.uint8)
    # keep the top-k a set of distinct experts
    clash = (root[:, :, 1:] == root[:, :, :1]).any(-1)
    root[:, :, 0][clash] = ((root[:, :, 0][clash].long() + 7) % E).to(torch.uint8)
    _, st = tree_routed_experts(seqs, routed)
    assert st["tree/r3_mismatch_tokens"] == bp1


def _reference_node_routing(seqs, routed):
    """Previous implementation: start from the emitting row, overwrite whole root paths row
    by row (later rows win)."""
    node = gather_nodes(seqs, routed)
    for r, p in _paths(seqs).items():
        node[torch.tensor(p)] = routed[r]
    return node


@pytest.mark.parametrize("cap", [10**9, 900])
def test_node_routing_matches_reference_and_stats_switch(cap):
    rows, seqs = _setup(4, cap)
    g = torch.Generator().manual_seed(4)
    routed = [torch.randint(0, E, (len(r), LAYERS, TOPK), generator=g, dtype=torch.uint8)
              for r in rows]     # rows disagree everywhere: worst case for the stats path
    ref = _reference_node_routing(seqs, routed)
    node, st = tree_routed_experts(seqs, routed)
    node_ns, st_ns = tree_routed_experts(seqs, routed, with_stats=False)
    assert torch.equal(node, ref) and torch.equal(node_ns, ref)
    assert st["tree/r3_mismatch_token_ratio"] > 0.99 and st_ns == {}


# ---------------------------------------------------------------------------------------
# chain_branch_points / build_tree_microbatch
# ---------------------------------------------------------------------------------------
from coda.backends.megatron.tree_batch import chain_branch_points  # noqa: E402


def test_chain_branch_points_rules():
    t = lambda *v: torch.tensor(v)  # noqa: E731
    toks = [t(1, 2, 3, 4), t(1, 2, 3, 9, 9), t(1, 2, 3), t(1, 2, 7, 7), t(5, 5), t(5, 5, 6)]
    groups = [0, 0, 0, 0, 1, 1]
    # row1 shares 3; row2 is a prefix of row1 -> capped to len-1=2 < 3 -> restart (0);
    # row3 shares 2 with row2 (>= 0) -> 2; row4 new group -> 0; row5 shares 2.
    assert chain_branch_points(toks, groups) == [0, 3, 0, 2, 0, 2]


def _mp_init():
    import os

    import torch.distributed as dist
    from megatron.core import parallel_state
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29536")
    torch.cuda.set_device(0)
    if not dist.is_initialized():
        dist.init_process_group("nccl", world_size=1, rank=0)
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(1, 1)


def test_build_tree_microbatch_structure():
    import pytest
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    _mp_init()
    from types import SimpleNamespace

    from coda.backends.megatron.cp_utils import prepare_packed_seq_params, prepare_routing_replay_data
    from coda.backends.megatron.tree_attn import build_attn_layout
    from coda.backends.megatron.tree_batch import build_tree_microbatch

    rows_a, _ = _setup(4)                  # one caterpillar trajectory ...
    rows_b, _ = _setup(5)                  # ... and another
    rows = rows_a + rows_b
    toks = [torch.tensor(r, device="cuda") for r in rows]
    resp = [min(40, len(r) - 1) for r in rows]
    groups = [0] * len(rows_a) + [1] * len(rows_b)
    g = torch.Generator().manual_seed(0)
    routed = [torch.randint(0, E, (len(r), LAYERS, TOPK), generator=g, dtype=torch.uint8) for r in rows]

    mb = build_tree_microbatch(toks, resp, groups, routed_rows=routed)
    psp = mb.packed_seq_params
    T = mb.tokens.numel()
    row_packed, _ = prepare_packed_seq_params(toks)
    assert mb.stats["tree/seqs"] == 2 and mb.stats["tree/nodes"] < mb.stats["tree/row_tokens"]
    assert T % 128 == 0 and T < row_packed.numel()
    assert psp.tree_pieces[-1].node_end == T and mb.position_ids.shape == (3, 1, T)
    # every row = its root path through the attention layout's K/V index (last piece of row)
    lay = build_attn_layout(psp.tree_pieces, "cpu")
    assert lay.kv_index.numel() >= T
    # loss index targets are exactly each row's response tokens
    idx = psp.tree_loss_index
    tgt = torch.cat([idx.node_targets, idx.extra_targets.to(idx.node_targets.device)])[idx.logp_src]
    want = torch.cat([toks[r][len(rows[r]) - resp[r]:] for r in range(len(rows))])
    assert torch.equal(tgt.cpu(), want.cpu())
    # node R3 pads to the same length as the node tokens
    cfg = SimpleNamespace(num_moe_experts=E, cp_partition_mode="zigzag", sequence_parallel=False)
    packed_r3 = prepare_routing_replay_data([mb.node_routing], cfg)
    assert packed_r3.shape[0] == T
