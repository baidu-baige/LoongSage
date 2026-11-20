"""GPU test: tree GDN conv + core vs. row-wise, with the exact fla kernels Megatron uses.

Builds a caterpillar with tree_pack, assigns random per-NODE inputs, then:
  * row-wise: each row gathers its root-path node inputs (prefix duplicated) and runs the
    kernel as its own varlen sequence (row-wise training);
  * tree:     tree_causal_conv1d / tree_gated_delta_rule on the de-duplicated node buffer.
Loss is a fixed weighting of each row's RESPONSE outputs. Outputs and node-input gradients
(row-wise grads scatter-added back to nodes) must agree to bf16 noise, measured against an
fp32 row-wise truth.
"""
from __future__ import annotations


import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

pytest.importorskip("megatron", reason="Megatron-Core is not installed")
fla_conv = pytest.importorskip("fla.modules.convolution").causal_conv1d
fla_gdr = pytest.importorskip("fla.ops.gated_delta_rule").chunk_gated_delta_rule

from coda.backends.megatron.tree_gdn import tree_causal_conv1d, tree_gated_delta_rule  # noqa: E402
from coda.backends.megatron.tree_pack import build_tree_seq, row_path_nodes  # noqa: E402

from .tree_test_utils import caterpillar as _caterpillar  # noqa: E402
from .tree_test_utils import cu_seqlens as _cu  # noqa: E402
from .tree_test_utils import rel_err as _rel  # noqa: E402




def _setup(seed=0):
    rows, bp, resp = _caterpillar(seed, n_rows=5, spine0=1500, grow=(300, 900), thorn=(200, 700))
    lens = [len(r) for r in rows]
    ts = build_tree_seq(0, len(rows), lens, bp, resp)
    paths = [row_path_nodes(ts, r) for r in range(len(rows))]
    resp_nodes = [ts.row_nodes[r] for r in range(len(rows))]
    return ts, lens, paths, resp_nodes, resp


def _rowwise_index(paths):
    """Concatenated root-path node indices and per-row lengths for the row-wise layout."""
    return [i for p in paths for i in p], [len(p) for p in paths]



def test_conv_tree_matches_rowwise():
    dev = "cuda"
    ts, lens, paths, resp_nodes, resp = _setup(1)
    T, D, W = ts.num_nodes, 512, 4
    torch.manual_seed(0)
    xn = torch.randn(1, T, D, device=dev, dtype=torch.bfloat16)
    wt = torch.randn(D, W, device=dev, dtype=torch.bfloat16)
    flat, rlens = _rowwise_index(paths)
    fidx = torch.tensor(flat, device=dev)
    ref = fla_conv(x=xn.index_select(1, fidx).contiguous(), weight=wt, bias=None,
                   activation="silu", cu_seqlens=_cu(rlens, dev))[0]
    tree = tree_causal_conv1d(xn, wt, None, "silu", ts.pieces, fla_conv)
    # every row-wise output position == tree output at the same node (conv is exact)
    assert torch.equal(ref[0], tree[0].index_select(0, fidx))


@pytest.mark.parametrize("level_tokens", [0.0, 2000.0, 1e9])   # one level per depth .. fewest calls
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gdn_core_tree_matches_rowwise(seed, level_tokens):
    dev = "cuda"
    ts, lens, paths, resp_nodes, resp = _setup(seed)
    T, H, K, V = ts.num_nodes, 8, 128, 128
    torch.manual_seed(seed)
    base = dict(
        q=torch.randn(1, T, H, K), k=torch.nn.functional.normalize(torch.randn(1, T, H, K), dim=-1),
        v=torch.randn(1, T, H, V), g=-torch.rand(1, T, H) * 0.1,
        beta=torch.rand(1, T, H).sigmoid(),
    )
    flat, rlens = _rowwise_index(paths)
    fidx = torch.tensor(flat, device=dev)
    # loss weights on each row's response nodes only
    w = torch.randn(1, T, H, V, device=dev)
    rw_mask = torch.zeros(1, sum(rlens), 1, 1, device=dev)
    off = 0
    for r, p in enumerate(paths):
        L = len(p)
        rw_mask[:, off + L - resp[r]:off + L] = 1.0
        off += L
    tree_mask = torch.zeros(1, T, 1, 1, device=dev)
    for r in range(len(paths)):
        tree_mask[:, resp_nodes[r]] = 1.0

    def run(dtype, tree):
        leaf = {n: t.to(dev, dtype).detach().requires_grad_(True) for n, t in base.items()}
        if tree:
            o = tree_gated_delta_rule(leaf["q"], leaf["k"], leaf["v"], leaf["g"], leaf["beta"],
                                      ts.pieces, fla_gdr, level_tokens=level_tokens)
            loss = (o.float() * w * tree_mask).sum()
            per_node_out = o
        else:
            sel = [leaf[n].index_select(1, fidx) for n in ("q", "k", "v", "g", "beta")]
            o, _ = fla_gdr(*sel, output_final_state=False, cu_seqlens=_cu(rlens, dev))
            loss = (o.float() * w.index_select(1, fidx) * rw_mask).sum()
            per_node_out = o
        loss.backward()
        return per_node_out.detach(), {n: t.grad for n, t in leaf.items()}

    o32, g32 = run(torch.float32, tree=False)
    orw, grw = run(torch.bfloat16, tree=False)
    otr, gtr = run(torch.bfloat16, tree=True)
    # outputs: tree node outputs, gathered to row-wise layout, vs row-wise
    otr_rw = otr.index_select(1, fidx)
    e_tree, e_rw = _rel(otr_rw, o32), _rel(orw, o32)
    assert e_tree < max(3 * e_rw, 2e-2), (e_tree, e_rw)
    # node-input gradients (row-wise scatter-added to nodes by autograd through index_select)
    for n in ("q", "k", "v", "g", "beta"):
        et, er = _rel(gtr[n], g32[n]), _rel(grw[n], g32[n])
        assert et < max(3 * er, 2e-2), (n, et, er)


def test_gdn_levels_layout():
    """Anchor choice: 0 keeps one call per tree depth with no re-read; a huge per-call
    cost collapses each tree to a single call; every node output is produced once."""
    from coda.backends.megatron.tree_gdn import _gdn_levels
    ts, *_ = _setup(0)
    lvl = []
    for p in ts.pieces:
        lvl.append(0 if p.parent_piece == -1 else lvl[p.parent_piece] + 1)
    depth = max(lvl) + 1
    for lt, n_calls in ((0.0, depth), (1e9, 1)):
        levels, perm = _gdn_levels(ts.pieces, "cpu", lt)
        assert len(levels) == n_calls
        own = sum(lv[4].numel() for lv in levels)
        read = sum(lv[0].numel() for lv in levels)
        assert own == ts.num_nodes and sorted(perm.tolist()) == list(range(ts.num_nodes))
        assert (read == ts.num_nodes) == (lt == 0.0)


def test_chain_anchors_matches_reference_dp():
    """Incremental DP picks exactly the anchors of the direct O(m^3) DP."""
    import random

    from coda.backends.megatron.tree_gdn import _chain_anchors

    def reference(lens, branches, base, lt):
        m = len(lens)
        best, back = [0.0] * m, [-1] * m
        for i in range(m):
            best[i] = float("inf")
            for j in range(-1, i):
                run, cost = (base if j == -1 else 0), lt
                for t in range(j + 1, i):
                    run += lens[t]
                    cost += branches[t] * run
                cost += (0.0 if j == -1 else best[j])
                if cost <= best[i]:
                    best[i], back[i] = cost, j
        ends, i = [], m - 1
        while i != -1:
            ends.append(i)
            i = back[i]
        return ends[::-1]

    for seed in range(500):
        rng = random.Random(seed)
        m = rng.randint(1, 25)
        lens = [rng.randint(1, 3000) for _ in range(m)]
        branches = [rng.randint(0, 4) for _ in range(m)]
        base = rng.choice([0, rng.randint(0, 5000)])
        lt = rng.choice([0, 0.0, 1, 3.7, 16000.0, 1e9, rng.uniform(0, 1e5)])
        assert _chain_anchors(lens, branches, base, lt) == reference(lens, branches, base, lt)


def test_gdn_cpu_cu_is_bit_identical():
    """Passing ``cu_seqlens_cpu`` to fla changes no output or gradient bit."""
    import inspect
    if "cu_seqlens_cpu" not in inspect.signature(fla_gdr).parameters:
        pytest.skip("fla without cu_seqlens_cpu")
    ts, *_ = _setup(1)
    T, dev = ts.num_nodes, "cuda"
    torch.manual_seed(0)
    base = dict(q=torch.randn(1, T, 2, 64), k=torch.nn.functional.normalize(torch.randn(1, T, 2, 64), dim=-1),
                v=torch.randn(1, T, 2, 64), g=torch.nn.functional.logsigmoid(torch.randn(1, T, 2)),
                beta=torch.rand(1, T, 2))

    def run(cpu_cu):
        leaf = {n: t.to(dev, torch.bfloat16).requires_grad_(True) for n, t in base.items()}
        o = tree_gated_delta_rule(*leaf.values(), ts.pieces, fla_gdr, level_tokens=500.0,
                                  cpu_cu=cpu_cu)
        o.float().square().sum().backward()
        return [o.detach()] + [t.grad for t in leaf.values()]

    assert all(torch.equal(a, b) for a, b in zip(run(False), run(True)))
