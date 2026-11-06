"""GPU test: tree loss gather (log-probs / entropy) vs. the row-wise path.

Logits come from per-node hidden states through a shared head (``logits = h @ W``), so a
shared node has identical logits in every row -- the property the real model provides
(verified layer by layer in the other tree tests). Row-wise uses
``prepare_packed_seq_params`` + ``compute_log_probs`` / ``compute_entropy``; tree uses
``tree_compute_log_probs`` / ``tree_compute_entropy``. Responses start exactly at each
row's branch point, so the first response token's logit sits on a shared node whose
default target belongs to the previous row: the extra-target path must handle it.
"""
from __future__ import annotations

import os
import random

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

pytest.importorskip("megatron", reason="Megatron-Core is not installed")

import torch.distributed as dist  # noqa: E402
from megatron.core import parallel_state  # noqa: E402

from coda.backends.megatron.cp_utils import prepare_packed_seq_params  # noqa: E402
from coda.backends.megatron.logits_utils import compute_entropy, compute_log_probs  # noqa: E402
from coda.backends.megatron.tree_loss import (  # noqa: E402
    build_loss_index, index_to, restrict_to_lm_rows, tree_compute_entropy, tree_compute_log_probs,
)
from coda.backends.megatron.tree_pack import build_tree_seq, merge_pieces  # noqa: E402

V, HD = 4096, 256


@pytest.fixture(scope="module", autouse=True)
def _mp():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29535")
    torch.cuda.set_device(0)
    if not dist.is_initialized():
        dist.init_process_group("nccl", world_size=1, rank=0)
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(1, 1)


def _rows(seed, n_rows=5):
    rng = random.Random(seed)
    tok = lambda n: [rng.randrange(V) for _ in range(n)]  # noqa: E731
    spine, rows, resp, bp = tok(800), [], [], []
    for k in range(n_rows):
        grow = tok(rng.randint(100, 400))
        th = tok(rng.randint(100, 400))
        rows.append(spine + grow + th)
        resp.append(len(grow) + len(th))      # response starts at the branch point
        spine = spine + grow
    for k, r in enumerate(rows):
        if k == 0:
            bp.append(0)
            continue
        m, i = min(len(r), len(rows[k - 1])), 0
        while i < m and r[i] == rows[k - 1][i]:
            i += 1
        bp.append(i)
    return rows, resp, bp


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_tree_loss_matches_rowwise(seed):
    dev = "cuda"
    rows, resp, bp = _rows(seed)
    lens = [len(r) for r in rows]
    ts = build_tree_seq(0, len(rows), lens, bp, resp)
    row_t = [torch.tensor(r) for r in rows]

    # row-wise, as in loss_function
    packed_tok, _ = prepare_packed_seq_params([t.to(dev) for t in row_t])
    targets = [torch.cat([t[1:], t.new_full((1,), 0)]).to(dev) for t in row_t]
    packed_tgt, _ = prepare_packed_seq_params(targets)
    P = packed_tok.numel()
    T = ts.num_nodes + (P - sum(lens))           # same tail padding as row-wise
    assert merge_pieces([ts], T)[-1].node_end == T

    idx = index_to(build_loss_index([ts], row_t, resp, T), dev)
    assert idx.extra_nodes.numel() == len(rows) - 1, idx.extra_nodes.numel()

    from coda.backends.megatron.tree_pack import row_path_nodes
    fidx = torch.tensor([i for r in range(len(rows)) for i in row_path_nodes(ts, r)], device=dev)

    torch.manual_seed(seed)
    h0 = torch.randn(T, HD, device=dev)
    W0 = torch.randn(HD, V, device=dev) * 0.2
    wl = [torch.randn(n, device=dev) for n in resp]
    we = [torch.randn(n, device=dev) for n in resp]

    idx_rows = index_to(restrict_to_lm_rows(build_loss_index([ts], row_t, resp, T)), dev)
    n_lm = idx_rows.lm_rows.numel()
    assert n_lm <= sum(resp)      # one logit row per response token at most

    def run(tree, extras=True, lm_rows=False):
        h = h0.clone().requires_grad_(True)
        W = W0.clone().requires_grad_(True)
        if tree and lm_rows:       # LM head only on the loss rows (output-layer row selection)
            logits = h.index_select(0, idx_rows.lm_rows) @ W
            ent = tree_compute_entropy(idx_rows, logits.clone())
            lp = tree_compute_log_probs(idx_rows, logits)
        elif tree:
            logits = h @ W
            ix = idx if extras else index_to(
                build_loss_index([ts], row_t, resp, T), dev)
            if not extras:      # negative control: ignore branch-point targets
                ix.logp_src = ix.logit_nodes.clone()
                ix.extra_nodes = ix.extra_nodes[:0]
            ent = tree_compute_entropy(ix, logits.clone())
            lp = tree_compute_log_probs(ix, logits)
        else:
            hr = torch.cat([h.index_select(0, fidx), h.new_zeros(P - fidx.numel(), HD)])
            logits = hr @ W
            ent = compute_entropy(lens, resp, logits.clone())
            lp = compute_log_probs(packed_tgt, lens, resp, logits)
        loss = sum((a * b).sum() for a, b in zip(lp, wl)) + sum((a * b).sum() for a, b in zip(ent, we))
        loss.backward()
        return torch.cat(lp).detach(), torch.cat(ent).detach(), h.grad, W.grad

    lp_r, en_r, dh_r, dW_r = run(False)
    lp_t, en_t, dh_t, dW_t = run(True)
    torch.testing.assert_close(lp_t, lp_r, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(en_t, en_r, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(dh_t, dh_r, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dW_t, dW_r, rtol=1e-4, atol=1e-3)

    # restricted LM head: same results, logits for n_lm rows instead of T
    lp_s, en_s, dh_s, dW_s = run(True, lm_rows=True)
    torch.testing.assert_close(lp_s, lp_r, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(en_s, en_r, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(dh_s, dh_r, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dW_s, dW_r, rtol=1e-4, atol=1e-3)
    assert n_lm < T

    # metric-only (no grad, token-chunked) entropy, on full and on restricted logits
    with torch.no_grad():
        for ix, hh in ((idx, h0), (idx_rows, h0.index_select(0, idx_rows.lm_rows))):
            logits = hh @ W0
            en_m = torch.cat(tree_compute_entropy(ix, logits, metric_only=True))
            torch.testing.assert_close(en_m, en_r.detach(), rtol=1e-5, atol=1e-5)
            assert torch.equal(logits, hh @ W0)    # output left untouched
    lp_bad, *_ = run(True, extras=False)
    n_bad = (lp_bad - lp_r).abs().gt(1e-3).sum().item()
    print(f"\n[seed={seed}] T={T} rowwise={P} resp_tokens={lp_r.numel()} "
          f"extras={idx.extra_nodes.numel()} max|dlogp|={(lp_t - lp_r).abs().max().item():.2e} "
          f"max|dW|={(dW_t - dW_r).abs().max().item():.2e} lm_rows={n_lm}/{T} "
          f"neg-control wrong tokens={n_bad}")
    assert n_bad == len(rows) - 1
