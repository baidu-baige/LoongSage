"""GPU test: a tree micro-batch built with the row packing (tokens, loss index, R3)."""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("megatron", reason="Megatron-Core is not installed")

from .test_batch import E, LAYERS, TOPK, _setup  # noqa: E402


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
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    _mp_init()
    from types import SimpleNamespace

    from coda.backends.megatron.cp_utils import prepare_packed_seq_params, prepare_routing_replay_data
    from coda.custom.tree_pack.index.batch import build_tree_microbatch
    from coda.custom.tree_pack.megatron.adapters.attention import build_attn_layout

    rows_a, _ = _setup(4)                  # one caterpillar trajectory ...
    rows_b, _ = _setup(5)                  # ... and another
    rows = rows_a + rows_b
    toks = [torch.tensor(r, device="cuda") for r in rows]
    resp = [min(40, len(r) - 1) for r in rows]
    groups = [0] * len(rows_a) + [1] * len(rows_b)
    g = torch.Generator().manual_seed(0)
    routed = [torch.randint(0, E, (len(r), LAYERS, TOPK), generator=g, dtype=torch.uint8) for r in rows]

    mb = build_tree_microbatch(toks, resp, groups, pack=lambda t: prepare_packed_seq_params([t]),
                               routed_rows=routed)
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
