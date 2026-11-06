"""GPU test: Megatron TEDotProductAttention (16 q / 2 kv heads, head_dim 256) in tree layout vs row-wise layout, through ``install_attention_tree_patch``.

Same protocol as the GDN tests: random per-node q/k/v, row-wise = each row's root path as its
own causal THD sequence; loss weights each row's response outputs; outputs and node grads are
compared against an fp32 SDPA row-wise truth. A negative control (same buffer, no
``tree_pieces``) must be clearly wrong.
"""
from __future__ import annotations

import os

os.environ["NVTE_FUSED_ATTN"] = "0"
os.environ["NVTE_FLASH_ATTN"] = "1"

import pytest  # noqa: E402
import torch  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
pytest.importorskip("transformer_engine")
pytest.importorskip("megatron", reason="Megatron-Core is not installed")

import torch.distributed as dist  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from megatron.core import parallel_state  # noqa: E402
from megatron.core.extensions.transformer_engine import TEDotProductAttention  # noqa: E402
from megatron.core.packed_seq_params import PackedSeqParams  # noqa: E402
from megatron.core.process_groups_config import ProcessGroupCollection  # noqa: E402
from megatron.core.transformer import TransformerConfig  # noqa: E402
from megatron.core.transformer.enums import AttnMaskType  # noqa: E402

from coda.backends.megatron.tree_attn import install_attention_tree_patch  # noqa: E402
from coda.backends.megatron.tree_pack import (  # noqa: E402
    build_tree_seq, merge_pieces, row_path_nodes,
)

from .tree_test_utils import caterpillar as _caterpillar  # noqa: E402
from .tree_test_utils import cu_seqlens as _cu  # noqa: E402
from .tree_test_utils import rel_err as _rel  # noqa: E402

HQ, HKV, D = 16, 2, 256


@pytest.fixture(scope="module")
def attn():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29534")
    torch.cuda.set_device(0)
    if not dist.is_initialized():
        dist.init_process_group("nccl", world_size=1, rank=0)
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(1, 1)
    install_attention_tree_patch()
    cfg = TransformerConfig(num_layers=1, hidden_size=2048, num_attention_heads=HQ,
                            num_query_groups=HKV, kv_channels=D, bf16=True,
                            params_dtype=torch.bfloat16, attention_dropout=0.0)
    pg = ProcessGroupCollection(tp=parallel_state.get_tensor_model_parallel_group(),
                                cp=parallel_state.get_context_parallel_group())
    return TEDotProductAttention(cfg, layer_number=1, attn_mask_type=AttnMaskType.causal,
                                 attention_type="self", pg_collection=pg)


def _psp(lens, dev, pieces=None):
    cu = _cu(lens, dev).to(torch.int32)
    p = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu,
                        max_seqlen_q=max(lens), max_seqlen_kv=max(lens))
    if pieces is not None:
        p.tree_pieces = pieces
    return p


def _sdpa_rows(q, k, v, rlens):
    outs, off = [], 0
    for L in rlens:
        s = slice(off, off + L)
        o = F.scaled_dot_product_attention(q[s].transpose(0, 1)[None], k[s].transpose(0, 1)[None],
                                           v[s].transpose(0, 1)[None], is_causal=True,
                                           enable_gqa=True)
        outs.append(o[0].transpose(0, 1).reshape(L, -1))
        off += L
    return torch.cat(outs)


@pytest.mark.parametrize("seed,pad", [(0, 0), (1, 37), (2, 0)])
def test_attention_tree_matches_rowwise(attn, seed, pad):
    dev = "cuda"
    rows, bp, resp = _caterpillar(seed, n_rows=5, spine0=1500, grow=(300, 900), thorn=(200, 700))
    lens = [len(r) for r in rows]
    ts = build_tree_seq(0, len(rows), lens, bp, resp)
    T = ts.num_nodes + pad
    pieces = merge_pieces([ts], T)
    paths = [row_path_nodes(ts, r) for r in range(len(rows))]
    fidx = torch.tensor([i for p in paths for i in p], device=dev)
    rlens = [len(p) for p in paths]

    torch.manual_seed(seed)
    base = dict(q=torch.randn(T, HQ, D), k=torch.randn(T, HKV, D), v=torch.randn(T, HKV, D))
    w = torch.randn(T, HQ * D, device=dev)
    rw_mask = torch.zeros(sum(rlens), 1, device=dev)
    off = 0
    for r, L in enumerate(rlens):
        rw_mask[off + L - resp[r]:off + L] = 1.0
        off += L

    def run(mode):
        dtype = torch.float32 if mode == "sdpa32" else torch.bfloat16
        leaf = {n: t.to(dev, dtype).requires_grad_(True) for n, t in base.items()}
        if mode == "tree":
            out = attn(leaf["q"], leaf["k"], leaf["v"], None, AttnMaskType.causal,
                       packed_seq_params=_psp([T], dev, pieces))
            out_rw = out.reshape(T, -1).index_select(0, fidx)
        else:
            q, k, v = (leaf[n].index_select(0, fidx) for n in ("q", "k", "v"))
            if mode == "sdpa32":
                out_rw = _sdpa_rows(q, k, v, rlens)
            else:
                out_rw = attn(q, k, v, None, AttnMaskType.causal,
                              packed_seq_params=_psp(rlens, dev)).reshape(sum(rlens), -1)
        (out_rw.float() * w.index_select(0, fidx) * rw_mask).sum().backward()
        return out_rw.detach().float(), {n: t.grad.float() for n, t in leaf.items()}

    o32, g32 = run("sdpa32")
    orw, grw = run("rowwise")
    otr, gtr = run("tree")
    report = {"out": (_rel(otr, o32), _rel(orw, o32))}
    for n in ("q", "k", "v"):
        report["d" + n] = (_rel(gtr[n], g32[n]), _rel(grw[n], g32[n]))
    print(f"\n[seed={seed} pad={pad} T={T} rowwise={sum(rlens)}]")
    for k, (et, er) in report.items():
        print(f"  {k:4s} tree={et:.3e} rowwise={er:.3e}")
    for k, (et, er) in report.items():
        assert et < max(3 * er, 2e-2), (k, et, er)

    # Negative control: same buffer as one plain causal sequence (thorns see siblings).
    with torch.no_grad():
        bad = attn(*(base[n].to(dev, torch.bfloat16) for n in ("q", "k", "v")), None,
                   AttnMaskType.causal, packed_seq_params=_psp([T], dev))
    e_bad = _rel(bad.reshape(T, -1).index_select(0, fidx).float(), o32)
    print(f"  negative control (no tree_pieces) out err={e_bad:.3e}")
    # Random q/k average over thousands of keys, so wrong extra keys shift outputs only by a
    # few percent; require it to sit far above bf16 noise.
    assert e_bad > 5 * report["out"][1], (e_bad, report["out"][1])


# ---------------------------------------------------------------------------------------
# RoPE. mRoPE models use Bridge's Qwen3VL mRoPE: per-token absolute frequencies
# computed from explicit [3, 1, T] position_ids, applied in THD without looking at
# cu_seqlens. Tree mode therefore only needs position_ids = node depth on all 3 axes.
# ---------------------------------------------------------------------------------------
rope_mod = pytest.importorskip("megatron.bridge.models.qwen_vl.modelling_qwen3_vl.rope")

MROPE_SECTION = [11, 11, 10]   # partial_rotary 0.25, theta 1e7


def _rowwise_positions_from_bridge(rows, dev):
    """Row-wise positions: coda packs rows (pad to 128 each), the Bridge
    model calls get_rope_index(packed_seq_params=...) because coda passes position_ids=None."""
    from coda.backends.megatron.cp_utils import prepare_packed_seq_params
    toks = [torch.tensor(r, device=dev) % 1000 for r in rows]   # stay clear of vision ids
    packed, psp = prepare_packed_seq_params(toks, cp_partition_mode="zigzag")
    pos, _ = rope_mod.get_rope_index(2, 10**6, 10**6 + 1, 10**6 + 2, packed.unsqueeze(0),
                                     packed_seq_params=psp)
    cu = psp.cu_seqlens_q.tolist()
    return [pos[0, 0, cu[i]:cu[i] + len(r)] for i, r in enumerate(rows)]


def test_tree_positions_match_bridge_rowwise_positions(attn):
    from coda.backends.megatron.tree_pack import merge_positions
    dev = "cuda"
    rows, bp, resp = _caterpillar(3, n_rows=5, spine0=1500, grow=(300, 900), thorn=(200, 700))
    lens = [len(r) for r in rows]
    ts = build_tree_seq(0, len(rows), lens, bp, resp)
    tree_pos = torch.tensor(merge_positions([ts], ts.num_nodes + 11), device=dev)
    for r, ref in enumerate(_rowwise_positions_from_bridge(rows, dev)):
        got = tree_pos.index_select(0, torch.tensor(row_path_nodes(ts, r), device=dev))
        assert torch.equal(got, ref.to(got.dtype)), r


def _mrope_emb(pos_1d, dev):
    rot = rope_mod.Qwen3VLMultimodalRotaryEmbedding(
        kv_channels=D, rotary_percent=0.25, rotary_base=10_000_000,
        cp_group=parallel_state.get_context_parallel_group())
    pid = pos_1d.view(1, 1, -1).expand(3, 1, -1).to(dev)
    return rot(pid, MROPE_SECTION)                      # [T, 1, 1, rot_dim]


class _RopeCfg:
    apply_rope_fusion = False
    apply_rotary_pos_emb_in_fp32 = False
    rotary_interleaved = False


@pytest.mark.parametrize("seed,pad", [(0, 0), (1, 37)])
def test_rope_attention_tree_matches_rowwise(attn, seed, pad):
    """RoPE + attention, tree positions vs row-wise positions; negative control uses the
    positions the model would derive for the tree buffer if position_ids were not passed."""
    from coda.backends.megatron.tree_pack import merge_positions
    dev = "cuda"
    rows, bp, resp = _caterpillar(seed, n_rows=5, spine0=1500, grow=(300, 900), thorn=(200, 700))
    lens = [len(r) for r in rows]
    ts = build_tree_seq(0, len(rows), lens, bp, resp)
    T = ts.num_nodes + pad
    pieces = merge_pieces([ts], T)
    paths = [row_path_nodes(ts, r) for r in range(len(rows))]
    fidx = torch.tensor([i for p in paths for i in p], device=dev)
    rlens = [len(p) for p in paths]
    rw_pos = torch.cat([torch.arange(L) for L in rlens])
    tree_pos = torch.tensor(merge_positions([ts], T))

    torch.manual_seed(seed)
    base = dict(q=torch.randn(T, HQ, D), k=torch.randn(T, HKV, D), v=torch.randn(T, HKV, D))
    w = torch.randn(T, HQ * D, device=dev)
    rw_mask = torch.zeros(sum(rlens), 1, device=dev)
    off = 0
    for r, L in enumerate(rlens):
        rw_mask[off + L - resp[r]:off + L] = 1.0
        off += L

    def rope(t, pos):
        return rope_mod.apply_rotary_pos_emb_absolute(t, _mrope_emb(pos, dev), _RopeCfg(),
                                                      cu_seqlens=torch.zeros(2, device=dev))

    def run(mode, positions=None):
        dtype = torch.float32 if mode == "sdpa32" else torch.bfloat16
        leaf = {n: t.to(dev, dtype).requires_grad_(True) for n, t in base.items()}
        if mode in ("tree", "tree_wrong_pos"):
            pos = tree_pos if mode == "tree" else torch.arange(T)
            q, k = rope(leaf["q"], pos), rope(leaf["k"], pos)
            out = attn(q, k, leaf["v"], None, AttnMaskType.causal,
                       packed_seq_params=_psp([T], dev, pieces))
            out_rw = out.reshape(T, -1).index_select(0, fidx)
        else:
            q, k, v = (leaf[n].index_select(0, fidx) for n in ("q", "k", "v"))
            q, k = rope(q, rw_pos), rope(k, rw_pos)
            if mode == "sdpa32":
                out_rw = _sdpa_rows(q, k, v, rlens)
            else:
                out_rw = attn(q, k, v, None, AttnMaskType.causal,
                              packed_seq_params=_psp(rlens, dev)).reshape(sum(rlens), -1)
        (out_rw.float() * w.index_select(0, fidx) * rw_mask).sum().backward()
        return out_rw.detach().float(), {n: t.grad.float() for n, t in leaf.items()}

    o32, g32 = run("sdpa32")
    orw, grw = run("rowwise")
    otr, gtr = run("tree")
    obad, _ = run("tree_wrong_pos")
    report = {"out": (_rel(otr, o32), _rel(orw, o32))}
    for n in ("q", "k", "v"):
        report["d" + n] = (_rel(gtr[n], g32[n]), _rel(grw[n], g32[n]))
    e_bad = _rel(obad, o32)
    print(f"\n[rope seed={seed} pad={pad} T={T}]")
    for k, (et, er) in report.items():
        print(f"  {k:4s} tree={et:.3e} rowwise={er:.3e}")
    print(f"  negative control (positions 0..T-1) out err={e_bad:.3e}")
    for k, (et, er) in report.items():
        assert et < max(3 * er, 2e-2), (k, et, er)
    assert e_bad > 5 * report["out"][1], (e_bad, report["out"][1])
