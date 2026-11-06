"""GPU test: a real Megatron GatedDeltaNet layer (Qwen-Next dims, TP=1) in tree layout vs
row-wise layout, through the coda monkeypatch (``install_gdn_tree_patch``).

Row-wise: every row is its own THD sequence holding its full root
path (prefix duplicated). Tree = the de-duplicated node buffer with ``tree_pieces`` attached
to ``packed_seq_params``. The loss weights each row's response outputs; we compare layer
outputs, hidden-state input grads (row-wise scatter-added to nodes) and all parameter grads,
each against an fp32 row-wise truth so the tolerance is relative to bf16 noise.
"""
from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
pytest.importorskip("fla")
pytest.importorskip("megatron.core.ssm.gated_delta_net")

import torch.distributed as dist  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from megatron.core import parallel_state  # noqa: E402
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (  # noqa: E402
    get_experimental_attention_variant_module_spec,
)
from megatron.core.packed_seq_params import PackedSeqParams  # noqa: E402
from megatron.core.process_groups_config import ProcessGroupCollection  # noqa: E402
from megatron.core.ssm.gated_delta_net import GatedDeltaNet  # noqa: E402
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed  # noqa: E402
from megatron.core.transformer import TransformerConfig  # noqa: E402

from coda.backends.megatron.tree_gdn import install_gdn_tree_patch  # noqa: E402
from coda.backends.megatron.tree_pack import (  # noqa: E402
    build_tree_seq, merge_pieces, row_path_nodes,
)

from .tree_test_utils import caterpillar as _caterpillar  # noqa: E402
from .tree_test_utils import cu_seqlens as _cu  # noqa: E402
from .tree_test_utils import rel_err as _rel  # noqa: E402


@pytest.fixture(scope="module")
def gdn():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29533")
    torch.cuda.set_device(0)
    if not dist.is_initialized():
        dist.init_process_group("nccl", world_size=1, rank=0)
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(123)
    install_gdn_tree_patch()
    cfg = TransformerConfig(
        hidden_size=2048, linear_conv_kernel_dim=4, linear_key_head_dim=128,
        linear_value_head_dim=128, linear_num_key_heads=16, linear_num_value_heads=32,
        num_layers=1, normalization="RMSNorm", use_cpu_initialization=True,
        layernorm_zero_centered_gamma=True, num_attention_heads=16, num_query_groups=2,
        activation_func=F.silu, bf16=True, experimental_attention_variant="gated_delta_net",
        linear_attention_freq=[1], transformer_impl="transformer_engine",
    )
    pg = ProcessGroupCollection(tp=parallel_state.get_tensor_model_parallel_group(),
                                cp=parallel_state.get_context_parallel_group(),
                                tp_cp=parallel_state.get_tensor_and_context_parallel_group())
    sub = get_experimental_attention_variant_module_spec(config=cfg).submodules
    m = GatedDeltaNet(cfg, submodules=sub, layer_number=1, bias=False, conv_bias=False,
                      conv_init=1.0, use_qk_l2norm=True, A_init_range=(1, 16), pg_collection=pg)
    # Default init (A in [1,16]) forgets within a few tokens, so a missing branch state would
    # be invisible. Trained models have long-memory heads; use slow decay so the test can
    # tell a correct state hand-off from a wrong one.
    with torch.no_grad():
        m.A_log.copy_(torch.empty_like(m.A_log).uniform_(1e-3, 5e-2).log())
        m.dt_bias.fill_(-2.0)
    return m.cuda()


def _psp(lens, dev, pieces=None):
    cu = _cu(lens, dev).to(torch.int32)
    p = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu,
                        max_seqlen_q=max(lens), max_seqlen_kv=max(lens))
    if pieces is not None:
        p.tree_pieces = pieces
    return p


@pytest.mark.parametrize("seed,pad", [(0, 0), (1, 37)])
def test_gdn_layer_tree_matches_rowwise(gdn, seed, pad):
    dev = "cuda"
    rows, bp, resp = _caterpillar(seed, n_rows=4, spine0=1200, grow=(200, 600),
                                  thorn=(150, 500))
    lens = [len(r) for r in rows]
    ts = build_tree_seq(0, len(rows), lens, bp, resp)
    T = ts.num_nodes + pad                 # trailing alignment padding as its own root
    pieces = merge_pieces([ts], T)
    paths = [row_path_nodes(ts, r) for r in range(len(rows))]
    fidx = torch.tensor([i for p in paths for i in p], device=dev)
    rlens = [len(p) for p in paths]

    torch.manual_seed(seed)
    H = gdn.config.hidden_size
    x_nodes = torch.randn(T, 1, H) * 0.5
    w = torch.randn(T, 1, H, device=dev)
    rw_mask = torch.zeros(sum(rlens), 1, 1, device=dev)
    off = 0
    for r, L in enumerate(rlens):
        rw_mask[off + L - resp[r]:off + L] = 1.0
        off += L
    tree_mask = torch.zeros(T, 1, 1, device=dev)
    for r in range(len(rows)):
        tree_mask[ts.row_nodes[r]] = 1.0
    assert tree_mask.sum().item() == rw_mask.sum().item()

    def run(model, dtype, tree):
        model.zero_grad(set_to_none=True)
        x = x_nodes.to(dev, dtype).requires_grad_(True)
        if tree:
            out, _ = model(x, None, packed_seq_params=_psp([T], dev, pieces))
            loss = (out.float() * w * tree_mask).sum()
            out_rw = out.index_select(0, fidx)
        else:
            xin = x.index_select(0, fidx)
            out, _ = model(xin, None, packed_seq_params=_psp(rlens, dev))
            loss = (out.float() * w.index_select(0, fidx) * rw_mask).sum()
            out_rw = out
        loss.backward()
        grads = {n: p.grad.detach().float().clone() for n, p in model.named_parameters()
                 if p.grad is not None}
        return (out_rw.detach().float() * rw_mask), x.grad.detach().float(), grads

    # Same (bf16-rounded) weights for the fp32 truth and the bf16 runs. The module holds
    # process groups, so it cannot be deep-copied; cast it in place instead.
    m = gdn.bfloat16().float()
    o32, x32, g32 = run(m, torch.float32, tree=False)
    m16 = m.bfloat16()
    orw, xrw, grw = run(m16, torch.bfloat16, tree=False)
    otr, xtr, gtr = run(m16, torch.bfloat16, tree=True)

    report = {"out": (_rel(otr, o32), _rel(orw, o32)), "dx": (_rel(xtr, x32), _rel(xrw, x32))}
    for n in g32:
        report[n] = (_rel(gtr[n], g32[n]), _rel(grw[n], g32[n]))
    print(f"\n[seed={seed} pad={pad} T={T} rowwise={sum(rlens)}]")
    for k, (et, er) in report.items():
        print(f"  {k:40s} tree={et:.3e} rowwise={er:.3e}")
    for k, (et, er) in report.items():
        assert et < max(3 * er, 2e-2), (k, et, er)
    # tree vs bf16 row-wise directly (same kernels, only the layout differs)
    assert _rel(otr, orw) < 2e-2 and _rel(xtr, xrw) < 2e-2

    # Negative control: the same node buffer WITHOUT tree_pieces is one plain sequence, so
    # thorns see their siblings' tokens -- must be clearly wrong, proving the patch is live.
    with torch.no_grad():
        x = x_nodes.to(dev, torch.bfloat16)
        bad, _ = m16(x, None, packed_seq_params=_psp([T], dev))
    e_bad = _rel(bad.index_select(0, fidx).float() * rw_mask, o32)
    print(f"  negative control (no tree_pieces) out err={e_bad:.3e}")
    assert e_bad > 0.1
