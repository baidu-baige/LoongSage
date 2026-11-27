from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

pytest.importorskip("megatron.bridge")
from megatron.bridge.models.conversion.param_mapping import (  # noqa: E402
    ColumnParallelMapping,
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
    RowParallelMapping,
)

from coda.backends.megatron import qat  # noqa: E402
from coda.utils.quant_formats import CkptQuant, QuantScheme, UnsupportedQuant  # noqa: E402

FP8 = QuantScheme("fp8_block", (32, 32))
FP4 = QuantScheme("mxfp4", (1, 32))


@pytest.mark.parametrize(
    "mapping,n,local,full,block,ok",
    [
        (ReplicatedMapping("x", "a"), 1, (64, 96), (64, 96), (32, 32), True),
        (ColumnParallelMapping("x", "a"), 1, (32, 96), (64, 96), (32, 32), True),
        (ColumnParallelMapping("x", "a"), 1, (16, 96), (64, 96), (32, 32), False),  # shard not tile aligned
        (RowParallelMapping("x", "a"), 1, (64, 32), (64, 96), (32, 32), True),
        (RowParallelMapping("x", "a"), 1, (64, 16), (64, 128), (1, 32), False),
        (GatedMLPMapping("x", gate="g", up="u"), 2, (64, 96), (64, 96), (32, 32), True),  # EP expert
        (GatedMLPMapping("x", gate="g", up="u"), 2, (64, 96), (128, 96), (32, 32), True),  # TP2 [g_i; u_i]
        (GatedMLPMapping("x", gate="g", up="u"), 2, (32, 96), (64, 96), (32, 32), False),  # seam at 16
        (GatedMLPMapping("x", gate="g", up="u"), 2, (32, 96), (64, 96), (1, 32), True),
        (QKVMapping("x", q="q", k="k", v="v"), 3, (96, 96), (32, 96), (32, 32), False),  # interleaved
    ],
)
def test_local_layout_ok(mapping, n, local, full, block, ok):
    assert qat._local_layout_ok(mapping, n, torch.Size(local), full, block) is ok




def _toy():
    m = torch.nn.Module()
    m.wq_b = torch.nn.Linear(64, 96, bias=False)
    m.wo_a = torch.nn.Linear(64, 64, bias=False)
    m.experts = torch.nn.Module()
    m.experts.weight0 = torch.nn.Parameter(torch.randn(128, 64))
    m.norm = torch.nn.LayerNorm(64)
    for p in m.parameters():
        p.data = p.data.bfloat16()
    return m


def _bridge(m, arch="DeepseekV41ForCausalLM"):
    def task(name, module, param, mapping):
        return SimpleNamespace(param_name=name, megatron_module=module, param_weight=param, mapping=mapping)

    tasks = [
        task("wq_b.weight", m.wq_b, m.wq_b.weight, ReplicatedMapping("wq_b.weight", "layers.0.attn.wq_b.weight")),
        task("wo_a.weight", m.wo_a, m.wo_a.weight, ReplicatedMapping("wo_a.weight", "layers.0.attn.wo_a.weight")),
        task("experts.weight0", m.experts, m.experts.weight0,
             GatedMLPMapping("experts.weight0", gate="layers.0.ffn.experts.0.w1.weight",
                             up="layers.0.ffn.experts.0.w3.weight")),
        task("norm.weight", m.norm, m.norm.weight, ReplicatedMapping("norm.weight", "layers.0.attn_norm.weight")),
        None,
        task("other_stage.weight", None, None, ReplicatedMapping("o", "layers.9.attn.wq_b.weight")),
    ]
    return SimpleNamespace(get_conversion_tasks=lambda chunks: tasks,
                           hf_pretrained=SimpleNamespace(config=SimpleNamespace(architectures=[arch])))


CKPT = {
    "layers.0.attn.wq_b.weight": CkptQuant(FP8, "layers.0.attn.wq_b.scale", (96, 64)),
    "layers.0.attn.wo_a.weight": CkptQuant(FP8, "layers.0.attn.wo_a.scale", (64, 64)),
    "layers.0.ffn.experts.0.w1.weight": CkptQuant(FP4, "layers.0.ffn.experts.0.w1.scale", (64, 64)),
    "layers.0.ffn.experts.0.w3.weight": CkptQuant(FP4, "layers.0.ffn.experts.0.w3.scale", (64, 64)),
    "layers.9.attn.wq_b.weight": CkptQuant(FP8, "layers.9.attn.wq_b.scale", (96, 64)),
}
GOOD_ROLLOUT = {"backend": "sglang", "env_vars": {"SGLANG_OPT_FP8_WO_A_GEMM": "0"},
                "sglang_args": {"flashinfer_mxfp4_moe_precision": "bf16", "disable_shared_experts_fusion": True}}


def _cfg(enable, rollout=GOOD_ROLLOUT, rounding="threshold"):
    return OmegaConf.create({"run_mode": "default", "trainer": {"qat": enable, "qat_weight_rounding": rounding},
                             "rollout": rollout})


def _plan(monkeypatch, m, enable, ckpt=CKPT, rollout=GOOD_ROLLOUT, unsupported=None, rounding="threshold"):
    monkeypatch.setattr(qat, "scan_ckpt_quant", lambda path: (ckpt, unsupported or {}))
    monkeypatch.setattr(qat.os.path, "isdir", lambda path: True)
    monkeypatch.setattr(qat.torch.distributed, "get_rank", lambda: 0)
    return qat.QuantPlan(_bridge(m), [m], "/ckpt", _cfg(enable, rollout, rounding))


@pytest.mark.parametrize("rounding", ["threshold", "nearest"])
def test_plan_snap_and_hooks(monkeypatch, rounding):
    m = _toy()
    keep = m.norm.weight.clone()
    plan = _plan(monkeypatch, m, enable=True, rounding=rounding)
    assert plan.quantized and [n for n, _, _ in plan.snap_params] == ["wq_b.weight", "wo_a.weight", "experts.weight0"]
    assert [n for n, _, _ in plan.act_modules] == ["wq_b"]  # wo_a: bf16 act; experts: W4A16
    before = [p.clone() for _, p, _ in plan.snap_params]
    assert plan.snap_() == 3
    for (_, p, scheme), thr, b in zip(plan.snap_params, plan._snap_thresholds, before):
        assert (thr is None) == (rounding == "nearest")
        assert not torch.equal(p, b) and torch.equal(p, scheme.snap_(b, thr))
        assert torch.equal(scheme.snap_(p.clone()), p)  # on the grid
    assert torch.equal(m.norm.weight, keep)


def test_plan_rejects_unknown_rounding(monkeypatch):
    with pytest.raises(ValueError, match="qat_weight_rounding"):
        _plan(monkeypatch, _toy(), enable=True, rounding="stochastic")


@pytest.mark.parametrize(
    "mapping,n,full,tp",
    [
        (ColumnParallelMapping("x", "a"), 1, (128, 64), 2),  # dim-0 shard
        (RowParallelMapping("x", "a"), 1, (64, 128), 4),  # dim-1 shard
        (GatedMLPMapping("x", gate="g", up="u"), 2, (128, 64), 2),  # [g_r; u_r]
        (GatedMLPMapping("x", gate="g", up="u"), 2, (64, 64), 1),  # EP expert, unsharded
    ],
)
def test_thresholds_follow_checkpoint_layout(monkeypatch, mapping, n, full, tp):
    """Every TP rank's local thresholds are the matching slices of the full checkpoint tensors'."""
    names = qat.hf_names(mapping)
    O, I = full
    whole = {nm: qat._thresholds_fn(mapping, [nm], torch.Size(full), full)(0, O, "cpu") for nm in names}
    dim0 = isinstance(mapping, (ColumnParallelMapping, GatedMLPMapping))
    o, i = (O // tp, I) if dim0 else (O, I // tp)
    for rank in range(tp):
        monkeypatch.setattr(type(mapping), "tp_rank", property(lambda self, r=rank: r))
        fn = qat._thresholds_fn(mapping, names, torch.Size((o * n, i)), full)
        got = torch.cat([fn(r0, min(r0 + 24, o * n), "cpu") for r0 in range(0, o * n, 24)])  # uneven chunks
        rs = slice(rank * o, (rank + 1) * o) if dim0 else slice(None)
        cs = slice(None) if dim0 else slice(rank * i, (rank + 1) * i)
        torch.testing.assert_close(got, torch.cat([whole[nm][rs, cs] for nm in names]), rtol=0, atol=0)


def test_plan_without_qat_only_encodes(monkeypatch):
    plan = _plan(monkeypatch, _toy(), enable=False)
    assert plan.quantized and not plan.snap_params and not plan.act_modules


def test_plan_bf16_ckpt(monkeypatch):
    plan = _plan(monkeypatch, _toy(), enable=False, ckpt={})
    assert not plan.quantized
    with pytest.raises(ValueError, match="no trained tensor"):
        _plan(monkeypatch, _toy(), enable=True, ckpt={})
    # only untrained tensors (e.g. the engram table) quantized
    table = {"layers.1.engram.embed.weight": CkptQuant(FP8, "layers.1.engram.embed.scale", (64, 64))}
    assert not _plan(monkeypatch, _toy(), enable=False, ckpt=table).quantized
    with pytest.raises(ValueError, match="no trained tensor"):
        _plan(monkeypatch, _toy(), enable=True, ckpt=table)


def _unsupported(name, dtype="F8_E4M3", shape=(96, 64)):
    return {name: UnsupportedQuant("unsupported quantized layout", name[: -len(".weight")] + ".scale", dtype, shape)}


def test_plan_unsupported_layout(monkeypatch, caplog):
    # e.g. MTP e_proj kept FP8 with fp32 scales in the BF16 DeepSeek-V4 checkpoint, never mapped
    unused = _unsupported("mtp.0.e_proj.weight")
    with caplog.at_level("INFO", logger=qat.__name__):
        assert not _plan(monkeypatch, _toy(), enable=False, ckpt={}, unsupported=unused).quantized
    assert "skipping 1 unmapped tensors" in caplog.text and "mtp.0.e_proj.weight" in caplog.text
    assert _plan(monkeypatch, _toy(), enable=True, unsupported=unused).snap_params
    # a mapped tensor with an unsupported layout: export-checked without QAT, an error with QAT
    for name in ("layers.0.attn.wq_b.weight", "layers.9.attn.wq_b.weight"):  # also on another PP stage
        caplog.clear()
        with caplog.at_level("INFO", logger=qat.__name__):
            assert _plan(monkeypatch, _toy(), enable=False, ckpt={}, unsupported=_unsupported(name)).quantized
        assert "1 trained tensors" in caplog.text and name in caplog.text and "skipping" not in caplog.text
        ckpt = {n: q for n, q in CKPT.items() if n != name}
        with pytest.raises(NotImplementedError, match=name):
            _plan(monkeypatch, _toy(), enable=True, ckpt=ckpt, unsupported=_unsupported(name))


def test_adapter_rejects_unmodeled_rollout(monkeypatch):
    bad = {"backend": "sglang", "env_vars": {}, "sglang_args": {}}
    with pytest.raises(ValueError, match="flashinfer_mxfp4_moe_precision.*SGLANG_OPT_FP8_WO_A_GEMM.*shared_experts"):
        _plan(monkeypatch, _toy(), enable=True, rollout=bad)
    _plan(monkeypatch, _toy(), enable=False, rollout=bad)  # no QAT -> no numerics constraints


def test_check_export(monkeypatch):
    m = _toy()
    plan = _plan(monkeypatch, m, enable=True)
    plan.snap_()
    q, s = FP8.quant(m.wq_b.weight.data)
    q4, s4 = FP4.quant(m.experts.weight0.data[:64])
    other = torch.randn(64).bfloat16()
    good = [
        ("layers.0.attn.wq_b.weight", q), ("layers.0.attn.wq_b.scale", s),
        ("layers.0.ffn.experts.0.w1.scale", s4), ("layers.0.ffn.experts.0.w1.weight", q4),  # any order
        ("layers.0.attn_norm.weight", other),
    ]
    out = list(plan.check_export(good))
    assert [n for n, _ in out] == [n for n, _ in good] and out[-1][1] is other
    # bridge exported bf16 for a tensor the checkpoint stores quantized (same shape for FP8)
    with pytest.raises(RuntimeError, match="must re-quantize"):
        list(plan.check_export([("layers.0.attn.wq_b.weight", m.wq_b.weight.data)]))
    with pytest.raises(RuntimeError, match="must re-quantize"):
        list(plan.check_export([("layers.0.ffn.experts.0.w1.weight", m.experts.weight0.data[:64])]))


def test_check_export_unsupported_layout(monkeypatch):
    name = "layers.0.attn.wq_b.weight"
    ckpt = {n: q for n, q in CKPT.items() if n != name}
    plan = _plan(monkeypatch, _toy(), enable=False, ckpt=ckpt, unsupported=_unsupported(name))
    w = torch.randn(96, 64)
    stored = [(name, w.to(torch.float8_e4m3fn)), ("layers.0.attn.wq_b.scale", torch.ones(1, 1))]
    assert [n for n, _ in plan.check_export(stored)] == [n for n, _ in stored]
    for bad in (w.bfloat16(), w[:32].to(torch.float8_e4m3fn)):
        with pytest.raises(RuntimeError, match=r"stores F8_E4M3 \(96, 64\).*must re-quantize"):
            list(plan.check_export([(name, bad)]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_activation_fake_quant_matches_sglang_and_is_ste(monkeypatch):
    pytest.importorskip("sglang.kernels.ops.quantization.fp8_kernel")
    from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_group_quant_fp8

    x = (torch.randn(3, 5, 128, device="cuda") * 3).bfloat16().requires_grad_()
    y = qat._FakeQuantActFP8.apply(x, 32)
    q, s = sglang_per_token_group_quant_fp8(x.detach().reshape(-1, 128).contiguous(), 32, scale_ue8m0=True)
    ref = (q.float().view(15, 4, 32) * s.view(15, 4, 1)).view(3, 5, 128).bfloat16()
    assert torch.equal(y, ref)
    assert not torch.equal(y, x)
    y.float().sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))

    m = _toy().cuda()
    plan = _plan(monkeypatch, m, enable=True)
    assert plan.register_activation_hooks() == 1
    inp = torch.randn(4, 64, device="cuda").bfloat16()
    torch.testing.assert_close(m.wq_b(inp), torch.nn.functional.linear(qat._FakeQuantActFP8.apply(inp, 32), m.wq_b.weight),
                               rtol=0, atol=0)
    assert torch.equal(m.wo_a(inp), torch.nn.functional.linear(inp, m.wo_a.weight))


# ---------------------------------------------------------------- ERNIE 5.5


class Ernie55LatentProjection(torch.nn.Module):
    """Stand-in with the forward of the bridge's ``Ernie55LatentProjection`` (fp32 weight, cast at use)."""

    def __init__(self, o, i):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(o, i) * 0.02)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight.to(x.dtype)), None


ERNIE_ROLLOUT = {"backend": "sglang", "env_vars": {},
                 "sglang_args": {"moe_runner_backend": "flashinfer_mxfp4", "flashinfer_mxfp4_moe_precision": "bf16",
                                 "disable_shared_experts_fusion": True}}


def _ernie(monkeypatch, rounding="threshold", rollout=ERNIE_ROLLOUT):
    em = pytest.importorskip("megatron.bridge.models.ernie.ernie55_mappings")
    # 2 heads of nope 24 + v 32 = 56 rows: heads and the k/v split are not 32-row aligned.
    dims = em.Ernie55DLayerDims(num_heads=2, kv_lora_rank=64, qk_nope_head_dim=24, qk_rope_head_dim=8, v_head_dim=32)
    m = torch.nn.Module()
    m.attn = torch.nn.Module()
    m.attn.k_b_proj = torch.nn.Parameter(torch.randn(2 * 64, 24).bfloat16())
    m.attn.v_b_proj = torch.nn.Parameter(torch.randn(2 * 32, 64).bfloat16())
    m.wq = torch.nn.Linear(64, 64, bias=False).bfloat16()
    m.latent = Ernie55LatentProjection(64, 96)
    kv = "model.layers.7.self_attn.kv_b_proj.weight"

    def task(name, module, param, mapping):
        return SimpleNamespace(param_name=name, megatron_module=module, param_weight=param, mapping=mapping)

    tasks = [
        task("attn.k_b_proj", m.attn, m.attn.k_b_proj,
             em._Ernie55KVUpMapping("attn.k_b_proj", kv, hf_dtype=torch.bfloat16, part="k", dims=dims)),
        task("attn.v_b_proj", m.attn, m.attn.v_b_proj,
             em._Ernie55KVUpMapping("attn.v_b_proj", kv, hf_dtype=torch.bfloat16, part="v", dims=dims)),
        task("wq.weight", m.wq, m.wq.weight, ReplicatedMapping("wq.weight", "model.layers.7.self_attn.q_a_proj.weight")),
        task("latent.weight", m.latent, m.latent.weight,
             ReplicatedMapping("latent.weight", "model.layers.7.block_sparse_moe.fc1_latent_proj.weight")),
    ]
    bridge = SimpleNamespace(get_conversion_tasks=lambda chunks: tasks,
                             hf_pretrained=SimpleNamespace(config=SimpleNamespace(architectures=["Ernie5_5_MoeForCausalLM"])))
    ckpt = {
        kv: CkptQuant(FP8, kv + "_scale_inv", (112, 64)),
        "model.layers.7.self_attn.q_a_proj.weight": CkptQuant(FP8, "q_scale_inv", (64, 64)),
        "model.layers.7.block_sparse_moe.fc1_latent_proj.weight": CkptQuant(FP8, "l_scale_inv", (64, 96)),
    }
    monkeypatch.setattr(qat, "scan_ckpt_quant", lambda path: (ckpt, {}))
    monkeypatch.setattr(qat.os.path, "isdir", lambda path: True)
    monkeypatch.setattr(qat.torch.distributed, "get_rank", lambda: 0)
    plan = qat.QuantPlan(bridge, [m], "/ckpt", _cfg(True, rollout, rounding))
    return plan, m, dims, kv, em


@pytest.mark.parametrize("rounding", ["threshold", "nearest"])
def test_ernie_kv_b_proj_joint_snap(monkeypatch, rounding):
    plan, m, dims, kv, em = _ernie(monkeypatch, rounding)
    assert [n for n, _, _ in plan.snap_params] == ["wq.weight"] and len(plan.custom_snaps) == 1
    assert not plan.act_modules  # weight QAT only
    full = em.join_kv_b_proj(m.attn.k_b_proj.data, m.attn.v_b_proj.data, dims)
    thr = None
    if rounding == "threshold":
        thr = qat._thresholds_fn(SimpleNamespace(tp_rank=0), [kv], full.shape, tuple(full.shape))
    expect = FP8.snap_(full.clone(), thr)
    assert plan.snap_() == 2
    got = em.join_kv_b_proj(m.attn.k_b_proj.data, m.attn.v_b_proj.data, dims)
    assert torch.equal(got, expect) and not torch.equal(got, full)
    assert torch.equal(FP8.snap_(got.clone()), got)  # on the checkpoint tensor's grid


def test_ernie_fp32_latent_forward_fake_quant(monkeypatch):
    plan, m, *_ = _ernie(monkeypatch)
    assert [n for n, _, _ in plan.fake_quant_params] == ["latent.weight"]
    master = m.latent.weight.data.clone()
    plan.snap_()
    assert torch.equal(m.latent.weight.data, master)  # never snapped in place
    x = torch.randn(5, 96).bfloat16()
    y, _ = m.latent(x)
    q = FP8.snap_(master.clone())
    assert torch.equal(y, torch.nn.functional.linear(x, q.bfloat16()))
    assert torch.equal(m.latent.weight.data, master)  # restored after the forward
    y.float().sum().backward()  # straight-through: dL/dW = g^T x, independent of W
    ref = torch.ones(5, 64).bfloat16().t().float() @ x.float()
    torch.testing.assert_close(m.latent.weight.grad, ref, rtol=1e-2, atol=1e-2)


def test_ernie_adapter_rejects_unmodeled_rollout(monkeypatch):
    bad = {"backend": "sglang", "env_vars": {}, "sglang_args": {}}
    with pytest.raises(ValueError, match="moe_runner_backend.*flashinfer_mxfp4_moe_precision.*shared_experts"):
        _ernie(monkeypatch, rollout=bad)


def test_fp32_quantized_param_needs_adapter(monkeypatch):
    m = _toy()
    m.wq_b.weight.data = m.wq_b.weight.data.float()
    with pytest.raises(NotImplementedError, match="fp32 quantized param"):
        _plan(monkeypatch, m, enable=True)
