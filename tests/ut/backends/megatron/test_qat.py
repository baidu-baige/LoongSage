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
from coda.utils.quant_formats import CkptQuant, QuantScheme  # noqa: E402

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


def _cfg(enable, rollout=GOOD_ROLLOUT):
    return OmegaConf.create({"run_mode": "default", "trainer": {"qat": enable}, "rollout": rollout})


def _plan(monkeypatch, m, enable, ckpt=CKPT, rollout=GOOD_ROLLOUT, unsupported=None):
    monkeypatch.setattr(qat, "scan_ckpt_quant", lambda path: (ckpt, unsupported or {}))
    monkeypatch.setattr(qat.os.path, "isdir", lambda path: True)
    monkeypatch.setattr(qat.torch.distributed, "get_rank", lambda: 0)
    return qat.QuantPlan(_bridge(m), [m], "/ckpt", _cfg(enable, rollout))


def test_plan_snap_and_hooks(monkeypatch):
    m = _toy()
    keep = m.norm.weight.clone()
    plan = _plan(monkeypatch, m, enable=True)
    assert plan.quantized and [n for n, _, _ in plan.snap_params] == ["wq_b.weight", "wo_a.weight", "experts.weight0"]
    assert [n for n, _, _ in plan.act_modules] == ["wq_b"]  # wo_a: bf16 act; experts: W4A16
    before = [p.clone() for _, p, _ in plan.snap_params]
    assert plan.snap_() == 3
    for (_, p, scheme), b in zip(plan.snap_params, before):
        assert not torch.equal(p, b) and torch.equal(p, scheme.snap_(b))
    assert torch.equal(m.norm.weight, keep)


def test_plan_without_qat_only_encodes(monkeypatch):
    plan = _plan(monkeypatch, _toy(), enable=False)
    assert plan.quantized and not plan.snap_params and not plan.act_modules


def test_plan_bf16_ckpt(monkeypatch):
    plan = _plan(monkeypatch, _toy(), enable=False, ckpt={})
    assert not plan.quantized
    with pytest.raises(ValueError, match="needs a quantized checkpoint"):
        _plan(monkeypatch, _toy(), enable=True, ckpt={})
    # only untrained tensors (e.g. the engram table) quantized
    table = {"layers.1.engram.embed.weight": CkptQuant(FP8, "layers.1.engram.embed.scale", (64, 64))}
    assert not _plan(monkeypatch, _toy(), enable=False, ckpt=table).quantized
    with pytest.raises(ValueError, match="no trained tensor"):
        _plan(monkeypatch, _toy(), enable=True, ckpt=table)


def test_plan_unsupported_layout(monkeypatch, caplog):
    # e.g. MTP e_proj kept FP8 with fp32 scales in the BF16 DeepSeek-V4 checkpoint, never mapped
    unused = {"mtp.0.e_proj.weight": "unsupported quantized layout"}
    with caplog.at_level("INFO", logger=qat.__name__):
        assert not _plan(monkeypatch, _toy(), enable=False, ckpt={}, unsupported=unused).quantized
    assert "skipping 1 unmapped tensors" in caplog.text and "mtp.0.e_proj.weight" in caplog.text
    assert _plan(monkeypatch, _toy(), enable=True, unsupported=unused).snap_params
    # a mapped tensor with an unsupported layout still fails, also on another PP stage
    for name in ("layers.0.attn.wq_b.weight", "layers.9.attn.wq_b.weight"):
        with pytest.raises(NotImplementedError, match=name):
            _plan(monkeypatch, _toy(), enable=False, ckpt={}, unsupported={name: "unsupported quantized layout"})


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
