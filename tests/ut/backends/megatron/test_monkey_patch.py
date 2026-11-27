"""Unit tests for patches in coda/backends/megatron/monkey_patch.py.

The FA4 causal-window tests use a fake ``flash_attn.cute.interface`` carrying a copy of
the window resolver from flash-attn-4 4.0.0b30+, so no GPU or FA4 install is required.
Megatron-Core must still be installed, because ``monkey_patch`` imports it at module scope.
"""

import os
import sys
import types

import pytest

pytest.importorskip("megatron", reason="Megatron-Core is not installed")

from coda.backends.megatron import monkey_patch  # noqa: E402


def _resolve_b30(causal, window_size_left, window_size_right, mask_mod=None):
    """``_resolve_causal_local_window`` as shipped in flash-attn-4 4.0.0b30 - b39."""
    if mask_mod is not None:
        return False, False, window_size_left, window_size_right
    if (window_size_left is not None and window_size_right is not None
            and window_size_left < 0 and window_size_right < 0):
        window_size_left = None
        window_size_right = None
    if causal:
        window_size_right = 0
    if window_size_left is not None or window_size_right is not None:
        if window_size_left is None and window_size_right == 0:
            causal, local = True, False
            window_size_right = None
        else:
            causal, local = False, True
    else:
        local = False
    return causal, local, window_size_left, window_size_right


@pytest.fixture
def fake_fa4(monkeypatch):
    module = types.ModuleType("flash_attn.cute.interface")
    module._resolve_causal_local_window = _resolve_b30
    monkeypatch.setitem(sys.modules, "flash_attn.cute.interface", module)
    return module


def test_unpatched_resolver_turns_te_causal_into_local(fake_fa4):
    assert fake_fa4._resolve_causal_local_window(True, -1, 0) == (False, True, -1, 0)


@pytest.mark.parametrize(
    "args, expected",
    [
        ((True, -1, 0), (True, False, None, None)),  # TE causal -> plain causal
        ((True, 128, 0), (False, True, 128, 0)),  # sliding window stays local
        ((False, 7, -8), (False, True, 7, -8)),  # non-causal window untouched
        ((False, -1, -1), (False, False, None, None)),  # unbounded stays unbounded
    ],
)
def test_patched_resolver(fake_fa4, args, expected):
    monkey_patch._patch_fa4_causal_unbounded_window()
    assert fake_fa4._resolve_causal_local_window(*args) == expected


def test_patch_is_applied_once(fake_fa4):
    monkey_patch._patch_fa4_causal_unbounded_window()
    patched = fake_fa4._resolve_causal_local_window
    monkey_patch._patch_fa4_causal_unbounded_window()
    assert fake_fa4._resolve_causal_local_window is patched


def test_missing_fa4_is_ignored(monkeypatch):
    monkeypatch.setitem(sys.modules, "flash_attn.cute.interface", None)
    monkey_patch._patch_fa4_causal_unbounded_window()


_NVTE_ATTN_VARS = (
    "NVTE_FLASH_ATTN",
    "NVTE_FUSED_ATTN",
    "NVTE_UNFUSED_ATTN",
    "NVTE_FLASH_ATTN_V2",
    "NVTE_FLASH_ATTN_V3",
    "NVTE_FLASH_ATTN_V4",
)


@pytest.fixture
def clean_nvte_env(monkeypatch):
    # setenv first so monkeypatch records each var (even if absent) and restores it on
    # teardown; set_attention_backend writes os.environ directly.
    for name in _NVTE_ATTN_VARS:
        monkeypatch.setenv(name, "x")
        monkeypatch.delenv(name)


def _attn_config(backend):
    from megatron.core.transformer.enums import AttnBackend

    return types.SimpleNamespace(
        attention_backend=AttnBackend[backend],
        flash_attention_version=None,
        batch_invariant_mode=False,
    )


def _set_via_model_modules(*backends):
    """Call set_attention_backend the way VisionModule then LanguageModule do."""
    from megatron.core.models.common.language_module import language_module
    from megatron.core.models.common.vision_module import vision_module

    callers = (vision_module, language_module)
    for i, backend in enumerate(backends):
        callers[i % 2].set_attention_backend(_attn_config(backend))


@pytest.mark.parametrize("order", [("auto", "flash"), ("flash", "auto")])
def test_auto_sub_model_inherits_explicit_backend(clean_nvte_env, order):
    _set_via_model_modules(*order)
    assert (os.environ["NVTE_FLASH_ATTN"], os.environ["NVTE_FUSED_ATTN"], os.environ["NVTE_UNFUSED_ATTN"]) == (
        "1",
        "0",
        "0",
    )


def test_auto_alone_leaves_env_unset(clean_nvte_env):
    _set_via_model_modules("auto")
    assert all(name not in os.environ for name in _NVTE_ATTN_VARS)


def test_conflicting_explicit_backends_still_fail(clean_nvte_env):
    with pytest.raises(AssertionError, match="NVTE_FLASH_ATTN"):
        _set_via_model_modules("flash", "fused")


def test_attention_backend_patch_is_applied_once():
    from megatron.core.transformer import utils as transformer_utils

    patched = transformer_utils.set_attention_backend
    assert getattr(patched, "_coda_auto_inherits", False)
    monkey_patch._patch_auto_attention_backend_inherits()
    assert transformer_utils.set_attention_backend is patched


@pytest.fixture
def fake_dsv4(monkeypatch):
    """Fake DeepSeek-V4 bridge/config modules recording layout and finalize calls."""
    calls = []

    class FakeDSv4Config:
        def __init__(self, pp, layout=None):
            self.transformer = types.SimpleNamespace(
                pipeline_model_parallel_size=pp, pipeline_model_parallel_layout=layout
            )

        def finalize(self):
            calls.append("finalize")

    def set_layout(cfg):
        calls.append("set_layout")
        cfg.transformer.pipeline_model_parallel_layout = "auto"

    bridge_mod = types.ModuleType("megatron.bridge.models.deepseek.deepseek_v4_bridge")
    bridge_mod.set_deepseek_v4_pipeline_model_parallel_layout = set_layout
    config_mod = types.ModuleType("megatron.bridge.models.deepseek.deepseek_v4_config")
    config_mod.DeepSeekV4ModelConfig = FakeDSv4Config
    monkeypatch.setitem(sys.modules, bridge_mod.__name__, bridge_mod)
    monkeypatch.setitem(sys.modules, config_mod.__name__, config_mod)
    monkey_patch._patch_dsv4_builder_pipeline_layout()
    return FakeDSv4Config, calls


@pytest.mark.parametrize(
    "pp, layout, expected_calls, expected_layout",
    [
        (6, None, ["set_layout", "finalize"], "auto"),  # PP > 1 without layout: auto-set first
        (1, None, ["finalize"], None),  # PP = 1 needs no layout
        (None, None, ["finalize"], None),  # unset PP counts as 1
        (6, "user", ["finalize"], "user"),  # never override a user layout
    ],
)
def test_dsv4_builder_pipeline_layout(fake_dsv4, pp, layout, expected_calls, expected_layout):
    config_cls, calls = fake_dsv4
    cfg = config_cls(pp, layout)
    cfg.finalize()
    assert calls == expected_calls
    assert cfg.transformer.pipeline_model_parallel_layout == expected_layout


def test_dsv4_builder_pipeline_layout_is_applied_once(fake_dsv4):
    config_cls, _ = fake_dsv4
    patched = config_cls.finalize
    monkey_patch._patch_dsv4_builder_pipeline_layout()
    assert config_cls.finalize is patched


# ---------------------------------------------------------------- ERNIE 5.5 quantized checkpoint


class _FakeStateDict(dict):
    def __init__(self, tensors, path):
        super().__init__(tensors)
        self.source = types.SimpleNamespace(path=path)


@pytest.fixture
def ernie_quant_ckpt(tmp_path):
    """A tiny ERNIE-Preview-like checkpoint: fp32 power-of-two ``*_scale_inv`` 128x128 tiles (one
    with a partial last tile, one on a weight without ``.weight``), an MXFP4 expert, a bf16 tensor."""
    import json

    import torch
    from safetensors.torch import save_file

    from coda.utils.quant_formats import QuantScheme, e8m0_to_float

    pytest.importorskip("megatron.bridge.models.ernie.ernie55_bridge")
    g = torch.Generator().manual_seed(0)
    blk, fp4 = QuantScheme("fp8_block", (128, 128)), QuantScheme("mxfp4", (1, 32))
    tensors = {}
    for name, (o, i) in {"l.kv_a.weight": (576, 256), "l.o_group": (256, 128)}.items():
        q, s = blk.quant((torch.randn(o, i, generator=g) * 0.05).bfloat16())
        tensors[name], tensors[name + "_scale_inv"] = q, e8m0_to_float(s)
    q, s = fp4.quant((torch.randn(64, 128, generator=g) * 0.05).bfloat16())
    tensors["l.experts.0.w1.weight"], tensors["l.experts.0.w1.weight_scale_inv"] = q, s
    tensors["l.norm.weight"] = torch.ones(64).bfloat16()
    save_file(tensors, str(tmp_path / "model.safetensors"))
    qcfg = {"quant_method": "fp8", "scale_fmt": "ue8m0", "weight_block_size": [128, 128]}
    with open(tmp_path / "config.json", "w") as f:
        json.dump({"quantization_config": qcfg}, f)
    bridge = types.SimpleNamespace(hf_config=types.SimpleNamespace(quantization_config=qcfg, _name_or_path=str(tmp_path)))
    schemes = {"l.kv_a.weight": blk, "l.o_group": blk, "l.experts.0.w1.weight": fp4}
    return bridge, _FakeStateDict(tensors, str(tmp_path)), schemes


def test_ernie55_quantized_import_export_round_trip(ernie_quant_ckpt):
    import torch
    from megatron.bridge.models.ernie.ernie55_bridge import Ernie55Bridge

    bridge, sd, schemes = ernie_quant_ckpt
    loaded = Ernie55Bridge.maybe_modify_loaded_hf_weight(bridge, {n: n for n in [*schemes, "l.norm.weight"]}, sd)
    for name, scheme in schemes.items():
        assert loaded[name].dtype == torch.bfloat16
        torch.testing.assert_close(loaded[name].float(), scheme.dequant(sd[name], sd[name + "_scale_inv"], torch.float32),
                                   rtol=0, atol=0)
    assert loaded["l.norm.weight"] is sd["l.norm.weight"]
    assert loaded["l.kv_a.weight"].shape == (576, 256)

    out = Ernie55Bridge.maybe_modify_converted_hf_weight(bridge, None, dict(loaded), sd)
    assert set(out) == {*loaded, *(n + "_scale_inv" for n in schemes)}
    for name, scheme in schemes.items():
        q, s = out[name], out[name + "_scale_inv"]
        assert q.dtype == sd[name].dtype and q.shape == sd[name].shape
        assert s.dtype == sd[name + "_scale_inv"].dtype and s.shape == sd[name + "_scale_inv"].shape
        # the exported values are the imported ones (a snapped weight is a fixed point)
        torch.testing.assert_close(scheme.dequant(q, s, torch.float32), loaded[name].float(), rtol=0, atol=0)
    assert out["l.norm.weight"] is loaded["l.norm.weight"]
    # plain-dtype export requested: untouched
    task = types.SimpleNamespace(weight_dtype=torch.bfloat16)
    assert Ernie55Bridge.maybe_modify_converted_hf_weight(bridge, task, dict(loaded), sd) == loaded


def test_ernie55_quantized_import_rejects_bad_scales(ernie_quant_ckpt):
    import torch
    from megatron.bridge.models.ernie.ernie55_bridge import Ernie55Bridge

    bridge, sd, _ = ernie_quant_ckpt
    sd["l.o_group_scale_inv"] = sd["l.o_group_scale_inv"] * 1.5
    with pytest.raises(RuntimeError, match="non-power-of-two"):
        Ernie55Bridge.maybe_modify_loaded_hf_weight(bridge, "l.o_group", sd)
    del sd["l.kv_a.weight_scale_inv"]
    with pytest.raises(RuntimeError, match="without l.kv_a.weight_scale_inv"):
        Ernie55Bridge.maybe_modify_loaded_hf_weight(bridge, "l.kv_a.weight", sd)
    assert sd["l.norm.weight"].dtype == torch.bfloat16


def test_ernie55_key_check_ignores_scale_companions(monkeypatch):
    ernie55_mappings = pytest.importorskip("megatron.bridge.models.ernie.ernie55_mappings")
    captured = []
    monkeypatch.setattr(ernie55_mappings, "check_ernie55_checkpoint_keys",
                        lambda hf_config, hf_keys: captured.append(set(hf_keys)))
    monkey_patch._patch_ernie55_quantized_checkpoint()  # wraps the recorder
    keys = {"m.a.weight", "m.a.weight_scale_inv", "m.o_group", "m.o_group_scale_inv", "m.orphan_scale_inv"}
    ernie55_mappings.check_ernie55_checkpoint_keys({}, keys)
    assert captured == [{"m.a.weight", "m.o_group", "m.orphan_scale_inv"}]
