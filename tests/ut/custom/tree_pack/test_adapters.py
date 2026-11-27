"""Adapter coverage: every sequence mixer must be matched, or the model is rejected."""
from __future__ import annotations

import pytest

pytest.importorskip("megatron", reason="Megatron-Core is not installed")
pytest.importorskip("transformer_engine")

from megatron.core.extensions.transformer_engine import TEDotProductAttention  # noqa: E402
from megatron.core.ssm.gated_delta_net.gdn import GatedDeltaNet  # noqa: E402
from megatron.core.transformer.attention import SelfAttention  # noqa: E402

import coda.custom.tree_pack.megatron.adapters as adapters  # noqa: E402


def _bare(cls, **attrs):
    """An instance without running ``__init__`` (only its type and attributes matter)."""
    obj = cls.__new__(cls)
    for k, v in attrs.items():
        object.__setattr__(obj, k, v)
    return obj


class _TEAttnSubclass(TEDotProductAttention):
    pass


class _OtherMixer:
    pass


def _match(monkeypatch, mixers):
    monkeypatch.setattr(adapters, "sequence_mixers",
                        lambda chunks: [(f"layers.{i}.self_attention", m) for i, m in enumerate(mixers)])
    return adapters.match_adapters([])


def test_attention_and_gdn_layers_are_covered(monkeypatch):
    attn = _bare(SelfAttention, core_attention=_bare(TEDotProductAttention))
    used, reason = _match(monkeypatch, [_bare(GatedDeltaNet), attn, _bare(GatedDeltaNet)])
    assert reason is None
    assert sorted(a.name for a in used) == ["attention", "gdn"]


@pytest.mark.parametrize("mixer", [
    _OtherMixer(),
    _bare(SelfAttention, core_attention=_bare(_TEAttnSubclass)),   # forward may be overridden
])
def test_uncovered_mixer_rejects_the_model(monkeypatch, mixer):
    used, reason = _match(monkeypatch, [_bare(GatedDeltaNet), mixer])
    assert used == []
    assert type(mixer).__name__ in reason and "layers.1.self_attention" in reason


def test_adapter_names_are_unique():
    with pytest.raises(ValueError):
        adapters.register_layer_adapter(adapters.LayerAdapter("gdn", lambda m: False, lambda o: None))
