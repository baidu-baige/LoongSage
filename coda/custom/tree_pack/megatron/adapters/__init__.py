"""Layer adapters: run a model's sequence-mixing layers on the tree layout.

Everything outside the sequence mixers (MLPs, norms, projections, MoE) is per-token and
runs on the node buffer unchanged. Each transformer layer's ``self_attention`` module mixes
tokens, so every one of them must be matched by an adapter, or tree packing is not used for
the model. New model families add an adapter module and register it here.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LayerAdapter:
    """``matches(module)``: whether ``install`` makes ``module`` run correctly on the tree
    layout; ``install(options)``: idempotent monkeypatch (``options`` = packer options)."""
    name: str
    matches: Callable[[torch.nn.Module], bool]
    install: Callable[[Mapping], None]


_ADAPTERS: dict[str, LayerAdapter] = {}


def register_layer_adapter(adapter: LayerAdapter) -> LayerAdapter:
    if adapter.name in _ADAPTERS:
        raise ValueError(f"layer adapter {adapter.name!r} is already registered")
    _ADAPTERS[adapter.name] = adapter
    return adapter


def sequence_mixers(model_chunks: Sequence[torch.nn.Module]) -> list[tuple[str, torch.nn.Module]]:
    """``(name, module)`` of the ``self_attention`` of every transformer layer."""
    from megatron.core.transformer.transformer_layer import TransformerLayer

    return [(f"{name}.self_attention", layer.self_attention)
            for chunk in model_chunks for name, layer in chunk.named_modules()
            if isinstance(layer, TransformerLayer)]


def match_adapters(model_chunks: Sequence[torch.nn.Module]) -> tuple[list[LayerAdapter], str | None]:
    """Adapters covering every sequence mixer of the model, or ``([], reason)``."""
    used: dict[str, LayerAdapter] = {}
    for name, mixer in sequence_mixers(model_chunks):
        adapter = next((a for a in _ADAPTERS.values() if a.matches(mixer)), None)
        if adapter is None:
            return [], f"no tree adapter for {type(mixer).__name__} ({name})"
        used[adapter.name] = adapter
    return list(used.values()), None


from coda.custom.tree_pack.megatron.adapters import attention, gdn  # noqa: E402,F401  (register built-ins)
