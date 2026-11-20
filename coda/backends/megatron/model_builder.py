"""Apply ``megatron.model`` overrides to a Megatron-Bridge builder config.

Model construction goes through the builder-backed path directly at the call site:
``bridge.get_model_config()`` returns a ``ModelConfig`` with a nested ``transformer``
config, :func:`apply_model_overrides` coerces and applies the yaml overrides, and
``bridge.get_model(cfg, ...)`` builds it. Every model family LoongSage uses (Qwen3,
ERNIE 5.5, DeepSeek-V4.1) supports this path.
"""

from enum import Enum
from typing import Any, Optional, get_type_hints

import torch

from coda.utils.types import to_torch_dtype


def _convert_overrides(overrides: dict[str, Any], hints: dict[str, Any]) -> dict[str, Any]:
    """Turn yaml strings into the dtypes / Enums the config fields declare."""
    converted = {}
    for k, v in overrides.items():
        hint = hints.get(k)
        if hint is torch.dtype or hint is Optional[torch.dtype]:
            v = to_torch_dtype(v)
        elif isinstance(hint, type) and issubclass(hint, Enum) and isinstance(v, str):
            # e.g. attention_backend: flash -> AttnBackend.flash. Megatron's enums
            # are plain Enum (no str mixin), so a raw string silently compares
            # unequal to every member instead of failing loudly.
            v = hint[v]
        converted[k] = v
    return converted


def apply_model_overrides(model_config: Any, overrides: dict[str, Any]) -> None:
    """Apply flat ``megatron.model`` overrides to a builder-backed ``ModelConfig``.

    Flat assignment routes each field to the outer config or its nested transformer
    config and raises on names neither declares. Call between ``bridge.get_model_config()``
    and ``bridge.get_model()``.
    """
    hints = {**get_type_hints(type(model_config.transformer)), **get_type_hints(type(model_config))}
    for k, v in _convert_overrides(overrides, hints).items():
        setattr(model_config, k, v)

    # DeepSeek-V4 hash-routed MoE needs an explicit pipeline_model_parallel_layout at PP > 1
    # (the hash layers must co-locate with the embedding on the first stage). The provider path
    # auto-set this in ModelProvider.apply_overrides_and_finalize; the builder path's finalize
    # does not, so port it here. Only DeepSeek-V4 (dsv4_hybrid); never overrides a user layout.
    transformer = getattr(model_config, "transformer", model_config)
    if (
        getattr(transformer, "experimental_attention_variant", None) == "dsv4_hybrid"
        and (getattr(model_config, "pipeline_model_parallel_size", 1) or 1) > 1
        and getattr(model_config, "pipeline_model_parallel_layout", None) is None
    ):
        from megatron.bridge.models.deepseek.deepseek_v4_bridge import (
            set_deepseek_v4_pipeline_model_parallel_layout,
        )

        set_deepseek_v4_pipeline_model_parallel_layout(model_config)

