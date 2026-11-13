"""Build a Megatron model from an HF checkpoint through Megatron-Bridge.

Megatron-Bridge has two build paths:

* provider: ``bridge.to_megatron_provider()`` returns a ``GPTModelProvider`` that is
  both the config and the builder. Most model families use it.
* builder-backed: ``bridge.get_model_config()`` returns a ``ModelConfig`` with a
  nested ``transformer`` config, and ``bridge.get_model(cfg, ...)`` builds it with
  the family's ``ModelBuilder``. Families that only implement this path (ERNIE 5.5)
  raise ``NotImplementedError`` from ``to_megatron_provider()``.

:func:`build_bridge_model` takes the provider path when the family supports it and
falls back to the builder-backed path otherwise.
"""

import logging
from enum import Enum
from typing import Any, Callable, Optional, get_type_hints

import torch
from megatron.bridge import AutoBridge
from megatron.bridge.models import GPTModelProvider

from coda.utils.types import to_torch_dtype

logger = logging.getLogger(__name__)


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


def build_bridge_model(
    bridge: AutoBridge,
    overrides: dict[str, Any],
    *,
    load_weights: bool,
    configure: Callable[[Any], None] | None = None,
    **build_kwargs: Any,
) -> list:
    """Build the distributed model stages for ``bridge``'s HF checkpoint.

    Args:
        bridge: Bridge created with ``AutoBridge.from_hf_pretrained``.
        overrides: Flat ``megatron.model`` settings to apply before building.
        load_weights: Load the HF weights (False when a dist checkpoint is restored later).
        configure: Optional hook run on the finalized provider (provider path) or on the
            overridden model config (builder-backed path) before the model is built.
        **build_kwargs: ``ddp_config``, ``wrap_with_ddp``, ``mixed_precision_wrapper``.

    Returns:
        The model stages (one per virtual pipeline stage).
    """
    try:
        provider = bridge.to_megatron_provider(load_weights=load_weights)
    except NotImplementedError:
        provider = None

    if provider is not None:
        # Apply overrides and finalize via the bridge helper so provider paths
        # that need post-override setup (e.g. DeepSeek-V4 hash MoE auto-setting
        # pipeline_model_parallel_layout when PP > 1) run before finalize().
        provider.apply_overrides_and_finalize(
            overrides=_convert_overrides(overrides, get_type_hints(GPTModelProvider))
        )
        if configure is not None:
            configure(provider)
        return provider.provide_distributed_model(**build_kwargs)

    model_config = bridge.get_model_config()
    hints = {**get_type_hints(type(model_config.transformer)), **get_type_hints(type(model_config))}
    # Flat assignment routes each field to the outer config or its nested
    # transformer config and raises on names neither declares.
    for k, v in _convert_overrides(overrides, hints).items():
        setattr(model_config, k, v)
    if configure is not None:
        configure(model_config)
    logger.info(f"Building {type(model_config).__name__} through the builder-backed Megatron-Bridge path")
    # get_model finalizes the config and loads the HF weights before wrapping.
    return bridge.get_model(model_config, load_weights=load_weights, **build_kwargs)
