"""Pluggable micro-batch packing for the Megatron backend.

A packer decides how the Segment rows of a micro-batch become model inputs. The default
``rows`` packer is the built-in behaviour (one packed sequence per row). Other packers
register with ``@register_packer("name")`` in ``coda/custom/`` (see
``docs/zh/custom-extensions.md``) and are selected with ``trainer.packer``;
``trainer.packer_options`` is passed to them as-is.

Every hook of the base class returns ``None`` (or does nothing), which tells the caller
to run its built-in row path. Kept free of Megatron imports so the controller can use it.
"""

from __future__ import annotations

import logging
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Callable

from omegaconf import DictConfig

from coda.utils.registry import Registry

logger = logging.getLogger(__name__)

PACKER_REGISTRY = Registry("micro_batch_packer")
register_packer = PACKER_REGISTRY.register


@dataclass
class PackPlan:
    """How to cut each mini-batch into micro-batches.

    ``item_rows[i][j]`` are the rows of item ``j`` in mini-batch ``i``; items are
    bin-packed by ``item_costs`` under ``max_budget`` and, when ``lm_budget`` is set, by
    ``item_lms`` under ``lm_budget``. Under VPP the micro-batch count is rounded up.
    """
    item_rows: list[list[list[int]]]
    item_costs: list[list[int]]
    max_budget: int
    item_lms: list[list[int]] | None = None
    lm_budget: int | None = None


@dataclass
class PackedInputs:
    """Model inputs of one micro-batch.

    ``packed_seq_params`` may carry ``row_outputs`` (see ``row_outputs``) when the logits
    are not laid out one row after another.
    """
    tokens: Any
    packed_seq_params: Any
    position_ids: Any
    replay_batch: dict                       # batch handed to setup_routing_replay
    forward_ctx: AbstractContextManager      # wraps the model forward


class MicroBatchPacker:
    """Built-in row packing; subclasses override the hooks they need."""

    # Per-group DP loads for split_traj_group_by_dp (None: built-in loads).
    dp_loads: Callable[[list], list[int]] | None = None

    def __init__(self, config: DictConfig):
        self.config = config

    @classmethod
    def unsupported_reason(cls, config: DictConfig) -> str | None:
        """Why this packer cannot run with ``config`` (None if it can)."""
        return None

    def setup(self, model) -> str | None:
        """Called once the model is built; a returned reason falls back to rows."""
        return None

    def applies(self, *, pg_policy=None) -> bool:
        """Whether a forward-only pass with this PG-KL policy uses the packer."""
        return True

    def extra_keys(self) -> list[str]:
        """Rollout keys ``pack`` needs besides the row path's."""
        return []

    def plan(self, rollout_data: dict, mini_batch_ranges: list[tuple[int, int]],
             max_budget: int) -> PackPlan | None:
        return None

    def pack(self, batch: dict, model_chunk, mb_key: int, *, training: bool,
             routed_rows: Callable[[], list]) -> PackedInputs | None:
        """``mb_key`` is the same for every VPP chunk of a micro-batch."""
        return None

    def step_metrics(self, step: int, metrics: dict) -> None:
        """Add packer metrics after a train step (``metrics`` is empty off the last PP stage)."""


register_packer("rows")(MicroBatchPacker)


def row_outputs(packed_seq_params):
    """Per-row ``log_probs(logits, temperature)`` / ``entropy(logits, temperature,
    with_grad)`` attached by a packer, or None for the row layout."""
    return getattr(packed_seq_params, "row_outputs", None)


def get_packer_class(name: str) -> type[MicroBatchPacker]:
    """Look ``name`` up in the packer registry (custom packers register when ``coda/custom``
    is imported, before the trainer and the workers resolve it)."""
    if name not in PACKER_REGISTRY:
        raise ValueError(
            f"Unknown trainer.packer {name!r}; registered: {PACKER_REGISTRY.keys()} "
            "(a coda/custom module that failed to import is logged as 'Skipped custom module')")
    return PACKER_REGISTRY.get(name)


def build_packer(config: DictConfig, model=None) -> MicroBatchPacker:
    """The packer selected by ``trainer.packer``, or rows when it cannot run (with
    ``model``, also when its ``setup`` rejects the built model)."""
    name = str(config.trainer.get("packer", "rows"))
    cls = get_packer_class(name)
    reason = cls.unsupported_reason(config)
    packer = cls(config) if reason is None else None
    if packer is not None and model is not None:
        reason = packer.setup(model)
    if reason:
        logger.warning("[packer] %s disabled, falling back to rows: %s", name, reason)
        return MicroBatchPacker(config)
    return packer
