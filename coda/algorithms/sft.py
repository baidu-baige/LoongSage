"""Supervised fine-tuning (SFT) expressed as a degenerate RL algorithm.

SFT reuses the RL training path unchanged: the ``sft`` policy loss is the
per-token negative log-likelihood, and the ``sft`` advantage is a constant 1 so
nothing reward-related is needed.  Mask weighting and normalisation are left to
``_aggregate_loss`` in the Megatron loss function; with ``loss_agg_mode:
token-mean`` the result is the masked token-level cross-entropy averaged over
every unmasked token of the global batch.

Select both with ``algorithm.policy_loss: sft`` and
``algorithm.advantage_estimator: sft`` (``run_mode: sft`` enforces this).
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig

from coda.algorithms.registry import register_advantage, register_policy_loss
from coda.utils.types import RolloutBatch


@register_policy_loss("sft")
def sft_loss(
    config: DictConfig,
    old_log_prob: list[torch.Tensor] | None,
    log_prob: list[torch.Tensor],
    advantages: list[torch.Tensor] | None,
    loss_masks: list[torch.Tensor],
    **kwargs,
) -> tuple[list[torch.Tensor], dict[str, float]]:
    """Per-token negative log-likelihood; ``old_log_prob`` and ``advantages`` are ignored."""
    return [-lp for lp in log_prob], {}


@register_advantage("sft")
def sft_advantage(
    config: DictConfig,
    rollout_data: RolloutBatch,
) -> list[torch.Tensor]:
    """Constant advantage of 1: SFT has no reward signal to normalise."""
    device = rollout_data["loss_masks"][0].device if rollout_data["loss_masks"] else None
    return [
        torch.ones(length, dtype=torch.float32, device=device)
        for length in rollout_data["response_lengths"]
    ]
