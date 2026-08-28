"""Off-Policy Sequence Masking (OPSM).

Computes a per-trajectory mask that zeros gradients for sequences satisfying
``advantage < 0`` AND ``seq_kl > delta``, preventing catastrophic updates
that push the policy further away from already-stale negative trajectories.

The returned masks are intended to be multiplied into the per-token policy
loss *after* ``compute_policy_loss`` so that filtered sequences contribute
zero to the numerator while the loss-aggregation denominator (built from
the original ``loss_masks``) is left unchanged. This matches the canonical
"drop the gradient" semantics of OPSM and keeps downstream metrics like
``pg_clipfrac`` / ``ppo_kl`` defined over the full pre-OPSM batch.

Reference: slime/utils/ppo_utils.py:compute_opsm_mask
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig


def compute_opsm_mask(
    config: DictConfig,
    log_probs: list[torch.Tensor],
    old_log_probs: list[torch.Tensor],
    advantages: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
) -> tuple[list[torch.Tensor], dict[str, float]]:
    """Compute Off-Policy Sequence Masking (OPSM) per-trajectory mask.

    Args:
        config: with sub-config field ``config.algorithm.opsm.delta``.
        log_probs: Per-trajectory current-policy log-probs (forward output).
        old_log_probs: Per-trajectory old-policy log-probs.
        advantages: Per-trajectory advantage tensors. For sequence-level advantage
            (e.g. GRPO) every token shares the same value.
        loss_masks: Per-trajectory binary token masks (1 = include in loss).

    Returns:
        ``(opsm_mask_list, metrics)``

        * ``opsm_mask_list`` -- list of per-trajectory masks (1 = keep, 0 = drop)
          shaped like the corresponding ``advantages`` entry. Multiply
          element-wise into the per-token policy loss.
        * ``metrics`` -- ``train/opsm_clipfrac`` accumulating per-sequence drop
          fractions; for sequence-level advantage this collapses to a count
          of fully-dropped sequences.
    """
    delta = float(config.algorithm.opsm.delta)
    device = advantages[0].device

    opsm_mask_list: list[torch.Tensor] = []
    opsm_clipfrac = torch.tensor(0.0, device=device)

    for log_prob, old_log_prob, advantage, loss_mask in zip(
        log_probs, old_log_probs, advantages, loss_masks, strict=False
    ):
        # Sequence-level KL.
        seq_kl = ((old_log_prob - log_prob) * loss_mask).sum() / torch.clamp_min(loss_mask.sum(), 1)

        # Per-token drop mask: 1 where (advantage < 0 AND seq_kl > delta), else 0.
        mask = ((advantage < 0) & (seq_kl > delta)).float()
        opsm_clipfrac += mask.sum() / torch.clamp_min(loss_mask.sum(), 1)

        # Keep mask: 1 - drop mask.
        opsm_mask_list.append(1 - mask)

    metrics = {"train/opsm_clipfrac": opsm_clipfrac.item()}
    return opsm_mask_list, metrics
