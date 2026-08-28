"""Second-Moment Trust Policy Optimization (M2PO).

Masks response tokens with the largest squared log-ratio ``log(π_old / π_rollout)²``
until the average second moment of the remaining valid tokens falls at or below
a configured ``threshold``. This uses the same policy gap as IS correction but,
instead of clamping / dropping individual off-bound tokens, iteratively removes
the worst-offending tokens across the whole batch to bound the aggregate
second moment.
"""

from __future__ import annotations

import logging

import torch
from omegaconf import DictConfig


logger = logging.getLogger(__name__)


def apply_m2po_masking(
    config: DictConfig,
    loss_masks: list[torch.Tensor],
    old_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
) -> tuple[list[torch.Tensor], dict[str, float]]:
    """Apply M2PO second-moment masking to off-policy response tokens.

    M2PO uses the same policy gap as IS correction, ``π_old / π_rollout``.
    Tokens with the largest squared log-ratio are masked until the average
    second moment of the remaining valid tokens falls to or below ``threshold``.
    """
    metrics: dict[str, float] = {}
    threshold = float(config.threshold)
    flat_m2: list[torch.Tensor] = []
    trajectory_indices: list[torch.Tensor] = []
    token_indices: list[torch.Tensor] = []
    m2_sum = torch.tensor(0.0, device=loss_masks[0].device)
    valid_tokens = 0
    skipped_trajectories = 0

    for trajectory_idx, (old_lp, rollout_lp, mask_i) in enumerate(
        zip(old_log_probs, rollout_log_probs, loss_masks)
    ):
        valid_mask = mask_i.bool()
        if not valid_mask.any():
            skipped_trajectories += 1
            continue

        delta = old_lp - rollout_lp
        m2 = delta * delta
        valid_m2 = m2[valid_mask]
        valid_tokens += int(valid_m2.numel())
        flat_m2.append(valid_m2)

        valid_token_indices = torch.where(valid_mask)[0]
        trajectory_indices.append(
            torch.full_like(valid_token_indices, trajectory_idx, dtype=torch.long)
        )
        token_indices.append(valid_token_indices)
        m2_sum += valid_m2.sum()

    metrics["m2"] = m2_sum.item()
    metrics["clip_count"] = 0.0
    metrics["valid_tokens"] = float(valid_tokens)

    if not flat_m2:
        logger.info("[m2po] no valid tokens found, keep original loss masks")
        return loss_masks, metrics

    all_m2 = torch.cat(flat_m2)
    all_trajectory_indices = torch.cat(trajectory_indices)
    all_token_indices = torch.cat(token_indices)

    sorted_m2, sorted_indices = torch.sort(all_m2, descending=True)
    suffix_sums = sorted_m2.flip(0).cumsum(0).flip(0)
    counts = torch.arange(
        sorted_m2.numel(), 0, -1, device=sorted_m2.device, dtype=sorted_m2.dtype
    )
    avg_m2_suffix = suffix_sums / counts

    below_threshold = torch.where(avg_m2_suffix <= threshold)[0]
    if len(below_threshold) > 0:
        num_to_mask = below_threshold[0].item()
    else:
        logger.warning(
            "[m2po] all %d valid tokens exceed threshold=%s; "
            "masking all but the smallest-m2 token",
            sorted_m2.numel(),
            threshold,
        )
        num_to_mask = sorted_m2.numel() - 1

    if num_to_mask <= 0:
        logger.info(
            "[m2po] avg_m2 already below threshold, keep all %d valid tokens",
            sorted_m2.numel(),
        )
        return loss_masks, metrics

    new_masks = [mask_i.clone() for mask_i in loss_masks]
    masked_indices = sorted_indices[:num_to_mask]
    for trajectory_idx, token_idx in zip(
        all_trajectory_indices[masked_indices].tolist(),
        all_token_indices[masked_indices].tolist(),
    ):
        new_masks[trajectory_idx][token_idx] = 0

    metrics["clip_count"] = float(num_to_mask)
    logger.info(
        "[m2po] masked_tokens=%d valid_tokens=%d threshold=%s m2_sum=%.6f",
        num_to_mask,
        sorted_m2.numel(),
        threshold,
        metrics["m2"],
    )
    return new_masks, metrics
