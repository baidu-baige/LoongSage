"""IS correction for train-inference mismatch.

Applies importance-sampling weights (π_old / π_rollout) to per-token losses,
either by clipping the weights into a safe range or by masking out-of-bound
tokens/sequences and removing them from the loss denominator.
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig


_VALID_LEVELS = ("token", "sequence", "geometric")
_VALID_ACTIONS = ("clip", "mask")


def _compute_correction_weights(
    old_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    level: str,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Compute per-trajectory IS correction weights at the requested level.

    Args:
        old_log_probs: Per-trajectory response log-probs from the behavior policy.
        rollout_log_probs: Per-trajectory response log-probs from the rollout policy.
        level: Granularity of the weight — ``"token"``, ``"sequence"``, or
            ``"geometric"``.
        loss_masks: Per-trajectory binary token masks (1 = include in loss).

    Returns:
        Tuple of (weights, nan_masks).
        weights: List of per-trajectory weight tensors, each of shape ``(response_length,)``.
        nan_masks: List of per-trajectory float tensors indicating clamped tokens.
    """
    assert level in _VALID_LEVELS, (
        f"Invalid is_correction.level={level!r}, must be one of {_VALID_LEVELS}"
    )

    weights: list[torch.Tensor] = []
    nan_masks: list[torch.Tensor] = []

    for old_lp, rollout_lp, mask_i in zip(old_log_probs, rollout_log_probs, loss_masks):
        delta = old_lp - rollout_lp  # (resp_len,)
        nan_masks.append(((delta < -20.0) | (delta > 20.0)).float())
        delta = torch.clamp(delta, min=-20.0, max=20.0)
        if level == "token":
            w = delta.exp()
        elif level == "sequence":
            # Full sequence probability ratio: exp(Σ Δlogp), masked
            w = (delta * mask_i).sum().exp().expand_as(delta)
        else:
            # Geometric mean of token ratios: exp(mean(Δlogp)), masked
            num_valid = mask_i.sum().clamp(min=1.0)
            w = ((delta * mask_i).sum() / num_valid).exp().expand_as(delta)
        weights.append(w)
    return weights, nan_masks


def apply_is_correction(
    config: DictConfig,
    loss_masks: list[torch.Tensor],
    old_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_agg_mode: str = "token-mean",
) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, float]]:
    """Compute IS correction weights and optionally update loss masks.

    Args:
        config: ``config.algorithm.is_correction`` sub-config with fields
            ``action``, ``level``, ``lower_bound``, ``upper_bound``.
        loss_masks: Per-trajectory binary token masks (1 = include in loss).
        old_log_probs: Behavior-policy log-probs (used as reference in policy
            loss IS ratio).
        rollout_log_probs: Original rollout-policy log-probs.
        loss_agg_mode: How to aggregate per-token values — ``"token-mean"`` or
            ``"seq-mean-token-mean"``.

    Returns:
        ``(is_weights, updated_loss_masks, metrics)``

        * ``is_weights`` – list of clamped IS weight tensors, one per trajectory.
          The caller is responsible for multiplying these into per-token losses.
        * For ``action="clip"``: loss_masks are unchanged.
        * For ``action="mask"``: out-of-bound tokens (token level) or sequences
          (sequence/geometric level) are additionally zeroed in loss_masks,
          excluding them from the loss denominator.
        * ``metrics`` contains raw sums; the caller is responsible for dividing
          by the appropriate count to obtain means.
    """
    # Always compute is_approx_k3_kl as a monitoring metric
    num_tokens = torch.tensor(0.0, device=loss_masks[0].device)
    num_seqs = torch.tensor(0.0, device=loss_masks[0].device)
    is_approx_k3_kl = torch.tensor(0.0, device=loss_masks[0].device)
    for mask_i, old_lp, rollout_lp in zip(loss_masks, old_log_probs, rollout_log_probs):
        num_tokens += mask_i.sum()
        if mask_i.any():
            num_seqs += 1
        ratio = old_lp - rollout_lp
        per_token_kl = (torch.exp(ratio) - ratio - 1.0) * mask_i
        if loss_agg_mode == "seq-mean-token-mean":
            is_approx_k3_kl += per_token_kl.sum() / torch.clamp_min(mask_i.sum(), 1)
        else:
            is_approx_k3_kl += per_token_kl.sum()

    metrics: dict[str, float] = {
        "train/is_approx_k3_kl": is_approx_k3_kl.item(),
        "num_tokens": num_tokens.item(),
        "num_seqs": num_seqs.item(),
    }

    if not config.enable:
        # IS correction disabled — return unit weights and original masks
        is_weights = [torch.ones_like(m) for m in loss_masks]
        return is_weights, loss_masks, metrics

    # Full IS correction path
    action = config.action
    level = config.level
    lower = float(config.lower_bound)
    upper = float(config.upper_bound)

    assert action in _VALID_ACTIONS, (
        f"Invalid is_correction.action={action!r}, must be one of {_VALID_ACTIONS}"
    )

    weights, nan_masks = _compute_correction_weights(old_log_probs, rollout_log_probs, loss_masks, level)

    new_masks: list[torch.Tensor] = []
    is_clip_sum = torch.tensor(0.0, device=loss_masks[0].device)
    nan_sum = torch.tensor(0.0, device=loss_masks[0].device)
    for mask_i, w_i, nan_i in zip(loss_masks, weights, nan_masks):
        keep_f = ((w_i >= lower) & (w_i <= upper)).float()
        new_mask = mask_i * keep_f
        new_masks.append(new_mask)
        is_clip_sum += mask_i.sum() - new_mask.sum()
        nan_sum += (nan_i * mask_i).sum()

    is_weights = [w.clamp(lower, upper) for w in weights]

    metrics["train/is_clip_ratio"] = is_clip_sum.item()
    metrics["train/is_nan_ratio"] = nan_sum.item()

    # action == "mask": zero out mask for Out Of Bounds tokens/sequences so they are
    # excluded from the loss denominator;
    return is_weights, new_masks if action == "mask" else loss_masks, metrics
