"""Built-in policy-loss implementations.

Registers GRPO and GSPO clipped surrogate losses into the policy-loss registry.
"""

from __future__ import annotations

import torch
from omegaconf import DictConfig

from coda.algorithms.registry import register_policy_loss, get_policy_loss


@register_policy_loss("grpo")
def grpo_loss(
    config: DictConfig,
    old_log_prob: list[torch.Tensor],
    log_prob: list[torch.Tensor],
    advantages: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs,
) -> tuple[list[torch.Tensor], dict[str, float]]:
    """GRPO clipped surrogate loss (per-token).

    Returns a list of per-token loss tensors (one per trajectory) and a metrics dict.
    """
    clip_ratio_low = config.clip_ratio_low
    clip_ratio_high = config.clip_ratio_high
    clip_ratio_c = config.clip_ratio_c

    response_lengths = [len(seq) for seq in old_log_prob]
    old_log_prob = torch.cat(old_log_prob)
    log_prob = torch.cat(log_prob)
    advantages = torch.cat(advantages)
    loss_masks = torch.cat(loss_masks)

    # Importance-sampling ratio
    neg_kl = log_prob - old_log_prob
    neg_kl_clamp_mask = ((neg_kl < -20.0) | (neg_kl > 20.0)).float()
    neg_kl = torch.clamp(neg_kl, min=-20.0, max=20.0)
    ratio = torch.exp(neg_kl)

    # Clipped surrogate
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high) * advantages
    surr = torch.min(surr1, surr2)
    # Dual-clip: when advantage < 0, clip again with clip_ratio_c as lower bound
    dual_clip_val = clip_ratio_c * advantages
    surr3 = torch.where(advantages < 0, torch.max(surr, dual_clip_val), surr)
    per_token_loss = -surr3

    per_token_loss = per_token_loss.split(response_lengths)

    # Metrics
    with torch.no_grad():
        neg_kl_clamp_sum = (neg_kl_clamp_mask * loss_masks).sum().item()
        approx_kl = ((-neg_kl) * loss_masks).sum().item()
        clip_sum = (torch.gt(surr1, surr2).float() * loss_masks).sum().item()
        dual_clip_sum = (torch.gt(surr3, surr).float() * loss_masks).sum().item()

    metrics = {
        "train/approx_kl": approx_kl,
        "train/clip_ratio": clip_sum,
        "train/dual_clip_ratio": dual_clip_sum,
        "train/nan_ratio": neg_kl_clamp_sum,
    }

    return per_token_loss, metrics


@register_policy_loss("gspo")
def gspo_loss(
    config: DictConfig,
    old_log_prob: list[torch.Tensor],
    log_prob: list[torch.Tensor],
    advantages: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs,
) -> tuple[list[torch.Tensor], dict[str, float]]:
    """GSPO sequence-level clipped surrogate loss (per-token).

    Uses a sequence-level importance ratio with per-token gradient routing.
    See https://arxiv.org/pdf/2507.18071 for details.

    Returns a list of per-token loss tensors (one per trajectory) and a metrics dict.
    """
    clip_ratio_low = config.clip_ratio_low
    clip_ratio_high = config.clip_ratio_high

    per_token_losses = []
    all_neg_kl: list[torch.Tensor] = []
    all_neg_kl_clamp_mask: list[torch.Tensor] = []
    all_clipped: list[torch.Tensor] = []

    for lp, olp, adv, mask_i in zip(log_prob, old_log_prob, advantages, loss_masks):
        raw_neg_kl = lp - olp  # (resp_len_i,)
        neg_kl_clamp_mask = ((raw_neg_kl < -20.0) | (raw_neg_kl > 20.0)).float()
        neg_kl = torch.clamp(raw_neg_kl, min=-20.0, max=20.0)

        # Sequence-level ratio in log space (scalar)
        seq_length = mask_i.sum().clamp(min=1)
        neg_kl_seq = (neg_kl * mask_i).sum() / seq_length

        # Combined token-level ratio: gradient flows only through lp, not through neg_kl_seq
        log_seq_ratio = lp - lp.detach() + neg_kl_seq.detach()
        log_seq_ratio = torch.clamp(log_seq_ratio, max=10.0)
        seq_ratio = torch.exp(log_seq_ratio)

        surr1 = seq_ratio * adv
        surr2 = torch.clamp(seq_ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high) * adv
        per_token_losses.append(-torch.min(surr1, surr2))

        all_neg_kl.append(neg_kl.detach())
        all_neg_kl_clamp_mask.append(neg_kl_clamp_mask.detach())
        all_clipped.append(torch.gt(surr1, surr2).detach())

    with torch.no_grad():
        neg_kl_flat = torch.cat(all_neg_kl)
        neg_kl_clamp_mask_flat = torch.cat(all_neg_kl_clamp_mask)
        clipped_flat = torch.cat(all_clipped)
        mask_flat = torch.cat(loss_masks)
        approx_kl = ((-neg_kl_flat) * mask_flat).sum().item()
        clip_sum = (clipped_flat.float() * mask_flat).sum().item()
        neg_kl_clamp_sum = (neg_kl_clamp_mask_flat * mask_flat).sum().item()


    metrics = {
        "train/approx_kl": approx_kl,
        "train/clip_ratio": clip_sum,
        "train/nan_ratio": neg_kl_clamp_sum,
    }

    return per_token_losses, metrics

def compute_policy_loss(
    config: DictConfig,
    old_log_prob: list[torch.Tensor],
    log_prob: list[torch.Tensor],
    advantages: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs,
) -> tuple[list[torch.Tensor], dict[str, float]]:
    """Dispatch to the registered policy loss function specified by ``config.policy_loss``.

    Args:
        config: Algorithm config. Must contain ``policy_loss`` (str) identifying
            the registered loss (e.g. ``"grpo"``, ``"gspo"``), plus any
            loss-specific hyper-parameters (e.g. ``clip_ratio_low``).
        old_log_prob: Per-trajectory log-probabilities from the old/rollout policy.
            Each tensor has shape ``(response_len_i,)``.
        log_prob: Per-trajectory log-probabilities from the current policy (with gradient).
            Each tensor has shape ``(response_len_i,)``.
        advantages: Per-token advantage estimates, one tensor per trajectory with
            shape ``(response_len_i,)``.
        loss_masks: per-trajectory binary token masks (1 = include).
            Used by loss functions to restrict metric
            computation (e.g. ``approx_kl``, ``clip_ratio``) and, for GSPO,
            the sequence-level ratio to valid tokens only.
        **kwargs: Extra keyword arguments forwarded verbatim to the loss function.

    Returns:
        ``(per_token_loss, metrics)`` where

        * ``per_token_loss`` – list of per-token loss tensors, one per trajectory,
          each with shape ``(response_len_i,)`` and gradient attached.
        * ``metrics`` – flat dict of scalar monitoring values (e.g.
          ``approx_kl``, ``clip_ratio``).  Values are **sums** (not
          averages); the caller is responsible for dividing by the
          appropriate count to obtain means.
    """
    user_loss_fn = get_policy_loss(config.policy_loss)
    return user_loss_fn(config, old_log_prob, log_prob, advantages, loss_masks=loss_masks, **kwargs)
