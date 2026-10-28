"""Built-in advantage estimator implementations.

Registers the GRPO and GAE (PPO) advantage estimators into the advantage
registry, and provides the top-level ``compute_advantages_and_returns``
dispatcher, which also applies the optional step-wide masked whitening.

Estimator contract: ``fn(config.algorithm, rollout_data) -> (advantages, returns)``,
both per-Segment tensor lists of shape ``(response_length,)``. ``returns`` are
the critic's value targets; critic-free estimators return their advantages
again.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence

from omegaconf import DictConfig

from coda.utils.types import RolloutBatch
from coda.algorithms.registry import register_advantage, get_advantage

logger = logging.getLogger(__name__)


_VALID_ADVANTAGE_NORMS = ("none", "group_mean", "group_zscore")


def _group_normalize(
    reward_tensor: torch.Tensor,
    prompt_ids: list[str],
    divide_std: bool,
) -> list[float]:
    """Normalize rewards within each prompt group (group-level, vectorised)."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, pid in enumerate(prompt_ids):
        groups[pid].append(i)

    group_sizes = [len(v) for v in groups.values()]
    assert len(set(group_sizes)) == 1, (
        f"All prompt groups must have the same size, got {group_sizes}"
    )

    indices_2d = torch.tensor(list(groups.values()), dtype=torch.long)
    group_rewards = reward_tensor[indices_2d]  # (num_prompts, group_size)

    centered = group_rewards - group_rewards.mean(dim=1, keepdim=True)

    if divide_std:
        group_size = group_rewards.size(1)
        if group_size >= 2:
            std = group_rewards.std(dim=1, unbiased=True, keepdim=True)
            centered = centered / (std + 1e-6)
        else:
            logger.warning(
                "Skipping std normalization (group_zscore) because "
                "num_trajectories_per_prompt=%d < 2; falling back to group_mean.",
                group_size,
            )

    advantages_tensor = torch.empty(len(reward_tensor), dtype=torch.float32)
    advantages_tensor[indices_2d.flatten()] = centered.flatten()
    return advantages_tensor.tolist()


@register_advantage("grpo")
def grpo_advantage(
    config: DictConfig,
    rollout_data: RolloutBatch,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """GRPO advantage: per-prompt-group reward normalisation.

    Controlled by ``config.advantage_norm_mode``:
    * ``"none"``:           raw rewards, no normalisation.
    * ``"group_mean"``:     subtract per-prompt-group mean (x - mean).
    * ``"group_zscore"``:   per-prompt-group z-score ((x - mean) / std).

    Step-wide normalisation is the separate ``use_masked_whiten`` step applied
    by :func:`compute_advantages_and_returns`.

    Rows are per-Segment. Advantage is trajectory-level, so rows are first
    deduplicated to their parent trajectory (via ``dp_local_traj_idx``, which is
    a dense 0-based index assigned in trajectory order by ``put_dp_shards_to_ray``),
    normalised at trajectory granularity, then the per-trajectory scalar is
    broadcast back to every Segment row's response tokens.

    Returns:
        ``(advantages, advantages)``: per-Segment tensors of shape
        ``(response_length,)``; GRPO has no critic, so returns equal advantages.
    """
    rewards = rollout_data["rewards"]
    prompt_ids = rollout_data["prompt_id"]
    dp_local_traj_idxs = rollout_data["dp_local_traj_idx"]
    response_lengths = rollout_data["response_lengths"]

    # Deduplicate Segment rows to their trajectory. dp_local_traj_idx is a
    # dense 0-based index (see put_dp_shards_to_ray), so it doubles as the
    # position in the deduplicated traj_rewards/traj_prompt_ids arrays.
    num_traj = dp_local_traj_idxs[-1] + 1
    traj_rewards: list[float] = [None] * num_traj
    traj_prompt_ids: list[str] = [None] * num_traj
    for row, tid in enumerate(dp_local_traj_idxs):
        if traj_rewards[tid] is None:
            traj_rewards[tid] = rewards[row]
            traj_prompt_ids[tid] = prompt_ids[row]

    reward_tensor = torch.tensor(traj_rewards, dtype=torch.float32)

    norm = config.get("advantage_norm_mode", "group_zscore")
    assert norm in _VALID_ADVANTAGE_NORMS, (
        f"Invalid advantage_norm_mode={norm!r}, must be one of {_VALID_ADVANTAGE_NORMS}"
    )

    if norm == "none":
        traj_scalar = reward_tensor.tolist()
    else:
        traj_scalar = _group_normalize(reward_tensor, traj_prompt_ids, divide_std=norm == "group_zscore")

    # Broadcast each trajectory's scalar back to its Segment rows, then to tokens.
    row_scalar = [traj_scalar[tid] for tid in dp_local_traj_idxs]
    device = torch.cuda.current_device()
    adv_tensor = torch.tensor(row_scalar, dtype=torch.float32, device=device)
    lengths_tensor = torch.tensor(response_lengths, dtype=torch.long, device=device)
    expanded = torch.repeat_interleave(adv_tensor, lengths_tensor)
    advantages = list(expanded.split(response_lengths))
    return advantages, advantages


@register_advantage("gae")
@torch.no_grad()
def gae_advantage(
    config: DictConfig,
    rollout_data: RolloutBatch,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """PPO advantage: GAE over the critic's pre-update ``old_values``.

    Runs on the critic worker (the only place the values exist); the actor
    reads the result as ``critic_advantages``. Uses the rollout's
    ``token_rewards`` and the raw response mask, ``config.ppo.gamma`` /
    ``config.ppo.lam``.

    Returns:
        ``(advantages, returns)`` per Segment row in fp32,
        ``returns = advantages + old_values``.
    """
    traj_idx = rollout_data["dp_local_traj_idx"]
    if len(set(traj_idx)) != len(traj_idx):
        raise ValueError(
            "advantage_estimator=gae supports one Segment per trajectory only "
            "(context compression / subagent trajectories are not supported yet)."
        )
    gamma, lam = float(config.ppo.gamma), float(config.ppo.lam)
    old_values = rollout_data["old_values"]
    response_lengths = rollout_data["response_lengths"]
    device = old_values[0].device

    def pad_to_max_response_len(rows):
        """Right-pad per-row response tensors with 0 into ``[num_rows, max_response_len]``."""
        return pad_sequence([t.to(device=device, dtype=torch.float32) for t in rows], batch_first=True)

    padded_rewards = pad_to_max_response_len(rollout_data["token_rewards"])
    padded_values = pad_to_max_response_len(old_values)
    # Padding positions get mask 0, so the recursion carries zeros through them.
    padded_masks = pad_to_max_response_len(rollout_data["raw_loss_masks"])

    num_rows, max_response_len = padded_values.shape
    next_value = torch.zeros(num_rows, device=device)       # V(s_{t+1}) of the next unmasked token
    next_advantage = torch.zeros(num_rows, device=device)   # A_{t+1} of the next unmasked token
    padded_advantages = torch.empty_like(padded_values)
    for t in reversed(range(max_response_len)):
        mask_t = padded_masks[:, t]
        delta_t = padded_rewards[:, t] + gamma * next_value - padded_values[:, t]
        advantage_t = delta_t + gamma * lam * next_advantage
        # mask == 0: not a step, keep carrying the values from the right.
        next_value = padded_values[:, t] * mask_t + (1 - mask_t) * next_value
        next_advantage = advantage_t * mask_t + (1 - mask_t) * next_advantage
        padded_advantages[:, t] = next_advantage
    padded_returns = padded_advantages + padded_values

    return (
        [padded_advantages[i, :n] for i, n in enumerate(response_lengths)],
        [padded_returns[i, :n] for i, n in enumerate(response_lengths)],
    )


def masked_whiten(advantages: list[torch.Tensor], masks: list[torch.Tensor]) -> list[torch.Tensor]:
    """Whiten advantages with mean / variance over the masked tokens of all DP ranks.

    The step is sharded over DP, so the statistics are all-reduced over the DP
    group. CP ranks hold the full rows, so CP is not part of the group.
    Unbiased variance, ``eps = 1e-8``; every position is transformed
    (masked-out ones never reach the loss).
    """
    from megatron.core import parallel_state as mpu  # Lazy: megatron may be absent.

    flat = torch.cat(advantages).float()
    mask = torch.cat(masks).to(device=flat.device, dtype=torch.float32)
    stats = torch.stack([(flat * mask).sum(), (flat * flat * mask).sum(), mask.sum()])
    dist.all_reduce(stats, group=mpu.get_data_parallel_group())
    total, total_sq, count = stats.tolist()
    if count == 0:
        raise ValueError("masked_whiten: no unmasked token across the DP group.")

    mean = total / count
    var = total_sq / count - mean ** 2
    if count >= 2:
        var = var * count / (count - 1)
    whitened = (flat - mean) * torch.rsqrt(torch.tensor(var + 1e-8, device=flat.device))
    return list(whitened.split([a.numel() for a in advantages]))


def _advantage_metrics(
    advantages: list[torch.Tensor],
    returns: list[torch.Tensor],
    rollout_data: RolloutBatch,
) -> dict[str, float]:
    """Monitoring metrics over the computed advantages, reduced across DP.

    ``train/adv_zero_ratio`` is the fraction of Segment rows with zero advantage.
    With a critic (``old_values`` present), also ``critic/value_mean`` /
    ``critic/returns_mean``: token means of ``old_values`` / ``returns`` over the
    raw response mask GAE uses. All stats share one all-reduce over the DP group,
    so they reflect the full training batch.
    """
    from megatron.core import parallel_state as mpu

    device = torch.cuda.current_device()
    local_zero_count = sum(1 for a in advantages if bool((a == 0).all()))
    stats = [
        torch.tensor(float(local_zero_count), device=device),
        torch.tensor(float(len(advantages)), device=device),
    ]
    old_values = rollout_data.get("old_values")
    if old_values is not None:
        mask = torch.cat(rollout_data["raw_loss_masks"]).to(device=device, dtype=torch.float32)
        values = torch.cat(old_values).to(device=device, dtype=torch.float32)
        rets = torch.cat(returns).to(device=device, dtype=torch.float32)
        stats += [(values * mask).sum(), (rets * mask).sum(), mask.sum()]
    stats = torch.stack(stats)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=mpu.get_data_parallel_group())
    global_zero_count, global_total, *value_stats = stats.tolist()

    metrics = {"train/adv_zero_ratio": global_zero_count / global_total if global_total > 0 else 0.0}
    if value_stats:
        value_sum, returns_sum, num_tokens = value_stats
        num_tokens = max(num_tokens, 1.0)
        metrics["critic/value_mean"] = value_sum / num_tokens
        metrics["critic/returns_mean"] = returns_sum / num_tokens
    return metrics


def compute_advantages_and_returns(
    config: DictConfig,
    rollout_data: RolloutBatch,
) -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, float]]:
    """Dispatch to the registered advantage estimator, then optionally whiten.

    Returns ``(advantages, returns, metrics)``. With ``use_masked_whiten`` the
    advantages are whitened over the raw response mask of the whole step;
    ``returns`` keep the un-whitened values. *metrics* are monitoring stats.
    """
    advantage_fn = get_advantage(config.advantage_estimator)
    advantages, returns = advantage_fn(config, rollout_data)
    if config.use_masked_whiten:
        advantages = masked_whiten(advantages, rollout_data["raw_loss_masks"])
    return advantages, returns, _advantage_metrics(advantages, returns, rollout_data)
