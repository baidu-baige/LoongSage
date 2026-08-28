"""Built-in advantage estimator implementations.

Registers GRPO advantage estimation into the advantage registry, and
provides the top-level ``compute_advantages`` dispatcher.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import torch
import torch.distributed as dist

from omegaconf import DictConfig

logger = logging.getLogger(__name__)

from coda.utils.types import RolloutBatch
from coda.algorithms.registry import register_advantage, get_advantage


_VALID_ADVANTAGE_NORMS = ("none", "group_mean", "group_zscore", "batch_mean", "batch_zscore")


def _parse_advantage_norm(norm: str) -> tuple[str, bool]:
    """Parse ``advantage_norm_mode`` into ``(scope, divide_std)``."""
    scope, mode = norm.rsplit("_", 1)
    return scope, mode == "zscore"


def _batch_normalize(
    reward_tensor: torch.Tensor, divide_std: bool,
) -> list[float]:
    """Normalize rewards across all DP ranks (batch-level)."""
    # Lazy import: this package is auto-imported, megatron may be absent.
    from megatron.core import parallel_state as mpu

    local_sum = reward_tensor.sum()
    local_count = torch.tensor(float(len(reward_tensor)))
    local_sq_sum = (reward_tensor ** 2).sum()

    stats = torch.stack([local_sum, local_count, local_sq_sum]).cuda()
    dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=mpu.get_data_parallel_group())
    global_sum, global_count, global_sq_sum = stats[0], stats[1], stats[2]

    mean = global_sum / global_count
    centered = reward_tensor.cuda() - mean

    if divide_std:
        # E[(X-μ)²] = E[X² - 2Xμ + μ²]
        #           = E[X²] - 2μ·E[X] + μ²
        #           = E[X²] - 2μ² + μ²
        #           = E[X²] - μ²
        var = global_sq_sum / global_count - mean ** 2
        if global_count >= 2:
            var = var * global_count / (global_count - 1)
        centered = centered * torch.rsqrt(var + 1e-6)

    return centered.cpu().tolist()


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
) -> list[torch.Tensor]:
    """GRPO advantage: reward normalisation.

    Controlled by ``config.advantage_norm_mode``:
    * ``"none"``:           raw rewards, no normalisation.
    * ``"group_mean"``:     subtract per-prompt-group mean (x - mean).
    * ``"group_zscore"``:   per-prompt-group z-score ((x - mean) / std).
    * ``"batch_mean"``:     subtract global mean across all DP ranks (x - mean).
    * ``"batch_zscore"``:   global z-score across all DP ranks ((x - mean) / std).

    Rows are per-Segment. Advantage is trajectory-level, so rows are first
    deduplicated to their parent trajectory (via ``trajectory_id``, which is a
    dense 0-based index assigned in trajectory order by ``put_dp_shards_to_ray``),
    normalised at trajectory granularity, then the per-trajectory scalar is
    broadcast back to every Segment row's response tokens.

    Returns:
        Per-Segment advantage tensors, each of shape ``(response_length,)``.
    """
    rewards = rollout_data["rewards"]
    prompt_ids = rollout_data["prompt_id"]
    trajectory_ids = rollout_data["trajectory_id"]
    response_lengths = rollout_data["response_lengths"]

    # Deduplicate Segment rows to their trajectory. trajectory_id is a
    # dense 0-based index (see put_dp_shards_to_ray), so it doubles as the
    # position in the deduplicated traj_rewards/traj_prompt_ids arrays.
    num_traj = trajectory_ids[-1] + 1
    traj_rewards: list[float] = [None] * num_traj
    traj_prompt_ids: list[str] = [None] * num_traj
    for row, tid in enumerate(trajectory_ids):
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
        scope, divide_std = _parse_advantage_norm(norm)
        if scope == "batch":
            traj_scalar = _batch_normalize(reward_tensor, divide_std)
        else:
            traj_scalar = _group_normalize(reward_tensor, traj_prompt_ids, divide_std)

    # Broadcast each trajectory's scalar back to its Segment rows, then to tokens.
    row_scalar = [traj_scalar[tid] for tid in trajectory_ids]
    device = torch.cuda.current_device()
    adv_tensor = torch.tensor(row_scalar, dtype=torch.float32, device=device)
    lengths_tensor = torch.tensor(response_lengths, dtype=torch.long, device=device)
    expanded = torch.repeat_interleave(adv_tensor, lengths_tensor)
    return list(expanded.split(response_lengths))


def _advantage_metrics(advantages: list[torch.Tensor]) -> dict[str, float]:
    """Monitoring metrics over the computed advantages, reduced across DP.

    ``train/adv_zero_ratio`` is the fraction of Segment rows with zero advantage.
    Reduced across the DP group so the ratio reflects the full training batch.
    """
    from megatron.core import parallel_state as mpu

    local_zero_count = sum(1 for a in advantages if bool((a == 0).all()))
    zero_and_total = torch.tensor(
        [float(local_zero_count), float(len(advantages))], device=torch.cuda.current_device()
    )
    dist.all_reduce(zero_and_total, op=dist.ReduceOp.SUM, group=mpu.get_data_parallel_group())
    global_zero_count, global_total = zero_and_total.tolist()
    return {"train/adv_zero_ratio": global_zero_count / global_total if global_total > 0 else 0.0}


def compute_advantages(
    config: DictConfig,
    rollout_data: RolloutBatch,
) -> tuple[list[torch.Tensor], dict[str, float]]:
    """Dispatch to the registered advantage estimator.

    Returns ``(advantages, metrics)`` where *advantages* are per-Segment
    advantage tensors (the caller is responsible for writing them back into
    *rollout_data* if needed) and *metrics* are monitoring stats to report.
    """
    advantage_fn = get_advantage(config.advantage_estimator)
    advantages = advantage_fn(config, rollout_data)
    return advantages, _advantage_metrics(advantages)
