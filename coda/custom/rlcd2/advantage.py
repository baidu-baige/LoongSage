"""RLCD² advantage (arXiv 2610.17309, Eq. 6): A_i = clip(r_i - mean_group(r), -c, c), no division by the group std.

Config: algorithm.advantage_estimator: rlcd2; algorithm.rlcd2_adv_clip (default 5.0).
"""
from __future__ import annotations

import torch

from coda.algorithms.advantage import _group_normalize
from coda.algorithms.registry import register_advantage


@register_advantage("rlcd2")
def rlcd2_advantage(config, rollout_data) -> list[torch.Tensor]:
    rewards = rollout_data["rewards"]
    prompt_ids = rollout_data["prompt_id"]
    traj_idx = rollout_data["dp_local_traj_idx"]
    response_lengths = rollout_data["response_lengths"]

    # Rows are Segments; the advantage is per trajectory (same dedup as grpo_advantage).
    num_traj = traj_idx[-1] + 1
    traj_rewards: list[float] = [None] * num_traj
    traj_prompt_ids: list[str] = [None] * num_traj
    for row, tid in enumerate(traj_idx):
        if traj_rewards[tid] is None:
            traj_rewards[tid] = rewards[row]
            traj_prompt_ids[tid] = prompt_ids[row]

    centered = torch.tensor(
        _group_normalize(torch.tensor(traj_rewards, dtype=torch.float32), traj_prompt_ids, divide_std=False))
    clip = float(config.get("rlcd2_adv_clip", 5.0))
    traj_adv = centered.clamp(-clip, clip)

    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    row_adv = traj_adv[torch.tensor(traj_idx, dtype=torch.long)].to(device)
    lengths = torch.tensor(response_lengths, dtype=torch.long, device=device)
    return list(torch.repeat_interleave(row_adv, lengths).split(response_lengths))
