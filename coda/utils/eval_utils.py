"""Evaluation metric utilities.

Computes per-dataset evaluation metrics from per-prompt reward groups:

  - mean / std / count: average reward level, spread across samples, sample count.
  - pass@k: probability that at least one of k samples is correct, as an unbiased
    estimate from the n samples actually rolled out. pass@1 tracks single-shot
    accuracy; pass@n shows how much headroom repeated sampling still buys.

A *group* is the set of ``num_trajectories_per_prompt`` rewards produced for one
prompt. A sample counts as *correct* when its reward is > 0.
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coda.agentflow.trajectory_store import TrajectoryGroup

logger = logging.getLogger(__name__)


def _std(xs: list[float]) -> float:
    """Sample standard deviation (0.0 for fewer than 2 values)."""
    if len(xs) < 2:
        return 0.0
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)


def _pass_at_k(group_rewards: list[list[float]], k: int) -> float:
    """Unbiased ``pass@k = 1 - C(n-c, k) / C(n, k)`` averaged over prompt groups."""
    vals: list[float] = []
    for g in group_rewards:
        n = len(g)
        c = sum(1 for r in g if r > 0)
        if n == 0:
            continue
        if n - c < k:
            vals.append(1.0)
        else:
            vals.append(1.0 - math.comb(n - c, k) / math.comb(n, k))
    return sum(vals) / len(vals) if vals else 0.0


def _dataset_metrics(label: str, group_rewards: list[list[float]]) -> dict[str, float]:
    """Compute a flat ``{eval/<label>_<metric>: value}`` dict for one dataset.

    ``mean`` / ``std`` / ``count`` are always reported. pass@k needs at least one
    sample per prompt and is omitted otherwise, so a dataset with no samples reports
    ``count = 0`` and zeros for mean/std — callers must treat ``count == 0`` as
    "no data", not as zero reward.

    Args:
        label:         Dataset label used in the metric key prefix, e.g. ``"ds0"``.
        group_rewards: Per-prompt reward lists (each length = num_trajectories_per_prompt).
    """
    prefix = f"eval/{label}_"
    all_rewards = [r for g in group_rewards for r in g]
    metrics: dict[str, float] = {}
    if not all_rewards:
        metrics[f"{prefix}mean"] = 0.0
        metrics[f"{prefix}std"] = 0.0
        metrics[f"{prefix}count"] = 0.0
        return metrics

    metrics[f"{prefix}mean"] = sum(all_rewards) / len(all_rewards)
    metrics[f"{prefix}std"] = _std(all_rewards)
    metrics[f"{prefix}count"] = float(len(all_rewards))

    non_empty = [g for g in group_rewards if g]
    n = min(len(g) for g in non_empty) if non_empty else 0
    if n >= 1:
        metrics[f"{prefix}pass@1"] = _pass_at_k(group_rewards, 1)
    if n > 1:
        metrics[f"{prefix}pass@{n}"] = _pass_at_k(group_rewards, n)
    return metrics


def compute_eval_metrics(eval_traj_groups: list[TrajectoryGroup]) -> dict[str, float]:
    """Compute flat ``eval/...`` metrics from the eval trajectory groups of one round.

    Groups are bucketed by ``ds_index`` (the training source each eval set was derived
    from) and reported as ``eval/ds{ds_index}_<metric>``, mirroring the
    ``rollout_per_ds/ds{ds_index}_<metric>`` naming. Rewards come from ``traj.reward``.
    ``eval/mean`` averages the per-dataset means, skipping datasets that produced no
    samples.

    Args:
        eval_traj_groups: Groups routed out of the training batch by ``is_eval``; each group
                          is one eval prompt and its samples.
    """
    ds_index_to_traj_groups: dict[int, list[TrajectoryGroup]] = defaultdict(list)
    for g in eval_traj_groups:
        ds_index_to_traj_groups[g.trajectories[0].ds_index].append(g)

    metrics: dict[str, float] = {}
    mean_values: list[float] = []
    for ds_index, traj_groups in sorted(ds_index_to_traj_groups.items()):
        # ds_index is a data source's only identity: dataset.name denotes a data format
        # and may repeat across sources, so it is not part of the key.
        label = f"ds{ds_index}"
        group_rewards = [[float(t.reward or 0.0) for t in g.trajectories] for g in traj_groups]
        ds_metrics = _dataset_metrics(label, group_rewards)
        metrics.update(ds_metrics)
        # Datasets with no samples report count=0; averaging their mean would
        # pull eval/mean down as if they had scored 0.
        if ds_metrics.get(f"eval/{label}_count", 0.0) > 0:
            mean_values.append(ds_metrics[f"eval/{label}_mean"])
    if mean_values:
        metrics["eval/mean"] = sum(mean_values) / len(mean_values)
    return metrics
