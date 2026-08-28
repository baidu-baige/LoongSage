"""DP-rank dispatch utilities for TrajectoryGroup data.

This module provides three layers of functionality:

1. Balancing – split a flat list of TrajectoryGroup objects across *dp_size* data-parallel ranks so that the total
   token load is as equal as possible (split_traj_group_by_dp).

2. Serialisation – flatten each rank's subset into per-Segment training rows, store them in Ray Object Store, and
   return a list[Box] of ObjectRef wrappers (put_dp_shards_to_ray).

3. Deserialisation – retrieve and reconstruct one rank's shard from Ray Object Store (get_dp_shard_from_ray).
"""

from __future__ import annotations

import logging
from typing import Any
from collections import defaultdict

import ray
import torch

from coda.agentflow.trajectory_store import TrajectoryGroup
from coda.utils.seqlen_balancing import get_seqlen_balanced_partitions


logger = logging.getLogger(__name__)


def _get_group_partition_loads(groups: list[TrajectoryGroup]) -> list[int]:
    """Balance Segment count first, then token count."""
    token_scale = sum(group.token_length for group in groups) + 1
    return [group.segment_count * token_scale + group.token_length for group in groups]


class Box:
    """Lightweight wrapper that holds a single ray.ObjectRef.

    Exposes the ref via attribute access (box.ref) so that callers do not need to unpack a tuple or dict manually.
    """

    def __init__(self, ref: ray.ObjectRef) -> None:
        self.ref = ref
        self.teacher_worker_ref: list[ray.ObjectRef] = []


def split_traj_group_by_dp(
    traj_groups: list[TrajectoryGroup],
    dp_size: int,
    num_mini_batch: int,
) -> list[list[TrajectoryGroup]]:
    """Split traj_groups by DP rank with ds_index-stratified balancing.

    Guarantees:
    1. Each DP rank receives the same ds_index proportion
    2. Within each DP rank, trajectory order ensures contiguous minibatch slices also have equal ds_index proportions

    Algorithm:
    1. Group traj_groups by ds_index
    2. For each ds group, seqlen-balanced split into dp_size parts
    3. Within each DP shard, further partition each ds group into num_mini_batch parts and interleave
    """

    ds_groups = defaultdict(list)
    for group in traj_groups:
        ds_idx = group.trajectories[0].ds_index
        ds_groups[ds_idx].append(group)

    # Step 1: split each ds group across dp ranks
    dp_shards = [[] for _ in range(dp_size)]
    for ds_idx in sorted(ds_groups.keys()):
        groups = ds_groups[ds_idx]
        seqlen_list = _get_group_partition_loads(groups)
        index_parts = get_seqlen_balanced_partitions(seqlen_list, dp_size, equal_size=True)
        for rank, part in enumerate(index_parts):
            dp_shards[rank].extend([groups[i] for i in part])

    # Step 2: within each rank, reorder for minibatch ds_index balance
    for rank in range(dp_size):
        rank_ds_groups = defaultdict(list)
        for group in dp_shards[rank]:
            rank_ds_groups[group.trajectories[0].ds_index].append(group)

        partitions_by_ds = {}
        for ds_idx in sorted(rank_ds_groups.keys()):
            groups = rank_ds_groups[ds_idx]
            seqlen_list = _get_group_partition_loads(groups)
            index_parts = get_seqlen_balanced_partitions(seqlen_list, num_mini_batch, equal_size=True)
            partitions_by_ds[ds_idx] = [[groups[i] for i in part] for part in index_parts]

        reordered = []
        for mb_idx in range(num_mini_batch):
            for ds_idx in sorted(partitions_by_ds.keys()):
                reordered.extend(partitions_by_ds[ds_idx][mb_idx])
        dp_shards[rank] = reordered

    return dp_shards


def put_dp_shards_to_ray(dp_traj_groups_list: list[list[TrajectoryGroup]], dp_size: int) -> list[Box]:
    """Store pre-split DP shards in Ray Object Store and return Box wrappers.

    Args:
        dp_traj_groups_list: Already-split shards, one sub-list per DP rank (e.g. the output of
            split_traj_group_by_dp).
        dp_size: Number of DP ranks; must equal ``len(dp_traj_groups_list)``.

    Returns:
        A list of Box objects of the same length as dp_traj_groups_list. boxes[rank].ref is the ray.ObjectRef
        for that rank's shard.

    Example::

        shards = split_traj_group_by_dp(groups, dp_size=4, num_mini_batch=2)
        boxes = put_dp_shards_to_ray(shards, dp_size=4)
        # Pass boxes[rank] to each DP worker.
    """
    assert dp_size == len(dp_traj_groups_list), "Number of trajectory groups must match DP size"

    rollout_data_refs: list[Box] = []

    for dp_traj_groups in dp_traj_groups_list:
        prompt_id: list[str] = []
        trajectory_id: list[int] = []
        tokens: list[torch.Tensor] = []
        loss_masks: list[torch.Tensor] = []
        rollout_log_probs: list[torch.Tensor] = []
        rollout_weight_versions: list[torch.Tensor] = []
        token_rewards: list[torch.Tensor] = []
        rewards: list[float | None] = []
        response_lengths: list[int] = []
        total_lengths: list[int] = []
        rollout_routed_experts: list[torch.Tensor | None] = []
        metadata: list[dict] = []
        ds_indices: list[int] = []

        device = torch.cuda.current_device()
        local_tid = 0
        for traj_group in dp_traj_groups:
            for traj in traj_group.trajectories:
                traj_tokens = torch.tensor(traj.tokens, dtype=torch.long, device=device)
                traj_loss_masks = torch.tensor(traj.loss_masks, dtype=torch.long, device=device)
                traj_log_probs = torch.tensor(traj.rollout_log_probs, device=device)
                traj_weight_versions = torch.tensor(
                    traj.rollout_weight_versions, dtype=torch.int, device=device
                )
                traj_token_rewards = torch.tensor(traj.token_rewards, device=device)
                traj_experts = traj.rollout_routed_experts

                # One training row per trainable Segment: slice the trajectory's flat
                # arrays to each Segment's span (token-space vs response-space).  This is
                # the single flatten point — every downstream consumer sees Segment rows.
                for seg in traj.segments:
                    if not seg.trainable:
                        continue
                    ts, te = seg.token_start, seg.token_end
                    ls, le = seg.logprob_start, seg.logprob_end
                    prompt_id.append(traj.prompt_id)
                    trajectory_id.append(local_tid)
                    tokens.append(traj_tokens[ts:te])
                    loss_masks.append(traj_loss_masks[ls:le])
                    rollout_log_probs.append(traj_log_probs[ls:le])
                    rollout_weight_versions.append(traj_weight_versions[ls:le])
                    token_rewards.append(traj_token_rewards[ls:le])
                    rollout_routed_experts.append(
                        traj_experts[ts:te] if traj_experts is not None else None
                    )
                    total_lengths.append(te - ts)
                    response_lengths.append(le - ls)
                    rewards.append(traj.reward)
                    metadata.append(traj.metadata)
                    ds_indices.append(traj.ds_index)
                local_tid += 1

        rollout_data = {
            "prompt_id": prompt_id,
            "trajectory_id": trajectory_id,
            "tokens": tokens,
            "loss_masks": loss_masks,
            "rollout_log_probs": rollout_log_probs,
            "rollout_weight_versions": rollout_weight_versions,
            "token_rewards": token_rewards,
            "rewards": rewards,
            "response_lengths": response_lengths,
            "total_lengths": total_lengths,
            "metadata": metadata,
            "ds_indices": ds_indices,
        }
        if rollout_routed_experts and all(experts is not None for experts in rollout_routed_experts):
            rollout_data["rollout_routed_experts"] = rollout_routed_experts
        elif any(experts is None for experts in rollout_routed_experts):
            logger.info(
                "Omitting rollout_routed_experts for DP shard because %d/%d Segments have no R3 data",
                sum(experts is None for experts in rollout_routed_experts),
                len(rollout_routed_experts),
            )
        ref: ray.ObjectRef = ray.put(rollout_data)
        rollout_data_refs.append(Box(ref))

    return rollout_data_refs


def get_dp_shard_from_ray(rollout_data_ref: Box) -> dict[str, Any]:
    """Retrieve and deserialise one DP rank's shard from Ray Object Store.

    Converts segment/triplet indices into sliced tensors.
    """

    def _unzip(pairs: list[tuple]) -> tuple[list, ...]:
        """Transpose a list of N-tuples into N lists."""
        if not pairs:
            return ()
        return tuple(list(col) for col in zip(*pairs))

    data = ray.get(rollout_data_ref.ref)

    tokens = data["tokens"]
    masks = data["loss_masks"]
    logps = data["rollout_log_probs"]

    # Per-trajectory segment slicing → [traj][seg]
    seg_toks, seg_masks, seg_logps, seg_rwds = _unzip([
        _unzip([
            (
                tokens[i][s["token_start"]:s["token_end"]],
                masks[i][s["logprob_start"]:s["logprob_end"]],
                logps[i][s["logprob_start"]:s["logprob_end"]],
                s.get("reward"),
            )
            for s in segs
        ]) or ([], [], [], [])
        for i, segs in enumerate(data["segments"])
    ]) or ([], [], [], [])

    # Per-trajectory triplet slicing → [traj][seg][trip]
    trip_toks, trip_masks, trip_logps, trip_rwds, trip_meta = _unzip([
        _unzip([
            _unzip([
                (
                    tokens[i][s["token_start"]:s["token_end"]],
                    masks[i][s["logprob_start"]:s["logprob_end"]],
                    logps[i][s["logprob_start"]:s["logprob_end"]],
                    s.get("reward"),
                    s.get("metadata", {}),
                )
                for s in seg_trips
            ]) or ([], [], [], [], [])
            for seg_trips in trips
        ]) or ([], [], [], [], [])
        for i, trips in enumerate(data["triplets"])
    ]) or ([], [], [], [], [])

    result = {
        "prompt_id": data["prompt_id"],
        "tokens": tokens,
        "loss_masks": masks,
        "rollout_log_probs": logps,
        "rollout_weight_versions": data["rollout_weight_versions"],
        "token_rewards": data["token_rewards"],
        "rewards": data["rewards"],
        "response_lengths": data["response_lengths"],
        "total_lengths": data["total_lengths"],
        "segment_tokens": seg_toks,
        "segment_loss_masks": seg_masks,
        "segment_log_probs": seg_logps,
        "segment_rewards": seg_rwds,
        "triplet_tokens": trip_toks,
        "triplet_loss_masks": trip_masks,
        "triplet_log_probs": trip_logps,
        "triplet_rewards": trip_rwds,
        "triplet_metadata": trip_meta,
        "metadata": data.get("metadata", []),
        "ds_indices": data.get("ds_indices", []),
    }
    if "rollout_routed_experts" in data:
        result["rollout_routed_experts"] = data["rollout_routed_experts"]
    return result
