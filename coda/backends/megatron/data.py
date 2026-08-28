"""Data utilities for the Megatron training backend.

Provides:
* RolloutBatch shard utilities — ``group_rollout_batch``, ``merge_rollout_batch``,
  ``concat_rollout_batch``, ``slice_rollout_batch``, ``select_rollout_batch``.
* ``get_rollout_data`` — resolve Ray refs and merge teacher data.
* ``DataIterator`` — micro-batch level iteration.
* ``get_data_iterator`` — factory that handles static / dynamic batch
  construction, VPP replication, and DP-level synchronisation.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import ray
import torch
import torch.distributed as dist

from megatron.core import parallel_state as mpu
from omegaconf import DictConfig

from megatron.core.utils import get_model_config
from coda.utils.seqlen_balancing import get_seqlen_balanced_partitions
from coda.utils.types import RolloutBatch

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────
# RolloutBatch shard utilities
# ────────────────────────────────────────────────────────────────────────
def select_rollout_batch(shard: RolloutBatch, indices: list[int]) -> RolloutBatch:
    """Select a subset of trajectories from a shard by indices."""
    num_trajectories = len(shard.get("tokens", []))
    subset = {}
    for key, value in shard.items():
        if isinstance(value, list) and len(value) == num_trajectories:
            subset[key] = [value[i] for i in indices]
        else:
            subset[key] = value
    return subset


def concat_rollout_batch(shards: list[RolloutBatch]) -> RolloutBatch:
    """Concatenate multiple shards by appending list fields."""
    if len(shards) == 1:
        return shards[0]
    merged = {}
    num_trajectories_first = len(shards[0].get("tokens", []))
    for key in shards[0]:
        values = [s[key] for s in shards]
        if isinstance(values[0], list) and len(values[0]) == num_trajectories_first:
            merged[key] = sum(values, [])
        else:
            merged[key] = values[0]
    return merged


def slice_rollout_batch(shard: RolloutBatch, num_splits: int, split_idx: int) -> RolloutBatch:
    """Slice one equal split from a shard (by trajectory count)."""
    num_trajectories = len(shard.get("tokens", []))
    chunk_size = (num_trajectories + num_splits - 1) // num_splits
    start = split_idx * chunk_size
    end = min(start + chunk_size, num_trajectories)
    return select_rollout_batch(shard, list(range(start, end)))


def group_rollout_batch(rollout_data: RolloutBatch, key: str, allowed_keys=None) -> dict:
    """Split a RolloutBatch into buckets by the per-trajectory field ``key``.

    Inverse of :func:`merge_rollout_batch`.  All fields are treated as
    trajectory-aligned lists; values are appended to the corresponding bucket
    by index.

    Args:
        rollout_data: The batch to split.
        key: Field name whose values determine which bucket each trajectory goes to.
        allowed_keys: If provided, trajectories whose ``key`` value is not in this
            collection are silently skipped.

    Returns:
        ``dict[key_value, RolloutBatch]`` — one sub-batch per distinct key value.
    """
    grouped: dict = {}
    for i, k in enumerate(rollout_data[key]):
        if allowed_keys is not None and k not in allowed_keys:
            continue
        if k not in grouped:
            grouped[k] = {f: [] for f in rollout_data}
        for f, values in rollout_data.items():
            grouped[k][f].append(values[i])
    return grouped


def merge_rollout_batch(batches: list[RolloutBatch], key: str) -> RolloutBatch:
    """Reassemble multiple shards into one full batch using ``batch[key]``.

    Inverse of :func:`group_rollout_batch`. Each input ``batches[i]`` must
    contain a list under ``key`` whose values place that batch's trajectories
    into the combined output. The union of all ``batch[key]`` values across
    batches must form exactly the contiguous range ``[0, N)`` (no duplicates,
    no gaps), where ``N`` is the total trajectory count.

    Every other field present in any input batch becomes a length-N list in
    the output, with values placed at their indexed position. ``key`` itself
    is dropped from the output.
    """
    if not batches:
        return {}

    # Validate contiguity (no duplicates, no gaps)
    all_indices: list[int] = []
    for b in batches:
        all_indices.extend(b[key])
    n = len(all_indices)
    if sorted(all_indices) != list(range(n)):
        raise ValueError(
            f"{key} across batches must form a contiguous range [0, {n}); "
            f"got duplicates or gaps: sorted indices = {sorted(all_indices)}"
        )

    # Collect every field that appears anywhere (except the index key itself)
    field_keys: set[str] = set()
    for b in batches:
        field_keys.update(b.keys())
    field_keys.discard(key)

    # Place each batch's values at its indexed positions
    merged: RolloutBatch = {f: [None] * n for f in field_keys}
    for b in batches:
        positions = b[key]
        for f in field_keys:
            values = b.get(f)
            if values is None:
                continue
            for i, pos in enumerate(positions):
                merged[f][pos] = values[i]
    return merged


# ────────────────────────────────────────────────────────────────────────
# Data Fetching
# ────────────────────────────────────────────────────────────────────────
def get_rollout_data(rollout_data_ref) -> RolloutBatch:
    """Resolve a Ray rollout shard and merge teacher data if present.

    Static tensor fields and any teacher tensor fields supplied via
    ``teacher_worker_ref`` are moved to the current CUDA device.
    """
    dp_rank = mpu.get_data_parallel_rank()
    box = rollout_data_ref[dp_rank]
    rollout_data: RolloutBatch = ray.get(box.ref)
    loss_masks = rollout_data.get("loss_masks")
    rollout_data["raw_loss_masks"] = [mask.clone() for mask in loss_masks]

    teacher_tensor_fields: list[str] = []
    if box.teacher_worker_ref:
        all_teacher_data = ray.get(box.teacher_worker_ref)
        merged = merge_rollout_batch(all_teacher_data, "seq_index")  # seq_index dropped here

        n = len(rollout_data["tokens"])
        merged_n = len(merged["teacher_idx"])
        if merged_n != n:
            raise ValueError(
                f"Teacher merge length mismatch: merged teacher_idx has "
                f"{merged_n} entries but original rollout has {n} trajectories"
            )

        # Identify teacher tensor fields (skip teacher_idx + any other list[int]
        # bookkeeping like train_dp_ranks).
        for k, v in merged.items():
            if isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
                teacher_tensor_fields.append(k)
        rollout_data.update(merged)

    device = torch.cuda.current_device()
    for field in (
        "tokens", "loss_masks", "rollout_log_probs", "rollout_routed_experts", "raw_loss_masks",
        *teacher_tensor_fields,
    ):
        if rollout_data.get(field):
            rollout_data[field] = [t.to(device) for t in rollout_data[field]]
    return rollout_data

# ────────────────────────────────────────────────────────────────────────
# DataIterator
# ────────────────────────────────────────────────────────────────────────

class DataIterator:
    """Yields micro-batches from a ``RolloutBatch``.

    Supports two modes:
    * **Static** (``micro_batch_size`` is set): advance by a fixed window.
    * **Dynamic** (``micro_batch_indices`` is set): index-list driven.

    Exactly one of the two arguments must be provided.
    """

    def __init__(
        self,
        rollout_data: RolloutBatch,
        micro_batch_size: int | None = None,
        micro_batch_indices: list[list[int]] | None = None,
    ) -> None:
        assert (micro_batch_size is None) != (micro_batch_indices is None), (
            "Exactly one of micro_batch_size / micro_batch_indices must be set."
        )
        self.rollout_data = rollout_data
        self.micro_batch_size = micro_batch_size
        self.micro_batch_indices = micro_batch_indices
        if micro_batch_indices is not None:
            total_rows = sum(len(indices) for indices in micro_batch_indices)
            pad_rows = sum(
                index == -1
                for indices in micro_batch_indices
                for index in indices
            )
            self.padding_stats = (pad_rows, total_rows)
        self.offset: int = 0

    # ── public API ──────────────────────────────────────────────────────

    def get_next(self, keys: Sequence[str]) -> dict[str, list[object] | None]:
        """Return the next micro-batch filtered by *keys*.

        Missing keys are returned as ``None`` values.
        """
        if self.micro_batch_indices is not None:
            if self.offset >= len(self.micro_batch_indices):
                raise IndexError(
                    f"DataIterator exhausted: step {self.offset} >= "
                    f"{len(self.micro_batch_indices)} total micro-batches"
                )
            indices = self.micro_batch_indices[self.offset]
        else:
            total = len(self.rollout_data["tokens"])
            start = self.offset * self.micro_batch_size
            if start >= total:
                raise IndexError(
                    f"DataIterator exhausted: start index {start} >= "
                    f"total trajectories {total}"
                )
            end = start + self.micro_batch_size
            indices = list(range(start, end))

        self.offset += 1

        batch: dict[str, list[object] | None] = {}
        for key in keys:
            values = self.rollout_data.get(key)
            if values is None:
                batch[key] = None
            else:
                batch[key] = []
                for index in indices:
                    value = values[0] if index == -1 else values[index]
                    if index == -1 and key in {"loss_masks", "raw_loss_masks"}:
                        value = torch.zeros_like(value)
                    batch[key].append(value)
        return batch

    def get_global_padding_stats(self) -> tuple[int, float]:
        """Return global segment padding count and ratio."""
        stats = torch.tensor(
            self.padding_stats,
            dtype=torch.long,
            device=torch.cuda.current_device(),
        )
        dist.all_reduce(
            stats,
            op=dist.ReduceOp.SUM,
            group=mpu.get_data_parallel_group(),
        )
        padding_count, total_count = stats.tolist()
        return padding_count, padding_count / total_count

# ────────────────────────────────────────────────────────────────────────
# Factory
# ────────────────────────────────────────────────────────────────────────

def _get_min_num_microbatches(total_lengths: list[int], max_tokens_per_gpu: int) -> int:
    """
    Use first fit to get the number of micro batches.
    """
    batches = []
    for length in total_lengths:
        for i in range(len(batches)):
            if batches[i] + length <= max_tokens_per_gpu:
                batches[i] += length
                break
        else:
            batches.append(length)

    return len(batches)


def _trajectory_row_ranges(trajectory_ids: list[int]) -> list[tuple[int, int]]:
    """Group contiguous equal ``trajectory_id`` rows into [start, end) ranges.

    Segment rows of one trajectory are emitted contiguously at the data source,
    so each run of equal ids is that trajectory's span of Segment rows.

    e.g. trajectory_ids = [0, 0, 0, 1, 2, 2] -> [(0, 3), (3, 4), (4, 6)]
    """
    ranges: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(trajectory_ids) + 1):
        if i == len(trajectory_ids) or trajectory_ids[i] != trajectory_ids[start]:
            ranges.append((start, i))
            start = i
    return ranges


def get_data_iterator(
    config: DictConfig,
    model: torch.nn.Module | Sequence[torch.nn.Module],
    rollout_data: RolloutBatch,
    use_single_mini_batch: bool = False,
) -> tuple[list[DataIterator], list[int]]:
    """Build data iterators and per minibatch num_microbatches for a rollout.

    The rollout data will be split into mini-batches first, and then micro-batches.

    Args:
        config: Unified training configuration.
        model: Model (or list of model chunks for VPP).
        rollout_data: The complete rollout batch for this DP rank.
        use_single_mini_batch: Treat the whole DP shard as one mini-batch instead
            of splitting it by ``config.mini_batch_size``.

    Returns:
        ``(data_iterators, num_microbatches_list)``
        *data_iterators* has ``vpp_size`` elements (same data, independent
        offset).  *num_microbatches_list* has ``num_mini_batch`` elements.
    """
    dp_size = mpu.get_data_parallel_world_size(with_context_parallel=False)
    cp_size = mpu.get_context_parallel_world_size()
    vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
    if vpp_size is None:
        vpp_size = 1

    if use_single_mini_batch:
        # Non-flattened data (e.g. teacher forward): one mini-batch over every
        # row.  ``trajectory_id`` is not needed here — rows are counted from
        # ``tokens`` directly.
        mini_batch_ranges = [(0, len(rollout_data["tokens"]))]
    else:
        # Flattened Segment rows: cut mini-batches on trajectory boundaries so
        # num_mini_batch is equal across DP ranks even when per-trajectory
        # segment counts differ.  ``trajectory_id`` tags each Segment row with
        # its parent trajectory (contiguous runs); a trajectory's rows stay in
        # one mini-batch while micro-batches may split them.
        trajectory_row_ranges = _trajectory_row_ranges(rollout_data["trajectory_id"])
        mini_batch_size_per_dp = config.mini_batch_size // dp_size
        num_mini_batch = len(trajectory_row_ranges) // mini_batch_size_per_dp
        mini_batch_ranges = [
            (
                trajectory_row_ranges[i * mini_batch_size_per_dp][0],
                trajectory_row_ranges[(i + 1) * mini_batch_size_per_dp - 1][1],
            )
            for i in range(num_mini_batch)
        ]

    if not config.use_dynamic_batch_size:
        # ── Static batching ─────────────────────────────────────────────
        micro_batch_size = config.micro_batch_size
        num_microbatches_tensor = torch.tensor(
            [
                (end - start + micro_batch_size - 1) // micro_batch_size
                for start, end in mini_batch_ranges
            ],
            dtype=torch.int,
            device=torch.cuda.current_device(),
        )
        if dp_size > 1:
            dist.all_reduce(
                num_microbatches_tensor,
                op=dist.ReduceOp.MAX,
                group=mpu.get_data_parallel_group(),
            )

        if vpp_size > 1:
            # VPP requires num_microbatches to be divisible by
            # microbatch_group_size_per_vp_stage.  Static batching uses a fixed window
            # plus -1 padding, so we ceil UP to the next multiple (adding padding-only
            # micro-batches).  Flooring down — as the dynamic path does — would make
            # num_microbatches * micro_batch_size smaller than the actual Segment-row
            # count, dropping real rows and desyncing the micro_batch_indices chunks.
            _megatron_cfg = get_model_config(model[0])
            group_size = _megatron_cfg.microbatch_group_size_per_vp_stage
            num_microbatches_tensor = (
                (num_microbatches_tensor + group_size - 1) // group_size * group_size
            )

        num_microbatches_list = num_microbatches_tensor.tolist()
        micro_batch_indices = []
        for (start, end), num_microbatches in zip(mini_batch_ranges, num_microbatches_list):
            rows = list(range(start, end))
            rows.extend([-1] * (num_microbatches * micro_batch_size - len(rows)))
            micro_batch_indices.extend(
                rows[offset : offset + micro_batch_size]
                for offset in range(0, len(rows), micro_batch_size)
            )

        data_iterators = [
            DataIterator(rollout_data, micro_batch_indices=micro_batch_indices)
            for _ in range(vpp_size)
        ]
        return data_iterators, num_microbatches_list

    # ── Dynamic batching (first-fit bin packing) ───────────────────
    total_lengths = rollout_data["total_lengths"]
    max_budget = config.max_tokens_per_gpu * cp_size
    all_num_microbatches: list[int] = []

    # Partition each mini-batch's Segments into micro-batches
    for start, end in mini_batch_ranges:
        all_num_microbatches.append(
            _get_min_num_microbatches(total_lengths[start:end], max_budget)
        )

    # Sync max num_microbatches across DP ranks
    num_microbatches_tensor = torch.tensor(all_num_microbatches, dtype=torch.int, device=torch.cuda.current_device())
    dist.all_reduce(num_microbatches_tensor, op=dist.ReduceOp.MAX, group=mpu.get_data_parallel_group())

    if vpp_size > 1:
        # vpp requires the number of microbatches to be divisible by
        # microbatch_group_size_per_vp_stage
        _megatron_cfg = get_model_config(model[0])
        microbatch_group_size_per_vp_stage = _megatron_cfg.microbatch_group_size_per_vp_stage
        # Floor-divide to the nearest multiple of microbatch_group_size_per_vp_stage so
        # that VPP scheduling constraints are satisfied.  This may reduce num_microbatches
        # below the value required to keep each microbatch within max_budget, meaning
        # individual microbatches could exceed the token budget and risk OOM.
        num_microbatches_tensor = torch.clamp(
            num_microbatches_tensor // microbatch_group_size_per_vp_stage * microbatch_group_size_per_vp_stage,
            min=1,
        )

    num_microbatches_list = num_microbatches_tensor.tolist()

    # Balance each micro-batch across steps
    micro_batch_indices = []
    for i, num_mbs in enumerate(num_microbatches_list):
        start, end = mini_batch_ranges[i]
        lengths = total_lengths[start:end]
        local_num_mbs = min(num_mbs, len(lengths))
        partitions = get_seqlen_balanced_partitions(
            lengths, local_num_mbs, equal_size=False
        )

        # Check for OOM risks after forced seqlen balancing with clamped num_mbs
        for j in range(local_num_mbs):
            partition_token_sum = sum(lengths[idx] for idx in partitions[j])
            if (
                partition_token_sum > max_budget
                and mpu.get_tensor_model_parallel_rank() == 0
                and mpu.get_pipeline_model_parallel_rank() == 0
                and mpu.get_context_parallel_rank() == 0
            ):
                logger.info(
                    f"DP rank {mpu.get_data_parallel_rank()} "
                    f"minibatch {i} microbatch {j} has {partition_token_sum} tokens, "
                    f"exceeds max_tokens_per_gpu * cp_size ({max_budget}) "
                    f"by {partition_token_sum - max_budget} tokens."
                )
            for k in range(len(partitions[j])):
                partitions[j][k] += start
        partitions.extend([[-1]] * (num_mbs - local_num_mbs))
        micro_batch_indices.extend(partitions)

    assert sorted(
        row for partition in micro_batch_indices for row in partition if row >= 0
    ) == list(range(len(total_lengths)))

    data_iterators = [
        DataIterator(rollout_data, micro_batch_indices=micro_batch_indices)
        for _ in range(vpp_size)
    ]

    return data_iterators, num_microbatches_list
