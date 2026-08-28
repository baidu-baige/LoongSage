"""Unit tests for coda/utils/seqlen_balancing.py."""

import pytest

from coda.agentflow.trajectory_store import Segment, Trajectory, TrajectoryGroup
from coda.data_factory.data_processor import split_traj_group_by_dp
from coda.utils.seqlen_balancing import (
    get_seqlen_balanced_partitions,
    karmarkar_karp,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_indices_covered(partitions, n):
    """Return True when every index in range(n) appears exactly once."""
    seen = sorted(idx for part in partitions for idx in part)
    return seen == list(range(n))


def _partition_sums(seqlen_list, partitions):
    """Return the sum of seqlens for each partition."""
    return [sum(seqlen_list[i] for i in part) for part in partitions]


# ===========================================================================
# karmarkar_karp
# ===========================================================================

class TestKarmarkarKarpEqualSize:
    """Tests for karmarkar_karp with equal_size=True."""

    def test_basic_two_partitions(self):
        seqlens = [1, 2, 3, 4, 5, 6]
        parts = karmarkar_karp(seqlens, k_partitions=2, equal_size=True)
        assert len(parts) == 2
        assert _all_indices_covered(parts, len(seqlens))

    def test_each_partition_has_equal_item_count(self):
        seqlens = [1, 2, 3, 4, 5, 6]
        parts = karmarkar_karp(seqlens, k_partitions=2, equal_size=True)
        sizes = [len(p) for p in parts]
        assert len(set(sizes)) == 1, f"unequal partition sizes: {sizes}"

    def test_three_partitions(self):
        seqlens = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        parts = karmarkar_karp(seqlens, k_partitions=3, equal_size=True)
        assert len(parts) == 3
        assert _all_indices_covered(parts, len(seqlens))
        sizes = [len(p) for p in parts]
        assert len(set(sizes)) == 1

    def test_non_divisible_raises_assertion_error(self):
        with pytest.raises(AssertionError, match=r"% .* != 0"):
            karmarkar_karp([1, 2, 3, 4, 5], k_partitions=3, equal_size=True)

    def test_all_same_lengths(self):
        seqlens = [5, 5, 5, 5]
        parts = karmarkar_karp(seqlens, k_partitions=2, equal_size=True)
        assert _all_indices_covered(parts, 4)
        sums = _partition_sums(seqlens, parts)
        assert sums[0] == sums[1]

    def test_single_group_of_k(self):
        # len == k_partitions: each partition gets exactly one item
        seqlens = [3, 1, 4, 2]
        parts = karmarkar_karp(seqlens, k_partitions=4, equal_size=True)
        assert len(parts) == 4
        assert all(len(p) == 1 for p in parts)
        assert _all_indices_covered(parts, 4)

    def test_balance_quality(self):
        """Karmarkar-Karp is deterministic, so pin the exact partition sums.

        ``spread <= max(seqlens)`` would hold for almost any split of a total of
        55, so it cannot detect a regression in the algorithm.
        """
        seqlens = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        parts = karmarkar_karp(seqlens, k_partitions=2, equal_size=True)
        sums = _partition_sums(seqlens, parts)
        assert sorted(sums) == [27, 28]
        assert all(len(p) == 5 for p in parts)

    def test_k_equals_one(self):
        seqlens = [3, 1, 4, 2]
        parts = karmarkar_karp(seqlens, k_partitions=1, equal_size=True)
        assert len(parts) == 1
        assert _all_indices_covered(parts, 4)


class TestKarmarkarKarpUnequalSize:
    """Tests for karmarkar_karp with equal_size=False."""

    def test_basic_two_partitions(self):
        seqlens = [1, 2, 3, 4, 5]
        parts = karmarkar_karp(seqlens, k_partitions=2, equal_size=False)
        assert len(parts) == 2
        assert _all_indices_covered(parts, len(seqlens))

    def test_all_indices_covered(self):
        seqlens = [7, 3, 5, 2, 8, 1]
        parts = karmarkar_karp(seqlens, k_partitions=3, equal_size=False)
        assert _all_indices_covered(parts, len(seqlens))

    def test_balance_quality(self):
        """Deterministic sums for 3 unequal-size partitions of a total of 55."""
        seqlens = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        parts = karmarkar_karp(seqlens, k_partitions=3, equal_size=False)
        sums = _partition_sums(seqlens, parts)
        assert sorted(sums) == [18, 18, 19]

    def test_returns_k_partitions(self):
        seqlens = [1, 2, 3, 4, 5, 6, 7]
        for k in [2, 3, 4]:
            parts = karmarkar_karp(seqlens, k_partitions=k, equal_size=False)
            assert len(parts) == k

    def test_single_item_per_partition_possible(self):
        # 4 items, 4 partitions → each partition has exactly 1 item
        seqlens = [4, 3, 2, 1]
        parts = karmarkar_karp(seqlens, k_partitions=4, equal_size=False)
        assert len(parts) == 4
        assert _all_indices_covered(parts, 4)

    def test_all_same_lengths(self):
        seqlens = [5, 5, 5, 5, 5, 5]
        parts = karmarkar_karp(seqlens, k_partitions=3, equal_size=False)
        assert _all_indices_covered(parts, 6)


# ===========================================================================
# get_seqlen_balanced_partitions
# ===========================================================================

class TestGetSeqlenBalancedPartitions:
    """Tests for the public wrapper get_seqlen_balanced_partitions."""

    # --- input validation ---

    def test_fewer_items_than_k_raises(self):
        with pytest.raises(AssertionError, match="k_partitions"):
            get_seqlen_balanced_partitions([1, 2], k_partitions=5, equal_size=False)

    def test_equal_size_non_divisible_raises(self):
        with pytest.raises(AssertionError):
            get_seqlen_balanced_partitions([1, 2, 3, 4, 5], k_partitions=3, equal_size=True)

    # --- output structure ---

    def test_returns_k_partitions(self):
        seqlens = [4, 3, 2, 1]
        parts = get_seqlen_balanced_partitions(seqlens, k_partitions=2, equal_size=True)
        assert len(parts) == 2

    def test_all_indices_covered_equal_size(self):
        seqlens = [1, 2, 3, 4, 5, 6]
        parts = get_seqlen_balanced_partitions(seqlens, k_partitions=2, equal_size=True)
        assert _all_indices_covered(parts, len(seqlens))

    def test_all_indices_covered_unequal_size(self):
        seqlens = [7, 3, 5, 2, 8]
        parts = get_seqlen_balanced_partitions(seqlens, k_partitions=2, equal_size=False)
        assert _all_indices_covered(parts, len(seqlens))

    def test_no_empty_partitions(self):
        seqlens = [1, 2, 3, 4, 5, 6]
        parts = get_seqlen_balanced_partitions(seqlens, k_partitions=3, equal_size=True)
        for part in parts:
            assert len(part) > 0

    def test_partitions_are_sorted(self):
        # _check_and_sort_partitions should sort each partition's indices.
        seqlens = [10, 1, 5, 3, 8, 2]
        parts = get_seqlen_balanced_partitions(seqlens, k_partitions=2, equal_size=True)
        for part in parts:
            assert part == sorted(part), f"partition not sorted: {part}"

    def test_equal_size_partition_lengths(self):
        seqlens = [1, 2, 3, 4, 5, 6]
        parts = get_seqlen_balanced_partitions(seqlens, k_partitions=3, equal_size=True)
        sizes = [len(p) for p in parts]
        assert len(set(sizes)) == 1, f"unequal sizes: {sizes}"

    def test_equal_items_count(self):
        seqlens = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        parts = get_seqlen_balanced_partitions(seqlens, k_partitions=2, equal_size=True)
        assert sum(len(p) for p in parts) == len(seqlens)

    # --- balance quality ---

    def test_balance_quality_equal_size(self):
        seqlens = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        parts = get_seqlen_balanced_partitions(seqlens, k_partitions=2, equal_size=True)
        sums = _partition_sums(seqlens, parts)
        assert sorted(sums) == [27, 28]

    def test_balance_quality_unequal_size(self):
        seqlens = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        parts = get_seqlen_balanced_partitions(seqlens, k_partitions=3, equal_size=False)
        sums = _partition_sums(seqlens, parts)
        assert sorted(sums) == [18, 18, 19]

    # --- edge cases ---

    def test_exact_k_items_equal_size(self):
        # len == k_partitions: each partition gets exactly one item
        seqlens = [3, 1, 4, 2]
        parts = get_seqlen_balanced_partitions(seqlens, k_partitions=4, equal_size=True)
        assert len(parts) == 4
        assert all(len(p) == 1 for p in parts)
        assert _all_indices_covered(parts, 4)

    def test_exact_k_items_unequal_size(self):
        seqlens = [3, 1, 4, 2]
        parts = get_seqlen_balanced_partitions(seqlens, k_partitions=4, equal_size=False)
        assert _all_indices_covered(parts, 4)

    def test_identical_seqlens(self):
        seqlens = [5] * 6
        parts = get_seqlen_balanced_partitions(seqlens, k_partitions=3, equal_size=True)
        assert _all_indices_covered(parts, 6)
        sums = _partition_sums(seqlens, parts)
        assert max(sums) == min(sums)

    def test_k_equals_one(self):
        seqlens = [3, 1, 4, 1, 5]
        parts = get_seqlen_balanced_partitions(seqlens, k_partitions=1, equal_size=False)
        assert len(parts) == 1
        assert _all_indices_covered(parts, 5)


# ===========================================================================
# split_traj_group_by_dp
# ===========================================================================

def _make_group(token_lengths: list[int], pid: str) -> TrajectoryGroup:
    """Create a TrajectoryGroup where each trajectory carries a token list of the given length."""
    trajectories = []
    for i, n in enumerate(token_lengths):
        t = Trajectory(trajectory_id=f"{pid}_t{i}", prompt_id=pid)
        t.tokens = list(range(n))
        trajectories.append(t)
    return TrajectoryGroup(prompt_id=pid, trajectories=trajectories)


class TestSplitTrajGroupByDp:
    """Tests for split_traj_group_by_dp."""

    # --- output structure ---

    def test_returns_dp_size_partitions(self):
        groups = [_make_group([10], f"p{i}") for i in range(4)]
        result = split_traj_group_by_dp(groups, dp_size=2, num_mini_batch=1)
        assert len(result) == 2

    def test_all_groups_appear_exactly_once(self):
        groups = [_make_group([10], f"p{i}") for i in range(6)]
        result = split_traj_group_by_dp(groups, dp_size=3, num_mini_batch=1)
        recovered = [g for part in result for g in part]
        assert len(recovered) == len(groups)
        assert all(g in recovered for g in groups)

    def test_groups_stored_by_reference(self):
        groups = [_make_group([5, 5], f"p{i}") for i in range(4)]
        result = split_traj_group_by_dp(groups, dp_size=2, num_mini_batch=1)
        flat = [g for part in result for g in part]
        for g in groups:
            assert any(g is f for f in flat), "group reference not preserved"

    def test_each_rank_same_group_count(self):
        groups = [_make_group([i + 1], f"p{i}") for i in range(6)]
        result = split_traj_group_by_dp(groups, dp_size=3, num_mini_batch=1)
        sizes = [len(part) for part in result]
        assert len(set(sizes)) == 1, f"unequal rank sizes: {sizes}"

    # --- balance quality ---

    def test_token_sums_are_balanced(self):
        # 4 groups each with known token sums
        groups = [
            _make_group([30, 60], "p0"),   # 90
            _make_group([5, 15],  "p1"),   # 20
            _make_group([30, 20], "p2"),   # 50
            _make_group([15, 15], "p3"),   # 30
        ]
        result = split_traj_group_by_dp(groups, dp_size=2, num_mini_batch=1)
        sums = [sum(g.token_length for g in part) for part in result]
        # best balanced equal split: 110 vs 80, spread=30
        assert abs(sums[0] - sums[1]) <= 30

    def test_segment_count_is_balanced_before_tokens(self):
        segment_counts = [4, 3, 2, 1]
        groups = [
            _make_group([1000 if count <= 2 else count], f"p{i}")
            for i, count in enumerate(segment_counts)
        ]
        for group, count in zip(groups, segment_counts):
            group.trajectories[0].segments = [Segment() for _ in range(count)]

        result = split_traj_group_by_dp(groups, dp_size=2, num_mini_batch=1)
        segment_sums = [
            sum(group.segment_count for group in shard)
            for shard in result
        ]
        assert segment_sums == [5, 5]

    # --- input validation ---

    def test_fewer_groups_than_dp_size_raises(self):
        groups = [_make_group([10], f"p{i}") for i in range(2)]
        with pytest.raises(AssertionError):
            split_traj_group_by_dp(groups, dp_size=5, num_mini_batch=1)

    def test_non_divisible_raises(self):
        groups = [_make_group([10], f"p{i}") for i in range(5)]
        with pytest.raises(AssertionError):
            split_traj_group_by_dp(groups, dp_size=3, num_mini_batch=1)

    # --- edge cases ---

    def test_dp_size_equals_group_count(self):
        groups = [_make_group([i + 1], f"p{i}") for i in range(4)]
        result = split_traj_group_by_dp(groups, dp_size=4, num_mini_batch=1)
        assert len(result) == 4
        assert all(len(part) == 1 for part in result)

    def test_dp_size_one(self):
        groups = [_make_group([10], f"p{i}") for i in range(5)]
        result = split_traj_group_by_dp(groups, dp_size=1, num_mini_batch=1)
        assert len(result) == 1
        assert len(result[0]) == 5

    def test_identical_token_sums(self):
        groups = [_make_group([5, 5], f"p{i}") for i in range(4)]
        result = split_traj_group_by_dp(groups, dp_size=2, num_mini_batch=1)
        sums = [sum(g.token_length for g in part) for part in result]
        assert sums[0] == sums[1]
