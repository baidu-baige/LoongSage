"""
# Copied from
# https://github.com/volcengine/verl/blob/468adf22c43b744348051fccd7a5d830c6c3c36a/verl/utils/seqlen_balancing.py
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""

import heapq


def karmarkar_karp(seqlen_list: list[int], k_partitions: int, equal_size: bool):
    """Partition a list of sequence lengths into *k_partitions* balanced subsets using the Karmarkar-Karp (Largest
    Differencing Method) heuristic.

    See: https://en.wikipedia.org/wiki/Largest_differencing_method

    The algorithm maintains a min-heap of State objects.  Each State represents a candidate k-way partition.  On every
    iteration the two states with the largest *spread* (max_sum − min_sum) are popped and merged so that the heaviest
    set of one state is paired with the lightest set of the other. This greedy pairing drives the partition sums
    towards equality.

    Args:
        seqlen_list:  Sequence lengths of each item (non-negative integers).
        k_partitions: Number of partitions to produce.
        equal_size:   If True, every partition must contain exactly len(seqlen_list) // k_partitions items, and
                      len(seqlen_list) must be divisible by k_partitions.  If False, partition sizes may differ;
                      only the sum is balanced.

    Returns:
        A list of *k_partitions* sub-lists, each containing the **indices** of the items assigned to that partition.
        The ordering of indices within each sub-list and the ordering of partitions themselves are determined by the
        algorithm and are not guaranteed to be sorted.

    Raises:
        AssertionError: If equal_size=True and len(seqlen_list) % k_partitions != 0.

    Example::

        >>> karmarkar_karp([1, 2, 3, 4, 5, 6], k_partitions=2, equal_size=True)
        [[1, 4, 3], [0, 5, 2]]   # sums: 11, 10
    """
    # see: https://en.wikipedia.org/wiki/Largest_differencing_method
    class Set:
        """A single partition bucket: tracks the indices it contains and their total seqlen sum."""

        def __init__(self) -> None:
            self.sum = 0
            self.items = []

        def add(self, idx: int, val: int):
            """Add one item (index, seqlen) to this bucket."""
            self.items.append((idx, val))
            self.sum += val

        def merge(self, other):
            """Absorb all items from *other* into this bucket."""
            self.items.extend(other.items)
            self.sum += other.sum

        def __lt__(self, other):
            if self.sum != other.sum:
                return self.sum < other.sum
            if len(self.items) != len(other.items):
                return len(self.items) < len(other.items)
            return self.items < other.items

        def __gt__(self, other):
            if self.sum != other.sum:
                return self.sum > other.sum
            if len(self.items) != len(other.items):
                return len(self.items) > len(other.items)
            return self.items > other.items

    class State:
        """A candidate k-way partition, kept in a min-heap ordered by spread."""

        def __init__(self, items: list[tuple[int, int]], k: int) -> None:
            """Initialise a state from a list of (index, seqlen) pairs.

            Args:
                items: Either a single (idx, seqlen) pair or exactly *k* pairs (one per partition).
                k:     Number of partitions this state tracks.
            """
            self.k = k
            # sets should always be decreasing order
            self.sets = [Set() for _ in range(k)]
            assert len(items) in [1, k], f"{len(items)} not in [1, {k}]"
            for i, (idx, seqlen) in enumerate(items):
                self.sets[i].add(idx=idx, val=seqlen)
            self.sets = sorted(self.sets, reverse=True)

        def get_partitions(self):
            """Return a list of index lists, one per partition bucket."""
            return [[idx for idx, _ in s.items] for s in self.sets]

        def merge(self, other):
            """Merge *other* into this state, pairing heaviest↔lightest buckets."""
            for i in range(self.k):
                self.sets[i].merge(other.sets[self.k - 1 - i])
            self.sets = sorted(self.sets, reverse=True)

        @property
        def spread(self) -> int:
            """Difference between the heaviest and lightest partition sums."""
            return self.sets[0].sum - self.sets[-1].sum

        def __lt__(self, other):
            # heapq is least heap, let the state with largest spread to be popped first,
            # if the spread is the same, let the state who has the largest set to be popped first.
            if self.spread != other.spread:
                return self.spread > other.spread
            return self.sets[0] > other.sets[0]

        def __repr__(self) -> str:
            return "[" + ",".join("{" + ",".join(str(seqlen) for _, seqlen in s.items) + "}" for s in self.sets) + "]"

    sorted_seqlen_list = sorted([(seqlen, i) for i, seqlen in enumerate(seqlen_list)])
    states_pq = []
    if equal_size:
        assert len(seqlen_list) % k_partitions == 0, f"{len(seqlen_list)} % {k_partitions} != 0"
        for offset in range(0, len(sorted_seqlen_list), k_partitions):
            items = []
            for i in range(k_partitions):
                seqlen, idx = sorted_seqlen_list[offset + i]
                items.append((idx, seqlen))
            heapq.heappush(states_pq, State(items=items, k=k_partitions))
    else:
        for seqlen, idx in sorted_seqlen_list:
            heapq.heappush(states_pq, State(items=[(idx, seqlen)], k=k_partitions))

    while len(states_pq) > 1:
        state0 = heapq.heappop(states_pq)
        state1 = heapq.heappop(states_pq)
        # merge states
        state0.merge(state1)
        heapq.heappush(states_pq, state0)

    final_state = states_pq[0]
    return final_state.get_partitions()


def get_seqlen_balanced_partitions(seqlen_list: list[int], k_partitions: int, equal_size: bool):
    """Partition items into *k_partitions* groups with balanced sequence-length sums.

    This is the public entry point used by the training loop to balance the total token workload across data-parallel
    ranks and microbatches. It delegates to karmarkar_karp for the core assignment and then validates and sorts the
    result.

    Args:
        seqlen_list:  Sequence lengths of each item.  Must contain at least *k_partitions* elements.
        k_partitions: Number of output partitions (e.g. number of DP ranks).
        equal_size:   If True, every partition must hold exactly len(seqlen_list) // k_partitions items, and
                      len(seqlen_list) must be divisible by *k_partitions*. If False, only the sum of lengths is
                      balanced; partition sizes may vary.

    Returns:
        A list of *k_partitions* sub-lists.  Each sub-list contains the **sorted** indices of the items assigned to
        that partition.  Every index in range(len(seqlen_list)) appears in exactly one sub-list.

    Raises:
        AssertionError: If len(seqlen_list) < k_partitions.
        AssertionError: If equal_size=True and len(seqlen_list) % k_partitions != 0.
        AssertionError: If the algorithm produces an empty partition or a partition that does not cover all indices
                        (internal sanity check).

    Example::

        >>> get_seqlen_balanced_partitions([10, 1, 5, 3], k_partitions=2, equal_size=True)
        [[0, 1], [2, 3]]   # each partition has 2 items; indices are sorted
    """
    assert len(seqlen_list) >= k_partitions, f"number of items:[{len(seqlen_list)}] < k_partitions:[{k_partitions}]"

    def _check_and_sort_partitions(partitions):
        """Validate that partitions cover all indices exactly once, then sort each."""
        assert len(partitions) == k_partitions, f"{len(partitions)} != {k_partitions}"
        seen_idx = set()
        sorted_partitions = [None] * k_partitions
        for i, partition in enumerate(partitions):
            assert len(partition) > 0, f"the {i}-th partition is empty"
            for idx in partition:
                seen_idx.add(idx)
            sorted_partitions[i] = sorted(partition)
        assert seen_idx == set(range(len(seqlen_list)))
        return sorted_partitions

    partitions = karmarkar_karp(seqlen_list=seqlen_list, k_partitions=k_partitions, equal_size=equal_size)
    return _check_and_sort_partitions(partitions)