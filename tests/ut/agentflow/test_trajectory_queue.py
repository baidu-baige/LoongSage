"""Unit tests for coda/agentflow/trajectory_queue.py.

Run: python -m pytest tests/ut/agentflow/test_trajectory_queue.py -v
"""
import threading

import pytest
from coda.agentflow.trajectory_queue import TrajQueue
from coda.agentflow.trajectory_store import Trajectory, TrajectoryGroup


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_group(prompt_id: str, trajectory_ids: list[str]) -> TrajectoryGroup:
    """Build a TrajectoryGroup with the given prompt_id and trajectory ids."""
    trajs = [
        Trajectory(trajectory_id=tid, prompt_id=prompt_id, prompt=f"test_{tid}")
        for tid in trajectory_ids
    ]
    return TrajectoryGroup(prompt_id=prompt_id, trajectories=trajs)


class TestAdd:
    """Tests for TrajQueue.add method."""

    def test_add_single_trajectory(self):
        """Add a trajectory, verify it's in the queue."""
        queue = TrajQueue(group_sizes={0: 2})
        traj = Trajectory(trajectory_id="1", prompt_id="100", prompt="test")

        queue.add(traj)

        assert "100" in queue.groups
        assert len(queue.groups["100"]) == 1
        assert "1" in queue.groups["100"]
        assert queue.groups["100"]["1"].trajectory_id == "1"
        assert "1" in queue.tid_pid_map

    def test_add_duplicate_id(self):
        """Add trajectory with duplicate id, expect ValueError."""
        queue = TrajQueue(group_sizes={0: 2})
        traj1 = Trajectory(trajectory_id="1", prompt_id="100", prompt="test1")
        traj2 = Trajectory(trajectory_id="1", prompt_id="100", prompt="test2")

        queue.add(traj1)

        with pytest.raises(ValueError, match="Duplicate trajectory_id"):
            queue.add(traj2)

    def test_add_blocks_when_full(self):
        """Add blocks when maxsize is reached, unblocks after pop."""
        queue = TrajQueue(group_sizes={0: 2}, maxsize=2)
        queue.add(Trajectory(trajectory_id="1", prompt_id="100", prompt="t1"))
        queue.add(Trajectory(trajectory_id="2", prompt_id="100", prompt="t2"))

        entered = threading.Event()
        added = threading.Event()

        def add_third():
            entered.set()
            queue.add(Trajectory(trajectory_id="3", prompt_id="100", prompt="t3"))
            added.set()

        t = threading.Thread(target=add_third)
        t.start()

        # Wait until the writer thread has actually reached add(), so a slow
        # scheduler cannot make "still blocked" pass for the wrong reason.
        assert entered.wait(timeout=5)
        assert not added.wait(timeout=0.5), "add() must block while the queue is full"

        queue.pop()
        t.join(timeout=5)
        assert added.is_set()
        assert len(queue) == 2


class TestPop:
    """Tests for TrajQueue.pop method."""

    def test_pop_by_id(self):
        """Pop a specific trajectory by trajectory_id."""
        queue = TrajQueue(group_sizes={0: 2})
        traj1 = Trajectory(trajectory_id="1", prompt_id="100", prompt="test1")
        traj2 = Trajectory(trajectory_id="2", prompt_id="100", prompt="test2")
        queue.add(traj1)
        queue.add(traj2)

        result = queue.pop(trajectory_id="1")

        assert result is not None
        assert result.trajectory_id == "1"
        assert "1" not in queue.tid_pid_map
        assert "2" in queue.tid_pid_map

    def test_pop_any(self):
        """Pop any available trajectory when id is None."""
        queue = TrajQueue(group_sizes={0: 2})
        traj1 = Trajectory(trajectory_id="1", prompt_id="100", prompt="test1")
        traj2 = Trajectory(trajectory_id="2", prompt_id="101", prompt="test2")
        queue.add(traj1)
        queue.add(traj2)

        result = queue.pop()

        assert result is not None
        assert result.trajectory_id in ["1", "2"]
        assert result.trajectory_id not in queue.tid_pid_map

    def test_pop_nonexistent_id(self):
        """Pop with non-existent id, expect None."""
        queue = TrajQueue(group_sizes={0: 2})
        traj = Trajectory(trajectory_id="1", prompt_id="100", prompt="test")
        queue.add(traj)

        result = queue.pop(trajectory_id="999")

        assert result is None
        assert len(queue.groups["100"]) == 1

    def test_pop_empty_queue(self):
        """Pop from empty queue, expect None."""
        queue = TrajQueue(group_sizes={0: 2})

        result = queue.pop()

        assert result is None


class TestAddGroup:
    """Tests for TrajQueue.add_group method."""

    def test_add_group_success(self):
        """Add a TrajectoryGroup, verify all trajectories are in the queue."""
        queue = TrajQueue(group_sizes={0: 3})
        traj_group = _make_group("100", ["1", "2", "3"])

        queue.add_group(traj_group)

        assert "100" in queue.groups
        assert len(queue.groups["100"]) == 3
        assert all(tid in queue.tid_pid_map for tid in ["1", "2", "3"])

    def test_add_group_empty(self):
        """Add TrajectoryGroup with no trajectories — should be a no-op."""
        queue = TrajQueue(group_sizes={0: 2})

        queue.add_group(TrajectoryGroup(prompt_id="100", trajectories=[]))

        assert len(queue.groups) == 0
        assert len(queue.tid_pid_map) == 0

    def test_add_group_mixed_prompt_id(self):
        """Trajectories with different prompt_ids inside the group → ValueError."""
        queue = TrajQueue(group_sizes={0: 2})
        traj_group = TrajectoryGroup(
            prompt_id="100",
            trajectories=[
                Trajectory(trajectory_id="1", prompt_id="100", prompt="test1"),
                Trajectory(trajectory_id="2", prompt_id="101", prompt="test2"),
            ],
        )

        with pytest.raises(ValueError, match="same prompt_id"):
            queue.add_group(traj_group)

        # Verify no state was mutated
        assert len(queue.groups) == 0
        assert len(queue.tid_pid_map) == 0

    def test_add_group_duplicate_id(self):
        """Trajectory id already in queue inside the new group → ValueError."""
        queue = TrajQueue(group_sizes={0: 2})
        queue.add(Trajectory(trajectory_id="1", prompt_id="100", prompt="test1"))

        traj_group = TrajectoryGroup(
            prompt_id="100",
            trajectories=[
                Trajectory(trajectory_id="1", prompt_id="100", prompt="test2"),
                Trajectory(trajectory_id="2", prompt_id="100", prompt="test3"),
            ],
        )

        with pytest.raises(ValueError, match="Duplicate trajectory_id"):
            queue.add_group(traj_group)

        # Verify original trajectory still exists and "2" was not added
        assert len(queue.groups["100"]) == 1
        assert "1" in queue.tid_pid_map
        assert "2" not in queue.tid_pid_map

    def test_add_group_not_multiple_of_group_size(self):
        """Group length not a multiple of group_size → ValueError."""
        queue = TrajQueue(group_sizes={0: 3})

        with pytest.raises(ValueError, match="multiple of group_size"):
            queue.add_group(_make_group("100", ["1", "2"]))

    def test_add_group_multiple_of_group_size(self):
        """Group length of 2 × group_size should succeed."""
        queue = TrajQueue(group_sizes={0: 2})

        queue.add_group(_make_group("100", ["0", "1", "2", "3"]))

        assert len(queue.groups["100"]) == 4


class TestPopGroup:
    """Tests for TrajQueue.pop_group method."""

    def test_pop_group_by_prompt_id(self):
        """Pop group by specific prompt_id returns a TrajectoryGroup."""
        queue = TrajQueue(group_sizes={0: 2})
        queue.add_group(_make_group("100", ["1", "2"]))

        result = queue.pop_group(prompt_id="100")

        assert result is not None
        assert isinstance(result, TrajectoryGroup)
        assert result.prompt_id == "100"
        assert len(result.trajectories) == 2
        assert [t.trajectory_id for t in result.trajectories] == ["1", "2"]
        assert "100" not in queue.groups
        assert all(tid not in queue.tid_pid_map for tid in ["1", "2"])

    def test_pop_group_any(self):
        """Pop any ready group when prompt_id is None."""
        queue = TrajQueue(group_sizes={0: 2})
        queue.add_group(_make_group("100", ["1", "2"]))

        result = queue.pop_group()

        assert result is not None
        assert isinstance(result, TrajectoryGroup)
        assert len(result.trajectories) == 2
        assert "100" not in queue.groups

    def test_pop_group_insufficient_size(self):
        """Pop when group has fewer than group_size trajectories → None."""
        queue = TrajQueue(group_sizes={0: 3})
        queue.add(Trajectory(trajectory_id="1", prompt_id="100", prompt="test1"))
        queue.add(Trajectory(trajectory_id="2", prompt_id="100", prompt="test2"))

        result = queue.pop_group(prompt_id="100")

        assert result is None
        assert len(queue.groups["100"]) == 2

    def test_pop_group_nonexistent_prompt(self):
        """Pop with non-existent prompt_id → None."""
        queue = TrajQueue(group_sizes={0: 2})

        assert queue.pop_group(prompt_id="999") is None

    def test_pop_group_empty_queue(self):
        """Pop from empty queue → None."""
        queue = TrajQueue(group_sizes={0: 2})

        assert queue.pop_group() is None

    def test_pop_group_partial_group(self):
        """Pop group_size items; remaining trajectories stay in queue."""
        queue = TrajQueue(group_sizes={0: 2})
        queue.add_group(_make_group("100", ["1", "2", "3", "4"]))

        result = queue.pop_group(prompt_id="100")

        assert result is not None
        assert len(result.trajectories) == 2
        assert [t.trajectory_id for t in result.trajectories] == ["1", "2"]
        # Remaining two trajectories are still in the queue
        assert len(queue.groups["100"]) == 2
        assert list(queue.groups["100"].keys()) == ["3", "4"]


class TestWaitForGroup:
    """Tests for TrajQueue.wait_for_group method."""

    def test_wait_returns_immediately_when_ready(self):
        """wait_for_group returns immediately if a group is already ready."""
        queue = TrajQueue(group_sizes={0: 2})
        queue.add_group(_make_group("100", ["1", "2"]))

        result = queue.wait_for_group(timeout=1.0)

        assert result is not None
        assert isinstance(result, TrajectoryGroup)
        assert len(result.trajectories) == 2

    def test_wait_blocks_until_group_ready(self):
        """wait_for_group blocks until enough trajectories are added."""
        queue = TrajQueue(group_sizes={0: 2})
        result_holder = []
        waiting = threading.Event()

        def waiter():
            waiting.set()
            result_holder.append(queue.wait_for_group(timeout=5.0))

        t = threading.Thread(target=waiter)
        t.start()

        # Without this, a fast machine can complete both add() calls before the
        # waiter blocks, so the test would pass without exercising the wait.
        assert waiting.wait(timeout=5)

        queue.add(Trajectory(trajectory_id="1", prompt_id="100", prompt="t1"))
        # One trajectory is not a full group of 2, so the waiter must still block.
        assert not t.join(timeout=0.2) and t.is_alive()

        queue.add(Trajectory(trajectory_id="2", prompt_id="100", prompt="t2"))

        t.join(timeout=5.0)
        assert not t.is_alive(), "waiter thread should have completed"
        assert len(result_holder) == 1
        assert result_holder[0] is not None
        assert len(result_holder[0].trajectories) == 2

    def test_wait_timeout_returns_none(self):
        """wait_for_group returns None on timeout."""
        queue = TrajQueue(group_sizes={0: 2})

        assert queue.wait_for_group(timeout=0.1) is None

    def test_wait_with_will_collect_filters_groups(self):
        """wait_for_group skips groups where will_collect returns False."""
        queue = TrajQueue(group_sizes={0: 2})
        queue.add_group(_make_group("100", ["1", "2"]))
        queue.add_group(_make_group("200", ["3", "4"]))

        result = queue.wait_for_group(
            timeout=1.0,
            will_collect=lambda group: next(iter(group.values())).prompt_id == "200",
        )

        assert result is not None
        assert result.prompt_id == "200"
        # "100" group should still be in the queue
        assert "100" in queue.groups

    def test_wait_with_will_collect_all_rejected_times_out(self):
        """If will_collect rejects all ready groups, timeout returns None."""
        queue = TrajQueue(group_sizes={0: 2})
        queue.add_group(_make_group("100", ["1", "2"]))

        result = queue.wait_for_group(timeout=0.2, will_collect=lambda group: False)

        assert result is None
        # Group should still be in the queue
        assert "100" in queue.groups

    def test_wait_with_will_collect_none_returns_first_ready(self):
        """will_collect=None behaves like the old API (returns first ready)."""
        queue = TrajQueue(group_sizes={0: 2})
        queue.add_group(_make_group("100", ["1", "2"]))

        result = queue.wait_for_group(timeout=1.0, will_collect=None)

        assert result is not None
        assert result.prompt_id == "100"


class TestThreadSafety:
    """Tests for thread safety of TrajQueue."""

    def test_concurrent_add(self):
        """Multiple threads adding trajectories concurrently."""
        queue = TrajQueue(group_sizes={0: 10})
        num_threads = 10
        items_per_thread = 100
        errors = []
        lock = threading.Lock()

        def add_trajectories(thread_id):
            try:
                for i in range(items_per_thread):
                    traj = Trajectory(
                        trajectory_id=str(thread_id * items_per_thread + i),
                        prompt_id=str(thread_id),
                        prompt=f"test_{thread_id}_{i}"
                    )
                    queue.add(traj)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [
            threading.Thread(target=add_trajectories, args=(tid,))
            for tid in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        total_items = sum(len(group) for group in queue.groups.values())
        assert total_items == num_threads * items_per_thread

    def test_concurrent_pop(self):
        """Multiple threads popping trajectories concurrently."""
        queue = TrajQueue(group_sizes={0: 10})
        num_items = 100
        num_threads = 10

        for i in range(num_items):
            traj = Trajectory(trajectory_id=str(i), prompt_id=str(i % 10), prompt=f"test_{i}")
            queue.add(traj)

        popped_items = []
        lock = threading.Lock()

        def pop_trajectories():
            for _ in range(num_items // num_threads):
                result = queue.pop()
                if result is not None:
                    with lock:
                        popped_items.append(result)

        threads = [
            threading.Thread(target=pop_trajectories)
            for _ in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        popped_ids = [t.trajectory_id for t in popped_items]
        # Without the count check, a run where every pop() returned None would
        # trivially satisfy the uniqueness assertion.
        assert len(popped_ids) == num_items
        assert len(popped_ids) == len(set(popped_ids))
        assert len(queue) == 0

    def test_concurrent_mixed_operations(self):
        """Concurrent add / pop / add_group / pop_group must conserve trajectories.

        Asserting only "no exceptions" cannot catch a lost or duplicated
        trajectory, so this tracks how many went in and how many came out.
        """
        queue = TrajQueue(group_sizes={0: 5})
        num_iterations = 50
        errors = []
        popped_groups = []
        popped_singles = []
        added_single_ids = []
        added_group_ids = []
        lock = threading.Lock()

        def add_single():
            try:
                for i in range(num_iterations):
                    traj = Trajectory(
                        trajectory_id=str(i),
                        prompt_id=str(i % 10),
                        prompt=f"single_{i}"
                    )
                    queue.add(traj)
                    with lock:
                        added_single_ids.append(traj.trajectory_id)
            except Exception as e:
                with lock:
                    errors.append(e)

        def pop_single():
            for _ in range(num_iterations):
                result = queue.pop()
                if result is not None:
                    with lock:
                        popped_singles.append(result)

        def add_groups():
            try:
                for i in range(num_iterations):
                    traj_group = TrajectoryGroup(
                        prompt_id=str(100 + i),
                        trajectories=[
                            Trajectory(
                                trajectory_id=str(num_iterations + i * 5 + j),
                                prompt_id=str(100 + i),
                                prompt=f"group_{i}_{j}"
                            )
                            for j in range(5)
                        ],
                    )
                    queue.add_group(traj_group)
                    with lock:
                        added_group_ids.extend(
                            t.trajectory_id for t in traj_group.trajectories
                        )
            except Exception as e:
                with lock:
                    errors.append(e)

        def pop_groups():
            for _ in range(num_iterations):
                result = queue.pop_group()
                if result is not None:
                    with lock:
                        popped_groups.append(result)

        threads = [
            threading.Thread(target=add_single),
            threading.Thread(target=pop_single),
            threading.Thread(target=add_groups),
            threading.Thread(target=pop_groups),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []

        added_ids = set(added_single_ids) | set(added_group_ids)
        popped_ids = [t.trajectory_id for t in popped_singles]
        popped_ids += [
            t.trajectory_id for g in popped_groups for t in g.trajectories
        ]
        remaining_ids = [
            t.trajectory_id for group in queue.groups.values() for t in group.values()
        ]

        # Nothing may be popped twice, and popped + still-queued must account for
        # exactly what was added.
        assert len(popped_ids) == len(set(popped_ids))
        assert set(popped_ids) | set(remaining_ids) == added_ids
        assert len(popped_ids) + len(remaining_ids) == len(added_ids)


class TestIntegration:
    """Integration tests for TrajQueue."""

    def test_add_pop_workflow(self):
        """Add trajectories, pop them, verify correct behaviour."""
        queue = TrajQueue(group_sizes={0: 3})

        for i in range(6):
            traj = Trajectory(trajectory_id=str(i), prompt_id=str(i % 3), prompt=f"test_{i}")
            queue.add(traj)

        assert len(queue.groups) == 3
        assert len(queue.tid_pid_map) == 6

        popped = queue.pop(trajectory_id="0")
        assert popped.trajectory_id == "0"
        assert "0" not in queue.tid_pid_map

        popped = queue.pop()
        assert popped is not None
        assert popped.trajectory_id not in queue.tid_pid_map

        for _ in range(4):
            queue.pop()

        assert len(queue.groups) == 0
        assert len(queue.tid_pid_map) == 0

    def test_group_lifecycle(self):
        """Full lifecycle: add TrajectoryGroups then pop them all."""
        queue = TrajQueue(group_sizes={0: 3})

        for prompt_id in range(3):
            traj_group = TrajectoryGroup(
                prompt_id=str(prompt_id),
                trajectories=[
                    Trajectory(
                        trajectory_id=str(prompt_id * 3 + i),
                        prompt_id=str(prompt_id),
                        prompt=f"test_{prompt_id}_{i}"
                    )
                    for i in range(3)
                ],
            )
            queue.add_group(traj_group)

        assert len(queue.groups) == 3
        for prompt_id in range(3):
            assert len(queue.groups[str(prompt_id)]) == 3

        popped_groups = []
        for _ in range(3):
            group = queue.pop_group()
            assert group is not None
            assert isinstance(group, TrajectoryGroup)
            assert len(group.trajectories) == 3
            popped_groups.append(group)

        assert len(queue.groups) == 0
        assert len(queue.tid_pid_map) == 0

        all_ids = [t.trajectory_id for group in popped_groups for t in group.trajectories]
        assert len(all_ids) == len(set(all_ids))
        assert set(all_ids) == set(str(i) for i in range(9))

    def test_empty_trajectory_id_is_rejected(self):
        """Trajectories with empty trajectory_id are rejected."""
        queue = TrajQueue(group_sizes={0: 2})

        with pytest.raises(ValueError, match="trajectory_id and prompt_id must not be empty"):
            queue.add(Trajectory(trajectory_id="", prompt_id="100", prompt="test1"))

        assert queue.is_empty()

    def test_multiple_groups_same_prompt_pop_sequence(self):
        """Pop multiple groups sequentially from the same prompt_id."""
        queue = TrajQueue(group_sizes={0: 2})
        queue.add_group(_make_group("100", ["0", "1", "2", "3", "4", "5"]))

        group1 = queue.pop_group(prompt_id="100")
        assert len(group1.trajectories) == 2
        assert [t.trajectory_id for t in group1.trajectories] == ["0", "1"]

        group2 = queue.pop_group(prompt_id="100")
        assert len(group2.trajectories) == 2
        assert [t.trajectory_id for t in group2.trajectories] == ["2", "3"]

        group3 = queue.pop_group(prompt_id="100")
        assert len(group3.trajectories) == 2
        assert [t.trajectory_id for t in group3.trajectories] == ["4", "5"]

        assert "100" not in queue.groups


class TestLenAndIsEmpty:
    """Tests for TrajQueue.__len__ and is_empty methods."""

    def test_len_empty_queue(self):
        assert len(TrajQueue(group_sizes={0: 2})) == 0

    def test_len_after_add(self):
        queue = TrajQueue(group_sizes={0: 2})
        queue.add(Trajectory(trajectory_id="1", prompt_id="100", prompt="test1"))
        queue.add(Trajectory(trajectory_id="2", prompt_id="100", prompt="test2"))
        queue.add(Trajectory(trajectory_id="3", prompt_id="101", prompt="test3"))
        assert len(queue) == 3

    def test_len_after_pop(self):
        queue = TrajQueue(group_sizes={0: 2})
        queue.add(Trajectory(trajectory_id="1", prompt_id="100", prompt="test1"))
        queue.add(Trajectory(trajectory_id="2", prompt_id="100", prompt="test2"))
        queue.pop()
        assert len(queue) == 1

    def test_len_after_pop_group(self):
        queue = TrajQueue(group_sizes={0: 2})
        queue.add_group(_make_group("100", ["0", "1", "2", "3"]))
        queue.pop_group(prompt_id="100")
        assert len(queue) == 2

    def test_is_empty_true(self):
        assert TrajQueue(group_sizes={0: 2}).is_empty() is True

    def test_is_empty_false(self):
        queue = TrajQueue(group_sizes={0: 2})
        queue.add(Trajectory(trajectory_id="1", prompt_id="100", prompt="test"))
        assert queue.is_empty() is False

    def test_is_empty_after_popping_the_only_trajectory(self):
        queue = TrajQueue(group_sizes={0: 2})
        queue.add(Trajectory(trajectory_id="1", prompt_id="100", prompt="test"))
        queue.pop()
        assert queue.is_empty() is True


class TestPopAll:
    """Tests for TrajQueue.pop_all method."""

    def test_pop_all_empty_queue(self):
        queue = TrajQueue(group_sizes={0: 2})
        assert queue.pop_all() == []
        assert queue.is_empty()

    def test_pop_all_single_group(self):
        queue = TrajQueue(group_sizes={0: 2})
        queue.add_group(_make_group("100", ["0", "1", "2", "3"]))

        result = queue.pop_all()

        assert len(result) == 4
        assert [t.trajectory_id for t in result] == ["0", "1", "2", "3"]
        assert queue.is_empty()
        assert len(queue.groups) == 0

    def test_pop_all_multiple_groups(self):
        queue = TrajQueue(group_sizes={0: 2})
        for prompt_id in range(3):
            for i in range(2):
                queue.add(Trajectory(
                    trajectory_id=str(prompt_id * 2 + i),
                    prompt_id=str(prompt_id),
                    prompt=f"test_{prompt_id}_{i}"
                ))

        result = queue.pop_all()

        assert len(result) == 6
        assert {t.trajectory_id for t in result} == {str(i) for i in range(6)}
        assert queue.is_empty()

    def test_pop_all_clears_id_set(self):
        queue = TrajQueue(group_sizes={0: 2})
        queue.add(Trajectory(trajectory_id="1", prompt_id="100", prompt="test1"))
        queue.add(Trajectory(trajectory_id="2", prompt_id="100", prompt="test2"))

        queue.pop_all()

        assert len(queue.tid_pid_map) == 0

    def test_pop_all_with_mixed_trajectory_ids(self):
        queue = TrajQueue(group_sizes={0: 2})
        queue.add(Trajectory(trajectory_id="t1", prompt_id="100", prompt="test1"))
        queue.add(Trajectory(trajectory_id="t2", prompt_id="100", prompt="test2"))

        result = queue.pop_all()

        assert len(result) == 2
        assert queue.is_empty()


class TestPopAllGroup:
    """Tests for TrajQueue.pop_all_group method."""

    def test_pop_all_group_empty_queue(self):
        assert TrajQueue(group_sizes={0: 2}).pop_all_group() == []

    def test_pop_all_group_single_complete_group(self):
        queue = TrajQueue(group_sizes={0: 2})
        queue.add_group(_make_group("100", ["1", "2"]))

        result = queue.pop_all_group()

        assert len(result) == 1
        assert isinstance(result[0], TrajectoryGroup)
        assert len(result[0].trajectories) == 2
        assert [t.trajectory_id for t in result[0].trajectories] == ["1", "2"]
        assert queue.is_empty()

    def test_pop_all_group_multiple_complete_groups_same_prompt(self):
        queue = TrajQueue(group_sizes={0: 2})
        queue.add_group(_make_group("100", ["0", "1", "2", "3", "4", "5"]))

        result = queue.pop_all_group()

        assert len(result) == 3
        assert all(len(g.trajectories) == 2 for g in result)
        assert result[0].trajectories[0].trajectory_id == "0"
        assert result[1].trajectories[0].trajectory_id == "2"
        assert result[2].trajectories[0].trajectory_id == "4"
        assert queue.is_empty()

    def test_pop_all_group_multiple_prompts(self):
        queue = TrajQueue(group_sizes={0: 2})
        for prompt_id in range(3):
            for i in range(2):
                queue.add(Trajectory(
                    trajectory_id=str(prompt_id * 2 + i),
                    prompt_id=str(prompt_id),
                    prompt=f"test_{prompt_id}_{i}"
                ))

        result = queue.pop_all_group()

        assert len(result) == 3
        assert all(len(g.trajectories) == 2 for g in result)
        assert queue.is_empty()

    def test_pop_all_group_mixed_complete_and_incomplete(self):
        """Only complete groups are popped; incomplete group remains."""
        queue = TrajQueue(group_sizes={0: 3})
        for i in range(7):
            traj = Trajectory(trajectory_id=str(i), prompt_id=str(i // 3), prompt=f"test_{i}")
            queue.add(traj)

        result = queue.pop_all_group()

        assert len(result) == 2
        assert all(len(g.trajectories) == 3 for g in result)
        assert "2" in queue.groups
        assert len(queue.groups["2"]) == 1
        assert "6" in queue.groups["2"]

    def test_pop_all_group_no_complete_groups(self):
        queue = TrajQueue(group_sizes={0: 3})
        queue.add(Trajectory(trajectory_id="0", prompt_id="100", prompt="test_0"))
        queue.add(Trajectory(trajectory_id="1", prompt_id="100", prompt="test_1"))

        result = queue.pop_all_group()

        assert result == []
        assert len(queue.groups["100"]) == 2

    def test_pop_all_group_clears_popped_ids(self):
        """pop_all_group clears popped trajectory ids from tid_pid_map."""
        queue = TrajQueue(group_sizes={0: 2})
        queue.add(Trajectory(trajectory_id="1", prompt_id="100", prompt="test1"))
        queue.add(Trajectory(trajectory_id="2", prompt_id="100", prompt="test2"))
        queue.add(Trajectory(trajectory_id="3", prompt_id="101", prompt="test3"))
        queue.add(Trajectory(trajectory_id="4", prompt_id="101", prompt="test4"))

        queue.pop_all_group()

        assert len(queue.tid_pid_map) == 0
        assert queue.is_empty()

    def test_pop_all_group_partial_cleanup(self):
        """Fully-emptied prompt_id group is removed; partial group remains."""
        queue = TrajQueue(group_sizes={0: 2})
        for i in range(7):
            traj = Trajectory(
                trajectory_id=str(i),
                prompt_id="100" if i < 5 else "101",
                prompt=f"test_{i}"
            )
            queue.add(traj)

        result = queue.pop_all_group()

        assert len(result) == 3
        assert "100" in queue.groups
        assert len(queue.groups["100"]) == 1
        assert "101" not in queue.groups

    def test_pop_all_group_sequential_calls(self):
        queue = TrajQueue(group_sizes={0: 2})

        assert queue.pop_all_group() == []

        queue.add_group(_make_group("100", ["0", "1", "2", "3"]))

        result2 = queue.pop_all_group()
        assert len(result2) == 2
        assert queue.is_empty()

        assert queue.pop_all_group() == []


class TestObservability:
    """Tests for observability methods."""

    def test_group_count(self):
        queue = TrajQueue(group_sizes={0: 2})
        assert queue.group_count() == 0

        queue.add(Trajectory(trajectory_id="1", prompt_id="100", prompt="t1"))
        queue.add(Trajectory(trajectory_id="2", prompt_id="101", prompt="t2"))
        assert queue.group_count() == 2

        queue.add(Trajectory(trajectory_id="3", prompt_id="100", prompt="t3"))
        assert queue.group_count() == 2  # same prompt_id

    def test_pending_prompt_ids(self):
        queue = TrajQueue(group_sizes={0: 2})
        assert queue.pending_prompt_ids() == []

        queue.add(Trajectory(trajectory_id="1", prompt_id="100", prompt="t1"))
        queue.add(Trajectory(trajectory_id="2", prompt_id="200", prompt="t2"))
        assert set(queue.pending_prompt_ids()) == {"100", "200"}

    def test_group_size_of(self):
        queue = TrajQueue(group_sizes={0: 2})
        assert queue.group_size_of("100") == 0

        queue.add(Trajectory(trajectory_id="1", prompt_id="100", prompt="t1"))
        assert queue.group_size_of("100") == 1

        queue.add(Trajectory(trajectory_id="2", prompt_id="100", prompt="t2"))
        assert queue.group_size_of("100") == 2

        assert queue.group_size_of("999") == 0
