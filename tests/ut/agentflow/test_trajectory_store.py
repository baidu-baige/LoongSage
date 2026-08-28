"""Unit tests for ``coda.agentflow.trajectory_store``."""

from coda.agentflow.trajectory_store import (
    Segment,
    Trajectory,
    TrajectoryGroup,
    TrajectoryStatus,
    TrajectoryStore,
    Triplet,
)


def make_traj(trajectory_id: str, attempt_id: int = 0, reward: float | None = None) -> Trajectory:
    """Create a minimal trajectory for store-oriented tests."""
    kwargs = dict(
        trajectory_id=trajectory_id,
        prompt_id="prompt-0",
        attempt_id=attempt_id,
    )
    if reward is not None:
        kwargs["reward"] = reward
    return Trajectory(
        **kwargs,
    )


def make_example_trajectory() -> Trajectory:
    """Build the concrete example described in ``trajectory_store.py``."""
    return Trajectory(
        trajectory_id="traj-doc-example",
        prompt_id="prompt-0",
        tokens=[
            101, 102, 103,  # P1
            201, 202,       # R1
            301, 302,       # O1
            401,            # Vm
            501, 502,       # S1
            101, 102, 103,  # P2
            501, 502,       # S1'
            601, 602,       # R2
        ],
        loss_masks=[
            1, 1,           # R1
            0, 0, 0,        # O1 + Vm
            1, 1,           # S1
            0, 0, 0, 0, 0,  # P2 + S1'
            1, 1,           # R2
        ],
        rollout_log_probs=[
            -0.4, -0.5,
            0.0, 0.0, 0.0,
            -0.3, -0.2,
            0.0, 0.0, 0.0, 0.0, 0.0,
            -0.1, -0.2,
        ],
        token_rewards=[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
        segments=[
            Segment(
                token_start=0,
                token_end=10,
                logprob_start=0,
                logprob_end=7,
                triplets=[
                    Triplet(token_start=0, token_end=5, logprob_start=0, logprob_end=2),
                    Triplet(token_start=0, token_end=10, logprob_start=5, logprob_end=7),
                ],
            ),
            Segment(
                token_start=10,
                token_end=17,
                logprob_start=12,
                logprob_end=14,
                triplets=[
                    Triplet(token_start=10, token_end=17, logprob_start=12, logprob_end=14),
                ],
            ),
        ],
        num_turns=3,
        status=TrajectoryStatus.COMPLETED,
    )


class TestTrajectoryModels:
    """Tests for Trajectory / Segment / Triplet model invariants."""

    def test_trajectory_defaults(self) -> None:
        """Minimal trajectory should populate documented defaults."""
        traj = Trajectory(trajectory_id="t1", prompt_id="prompt-0")

        assert traj.prompt == ""
        assert traj.attempt_id == 0
        assert traj.start_rollout_weight_version == -1
        assert traj.end_rollout_weight_version == -1
        assert traj.tokens == []
        assert traj.loss_masks == []
        assert traj.rollout_log_probs == []
        assert traj.token_rewards == []
        assert traj.chat_completions == {}
        assert traj.metadata == {}
        assert traj.segments == []
        assert traj.status is TrajectoryStatus.PENDING
        assert traj.masked_out is False
        assert traj.active_segment_id == 0

    def test_segment_tree_field_defaults(self) -> None:
        """A bare Segment should default to a root, mainline, trainable node."""
        seg = Segment()

        assert seg.segment_id == 0
        assert seg.parent_segment_id is None
        assert seg.origin == "root"
        assert seg.depth == 0
        assert seg.trainable is True

    def test_example_segment_and_triplet_ranges(self) -> None:
        """Segment / Triplet ranges should slice the flat arrays exactly as documented."""
        traj = make_example_trajectory()
        seg0, seg1 = traj.segments
        turn0, turn1 = seg0.triplets
        turn2 = seg1.triplets[0]

        assert len(traj.tokens) == 17
        assert len(traj.loss_masks) == 14
        assert len(traj.rollout_log_probs) == 14
        assert len(traj.token_rewards) == 14

        assert traj.tokens[seg0.token_start:seg0.token_end] == [
            101, 102, 103, 201, 202, 301, 302, 401, 501, 502
        ]
        assert traj.loss_masks[seg0.logprob_start:seg0.logprob_end] == [1, 1, 0, 0, 0, 1, 1]
        assert traj.rollout_log_probs[seg0.logprob_start:seg0.logprob_end] == [
            -0.4, -0.5, 0.0, 0.0, 0.0, -0.3, -0.2
        ]

        assert traj.tokens[turn0.token_start:turn0.token_end] == [101, 102, 103, 201, 202]
        assert traj.loss_masks[turn0.logprob_start:turn0.logprob_end] == [1, 1]

        assert traj.tokens[turn1.token_start:turn1.token_end] == [
            101, 102, 103, 201, 202, 301, 302, 401, 501, 502
        ]
        assert traj.loss_masks[turn1.logprob_start:turn1.logprob_end] == [1, 1]
        assert traj.loss_masks[turn0.logprob_end:turn1.logprob_start] == [0, 0, 0]

        assert traj.tokens[seg1.token_start:seg1.token_end] == [101, 102, 103, 501, 502, 601, 602]
        assert traj.loss_masks[seg0.logprob_end:seg1.logprob_start] == [0, 0, 0, 0, 0]
        assert traj.tokens[turn2.token_start:turn2.token_end] == [101, 102, 103, 501, 502, 601, 602]
        assert traj.loss_masks[turn2.logprob_start:turn2.logprob_end] == [1, 1]
        assert traj.token_rewards[-1] == 1.0

class TestTrajectoryGroup:
    """Tests for TrajectoryGroup model."""

    def test_defaults(self) -> None:
        """Minimal TrajectoryGroup should have an empty trajectories list."""
        group = TrajectoryGroup(prompt_id="p0")
        assert group.prompt_id == "p0"
        assert group.trajectories == []

    def test_token_length_sums_tokens_across_trajectories(self) -> None:
        """token_length sums tokens; empty trajectories contribute 0."""
        assert TrajectoryGroup(prompt_id="p0").token_length == 0

        trajs = [
            Trajectory(trajectory_id="t0", prompt_id="p0", tokens=list(range(10))),
            Trajectory(trajectory_id="t1", prompt_id="p0", tokens=[]),
            Trajectory(trajectory_id="t2", prompt_id="p0", tokens=list(range(5))),
        ]
        assert TrajectoryGroup(prompt_id="p0", trajectories=trajs).token_length == 15


class TestTrajectoryStore:
    """Tests for TrajectoryStore CRUD and attempt semantics."""

    def test_add_and_get_latest_attempt(self) -> None:
        """get() without attempt_id should return only the latest attempt."""
        store = TrajectoryStore()
        store.add("t1", make_traj("t1", attempt_id=0))
        store.add("t1", make_traj("t1", attempt_id=1, reward=1.0))

        result = store.get(["t1"])

        assert list(result) == ["t1"]
        assert len(result["t1"]) == 1
        assert result["t1"][0].attempt_id == 1
        assert result["t1"][0].reward == 1.0

    def test_get_specific_attempt_id(self) -> None:
        """get(..., attempt_id=...) should filter attempts by their attempt id."""
        store = TrajectoryStore()
        for attempt_id in range(3):
            store.add("t1", make_traj("t1", attempt_id=attempt_id))

        result = store.get(["t1"], attempt_id=1)

        assert len(result["t1"]) == 1
        assert result["t1"][0].attempt_id == 1

    def test_get_omits_missing_ids(self) -> None:
        """Missing trajectory ids should be omitted from the returned mapping."""
        store = TrajectoryStore()
        store.add("t1", make_traj("t1"))

        result = store.get(["t1", "missing"])

        assert result == {"t1": [store.trajectory_data["t1"][-1]]}

    def test_update_replaces_matching_attempt_id(self) -> None:
        """update() should replace the attempt matching attempt_id, not the latest."""
        store = TrajectoryStore()
        store.add("t1", make_traj("t1", attempt_id=0, reward=0.0))
        store.add("t1", make_traj("t1", attempt_id=1, reward=1.0))

        updated = make_traj("t1", attempt_id=0, reward=9.0)
        updated.status = TrajectoryStatus.COMPLETED
        store.update("t1", updated)

        # attempt_id=0 should be updated, attempt_id=1 should be untouched
        assert [traj.attempt_id for traj in store.trajectory_data["t1"]] == [0, 1]
        assert store.trajectory_data["t1"][0].reward == 9.0
        assert store.trajectory_data["t1"][0].status is TrajectoryStatus.COMPLETED
        assert store.trajectory_data["t1"][1].reward == 1.0

    def test_update_ignores_stale_attempt(self) -> None:
        """update() with an unknown attempt_id should be a no-op (logs warning)."""
        store = TrajectoryStore()
        store.add("t1", make_traj("t1", attempt_id=0, reward=0.0))

        store.update("t1", make_traj("t1", attempt_id=99, reward=9.0))

        assert len(store.trajectory_data["t1"]) == 1
        assert store.trajectory_data["t1"][0].attempt_id == 0

    def test_update_warns_when_trajectory_id_missing(self) -> None:
        """update() should not create a new entry when trajectory_id is absent."""
        store = TrajectoryStore()

        store.update("nonexistent", make_traj("nonexistent"))

        assert "nonexistent" not in store.trajectory_data

    def test_get_trajectory_attempt_ids_preserves_insertion_order(self) -> None:
        """Attempt ids should be returned in the order they were appended."""
        store = TrajectoryStore()
        store.add("t1", make_traj("t1", attempt_id=2))
        store.add("t1", make_traj("t1", attempt_id=5))
        store.add("t2", make_traj("t2", attempt_id=1))

        result = store.get_trajectory_attempt_ids(["t1", "t2", "missing"])

        assert result == {"t1": [2, 5], "t2": [1]}

    def test_delete_removes_all_attempts_for_ids(self) -> None:
        """delete() should remove the full attempt history for each id."""
        store = TrajectoryStore()
        store.add("t1", make_traj("t1", attempt_id=0))
        store.add("t1", make_traj("t1", attempt_id=1))
        store.add("t2", make_traj("t2", attempt_id=0))

        store.delete(["t1"])

        assert "t1" not in store.trajectory_data
        assert "t2" in store.trajectory_data

    def test_clear_removes_everything(self) -> None:
        """clear() should empty the store completely."""
        store = TrajectoryStore()
        store.add("t1", make_traj("t1"))
        store.add("t2", make_traj("t2"))

        store.clear()

        assert store.trajectory_data == {}
