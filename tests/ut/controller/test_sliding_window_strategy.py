"""Unit tests for sliding window strategies in coda/controller/rollout_sampler.py.

Run: python -m pytest tests/ut/controller/test_sliding_window_strategy.py -v
"""
import pytest
from omegaconf import OmegaConf
from coda.controller import (
    SLIDING_WINDOW_STRATEGY_REGISTRY,
    list_sliding_window_strategies,
)
from coda.controller.rollout_sampler import (
    NoWindowStrategy,
    WindowGatedStrategy,
    WindowedFifoStrategy,
    create_strategy,
    _ThreadSafeStrategy,
)
from coda.agentflow.trajectory_store import Trajectory, TrajectoryGroup, TrajectoryStatus


def _make_config(window_size=5, stale_steps=0.0, sliding_window="no-window"):
    """Create an OmegaConf config for strategy tests."""
    return OmegaConf.create({
        "data_sources": [{"num_prompts_per_step": window_size}],
        "fully_async": {
            "stale_steps": stale_steps,
            "sliding_window": sliding_window,
        },
    })


def _make_dispatch_group(prompt_id):
    """Create a minimal TrajectoryGroup for dispatch bookkeeping tests."""
    return TrajectoryGroup(prompt_id=prompt_id, trajectories=[])


def _make_queue_group(prompt_id, status=TrajectoryStatus.COMPLETED):
    """Create a minimal TrajQueue-style group mapping for strategy tests."""
    traj = Trajectory(
        trajectory_id=f"{prompt_id}_t0",
        prompt_id=prompt_id,
        status=status,
    )
    return {traj.trajectory_id: traj}


class TestNoWindowStrategy:
    """Tests for NoWindowStrategy dispatch logic."""

    def test_compute_dispatch_count_empty(self):
        """Verify full capacity is available when nothing is running."""
        s = NoWindowStrategy(_make_config(window_size=5))
        assert s.compute_dispatch_count(buf_qsize=0) == 5

    def test_compute_dispatch_count_with_running(self):
        """Verify dispatched groups reduce available capacity."""
        s = NoWindowStrategy(_make_config(window_size=5))
        s.on_dispatched([_make_dispatch_group("p0"), _make_dispatch_group("p1")])
        assert s.compute_dispatch_count(buf_qsize=0) == 3

    def test_compute_dispatch_count_with_buf(self):
        """Verify buffered items reduce available dispatch count."""
        s = NoWindowStrategy(_make_config(window_size=5))
        assert s.compute_dispatch_count(buf_qsize=3) == 2

    def test_compute_dispatch_count_saturated(self):
        """Verify zero dispatch when at max inflight."""
        s = NoWindowStrategy(_make_config(window_size=2))
        s.on_dispatched([_make_dispatch_group("p0"), _make_dispatch_group("p1")])
        assert s.compute_dispatch_count(buf_qsize=0) == 0

    def test_stale_steps_expands_capacity(self):
        """Verify stale_steps multiplier increases effective capacity."""
        s = NoWindowStrategy(_make_config(window_size=5, stale_steps=1.0))
        assert s.compute_dispatch_count(buf_qsize=0) == 10

    def test_on_collected_decrements(self):
        """Verify collecting a prompt frees one slot."""
        s = NoWindowStrategy(_make_config(window_size=5))
        s.on_dispatched([_make_dispatch_group("p0")])
        s.on_collected("p0")
        assert s.compute_dispatch_count(buf_qsize=0) == 5

    def test_on_reset(self):
        """Verify reset restores full capacity."""
        s = NoWindowStrategy(_make_config(window_size=5))
        s.on_dispatched([_make_dispatch_group("p0"), _make_dispatch_group("p1")])
        s.on_reset()
        assert s.compute_dispatch_count(buf_qsize=0) == 5

    def test_will_collect_rejects_only_aborted_groups(self):
        """Base status guard rejects ABORTED; FAILED is left to the data filter."""
        s = NoWindowStrategy(_make_config(window_size=5))

        assert s.will_collect(_make_queue_group("p0", TrajectoryStatus.COMPLETED)) is True
        assert s.will_collect(_make_queue_group("p1", TrajectoryStatus.ABORTED)) is False
        assert s.will_collect(_make_queue_group("p2", TrajectoryStatus.FAILED)) is True


class TestWindowGatedStrategy:
    """Tests for WindowGatedStrategy gated dispatch logic."""

    def test_compute_dispatch_count_empty(self):
        """Verify full window is available when nothing is running."""
        s = WindowGatedStrategy(_make_config(window_size=3))
        assert s.compute_dispatch_count(buf_qsize=0) == 3

    def test_window_constrains_dispatch(self):
        """Verify no dispatch when window is fully occupied."""
        s = WindowGatedStrategy(_make_config(window_size=3))
        groups = [_make_dispatch_group(f"p{i}") for i in range(3)]
        s.on_dispatched(groups)
        # All 3 slots used, oldest seq=0, next_seq=3, spread=3 == window
        assert s.compute_dispatch_count(buf_qsize=0) == 0

    def test_collecting_oldest_opens_window(self):
        """Verify collecting the oldest prompt opens a slot."""
        s = WindowGatedStrategy(_make_config(window_size=3))
        groups = [_make_dispatch_group(f"p{i}") for i in range(3)]
        s.on_dispatched(groups)
        s.on_collected("p0")
        # min_seq=1, next_seq=3, spread allows 1 more
        assert s.compute_dispatch_count(buf_qsize=0) == 1

    def test_on_reset_clears_state(self):
        """Verify reset restores full window capacity."""
        s = WindowGatedStrategy(_make_config(window_size=3))
        s.on_dispatched([_make_dispatch_group("p0")])
        s.on_reset()
        assert s.compute_dispatch_count(buf_qsize=0) == 3


class TestWindowedFifoStrategy:
    """Tests for WindowedFifoStrategy FIFO collection logic."""

    def test_compute_dispatch_count_like_no_window(self):
        """Verify dispatch count matches window_size when idle."""
        s = WindowedFifoStrategy(_make_config(window_size=5))
        assert s.compute_dispatch_count(buf_qsize=0) == 5

    def test_will_collect_within_window(self):
        """Verify all prompts within window are collectible."""
        s = WindowedFifoStrategy(_make_config(window_size=3))
        dispatch_groups = [_make_dispatch_group(f"p{i}") for i in range(3)]
        queue_groups = [_make_queue_group(f"p{i}") for i in range(3)]
        s.on_dispatched(dispatch_groups)
        assert s.will_collect(queue_groups[0]) is True
        assert s.will_collect(queue_groups[1]) is True
        assert s.will_collect(queue_groups[2]) is True

    def test_will_collect_outside_window(self):
        """Verify prompts outside window are not collectible."""
        s = WindowedFifoStrategy(_make_config(window_size=2))
        dispatch_groups = [_make_dispatch_group(f"p{i}") for i in range(4)]
        queue_groups = [_make_queue_group(f"p{i}") for i in range(4)]
        s.on_dispatched(dispatch_groups)
        # min_inflight_seq=0, window=2: seq 0,1 ok; seq 2,3 outside
        assert s.will_collect(queue_groups[0]) is True
        assert s.will_collect(queue_groups[1]) is True
        assert s.will_collect(queue_groups[2]) is False
        assert s.will_collect(queue_groups[3]) is False

    def test_will_collect_after_oldest_collected(self):
        """Verify window slides forward after oldest is collected."""
        s = WindowedFifoStrategy(_make_config(window_size=2))
        dispatch_groups = [_make_dispatch_group(f"p{i}") for i in range(4)]
        queue_groups = [_make_queue_group(f"p{i}") for i in range(4)]
        s.on_dispatched(dispatch_groups)
        s.on_collected("p0")
        # min_inflight_seq=1, window=2: seq 1,2 ok; seq 3 outside
        assert s.will_collect(queue_groups[1]) is True
        assert s.will_collect(queue_groups[2]) is True
        assert s.will_collect(queue_groups[3]) is False

    def test_will_collect_unknown_prompt_raises(self):
        """Verify RuntimeError for unknown prompt_id."""
        s = WindowedFifoStrategy(_make_config(window_size=3))
        with pytest.raises(RuntimeError, match="not in window mapping"):
            s.will_collect(_make_queue_group("unknown"))

    def test_will_collect_rejects_dirty_group_before_window_lookup(self):
        """Dirty groups are rejected without requiring a window mapping."""
        s = WindowedFifoStrategy(_make_config(window_size=3))

        assert s.will_collect(_make_queue_group("unknown", TrajectoryStatus.ABORTED)) is False

    def test_on_reset_clears_state(self):
        """Verify reset restores full capacity."""
        s = WindowedFifoStrategy(_make_config(window_size=3))
        s.on_dispatched([_make_dispatch_group("p0")])
        s.on_reset()
        assert s.compute_dispatch_count(buf_qsize=0) == 3


class TestCreateStrategy:
    """Tests for the create_strategy factory function."""

    def test_creates_known_strategies(self):
        """Verify all registered strategies are created successfully."""
        for name in list_sliding_window_strategies():
            s = create_strategy(_make_config(window_size=5, sliding_window=name))
            assert isinstance(s, _ThreadSafeStrategy)

    def test_unknown_strategy_raises(self):
        """Verify ValueError for unregistered strategy name."""
        assert "nonexistent" not in SLIDING_WINDOW_STRATEGY_REGISTRY
        with pytest.raises(ValueError, match="Unknown sliding_window strategy"):
            create_strategy(_make_config(sliding_window="nonexistent"))

    def test_thread_safe_wrapper_delegates(self):
        """Verify thread-safe wrapper correctly delegates calls."""
        s = create_strategy(_make_config(window_size=5, sliding_window="no-window"))
        assert s.compute_dispatch_count(buf_qsize=0) == 5
        s.on_dispatched([_make_dispatch_group("p0")])
        assert s.compute_dispatch_count(buf_qsize=0) == 4
