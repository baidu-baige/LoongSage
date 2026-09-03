"""Unit tests for coda/controller/rollout_sampler.py.

Run: python -m pytest tests/ut/controller/test_rollout_sampler.py -v
"""
import sys
import os
import copy
from unittest.mock import Mock, AsyncMock, patch

import pytest
import asyncio
from omegaconf import OmegaConf

# Add the project root to sys.path to allow imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from coda.agentflow.trajectory_store import Trajectory, TrajectoryGroup, TrajectoryStatus


# ---------------------------------------------------------------------------
# Helper functions for creating test data
# ---------------------------------------------------------------------------

def make_minimal_config(**overrides):
    """Build a minimal config for RolloutSampler testing."""
    base = {
        "fully_async": {
            "enable": False,
            "sliding_window": "no-window",
            "stale_steps": 0,
        },
        "rollout": {
            "partial": False,
            "sampler": {
                "name": "dynamic",
                "num_oversample": 5,
                "max_refill_count": 10,
                "refill_ratio": 2,
                "timeout": 60.0,
            },
            "filter": None,
            "eval": {
                "interval": -1,
                "temperature": None,
            },
        },
        "data_sources": [
            {
                "dataset": {},
                # Mirrors conf/default.yaml: agent always exists, max_turns is optional.
                "agent": {"name": None},
                "num_trajectories_per_prompt": 2,
                "num_prompts_per_step": 5,
                "max_response_len_per_trajectory": 1024,
            }
        ],
        "trainer": {
            "mini_batch_size": 4,
        },
    }
    cfg = OmegaConf.create(base)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return cfg


def create_test_trajectory(
    trajectory_id: str = "test_prompt_step0_traj_0",
    prompt_id: str = "test_prompt_0",
    reward: float = 0.5,
    status: str = "completed",
    rollout_weight_versions: list[int] | None = None,
    start_rollout_weight_version: int = -1,
    end_rollout_weight_version: int = -1,
    is_correct: bool = False,
    num_turns: int = 0,
    ds_index: int = 0,
) -> Trajectory:
    """Create a test Trajectory for testing."""
    return Trajectory(
        trajectory_id=trajectory_id,
        prompt_id=prompt_id,
        prompt="Test prompt",
        tokens=[1, 2, 3, 4, 5],
        loss_masks=[0, 1, 1, 1, 0],
        rollout_log_probs=[0.0, -0.5, -0.3, -0.2, 0.0],
        rollout_weight_versions=rollout_weight_versions or [],
        start_rollout_weight_version=start_rollout_weight_version,
        end_rollout_weight_version=end_rollout_weight_version,
        reward=reward,
        is_correct=is_correct,
        num_turns=num_turns,
        token_rewards=[0.0, 0.0, 0.0, 0.0, reward],
        status=TrajectoryStatus(status),
        ds_index=ds_index,
    )


def create_test_trajectory_group(
    prompt_id: str = "test_prompt_0",
    num_trajectories: int = 2,
    step: int = 0,
    correctness: list[bool] | None = None,
    ds_index: int = 0,
) -> TrajectoryGroup:
    """Create a test TrajectoryGroup for testing.

    ``correctness`` overrides per-trajectory is_correct and its length wins over
    ``num_trajectories``.
    """
    flags = correctness if correctness is not None else [False] * num_trajectories
    trajs = [
        create_test_trajectory(
            trajectory_id=f"{prompt_id}_step{step}_traj_{i}",
            prompt_id=prompt_id,
            reward=0.5 + i * 0.1,
            is_correct=flag,
            ds_index=ds_index,
        )
        for i, flag in enumerate(flags)
    ]
    return TrajectoryGroup(prompt_id=prompt_id, trajectories=trajs)


# ===========================================================================
# RolloutSampler.__init__
# ===========================================================================

class TestRolloutSamplerInit:
    """Tests for RolloutSampler initialization."""

    def test_init_stores_dependencies(self):
        """Test that __init__ stores all dependencies correctly."""
        from coda.controller.rollout_sampler import RolloutSampler

        config = make_minimal_config()
        datasource = Mock()
        agentflow = Mock()
        traj_queue = Mock()
        agentflow.traj_queue = traj_queue

        sampler = RolloutSampler(config, [datasource], agentflow)

        assert sampler.config is config
        assert sampler.datasources == [datasource]
        assert sampler.agentflow is agentflow
        assert sampler.traj_queue is traj_queue

    def test_init_with_minimal_config(self):
        """Test that __init__ works with minimal config."""
        from coda.controller.rollout_sampler import RolloutSampler

        config = make_minimal_config()
        sampler = RolloutSampler(config, Mock(), Mock())

        assert sampler.config.rollout.sampler.name == "dynamic"

    def test_init_creates_data_filter(self):
        """Test that __init__ creates a DataFilter instance."""
        from coda.controller.rollout_sampler import RolloutSampler
        from coda.data_factory.data_filter import DataFilter

        config = make_minimal_config()
        sampler = RolloutSampler(config, Mock(), Mock())

        assert isinstance(sampler.data_filter, DataFilter)

    def test_init_sets_timeout(self):
        """Test that __init__ sets timeout from config."""
        from coda.controller.rollout_sampler import RolloutSampler

        config = make_minimal_config()
        sampler = RolloutSampler(config, Mock(), Mock())

        assert sampler.timeout == 60.0

    def test_init_creates_empty_metrics(self):
        """Test that __init__ creates empty metrics dict."""
        from coda.controller.rollout_sampler import RolloutSampler

        config = make_minimal_config()
        sampler = RolloutSampler(config, Mock(), Mock())

        assert sampler.metrics == {}


# ===========================================================================
# RolloutSampler.__call__
# ===========================================================================

class TestRolloutSamplerCall:
    """Tests for RolloutSampler.__call__ method."""

    @pytest.mark.asyncio
    async def test_call_dynamic_sampler(self):
        """Test __call__ with dynamic rollout sampler dispatches to dynamic_rollout."""
        from coda.controller.rollout_sampler import RolloutSampler

        config = make_minimal_config()
        sampler = RolloutSampler(config, Mock(), Mock())
        sampler.dynamic_rollout = AsyncMock(return_value=[])

        result = await sampler(step=0)

        assert result == []
        # __call__ passes (num_oversample, step) to dynamic_rollout
        sampler.dynamic_rollout.assert_called_once_with(5, 0)

    @pytest.mark.asyncio
    async def test_call_unknown_sampler_raises_value_error(self):
        """Test __call__ with unknown sampler raises ValueError."""
        from coda.controller.rollout_sampler import RolloutSampler

        config = make_minimal_config()
        config.rollout.sampler.name = "unknown"
        sampler = RolloutSampler(config, Mock(), Mock())

        with pytest.raises(ValueError, match="Unknown sampler"):
            await sampler(step=0)

    @pytest.mark.asyncio
    async def test_call_passes_step_and_batch_sizes(self):
        """Test __call__ passes correct arguments to dynamic_rollout."""
        from coda.controller.rollout_sampler import RolloutSampler

        config = make_minimal_config()
        config.data_sources[0].num_prompts_per_step = 4
        config.rollout.sampler.num_oversample = 4
        sampler = RolloutSampler(config, Mock(), Mock())
        sampler.dynamic_rollout = AsyncMock(return_value=[])

        await sampler(step=3)

        # __call__ passes (num_oversample, step) to dynamic_rollout
        sampler.dynamic_rollout.assert_called_once_with(4, 3)


# ===========================================================================
# RolloutSampler._pre_process
# ===========================================================================

class TestPreProcess:
    """Tests for RolloutSampler._pre_process method."""

    def test_pre_process_returns_traj_group_when_kept(self):
        """Test _pre_process returns TrajectoryGroup when filter passes."""
        from coda.controller.rollout_sampler import RolloutSampler

        sampler = RolloutSampler(make_minimal_config(), Mock(), Mock())
        # rewards 0.5, 0.6 — mixed, will pass the reward filter
        traj_group = create_test_trajectory_group("prompt_0", num_trajectories=2, step=0)

        result = sampler._pre_process(traj_group, step=0)

        assert result is not None
        assert isinstance(result, TrajectoryGroup)
        assert len(result.trajectories) == 2

    def test_pre_process_returns_none_when_filtered(self):
        """Test _pre_process returns None when data_filter drops the group."""
        from coda.controller.rollout_sampler import RolloutSampler

        sampler = RolloutSampler(make_minimal_config(), Mock(), Mock())
        sampler.data_filter = Mock()
        sampler.data_filter.apply = Mock(return_value=None)

        traj_group = create_test_trajectory_group("prompt_0", num_trajectories=2, step=0)
        result = sampler._pre_process(traj_group, step=0)

        assert result is None

    def test_pre_process_is_sync(self):
        """Test _pre_process is a regular (sync) method, not async."""
        import inspect
        from coda.controller.rollout_sampler import RolloutSampler

        assert not inspect.iscoroutinefunction(RolloutSampler._pre_process)

    def test_pre_process_raises_on_failed_trajectory(self):
        """Test _pre_process raises ValueError when a trajectory has FAILED status."""
        from coda.controller.rollout_sampler import RolloutSampler

        sampler = RolloutSampler(make_minimal_config(), Mock(), Mock())
        failed_traj = create_test_trajectory(
            trajectory_id="prompt_0_step0_traj_0",
            prompt_id="prompt_0",
            status="failed",
        )
        traj_group = TrajectoryGroup(prompt_id="prompt_0", trajectories=[failed_traj])

        with pytest.raises(ValueError, match="preprocess failed by dirty traj"):
            sampler._pre_process(traj_group, step=0)

    def test_pre_process_accumulates_metrics(self):
        """Test _pre_process calls _stat_metrics to accumulate metrics."""
        from coda.controller.rollout_sampler import RolloutSampler

        sampler = RolloutSampler(make_minimal_config(), Mock(), Mock())
        traj_group = create_test_trajectory_group("prompt_0", num_trajectories=2, step=1)

        sampler._pre_process(traj_group, step=1)

        assert 1 in sampler.metrics
        assert "rewards" in sampler.metrics[1]


# ===========================================================================
# RolloutSampler._stat_metrics
# ===========================================================================

class TestStatMetrics:
    """Tests for RolloutSampler._stat_metrics method."""

    def test_stat_metrics_initializes_step(self):
        """Test _stat_metrics initializes metrics dict for a new step."""
        from coda.controller.rollout_sampler import RolloutSampler

        sampler = RolloutSampler(make_minimal_config(), Mock(), Mock())
        traj_group = create_test_trajectory_group("prompt_0", num_trajectories=2, step=0)

        sampler._stat_metrics(traj_group, step=0)

        assert 0 in sampler.metrics

    def test_stat_metrics_frees_previous_step(self):
        """Test _stat_metrics frees memory from previous steps."""
        from coda.controller.rollout_sampler import RolloutSampler

        sampler = RolloutSampler(make_minimal_config(), Mock(), Mock())
        traj_group_0 = create_test_trajectory_group("prompt_0", num_trajectories=2, step=0)
        traj_group_1 = create_test_trajectory_group("prompt_1", num_trajectories=2, step=1)

        sampler._stat_metrics(traj_group_0, step=0)
        assert 0 in sampler.metrics

        sampler._stat_metrics(traj_group_1, step=1)
        # Previous step should be gone
        assert 0 not in sampler.metrics
        assert 1 in sampler.metrics

    def test_stat_metrics_accumulates_rewards(self):
        """Test _stat_metrics accumulates rewards correctly."""
        from coda.controller.rollout_sampler import RolloutSampler

        sampler = RolloutSampler(make_minimal_config(), Mock(), Mock())
        traj_group = create_test_trajectory_group("prompt_0", num_trajectories=2, step=0)
        rewards = [t.reward for t in traj_group.trajectories]

        sampler._stat_metrics(traj_group, step=0)

        assert sampler.metrics[0]["rewards"] == rewards

    def test_stat_metrics_computes_summary_stats(self):
        """Test _stat_metrics computes mean/min/max summary stats."""
        from coda.controller.rollout_sampler import RolloutSampler

        sampler = RolloutSampler(make_minimal_config(), Mock(), Mock())
        traj_group = create_test_trajectory_group("prompt_0", num_trajectories=2, step=0)

        sampler._stat_metrics(traj_group, step=0)

        metrics = sampler.metrics[0]
        assert "rollout/reward_mean" in metrics
        assert "rollout/reward_max" in metrics
        assert "rollout/reward_min" in metrics
        assert "rollout/completed_count" in metrics

    def test_stat_metrics_response_clip_ratio(self):
        """Test _stat_metrics reports response clip_ratio against the generation budget."""
        from coda.controller.rollout_sampler import RolloutSampler

        # response_len for the test trajectory is len(rollout_log_probs) == 4.
        config = make_minimal_config()
        config.data_sources[0].max_response_len_per_trajectory = 4
        sampler = RolloutSampler(config, Mock(), Mock())
        traj_group = create_test_trajectory_group("prompt_0", num_trajectories=2, step=0)

        sampler._stat_metrics(traj_group, step=0)

        # Both trajectories hit the budget (response_len == 4 >= 4) -> ratio 1.0.
        assert sampler.metrics[0]["rollout/response_length_clip_ratio"] == pytest.approx(1.0)

    def test_stat_metrics_turns_clip_ratio(self):
        """Test _stat_metrics reports turns_clip_ratio against agent.max_turns."""
        from coda.controller.rollout_sampler import RolloutSampler

        config = make_minimal_config()
        config.data_sources[0].agent.max_turns = 3
        sampler = RolloutSampler(config, Mock(), Mock())
        traj_group = TrajectoryGroup(
            prompt_id="prompt_0",
            trajectories=[
                create_test_trajectory(trajectory_id="t0", prompt_id="prompt_0", num_turns=3),
                create_test_trajectory(trajectory_id="t1", prompt_id="prompt_0", num_turns=5),
                create_test_trajectory(trajectory_id="t2", prompt_id="prompt_0", num_turns=2),
                create_test_trajectory(trajectory_id="t3", prompt_id="prompt_0", num_turns=0),
            ],
        )

        sampler._stat_metrics(traj_group, step=0)

        # num_turns >= 3 for two of four trajectories.
        assert sampler.metrics[0]["rollout/turns_clip_ratio"] == pytest.approx(0.5)

    def test_stat_metrics_turns_clip_ratio_zero_without_max_turns(self):
        """Data sources without agent.max_turns never count as turn-clipped."""
        from coda.controller.rollout_sampler import RolloutSampler

        sampler = RolloutSampler(make_minimal_config(), Mock(), Mock())
        traj_group = TrajectoryGroup(
            prompt_id="prompt_0",
            trajectories=[
                create_test_trajectory(trajectory_id="t0", prompt_id="prompt_0", num_turns=99),
            ],
        )

        sampler._stat_metrics(traj_group, step=0)

        assert sampler.metrics[0]["rollout/turns_clip_ratio"] == pytest.approx(0.0)

    def test_stat_metrics_accumulates_across_multiple_groups(self):
        """Test _stat_metrics accumulates data from multiple groups at same step."""
        from coda.controller.rollout_sampler import RolloutSampler

        sampler = RolloutSampler(make_minimal_config(), Mock(), Mock())
        group1 = create_test_trajectory_group("prompt_0", num_trajectories=2, step=0)
        group2 = create_test_trajectory_group("prompt_1", num_trajectories=2, step=0)

        sampler._stat_metrics(group1, step=0)
        sampler._stat_metrics(group2, step=0)

        assert sampler.metrics[0]["rollout/completed_count"] == 4


# ===========================================================================
# RolloutSampler._stat_metrics — group-level correctness ratios
# ===========================================================================

class TestStatMetricsGroupCorrectness:
    """Tests for rollout/all_correct_ratio and rollout/all_wrong_ratio."""

    def test_all_correct_and_all_wrong_groups(self):
        """One all-correct group and one all-wrong group -> 0.5 / 0.5."""
        from coda.controller.rollout_sampler import RolloutSampler

        sampler = RolloutSampler(make_minimal_config(), Mock(), Mock())
        sampler._stat_metrics(
            create_test_trajectory_group("prompt_0", correctness=[True, True]), step=0
        )
        sampler._stat_metrics(
            create_test_trajectory_group("prompt_1", correctness=[False, False]), step=0
        )

        metrics = sampler.metrics[0]
        assert metrics["rollout/all_correct_ratio"] == pytest.approx(0.5)
        assert metrics["rollout/all_wrong_ratio"] == pytest.approx(0.5)

    def test_mixed_group_counts_in_neither_numerator(self):
        """A group with both correct and wrong trajectories is neither all-correct nor all-wrong."""
        from coda.controller.rollout_sampler import RolloutSampler

        sampler = RolloutSampler(make_minimal_config(), Mock(), Mock())
        sampler._stat_metrics(
            create_test_trajectory_group("prompt_0", correctness=[True, False]), step=0
        )

        metrics = sampler.metrics[0]
        assert metrics["rollout/all_correct_ratio"] == pytest.approx(0.0)
        assert metrics["rollout/all_wrong_ratio"] == pytest.approx(0.0)

    def test_default_trajectories_count_as_all_wrong(self):
        """is_correct defaults to False, so an unset group is all-wrong rather than skipped."""
        from coda.controller.rollout_sampler import RolloutSampler

        sampler = RolloutSampler(make_minimal_config(), Mock(), Mock())
        sampler._stat_metrics(create_test_trajectory_group("prompt_0", num_trajectories=2), step=0)

        metrics = sampler.metrics[0]
        assert metrics["rollout/all_correct_ratio"] == pytest.approx(0.0)
        assert metrics["rollout/all_wrong_ratio"] == pytest.approx(1.0)
        # Unrelated rollout metrics must keep working.
        assert metrics["rollout/completed_count"] == 2

    def test_ratios_denominator_counts_groups_not_trajectories(self):
        """Denominator is the number of groups, independent of group size."""
        from coda.controller.rollout_sampler import RolloutSampler

        sampler = RolloutSampler(make_minimal_config(), Mock(), Mock())
        sampler._stat_metrics(
            create_test_trajectory_group("prompt_0", correctness=[True, True, True, True]), step=0
        )
        sampler._stat_metrics(
            create_test_trajectory_group("prompt_1", correctness=[False, True]), step=0
        )

        metrics = sampler.metrics[0]
        assert metrics["rollout/completed_count"] == 6
        assert metrics["rollout/all_correct_ratio"] == pytest.approx(0.5)
        assert metrics["rollout/all_wrong_ratio"] == pytest.approx(0.0)


# ===========================================================================
# RolloutSampler._stat_metrics — per-data-source breakdown
# ===========================================================================

class TestPerDsMetrics:
    """Tests for rollout_per_ds/ds{ds_index}_* metrics."""

    @staticmethod
    def _make_config(num_data_sources: int = 2):
        """Clone the single data source of make_minimal_config() N times."""
        cfg = make_minimal_config()
        ds = OmegaConf.to_container(cfg.data_sources[0], resolve=True)
        cfg.data_sources = [copy.deepcopy(ds) for _ in range(num_data_sources)]
        return cfg

    @staticmethod
    def _make_group(prompt_id, ds_index, rewards, num_turns=0, correctness=None):
        flags = correctness if correctness is not None else [False] * len(rewards)
        return TrajectoryGroup(
            prompt_id=prompt_id,
            trajectories=[
                create_test_trajectory(
                    trajectory_id=f"{prompt_id}_traj_{i}",
                    prompt_id=prompt_id,
                    reward=reward,
                    num_turns=num_turns,
                    is_correct=flag,
                    ds_index=ds_index,
                )
                for i, (reward, flag) in enumerate(zip(rewards, flags))
            ],
        )

    def test_single_data_source_reports_nothing(self):
        """A single source would only duplicate rollout/*, so nothing is reported."""
        from coda.controller.rollout_sampler import RolloutSampler

        sampler = RolloutSampler(make_minimal_config(), Mock(), Mock())
        sampler._stat_metrics(create_test_trajectory_group("prompt_0", num_trajectories=2), step=0)

        assert not any(k.startswith("rollout_per_ds/") for k in sampler.metrics[0])

    def test_reward_stats_are_split_per_data_source(self):
        """Each source reports its own reward stats; rollout/* stays pooled."""
        from coda.controller.rollout_sampler import RolloutSampler

        sampler = RolloutSampler(self._make_config(), Mock(), Mock())
        sampler._stat_metrics(self._make_group("prompt_0", 0, [0.0, 1.0]), step=0)
        sampler._stat_metrics(self._make_group("prompt_1", 1, [0.2, 0.2]), step=0)

        metrics = sampler.metrics[0]
        assert metrics["rollout_per_ds/ds0_reward_mean"] == pytest.approx(0.5)
        assert metrics["rollout_per_ds/ds0_reward_max"] == pytest.approx(1.0)
        assert metrics["rollout_per_ds/ds0_reward_min"] == pytest.approx(0.0)
        assert metrics["rollout_per_ds/ds1_reward_mean"] == pytest.approx(0.2)
        assert metrics["rollout/reward_mean"] == pytest.approx(0.35)

    def test_clip_ratios_use_each_data_source_threshold(self):
        """Clip ratios are per-source, matching the per-source budget they compare against."""
        from coda.controller.rollout_sampler import RolloutSampler

        cfg = self._make_config()
        # response_len for the test trajectory is len(rollout_log_probs) == 5.
        cfg.data_sources[0].max_response_len_per_trajectory = 4
        cfg.data_sources[0].agent.max_turns = 3
        cfg.data_sources[1].max_response_len_per_trajectory = 1024
        sampler = RolloutSampler(cfg, Mock(), Mock())

        sampler._stat_metrics(self._make_group("prompt_0", 0, [0.5, 0.5], num_turns=3), step=0)
        sampler._stat_metrics(self._make_group("prompt_1", 1, [0.5, 0.5], num_turns=3), step=0)

        metrics = sampler.metrics[0]
        assert metrics["rollout_per_ds/ds0_response_length_clip_ratio"] == pytest.approx(1.0)
        assert metrics["rollout_per_ds/ds1_response_length_clip_ratio"] == pytest.approx(0.0)
        assert metrics["rollout_per_ds/ds0_turns_clip_ratio"] == pytest.approx(1.0)
        assert metrics["rollout_per_ds/ds1_turns_clip_ratio"] == pytest.approx(0.0)
        assert metrics["rollout_per_ds/ds0_num_turns_max"] == 3

    def test_group_ratios_are_split_per_data_source(self):
        """Each source scores its own groups; rollout/* pools them."""
        from coda.controller.rollout_sampler import RolloutSampler

        sampler = RolloutSampler(self._make_config(), Mock(), Mock())
        sampler._stat_metrics(
            self._make_group("prompt_0", 0, [0.0, 0.0], correctness=[False, False]), step=0
        )
        sampler._stat_metrics(
            self._make_group("prompt_1", 1, [1.0, 1.0], correctness=[True, True]), step=0
        )

        metrics = sampler.metrics[0]
        assert metrics["rollout_per_ds/ds0_all_correct_ratio"] == pytest.approx(0.0)
        assert metrics["rollout_per_ds/ds0_all_wrong_ratio"] == pytest.approx(1.0)
        assert metrics["rollout_per_ds/ds1_all_correct_ratio"] == pytest.approx(1.0)
        assert metrics["rollout_per_ds/ds1_all_wrong_ratio"] == pytest.approx(0.0)
        # Pooled ratios still cover both groups.
        assert metrics["rollout/all_correct_ratio"] == pytest.approx(0.5)
        assert metrics["rollout/all_wrong_ratio"] == pytest.approx(0.5)


# ===========================================================================
# RolloutSampler.calculate_partial_ratio
# ===========================================================================

class TestCalculatePartialRatio:
    """Tests for RolloutSampler.calculate_partial_ratio method."""

    def test_empty_trajectories_returns_zero(self):
        """Test calculate_partial_ratio returns 0.0 for empty input."""
        from coda.controller.rollout_sampler import RolloutSampler

        sampler = RolloutSampler(make_minimal_config(), Mock(), Mock())

        result = sampler.calculate_partial_ratio([])

        assert result == 0.0

    def test_all_same_versions_returns_zero(self):
        """Test calculate_partial_ratio returns 0.0 when start == end for every trajectory."""
        from coda.controller.rollout_sampler import RolloutSampler

        sampler = RolloutSampler(make_minimal_config(), Mock(), Mock())
        trajectories = [
            create_test_trajectory(start_rollout_weight_version=1, end_rollout_weight_version=1),
            create_test_trajectory(start_rollout_weight_version=1, end_rollout_weight_version=1),
        ]

        result = sampler.calculate_partial_ratio(trajectories)

        assert result == 0.0

    def test_all_switched_versions_returns_one(self):
        """Test calculate_partial_ratio returns 1.0 when every trajectory crosses rollout versions."""
        from coda.controller.rollout_sampler import RolloutSampler

        sampler = RolloutSampler(make_minimal_config(), Mock(), Mock())
        trajectories = [
            create_test_trajectory(start_rollout_weight_version=0, end_rollout_weight_version=1),
            create_test_trajectory(start_rollout_weight_version=2, end_rollout_weight_version=3),
        ]

        result = sampler.calculate_partial_ratio(trajectories)

        assert result == pytest.approx(1.0)

    def test_mixed_version_switches_returns_correct_ratio(self):
        """Test calculate_partial_ratio returns correct ratio for mixed rollout version switches."""
        from coda.controller.rollout_sampler import RolloutSampler

        sampler = RolloutSampler(make_minimal_config(), Mock(), Mock())
        trajectories = [
            create_test_trajectory(start_rollout_weight_version=1, end_rollout_weight_version=1),
            create_test_trajectory(start_rollout_weight_version=0, end_rollout_weight_version=0),
            create_test_trajectory(start_rollout_weight_version=1, end_rollout_weight_version=2),
            create_test_trajectory(start_rollout_weight_version=-1, end_rollout_weight_version=-1),
        ]

        result = sampler.calculate_partial_ratio(trajectories)

        assert result == pytest.approx(0.25)

    def test_unset_version_returns_zero(self):
        """Test trajectories that never recorded a version (start == -1) are treated as not switched."""
        from coda.controller.rollout_sampler import RolloutSampler

        sampler = RolloutSampler(make_minimal_config(), Mock(), Mock())
        trajectories = [
            create_test_trajectory(start_rollout_weight_version=1, end_rollout_weight_version=1),
            create_test_trajectory(start_rollout_weight_version=-1, end_rollout_weight_version=-1),
            create_test_trajectory(start_rollout_weight_version=2, end_rollout_weight_version=4),
        ]

        result = sampler.calculate_partial_ratio(trajectories)

        assert result == pytest.approx(1 / 3)

# ===========================================================================
# RolloutSampler.calculate_max_partial_span
# ===========================================================================

class TestCalculateMaxPartialSpan:
    """Tests for RolloutSampler.calculate_max_partial_span method."""

    def test_empty_trajectories_returns_zero(self):
        """Test calculate_max_partial_span returns 0 for empty input."""
        from coda.controller.rollout_sampler import RolloutSampler

        sampler = RolloutSampler(make_minimal_config(), Mock(), Mock())

        result = sampler.calculate_max_partial_span([])

        assert result == 0

    def test_single_version_returns_zero(self):
        """Test trajectories with start == end have span 0."""
        from coda.controller.rollout_sampler import RolloutSampler

        sampler = RolloutSampler(make_minimal_config(), Mock(), Mock())
        trajectories = [
            create_test_trajectory(start_rollout_weight_version=3, end_rollout_weight_version=3),
            create_test_trajectory(start_rollout_weight_version=5, end_rollout_weight_version=5),
        ]

        result = sampler.calculate_max_partial_span(trajectories)

        assert result == 0

    def test_returns_max_span_across_trajectories(self):
        """Test that the maximum span across all trajectories is returned."""
        from coda.controller.rollout_sampler import RolloutSampler

        sampler = RolloutSampler(make_minimal_config(), Mock(), Mock())
        trajectories = [
            create_test_trajectory(start_rollout_weight_version=1, end_rollout_weight_version=2),  # span = 1
            create_test_trajectory(start_rollout_weight_version=3, end_rollout_weight_version=7),  # span = 4
            create_test_trajectory(start_rollout_weight_version=2, end_rollout_weight_version=5),  # span = 3
        ]

        result = sampler.calculate_max_partial_span(trajectories)

        assert result == 4

    def test_unset_versions_are_excluded(self):
        """Test that trajectories which never recorded a version (start == -1) are excluded."""
        from coda.controller.rollout_sampler import RolloutSampler

        sampler = RolloutSampler(make_minimal_config(), Mock(), Mock())
        trajectories = [
            create_test_trajectory(start_rollout_weight_version=-1, end_rollout_weight_version=-1),
            create_test_trajectory(start_rollout_weight_version=-1, end_rollout_weight_version=-1),
        ]

        result = sampler.calculate_max_partial_span(trajectories)

        assert result == 0

# ===========================================================================
# RolloutSampler._trigger_generation
# ===========================================================================

class TestTriggerGeneration:
    """Tests for RolloutSampler._trigger_generation method."""

    @pytest.mark.asyncio
    async def test_trigger_generation_calls_get(self):
        """Test _trigger_generation calls datasource.get with correct args."""
        from coda.controller.rollout_sampler import RolloutSampler

        datasource = Mock()
        datasource.get = Mock(return_value=[])
        agentflow = Mock()
        agentflow.generate = AsyncMock(return_value=None)

        agentflow.traj_queue = Mock()
        sampler = RolloutSampler(make_minimal_config(), [datasource], agentflow)

        task_set = set()
        sampler._trigger_generation(num=3, step=2, task_set=task_set, ds_index=0)

        datasource.get.assert_called_once_with(3, 2)

    @pytest.mark.asyncio
    async def test_trigger_generation_creates_tasks_for_each_group(self):
        """Test _trigger_generation creates an asyncio task for each traj_group."""
        from coda.controller.rollout_sampler import RolloutSampler

        traj_group_1 = create_test_trajectory_group("p0", step=0)
        traj_group_2 = create_test_trajectory_group("p1", step=0)

        datasource = Mock()
        datasource.get = Mock(return_value=[traj_group_1, traj_group_2])
        agentflow = Mock()

        async def mock_generate(_trajs):
            await asyncio.sleep(0)

        agentflow.generate = mock_generate

        agentflow.traj_queue = Mock()
        sampler = RolloutSampler(make_minimal_config(), [datasource], agentflow)

        task_set = set()
        sampler._trigger_generation(num=2, step=0, task_set=task_set, ds_index=0)

        assert len(task_set) == 2
        # Cleanup tasks
        for t in task_set:
            t.cancel()
        await asyncio.gather(*task_set, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_trigger_generation_tasks_added_to_task_set(self):
        """Test _trigger_generation tasks have done callbacks for removal."""
        from coda.controller.rollout_sampler import RolloutSampler

        traj_group = create_test_trajectory_group("p0", step=0)
        datasource = Mock()
        datasource.get = Mock(return_value=[traj_group])
        agentflow = Mock()

        async def fast_generate(_trajs):
            pass

        agentflow.generate = fast_generate

        agentflow.traj_queue = Mock()
        sampler = RolloutSampler(make_minimal_config(), [datasource], agentflow)
        task_set = set()
        sampler._trigger_generation(num=1, step=0, task_set=task_set, ds_index=0)

        # Allow task to complete and remove itself from set
        await asyncio.sleep(0.1)

        assert len(task_set) == 0


# ===========================================================================
# RolloutSampler._cleanup
# ===========================================================================

class TestCleanup:
    """Tests for RolloutSampler._cleanup method."""

    @pytest.mark.asyncio
    async def test_cleanup_aborts_agent_flow(self):
        """Test _cleanup calls agentflow.abort()."""
        from coda.controller.rollout_sampler import RolloutSampler

        agentflow = Mock()
        agentflow.abort = AsyncMock()
        traj_queue = Mock()
        traj_queue.is_empty = Mock(return_value=True)

        agentflow.traj_queue = traj_queue
        sampler = RolloutSampler(make_minimal_config(), Mock(), agentflow)

        with patch('coda.controller.rollout_sampler.track'):
            await sampler._cleanup(set(), step=0)

        agentflow.abort.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_cancels_running_tasks(self):
        """Test _cleanup cancels all running tasks."""
        from coda.controller.rollout_sampler import RolloutSampler

        agentflow = Mock()
        agentflow.abort = AsyncMock()
        traj_queue = Mock()
        traj_queue.is_empty = Mock(return_value=True)

        agentflow.traj_queue = traj_queue
        sampler = RolloutSampler(make_minimal_config(), Mock(), agentflow)

        task1 = asyncio.create_task(asyncio.sleep(10))
        task2 = asyncio.create_task(asyncio.sleep(10))
        running_tasks = {task1, task2}

        with patch('coda.controller.rollout_sampler.track'):
            await sampler._cleanup(running_tasks, step=0)

        assert task1.cancelled()
        assert task2.cancelled()

    @pytest.mark.asyncio
    async def test_cleanup_drains_remaining_items(self):
        """Test _cleanup drains remaining items from traj_queue."""
        from coda.controller.rollout_sampler import RolloutSampler

        agentflow = Mock()
        agentflow.abort = AsyncMock()
        traj_queue = Mock()
        remaining = [create_test_trajectory(f"prompt_0_step0_t{i}", "prompt_0") for i in range(3)]
        traj_queue.is_empty = Mock(side_effect=[False, True])
        traj_queue.pop_all = Mock(return_value=remaining)

        config = make_minimal_config()
        config.rollout.partial = False
        agentflow.traj_queue = traj_queue
        sampler = RolloutSampler(config, Mock(), agentflow)

        with patch('coda.controller.rollout_sampler.track'):
            await sampler._cleanup(set(), step=0)

        traj_queue.pop_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_with_partial_rollout_adds_complete_groups(self):
        """Test _cleanup adds complete groups to datasource when partial enabled."""
        from coda.controller.rollout_sampler import RolloutSampler

        datasource = Mock()
        datasource.add = Mock()
        agentflow = Mock()
        agentflow.abort = AsyncMock()
        traj_queue = Mock()
        # 2 trajectories for prompt_0 — matches num_trajectories_per_prompt=2
        remaining = [create_test_trajectory(f"prompt_0_step0_t{i}", "prompt_0") for i in range(2)]
        traj_queue.is_empty = Mock(side_effect=[False, True])
        traj_queue.pop_all = Mock(return_value=remaining)

        config = make_minimal_config()
        config.rollout.partial = True
        agentflow.traj_queue = traj_queue
        sampler = RolloutSampler(config, [datasource], agentflow)

        with patch('coda.controller.rollout_sampler.track'):
            await sampler._cleanup(set(), step=0)

        datasource.add.assert_called_once()
        call_args = datasource.add.call_args[0][0]
        assert len(call_args) == 1
        assert isinstance(call_args[0], TrajectoryGroup)
        assert call_args[0].prompt_id == "prompt_0"
        assert len(call_args[0].trajectories) == 2

    @pytest.mark.asyncio
    async def test_cleanup_incomplete_groups_not_added(self):
        """Test _cleanup does not add incomplete groups to datasource."""
        from coda.controller.rollout_sampler import RolloutSampler

        datasource = Mock()
        datasource.add = Mock()
        agentflow = Mock()
        agentflow.abort = AsyncMock()
        traj_queue = Mock()
        # Only 1 trajectory but num_trajectories_per_prompt=2, so it's incomplete
        remaining = [create_test_trajectory("prompt_0_step0_t0", "prompt_0")]
        traj_queue.is_empty = Mock(side_effect=[False, True])
        traj_queue.pop_all = Mock(return_value=remaining)

        config = make_minimal_config()
        config.rollout.partial = True
        agentflow.traj_queue = traj_queue
        sampler = RolloutSampler(config, [datasource], agentflow)

        with patch('coda.controller.rollout_sampler.track'):
            await sampler._cleanup(set(), step=0)

        # add should not be called (no complete groups)
        datasource.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_cleanup_multiple_prompt_groups(self):
        """Test _cleanup groups trajectories by prompt_id correctly."""
        from coda.controller.rollout_sampler import RolloutSampler

        datasource = Mock()
        datasource.add = Mock()
        agentflow = Mock()
        agentflow.abort = AsyncMock()
        traj_queue = Mock()
        # prompt_0: 2 trajs (complete), prompt_1: 1 traj (incomplete)
        remaining = (
            [create_test_trajectory(f"prompt_0_step0_t{i}", "prompt_0") for i in range(2)] +
            [create_test_trajectory("prompt_1_step0_t0", "prompt_1")]
        )
        traj_queue.is_empty = Mock(side_effect=[False, True])
        traj_queue.pop_all = Mock(return_value=remaining)

        config = make_minimal_config()
        config.rollout.partial = True
        agentflow.traj_queue = traj_queue
        sampler = RolloutSampler(config, [datasource], agentflow)

        with patch('coda.controller.rollout_sampler.track'):
            await sampler._cleanup(set(), step=0)

        datasource.add.assert_called_once()
        call_args = datasource.add.call_args[0][0]
        # Only prompt_0 is complete
        assert len(call_args) == 1
        assert call_args[0].prompt_id == "prompt_0"

    @pytest.mark.asyncio
    async def test_cleanup_without_partial_rollout(self):
        """Test _cleanup does not add groups when partial is disabled."""
        from coda.controller.rollout_sampler import RolloutSampler

        datasource = Mock()
        datasource.add = Mock()
        agentflow = Mock()
        agentflow.abort = AsyncMock()
        traj_queue = Mock()
        remaining = [create_test_trajectory(f"prompt_0_step0_t{i}", "prompt_0") for i in range(2)]
        traj_queue.is_empty = Mock(side_effect=[False, True])
        traj_queue.pop_all = Mock(return_value=remaining)

        config = make_minimal_config()
        config.rollout.partial = False
        agentflow.traj_queue = traj_queue
        sampler = RolloutSampler(config, [datasource], agentflow)

        with patch('coda.controller.rollout_sampler.track'):
            await sampler._cleanup(set(), step=0)

        datasource.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_cleanup_empty_queue_skips_pop_and_add(self):
        """Test _cleanup with empty queue skips pop_all and add."""
        from coda.controller.rollout_sampler import RolloutSampler

        datasource = Mock()
        datasource.add = Mock()
        agentflow = Mock()
        agentflow.abort = AsyncMock()
        traj_queue = Mock()
        traj_queue.is_empty = Mock(return_value=True)

        config = make_minimal_config()
        config.rollout.partial = True
        agentflow.traj_queue = traj_queue
        sampler = RolloutSampler(config, [datasource], agentflow)

        with patch('coda.controller.rollout_sampler.track'):
            await sampler._cleanup(set(), step=0)

        traj_queue.pop_all.assert_not_called()
        datasource.add.assert_not_called()


# ===========================================================================
# RolloutSampler.dynamic_rollout
# ===========================================================================

class TestDynamicRollout:
    """Tests for RolloutSampler.dynamic_rollout method."""

    @pytest.mark.asyncio
    async def test_dynamic_rollout_returns_expected_groups(self):
        """Test dynamic_rollout returns exactly num_prompt_per_step groups."""
        from coda.controller.rollout_sampler import RolloutSampler

        num_prompt_per_step = 2
        groups = [create_test_trajectory_group(f"p{i}", step=0) for i in range(num_prompt_per_step)]

        datasource = Mock()
        datasource.get = Mock(return_value=groups)

        agentflow = Mock()
        agentflow.generate = AsyncMock(return_value=None)
        agentflow.abort = AsyncMock()

        traj_queue = Mock()
        traj_queue.pop_group = Mock(side_effect=groups + [None] * 10)
        traj_queue.wait_for_group = Mock(return_value=None)
        traj_queue.is_empty = Mock(return_value=True)

        config = make_minimal_config()
        config.data_sources[0].num_prompts_per_step = num_prompt_per_step
        config.rollout.sampler.num_oversample = 0

        agentflow.traj_queue = traj_queue
        sampler = RolloutSampler(config, [datasource], agentflow)

        with patch('coda.controller.rollout_sampler.track'):
            result = await sampler.dynamic_rollout(0, step=0)

        assert len(result) == num_prompt_per_step

    @pytest.mark.asyncio
    async def test_dynamic_rollout_drops_filtered_groups(self):
        """Test dynamic_rollout drops groups that fail data filter."""
        from coda.controller.rollout_sampler import RolloutSampler

        group_ok = create_test_trajectory_group("pok", step=0)
        group_bad = create_test_trajectory_group("pbad", step=0)

        datasource = Mock()
        datasource.get = Mock(return_value=[group_ok, group_bad])

        agentflow = Mock()
        agentflow.generate = AsyncMock(return_value=None)
        agentflow.abort = AsyncMock()

        # Queue returns bad group first, then good group, then None
        traj_queue = Mock()
        traj_queue.pop_group = Mock(side_effect=[group_bad, group_ok, None, None, None])
        traj_queue.wait_for_group = Mock(return_value=None)
        traj_queue.is_empty = Mock(return_value=True)

        config = make_minimal_config()
        config.data_sources[0].num_prompts_per_step = 1
        config.rollout.sampler.num_oversample = 1

        agentflow.traj_queue = traj_queue
        sampler = RolloutSampler(config, [datasource], agentflow)
        # Filter drops group_bad but passes group_ok
        sampler.data_filter = Mock()
        sampler.data_filter.apply = Mock(side_effect=lambda g: None if g.prompt_id == "pbad" else g)

        with patch('coda.controller.rollout_sampler.track'):
            result = await sampler.dynamic_rollout(1, step=0)

        assert len(result) == 1
        assert result[0].prompt_id == "pok"

    @pytest.mark.asyncio
    async def test_dynamic_rollout_raises_when_no_tasks(self):
        """Test dynamic_rollout propagates OverflowError when nothing is left to run."""
        from coda.controller.rollout_sampler import RolloutSampler

        # 2 groups that will all be filtered out
        group1 = create_test_trajectory_group("p0", step=0)
        group2 = create_test_trajectory_group("p1", step=0)

        datasource = Mock()
        datasource.get = Mock(return_value=[group1, group2])

        agentflow = Mock()
        agentflow.generate = AsyncMock(return_value=None)
        agentflow.abort = AsyncMock()

        traj_queue = Mock()
        # Queue returns the 2 groups (both filtered), then None → triggers OverflowError (caught)
        traj_queue.pop_group = Mock(side_effect=[group1, group2, None])
        traj_queue.wait_for_group = Mock(return_value=None)
        traj_queue.is_empty = Mock(return_value=True)

        config = make_minimal_config()
        config.data_sources[0].num_prompts_per_step = 2
        config.rollout.sampler.num_oversample = 0
        config.rollout.sampler.max_refill_count = 0
        config.rollout.sampler.refill_ratio = 0

        agentflow.traj_queue = traj_queue
        sampler = RolloutSampler(config, [datasource], agentflow)
        # Filter drops all groups
        sampler.data_filter = Mock()
        sampler.data_filter.apply = Mock(return_value=None)

        with patch('coda.controller.rollout_sampler.track'):
            with pytest.raises(OverflowError, match="no more trajectory left to rollout"):
                await sampler.dynamic_rollout(0, step=0)

    @pytest.mark.asyncio
    async def test_dynamic_rollout_is_async(self):
        """Test dynamic_rollout is an async (coroutine) method."""
        import inspect
        from coda.controller.rollout_sampler import RolloutSampler

        assert inspect.iscoroutinefunction(RolloutSampler.dynamic_rollout)


# ===========================================================================
# Misc / import checks
# ===========================================================================

class TestImport:
    """Sanity checks for RolloutSampler public interface."""

    def test_import_and_public_interface(self):
        """RolloutSampler can be imported and exposes expected attributes."""
        from coda.controller.rollout_sampler import RolloutSampler

        assert hasattr(RolloutSampler, '__call__')
        assert hasattr(RolloutSampler, 'dynamic_rollout')
        assert hasattr(RolloutSampler, '_pre_process')
        assert hasattr(RolloutSampler, '_cleanup')
        assert hasattr(RolloutSampler, '_trigger_generation')
        assert hasattr(RolloutSampler, '_stat_metrics')
        assert hasattr(RolloutSampler, 'calculate_partial_ratio')


# ===========================================================================
# RolloutSampler._cleanup with wait=True
# ===========================================================================

class TestCleanupWait:
    """Tests for RolloutSampler._cleanup with wait=True (fully async mode)."""

    def test_cleanup_wait_true_does_not_cancel_tasks(self):
        """Test _cleanup(wait=True) waits for tasks instead of cancelling."""
        from coda.controller.rollout_sampler import RolloutSampler

        async def _run():
            agentflow = Mock()
            agentflow.abort = AsyncMock()
            traj_queue = Mock()
            traj_queue.is_empty = Mock(return_value=True)

            agentflow.traj_queue = traj_queue
            sampler = RolloutSampler(make_minimal_config(), Mock(), agentflow)

            completed = []

            async def slow_task():
                await asyncio.sleep(0.05)
                completed.append(True)

            task = asyncio.create_task(slow_task())

            with patch('coda.controller.rollout_sampler.track'):
                await sampler._cleanup({task}, step=0, wait=True)

            assert len(completed) == 1
            assert not task.cancelled()
            agentflow.abort.assert_not_called()

        asyncio.run(_run())

    def test_cleanup_wait_false_cancels_tasks(self):
        """Test _cleanup(wait=False) cancels tasks and calls abort."""
        from coda.controller.rollout_sampler import RolloutSampler

        async def _run():
            agentflow = Mock()
            agentflow.abort = AsyncMock()
            traj_queue = Mock()
            traj_queue.is_empty = Mock(return_value=True)

            agentflow.traj_queue = traj_queue
            sampler = RolloutSampler(make_minimal_config(), Mock(), agentflow)

            task = asyncio.create_task(asyncio.sleep(10))

            with patch('coda.controller.rollout_sampler.track'):
                await sampler._cleanup({task}, step=0, wait=False)

            assert task.cancelled()
            agentflow.abort.assert_called_once()

        asyncio.run(_run())


# ===========================================================================
# RolloutSampler._trigger_generation return value
# ===========================================================================

class TestTriggerGenerationReturnValue:
    """Tests for _trigger_generation returning dispatched traj_groups."""

    def test_trigger_generation_returns_traj_groups(self):
        """Test _trigger_generation returns the list of traj_groups from datasource."""
        from coda.controller.rollout_sampler import RolloutSampler

        async def _run():
            groups = [create_test_trajectory_group("p0"), create_test_trajectory_group("p1")]
            datasource = Mock()
            datasource.get = Mock(return_value=groups)
            agentflow = Mock()
            agentflow.generate = AsyncMock()

            agentflow.traj_queue = Mock()
            sampler = RolloutSampler(make_minimal_config(), [datasource], agentflow)

            task_set = set()
            result = sampler._trigger_generation(num=2, step=0, task_set=task_set, ds_index=0)

            assert result is groups
            assert len(result) == 2

            for t in task_set:
                t.cancel()
            await asyncio.gather(*task_set, return_exceptions=True)

        asyncio.run(_run())


# ===========================================================================
# RolloutSampler fully_async __init__ and __call__
# ===========================================================================

class TestFullyAsyncInit:
    """Tests for RolloutSampler initialization in fully_async mode."""

    def test_fully_async_init_creates_pipeline_buf(self):
        """Test fully_async init creates PipelineBuffer and collector thread."""
        from coda.controller.rollout_sampler import RolloutSampler

        config = make_minimal_config()
        config.fully_async.enable = True

        sampler = RolloutSampler(config, Mock(), Mock())

        assert sampler.is_fully_async is True
        assert sampler._groups_per_mini_batch == 2
        assert sampler.pipeline_buf is not None
        assert sampler._stopped is False
        assert sampler.step == 0

    def test_fully_async_pause_resume_stop(self):
        """Test pause/resume/stop state transitions."""
        from coda.controller.rollout_sampler import RolloutSampler

        config = make_minimal_config()
        config.fully_async.enable = True

        sampler = RolloutSampler(config, Mock(), Mock())

        assert sampler.is_paused() is False
        assert sampler.is_stopped() is False

        # Simulate pause (need to set _cleanup_done from another context)
        sampler._cleanup_done.clear()
        sampler._run_gate.clear()
        assert sampler.is_paused() is True

        # Simulate cleanup done and resume
        sampler._cleanup_done.set()
        sampler.resume()
        assert sampler.is_paused() is False
        assert sampler.step == 1

        sampler.stop()
        assert sampler.is_stopped() is True

    def test_call_fully_async_reads_from_pipeline_buf(self):
        """Test __call__ in fully_async mode reads from pipeline_buf."""
        from coda.controller.rollout_sampler import RolloutSampler

        async def _run():
            config = make_minimal_config()
            config.fully_async.enable = True

            sampler = RolloutSampler(config, Mock(), Mock())

            group = create_test_trajectory_group("p0")
            sampler.pipeline_buf.async_get = AsyncMock(side_effect=[group, group, None])

            result = await sampler(step=0)

            assert len(result) == 2
            assert result[0] is group

        asyncio.run(_run())
