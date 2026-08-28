"""Unit tests for TrainMetricsAggregator rule-based flush."""

from coda.utils.tracking import TrainMetricsAggregator


def test_flush_aggregates_by_naming_rule():
    """timing/* -> sum, */max -> max, */min -> min, everything else -> mean."""
    agg = TrainMetricsAggregator()

    # Two mini-batch records for the same step.
    agg.add(
        {
            "timing/train_actor": 1.0,
            "perf/train_memory_allocated_max": 100.0,
            "rollout/response_length_min": 5.0,
            "train/loss": 2.0,
            "train/clip_ratio": 0.1,
        },
        step=3,
    )
    agg.add(
        {
            "timing/train_actor": 2.5,
            "perf/train_memory_allocated_max": 120.0,
            "rollout/response_length_min": 3.0,
            "train/loss": 4.0,
            "train/clip_ratio": 0.3,
        },
        step=3,
    )

    flushed = agg.flush()
    assert flushed is not None
    step, result = flushed

    assert step == 3
    assert result["timing/train_actor"] == 3.5          # sum
    assert result["perf/train_memory_allocated_max"] == 120.0  # max
    assert result["rollout/response_length_min"] == 3.0  # min
    assert result["train/loss"] == 3.0                   # mean
    assert result["train/clip_ratio"] == 0.2             # mean


def test_flush_empty_returns_none():
    agg = TrainMetricsAggregator()
    assert agg.flush() is None


def test_add_skips_non_numeric_values():
    agg = TrainMetricsAggregator()
    agg.add({"train/loss": "not-a-number", "train/entropy": 1.0}, step=0)
    step, result = agg.flush()
    assert step == 0
    assert "train/loss" not in result
    assert result["train/entropy"] == 1.0
