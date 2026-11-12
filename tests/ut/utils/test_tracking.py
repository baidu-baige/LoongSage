"""Unit tests for coda/utils/tracking.py time_marker().

time_marker both reports timing/<name> and hands the caller the duration, which
the trainer divides by the step's token total.
"""
import logging
from unittest.mock import patch

import pytest

from coda.utils.tracking import time_marker


def test_marker_exposes_elapsed_after_the_block():
    with patch("coda.utils.tracking.track") as track:
        with time_marker("rollout", step=1) as marker:
            assert marker.elapsed == 0.0
        assert marker.elapsed > 0.0

    assert "timing/rollout" in track.call_args[0][0]


def test_can_still_be_used_without_binding():
    """Existing call sites do not bind the marker and must keep working."""
    with patch("coda.utils.tracking.track") as track:
        with time_marker("train", step=2):
            pass

    assert "timing/train" in track.call_args[0][0]


def test_a_raising_block_still_reports_its_duration(caplog):
    """Without this a failing phase leaves no timing behind, which hides regressions."""
    with patch("coda.utils.tracking.track") as track:
        with caplog.at_level(logging.INFO, logger="coda.utils.tracking"):
            with pytest.raises(RuntimeError, match="boom"):
                with time_marker("rollout", step=3) as marker:
                    raise RuntimeError("boom")

    assert marker.elapsed > 0.0
    assert track.call_args[1] == {"step": 3}
    assert any("rollout done" in record.getMessage() for record in caplog.records)
