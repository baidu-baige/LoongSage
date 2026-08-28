"""Unit tests for AgentFlow utility helpers."""

from coda.agentflow.utils import build_request_id


def test_build_request_id() -> None:
    """build_request_id should generate a stable attempt-scoped identifier."""
    assert build_request_id("traj-001", 3) == "traj-001#3"
