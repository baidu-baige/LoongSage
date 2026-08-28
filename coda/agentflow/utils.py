"""Utility helpers for AgentFlow internals."""

from __future__ import annotations

_REQUEST_ID_SEPARATOR = "#"
CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"


def build_request_id(trajectory_id: str, attempt_id: int) -> str:
    """Build a stable per-attempt request id.

    Used by the router's abort_session endpoint and AgentFlow's failure-handling
    path to identify the worker request to cancel.
    """
    return f"{trajectory_id}{_REQUEST_ID_SEPARATOR}{attempt_id}"


def parse_request_id(request_id: str) -> tuple[str, int]:
    """Parse a request id back into (trajectory_id, attempt_id)."""
    trajectory_id, attempt_id_str = request_id.rsplit(_REQUEST_ID_SEPARATOR, 1)
    return trajectory_id, int(attempt_id_str)
