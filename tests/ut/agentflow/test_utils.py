"""Unit tests for AgentFlow utility helpers."""

import pytest

from coda.agentflow.utils import ContextLengthExceededError, build_request_id, require_docker_image


def test_build_request_id() -> None:
    """build_request_id should generate a stable attempt-scoped identifier."""
    assert build_request_id("traj-001", 3) == "traj-001#3"


def test_context_length_exceeded_error() -> None:
    """ContextLengthExceededError is a RuntimeError and preserves the error payload."""
    err = ContextLengthExceededError("budget exhausted", error={"type": "context_length_exceeded"})
    assert isinstance(err, RuntimeError)
    assert str(err) == "budget exhausted"
    assert err.error == {"type": "context_length_exceeded"}

    default = ContextLengthExceededError()
    assert default.error == {}
    assert str(default)


def test_require_docker_image() -> None:
    """require_docker_image passes through valid values, raises ValueError otherwise."""
    assert require_docker_image({"docker_image": "example/image:latest"}) == "example/image:latest"
    with pytest.raises(ValueError, match="docker_image"):
        require_docker_image({})
    with pytest.raises(ValueError, match="docker_image"):
        require_docker_image(None)
    with pytest.raises(ValueError, match="docker_image"):
        require_docker_image({"docker_image": ""})
    with pytest.raises(ValueError, match="docker_image"):
        require_docker_image({"docker_image": None})
