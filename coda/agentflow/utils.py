"""Utility helpers for AgentFlow internals."""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from coda.agentflow.sandbox import SandboxClient

_REQUEST_ID_SEPARATOR = "#"
CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"


class ContextLengthExceededError(RuntimeError):
    """Raised when the router rejects a request because the trajectory token
    budget is exhausted (HTTP 400 with ``error.type == context_length_exceeded``).

    Callers treat this as a graceful stop of the trajectory, not a failure.
    """

    def __init__(self, message: str = "", *, error: dict | None = None) -> None:
        super().__init__(message or "context length exceeded (router token budget exhausted)")
        self.error: dict = error or {}


def build_request_id(trajectory_id: str, attempt_id: int) -> str:
    """Build a stable per-attempt request id.

    Used by the router's abort_session endpoint and AgentFlow's failure-handling
    path to identify the worker request to cancel.
    """
    return f"{trajectory_id}{_REQUEST_ID_SEPARATOR}{attempt_id}"


def require_docker_image(metadata: Mapping[str, Any] | None) -> str:
    """Return ``metadata['docker_image']`` or raise ValueError when missing."""
    image = str((metadata or {}).get("docker_image") or "")
    if not image:
        raise ValueError("metadata['docker_image'] is missing")
    return image


async def setup_sandbox(
    client: SandboxClient, metadata: Mapping[str, Any], prepare: Callable,
) -> tuple[str, dict[str, Any]]:
    """Return (sandbox_id, reward state) for a newly created, prepared sandbox.

    Creation and preparation share one worker thread so they happen as a unit: a
    sandbox that was created but not prepared is useless, and once retained it is
    indistinguishable from a prepared one whose preparation returned no state.
    """
    def create_and_prepare() -> tuple[str, dict[str, Any]]:
        sandbox_id = client.create(image=require_docker_image(metadata))
        try:
            return sandbox_id, dict(prepare(client, sandbox_id, metadata=copy.deepcopy(dict(metadata))))
        except BaseException:
            client.delete(sandbox_id)
            raise

    return await asyncio.to_thread(create_and_prepare)
