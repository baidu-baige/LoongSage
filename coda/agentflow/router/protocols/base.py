"""Protocol adapter contracts."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

from fastapi.responses import StreamingResponse
from starlette.requests import Request
from starlette.responses import Response

class ProtocolAdapter(ABC):
    """A single external wire protocol <-> coda-internal shape conversion."""

    @abstractmethod
    def parse_route(self, path: str) -> tuple[str, int] | None:
        """Return (trajectory_id, attempt_id), or None."""

    def normalize_headers(self, request: Request) -> dict[str, str]:
        """Return request headers with protocol-specific metadata normalized."""
        return {key.lower(): value for key, value in request.headers.items()}

    @abstractmethod
    def parse_request(self, body: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
        """Normalize the request body and return (messages, tools)."""

    @abstractmethod
    def build_response(
        self,
        body: dict[str, Any],
        assistant_message: dict[str, Any],
        finish_reason: str,
        usage: dict[str, Any],
        *,
        path: str | None = None,
        upstream_response: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Internal assistant_message -> wire-format response body (non-streaming)."""

    @abstractmethod
    def build_streaming_response(self, wire_response: dict[str, Any]) -> StreamingResponse:
        """Wire-format response body -> a fake-streaming SSE response."""

    @abstractmethod
    def build_error_response(
        self,
        body: dict[str, Any],
        *,
        error_type: str,
        message: str,
        status_code: int,
    ) -> Response:
        """Build a protocol-specific error response."""


def _route_re(suffix: str) -> "re.Pattern[str]":
    """Build the standard ``/{trajectory_id}/{attempt_id}/<suffix>`` route regex."""
    return re.compile(rf"^/(?P<trajectory_id>[^/]+)/(?P<attempt_id>\d+)/{suffix}$")
