"""Agent interfaces and helpers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import httpx

from coda.agentflow.sandbox.base import SandboxClient
from coda.agentflow.utils import CONTEXT_LENGTH_EXCEEDED, ContextLengthExceededError
from coda.reward.reward import Reward

logger = logging.getLogger(__name__)

_LLM_TIMEOUT_SECONDS = 600.0


class BaseAgent(ABC):
    """Shared plumbing for white-box agents.

    Concrete agents only need to implement :meth:`run_trajectory`. They may
    use :meth:`call_llm` for Router requests and :meth:`close` for shared
    resource cleanup, and call ``self.reward_fn`` themselves. AgentFlow
    also injects the data source's optional ``sandbox_client`` and resumed
    ``sandbox_id``; sandbox agents create an instance only when the ID is absent.
    """

    def __init__(
        self,
        router_url: str,
        reward_fn: Any,
        max_response_len_per_trajectory: int = 0,
        sandbox_client: SandboxClient | None = None,
        sandbox_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize common AgentFlow-provided parameters.

        Args:
            router_url: URL of the Router to send LLM requests to (required).
            reward_fn: Reward function injected by AgentFlow (required).
            max_response_len_per_trajectory: Response-area token budget per trajectory.
            sandbox_client: Shared backend client for this data source, when enabled.
            sandbox_id: Reused sandbox identifier, or ``None`` before first use.
            **kwargs: Agent-specific extension parameters (kept in ``extra_config``).
        """
        if not router_url:
            raise ValueError("router_url is required and must be a non-empty URL")
        if reward_fn is None:
            raise ValueError("reward_fn is required")
        self.router_url = router_url
        self.max_response_len_per_trajectory = int(max_response_len_per_trajectory or 0)
        self.reward_fn = reward_fn
        self.sandbox_client = sandbox_client
        self.sandbox_id = sandbox_id
        self.extra_config = kwargs
        self._llm_client: httpx.AsyncClient | None = None

    @abstractmethod
    async def run_trajectory(self, trajectory: dict[str, Any]) -> Reward:
        """Run a complete trajectory and return its reward.

        Args:
            trajectory: A dataset trajectory passed through from the training
                controller. AgentFlow does not prescribe its schema;
                each agent interprets the fields it needs.
        """

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    async def call_llm(
        self,
        messages: list[dict],
        *,
        max_tokens: int | None = None,
        extra_body: dict | None = None,
    ) -> dict:
        """POST to the Router's ``/v1/chat/completions`` and return the parsed body.

        The request body contains ``messages`` and ``max_tokens`` (defaults to
        ``max_response_len_per_trajectory``), plus anything in *extra_body*
        (e.g. ``{"tools": ..., "tool_choice": ...}``).

        Raises:
            ContextLengthExceededError: The Router returned HTTP 400 with
                ``error.type == context_length_exceeded`` — the trajectory-wide
                token budget is exhausted; callers should stop gracefully.
        """
        body: dict[str, Any] = {"model": "default", "messages": messages}
        body["max_tokens"] = self.max_response_len_per_trajectory if max_tokens is None else max_tokens
        if extra_body:
            body.update(extra_body)
        if self._llm_client is None:
            self._llm_client = httpx.AsyncClient(timeout=httpx.Timeout(_LLM_TIMEOUT_SECONDS))
        response = await self._llm_client.post(f"{self.router_url}/v1/chat/completions", json=body)
        if response.status_code == httpx.codes.BAD_REQUEST:
            try:
                err = response.json().get("error", {})
            except Exception:
                err = {}
            if err.get("type") == CONTEXT_LENGTH_EXCEEDED:
                raise ContextLengthExceededError(err.get("message", ""), error=err) from None
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close resources owned by this agent."""
        client, self._llm_client = self._llm_client, None
        if client is not None:
            await client.aclose()
