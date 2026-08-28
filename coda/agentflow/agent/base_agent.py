"""Agent interfaces and helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """Base class for all agents."""

    def __init__(
        self,
        router_url: str,
        completion_params: dict | None = None,
        max_response_len_per_trajectory: int = 0,
        temperature: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Initialize common AgentFlow-provided parameters.

        Args:
            router_url: URL of the Router to send LLM requests to.
            completion_params: Sampling parameters for LLM requests.
            max_response_len_per_trajectory: Response-area token budget per trajectory.
            temperature: Sampling temperature.
            **kwargs: Agent-specific extension parameters.
        """
        self.router_url = router_url
        self.completion_params = completion_params or {}
        self.max_response_len_per_trajectory = int(max_response_len_per_trajectory or 0)
        self.temperature = temperature
        self.extra_config = kwargs

    @abstractmethod
    async def run_trajectory(self, trajectory: dict[str, Any]) -> Any:
        """Run a complete trajectory.

        Args:
            trajectory: A dataset trajectory passed through from the training
                controller. AgentFlow does not prescribe its schema;
                each agent interprets the fields it needs.
        """

    @abstractmethod
    async def clear(self) -> None:
        """Clear agent resources."""
