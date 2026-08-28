"""Sandbox client abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any


class SandboxClient(ABC):
    """Abstract base class for sandbox execution environments."""

    @classmethod
    def from_config(
        cls,
        sandbox_config: Mapping[str, Any],
        **kwargs: Any,
    ) -> "SandboxClient | None":
        """Create a sandbox client instance from framework config.

        The default implementation removes ``type`` and instantiates ``cls``
        directly with the remaining config values. Sandbox implementations can
        override this when they need extra adaptation logic.
        """
        config = dict(sandbox_config)
        config.pop("type", None)
        return cls(**config)

    @property
    def sandbox_id(self) -> str | None:
        """Identifier of the live sandbox instance, or None when not created.

        Agents use this to detect a reusable sandbox (kept alive across a
        partial-rollout abort) and skip re-creation, so tool state accumulated
        in earlier turns is preserved.
        """
        return None

    @abstractmethod
    def create(self, **kwargs) -> str:
        """Create and start a sandbox instance. Returns a sandbox ID.

        Runtime-specific details such as image / pod spec may be passed via
        ``kwargs`` by the agent based on the current trajectory.
        """

    @abstractmethod
    def execute(self, command: str, **kwargs) -> dict[str, Any]:
        """Execute a shell command inside the sandbox."""

    @abstractmethod
    def delete(self, **kwargs) -> None:
        """Stop and destroy the sandbox instance."""
