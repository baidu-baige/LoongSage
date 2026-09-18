"""Stateless sandbox backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any


class SandboxClient(ABC):
    """Backend connector that can manage multiple sandbox instances."""

    @classmethod
    def from_config(cls, sandbox_config: Mapping[str, Any], **kwargs: Any) -> "SandboxClient":
        """Construct a backend client from framework configuration."""
        config = dict(sandbox_config)
        config.pop("type", None)
        return cls(**config)

    @abstractmethod
    def create(self, **kwargs: Any) -> str:
        """Create a sandbox instance and return its backend identifier."""

    @abstractmethod
    def execute(self, sandbox_id: str, command: str, **kwargs: Any) -> dict[str, Any]:
        """Execute a shell command in the identified sandbox."""

    @abstractmethod
    def delete(self, sandbox_id: str, **kwargs: Any) -> None:
        """Destroy the identified sandbox; a missing resource is a success."""
