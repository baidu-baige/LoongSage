"""Memory pool interfaces for OPD teacher weights."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class MemoryPool(ABC):
    """Interface for distributed teacher weight storage."""

    def __init__(self, config: Any):
        self.config = config

    @abstractmethod
    def fetch(self, teacher_idx: int) -> dict[str, Any]:
        """Fetch a teacher state dict from the pool."""

    @abstractmethod
    def store(self, teacher_idx: int, state_dict: dict[str, Any]):
        """Store a teacher state dict in the pool."""

    @abstractmethod
    def has(self, teacher_idx: int) -> bool:
        """Return whether a teacher exists in the pool."""


class DisabledMemoryPool(MemoryPool):
    """Disabled memory pool used before a concrete backend is configured."""

    def fetch(self, teacher_idx: int) -> dict[str, Any]:
        raise KeyError(
            f"Teacher {teacher_idx} is not in local CPU cache and opd.memory_pool.backend is disabled."
        )

    def store(self, teacher_idx: int, state_dict: dict[str, Any]):
        raise NotImplementedError("Cannot store teacher weights because opd.memory_pool.backend is disabled.")

    def has(self, teacher_idx: int) -> bool:
        return False


def build_memory_pool(config: Any) -> MemoryPool:
    """Build the configured OPD memory pool implementation."""
    backend = None
    if config is not None and "opd" in config and "memory_pool" in config.opd:
        backend = config.opd.memory_pool.get("backend")

    if backend is None or str(backend).lower() in ("", "none", "null"):
        return DisabledMemoryPool(config)
    raise NotImplementedError(f"Unsupported OPD memory pool backend: {backend}")
