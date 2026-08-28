"""Base interface for reward functions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from coda.reward.reward import Reward
from omegaconf import DictConfig


class RewardFunction(ABC):
    """Abstract base class for reward functions.

    Args:
        config: Full reward config dict. Subclasses read their own keys from ``self.config``.
    """

    def __init__(self, config: DictConfig) -> None:
        self.config = config

    @abstractmethod
    def __call__(
        self,
        messages: list[dict],
        label: Any,
        trajectory: dict,
        **kwargs: Any,
    ) -> Reward:
        """Compute reward from conversation history and label."""
