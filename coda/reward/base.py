"""Base interface for reward functions."""

from __future__ import annotations

from abc import ABC
from typing import Any

from coda.reward.reward import Reward
from omegaconf import DictConfig


class RewardFunction(ABC):
    """Base class for reward functions.

    Subclasses implement ``__call__`` (per-trajectory), ``group_rm`` (per prompt
    group, enabled by ``reward.group_rm: true``), or both. When both are
    implemented, ``__call__`` runs first and ``group_rm`` overrides the result.

    Args:
        config: Full reward config dict. Subclasses read their own keys from ``self.config``.
    """

    def __init__(self, config: DictConfig) -> None:
        self.config = config

    def __call__(
        self,
        messages: list[dict],
        label: Any,
        context: dict,
        **kwargs: Any,
    ) -> Reward:
        """Compute reward from conversation history, label, and rollout context.

        The default is a placeholder for group_rm-only subclasses; the group
        reward overwrites it once the whole group has finished.
        """
        return Reward(final_reward=0.0)

    async def group_rm(
        self,
        messages: list[list[dict]],
        label: Any,
        contexts: list[dict],
        **kwargs: Any,
    ) -> list[Reward]:
        """Compute rewards for all trajectories of one prompt group, in order.

        Awaited directly on the AgentFlow event loop: do not block inside it.
        Wrap blocking or CPU-heavy work with ``await asyncio.to_thread(...)``.

        Args:
            messages: Chat history of each trajectory.
            label: Label shared by the group.
            contexts: ``Trajectory.model_dump()`` of each trajectory, including
                any per-trajectory reward and metadata from ``__call__``.
        """
        raise NotImplementedError
