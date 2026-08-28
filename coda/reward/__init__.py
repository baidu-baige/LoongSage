"""Reward module for coda.

Usage:
    from coda.reward import Reward, RewardFunction, create_reward_fn

    # Create from config
    reward_fn = create_reward_fn({"name": "gsm8k"})

    # Or use directly
    from coda.reward.functions.gsm8k import GSM8KReward
    reward_fn = GSM8KReward(config={"name": "gsm8k"})
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from coda.reward.base import RewardFunction
from coda.reward.reward import Reward
from coda.utils.registry import Registry

REWARD_REGISTRY = Registry("reward")

register_reward = REWARD_REGISTRY.register
get_reward_class = REWARD_REGISTRY.get


def create_reward_fn(reward_config: Mapping[str, Any] | None) -> RewardFunction | None:
    """Create a reward function instance from config dict.

    The entire ``reward_config`` dict (including the ``name`` key) is forwarded
    to the class constructor as ``config=``.  Subclasses extract their own
    parameters from ``self.config`` — no unpacking happens here, so adding a
    new parameter to a reward function never requires touching this factory.
    """
    reward_config = dict(reward_config or {})
    name = reward_config.get("name")
    if not name:
        return None

    cls = get_reward_class(name)
    return cls(config=reward_config)


__all__ = [
    "Reward",
    "RewardFunction",
    "REWARD_REGISTRY",
    "register_reward",
    "get_reward_class",
    "create_reward_fn",
]

# Auto-discover built-in reward functions so their @register_reward decorators execute.
import pkgutil as _pkgutil  # noqa: E402
for _, _name, _ in _pkgutil.walk_packages(__path__, __name__ + "."):
    __import__(_name)
