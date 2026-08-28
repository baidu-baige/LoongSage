"""Reward data model."""

from typing import Any

from pydantic import BaseModel as PydanticModel, Field


class Reward(PydanticModel):
    """Reward for a single trajectory."""

    final_reward: float
    # Reserved: per-step process rewards (PRM). Not populated by any current
    # reward function; field exists for future multi-step reward support.
    completion_rewards: list[float] = Field(default_factory=list)
    is_valid: bool = True
    # Correctness override. None = derive it from final_reward > 0.
    is_correct: bool | None = None
    extra_info: dict[str, Any] = Field(default_factory=dict)
