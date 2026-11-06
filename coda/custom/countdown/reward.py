"""Countdown reward: format bonus for a well-formed ``<answer>`` tag plus a 0/1 answer reward."""

from __future__ import annotations

import logging
import re

from omegaconf import DictConfig

from coda.reward import register_reward
from coda.reward.base import RewardFunction
from coda.reward.reward import Reward

logger = logging.getLogger(__name__)

_ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
_NUMBER_PATTERN = re.compile(r"\d+")
_ALLOWED_CHARS = set("0123456789+-*/(). ")


def extract_equation(text: str) -> str | None:
    if not text:
        return None
    matches = _ANSWER_PATTERN.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def _parse_label(label) -> tuple[float | None, list[int]]:
    # dict labels arrive as {"target": T, "nums": [...]}; scalar labels are
    # wrapped by Dataset as {"value": ...} and treated as the target.
    if isinstance(label, dict):
        inner = label.get("value") if isinstance(label.get("value"), dict) else label
        target_raw = inner.get("target", inner.get("value"))
        nums_raw = inner.get("nums", [])
    else:
        target_raw, nums_raw = label, []

    try:
        target = float(target_raw) if target_raw is not None else None
    except (TypeError, ValueError):
        target = None

    nums: list[int] = []
    if nums_raw is not None:
        for n in nums_raw:
            try:
                nums.append(int(n))
            except (TypeError, ValueError):
                continue
    return target, nums


def evaluate_equation(equation: str) -> float | None:
    # Safe eval: char allowlist blocks identifiers/strings/calls; "**" banned to
    # avoid exponent blow-ups.
    if not equation or not all(c in _ALLOWED_CHARS for c in equation):
        return None
    if "**" in equation:
        return None
    try:
        return float(eval(equation, {"__builtins__": {}}, {}))  # noqa: S307 - inputs char-allowlisted above
    except (ArithmeticError, SyntaxError, ValueError, TypeError):
        return None


@register_reward("countdown")
class CountdownReward(RewardFunction):
    # config keys: tolerance (default 1e-6), format_reward (default 0.1)
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.tolerance = self.config.get("tolerance", 1e-6)
        self.format_reward = self.config.get("format_reward", 0.1)

    def __call__(self, messages: list[dict], label, context: dict | None = None, **kwargs) -> Reward:
        target, nums = _parse_label(label)

        equation = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                equation = extract_equation(msg["content"])
                if equation is not None:
                    break

        fmt_reward = self.format_reward if equation else 0.0

        answer_reward = 0.0
        if equation is not None and target is not None:
            used = [int(n) for n in _NUMBER_PATTERN.findall(equation)]
            # Each provided number must be used exactly once (multiset equality).
            if nums and sorted(used) == sorted(nums):
                value = evaluate_equation(equation)
                if value is not None and abs(value - target) <= self.tolerance:
                    answer_reward = 1.0

        logger.debug(
            "CountdownReward: target=%s nums=%s equation=%r answer=%.1f format=%.2f",
            target, nums, equation, answer_reward, fmt_reward,
        )
        return Reward(
            final_reward=answer_reward + fmt_reward,
            is_correct=bool(answer_reward),
            is_valid=True,
        )
