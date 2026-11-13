"""GSM8K reward for answer-only responses (the reply is just the final number)."""

from __future__ import annotations

import logging
import re

from omegaconf import DictConfig

from coda.reward import register_reward
from coda.reward.base import RewardFunction
from coda.reward.reward import Reward

logger = logging.getLogger(__name__)

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def _parse(text: str) -> float | None:
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


@register_reward("gsm8k_answer_only")
class GSM8KAnswerOnlyReward(RewardFunction):
    """1.0 when the whole reply (stripped, commas removed) is a number equal to ``label["ground_truth"]``.

    Otherwise ``format_weight * digit_fraction``, where ``digit_fraction`` is the share of non-space
    characters in the reply that are digits. The shaped term keeps GRPO groups from being all-zero on
    weak models; set ``format_weight: 0`` for a plain 0/1 reward.

    Config keys (all optional):
        format_weight (float): weight of the digit-fraction shaping term. Default 0.0.
        tolerance (float):     absolute tolerance for the numeric comparison. Default 1e-6.
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.format_weight = float(self.config.get("format_weight", 0.0))
        self.tolerance = float(self.config.get("tolerance", 1e-6))

    def __call__(self, messages: list[dict], label, context: dict | None = None, **kwargs) -> Reward:
        truth = _parse(str(label.get("ground_truth", ""))) if isinstance(label, dict) else _parse(str(label))
        reply = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                reply = msg["content"]
                break
        text = reply.strip().replace(",", "")
        pred = _parse(text) if _NUMBER.fullmatch(text) else None
        correct = truth is not None and pred is not None and abs(pred - truth) <= self.tolerance
        if correct:
            score = 1.0
        else:
            chars = [c for c in text if not c.isspace()]
            digit_fraction = sum(c.isdigit() for c in chars) / len(chars) if chars else 0.0
            score = self.format_weight * digit_fraction
        logger.debug(f"reply={reply!r} pred={pred} truth={truth} score={score}")
        return Reward(final_reward=score, is_correct=correct)
