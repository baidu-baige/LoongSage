"""GSM8K reward function."""

from __future__ import annotations

import logging
import re

from coda.reward.base import RewardFunction
from coda.reward import register_reward
from coda.reward.reward import Reward
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

# Reference patterns:
# - verl GSM8K reward extraction:
#   https://github.com/volcengine/verl/blob/main/verl/utils/reward_score/gsm8k.py
_SOLUTION_CLIP_CHARS = 300
_ANSWER_PATTERN = re.compile(r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)")


# ---------------------------------------------------------------------------
# Helpers (shared with gsm8k_agent for backward compatibility)
# ---------------------------------------------------------------------------


def _clip_solution_tail(text: str) -> str:
    """Clip long solutions to the tail where final answers usually appear."""
    if len(text) <= _SOLUTION_CLIP_CHARS:
        return text
    return text[-_SOLUTION_CLIP_CHARS:]


def _parse_number(text: str) -> float | None:
    """Parse one extracted numeric string."""
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def extract_answer(text: str) -> float | None:
    """Extract GSM8K answer using the strict ``#### <answer>`` format."""
    if not text:
        return None

    clipped = _clip_solution_tail(text)
    matches = _ANSWER_PATTERN.findall(clipped)
    if not matches:
        return None
    return _parse_number(matches[-1])


def extract_ground_truth(ground_truth_str: str) -> float | None:
    """Extract ground truth answer from GSM8K format string."""
    if not ground_truth_str:
        return None

    return extract_answer(ground_truth_str)


def normalize_answer(answer: float) -> float:
    """Normalize a numerical answer for stable comparison.

    Rounds to 6 decimal places to avoid floating-point noise when comparing
    model predictions against ground-truth values.

    Args:
        answer: Raw float answer extracted from text.

    Returns:
        The answer rounded to 6 decimal places.
    """
    return round(answer, 6)


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------


@register_reward("gsm8k")
class GSM8KReward(RewardFunction):
    """
    Reward function for GSM8K mathematical reasoning.

    Extracts the predicted answer from the last assistant message and compares it
    to the ground truth taken from the label (``label["value"]``, falling back to
    ``label["answer"]``, or the label itself when it is a plain string).
    Returns 1.0 for correct / 0.0 for incorrect.

    Config keys (all optional):
        tolerance (float): Absolute tolerance for float comparison. Default 1e-6.
    """

    def __init__(self, config: DictConfig):
        """Initialize from reward config dict."""
        super().__init__(config)
        self.tolerance = self.config.get("tolerance", 1e-6)

    def __call__(self, messages: list[dict], label, **kwargs) -> Reward:
        """Compute reward by comparing the last assistant answer to the label."""
        logger.info(f"GSM8KReward.__call__ label is : {label}")
        if isinstance(label, dict):
            ground_truth_str = label.get("value", "") or label.get("answer", "")
        elif isinstance(label, str):
            ground_truth_str = label
        else:
            ground_truth_str = ""
        logger.info(f"GSM8KReward.__call__ ground_truth_str is : {ground_truth_str}")
        ground_truth = extract_ground_truth(ground_truth_str) if ground_truth_str else None
        logger.info(f"GSM8KReward.__call__ ground_truth is : {ground_truth}")

        predicted = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                predicted = extract_answer(msg["content"])
                logger.info(f"GSM8KReward.__call__ Extracted prediction: {predicted}")
                if predicted is not None:
                    break

        if ground_truth is None:
            # No usable label: correctness is undecidable (a dataset problem, not a model one).
            logger.debug(f"Cannot compute reward: ground_truth={ground_truth}, predicted={predicted}")
            return Reward(final_reward=0.0)
        if predicted is None:
            # The model never emitted a parseable "#### <number>" answer: counts as incorrect.
            logger.debug(f"Cannot compute reward: ground_truth={ground_truth}, predicted={predicted}")
            return Reward(final_reward=0.0)

        correct = abs(predicted - ground_truth) <= self.tolerance
        score = 1.0 if correct else 0.0
        logger.debug(f"predicted={predicted}, ground_truth={ground_truth}, reward={score}")
        return Reward(final_reward=score)
