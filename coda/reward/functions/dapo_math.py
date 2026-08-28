"""DAPO Math reward function."""

from __future__ import annotations

import logging

from coda.reward.base import RewardFunction
from coda.reward import register_reward
from coda.reward.reward import Reward
from omegaconf import DictConfig
from coda.reward.functions.dapo_math_util import compute_score

logger = logging.getLogger(__name__)

@register_reward("dapo_math")
class DapoMathReward(RewardFunction):
    """DAPO Math reasoning reward function.

    Extracts the predicted answer from the last assistant message and compares
    it against ``label["ground_truth"]``.

    Answer extraction: uses regex ``Answer\\s*:\\s*([^\\n]+)`` to find the last
    ``Answer: <value>`` in the response (truncated to the last 300 characters),
    then normalizes the extracted value.
    Answer comparison: normalized string exact match.

    The base score is 1.0 for correct and -1.0 for incorrect, with an optional
    overlong penalty subtracted from the final reward.

    Config keys (all optional):
        overlong_penalty_length (float): Length threshold for overlong penalty.
            When > 0, responses exceeding (max_tokens - overlong_penalty_length)
            tokens receive a linearly increasing penalty. Default 0.0 (disabled).
    """

    def __init__(self, config: DictConfig):
        """Initialize from reward config dict."""
        super().__init__(config)
        self.overlong_penalty_length = float(self.config.get("overlong_penalty_length", 0.0))

    def __call__(self, messages: list[dict], label, trajectory: dict, **kwargs) -> Reward:
        """Compute reward by comparing the last assistant message's answer against the label."""
        ground_truth_str = label.get("ground_truth")
        solution_str = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                solution_str = msg["content"]
                break
        if solution_str.endswith("<|im_end|>"):
            solution_str = solution_str[: -len("<|im_end|>")]
        logger.info(f"extract answer text: {solution_str}")
        result = compute_score(solution_str, ground_truth_str)
        score, predicted = result["score"], result["pred"]
        response_length = len(trajectory["loss_masks"])
        penalty = self._overlong_penalty(response_length, kwargs.get("max_tokens"))
        final_score = score - penalty
        logger.info(f"predicted={predicted}, ground_truth={ground_truth_str}, score={score}, "
                    f"response_length={response_length} penalty={penalty}, reward={final_score}")
        return Reward(final_reward=final_score, is_correct=score > 0)

    def _overlong_penalty(self, response_length, max_length):
        if self.overlong_penalty_length <= 0:
            return 0
        expect_length = max(max_length - self.overlong_penalty_length, 0)
        exceed_length = response_length - expect_length
        if exceed_length <= 0:
            return 0
        else:
            return exceed_length / self.overlong_penalty_length
