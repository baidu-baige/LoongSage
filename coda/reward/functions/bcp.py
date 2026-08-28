"""BrowseComp-Plus (https://texttron.github.io/BrowseComp-Plus/) reward function.

Three-tool setup (search / open_page / finish):
- Outcome reward (0 or 1.0): answer from the finish tool call compared against
  ground-truth using normalised string matching.
- Process reward (up to 0.5): behavioural signals from structured tool_calls:
    +0.2  finish tool was called           (agent reached a conclusion)
    +0.2  no repeated search queries       (anti death-loop)
    +0.1  open_page was called at least once  (encourage deep verification)
"""

import json
import logging
import re
import string

from omegaconf import DictConfig

from coda.reward import register_reward
from coda.reward.base import RewardFunction
from coda.reward.reward import Reward

logger = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    """Lowercase, strip, remove articles / punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _outcome_reward(predicted: str | None, ground_truth: str) -> float:
    """Binary outcome reward: 1.0 if predicted matches ground_truth, else 0.0."""
    if not predicted:
        return 0.0
    expected = str(ground_truth).strip()
    if predicted.lower() == expected.lower():
        return 1.0
    norm_pred = _normalize(predicted)
    norm_exp = _normalize(expected)
    if norm_pred == norm_exp:
        return 1.0
    if norm_exp and norm_pred and (norm_exp in norm_pred or norm_pred in norm_exp):
        return 1.0
    return 0.0


@register_reward("bcp")
class BrowseCompPlusReward(RewardFunction):
    """BrowseComp-Plus reward function.

    Computes a combined process + outcome reward for agentic search tasks
    directly from the structured messages list — no regex, no solution_str.
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)

    def __call__(self, messages: list[dict], label, trajectory: dict, **kwargs) -> Reward:
        queries: list[str] = []
        n_search = n_open_page = n_finish = 0
        predicted_answer: str | None = None

        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    args = {}
                if name == "search":
                    n_search += 1
                    queries.extend(args.get("query_list", []))
                elif name == "open_page":
                    n_open_page += 1
                elif name == "finish":
                    n_finish += 1
                    predicted_answer = args.get("answer")

        # Process reward
        process = 0.0
        if n_finish > 0:                                  process += 0.2
        if queries and len(queries) == len(set(queries)): process += 0.2
        if n_open_page > 0:                               process += 0.1

        # Outcome reward
        if isinstance(label, str):
            ground_truth = label
        else:
            ground_truth = label.get("ground_truth") or label.get("value") or ""
        outcome = _outcome_reward(predicted_answer, ground_truth)

        score = process + outcome
        logger.info("ground_truth=%s predicted=%s acc=%s score=%s",
                    ground_truth, predicted_answer, outcome, score)
        return Reward(final_reward=score, is_correct=outcome > 0)
