"""RLCD² reward and agent (arXiv 2610.17309, §3.1-3.2).
"""
from __future__ import annotations

import time
from typing import Any

from coda.agentflow.agent import register_agent
from coda.agentflow.agent.base_agent import BaseAgent
from coda.reward import register_reward
from coda.reward.base import RewardFunction
from coda.reward.reward import Reward

from coda.custom.rlcd2.scorer import DecisionConfig, ScorerClient, thought_read_prompt, thought_text
from coda.custom.rlcd2.scoring_rules import RULES, score


OVERCONFIDENT = 0.9


def _argmax(x) -> int:
    return max(range(len(x)), key=lambda i: x[i])


@register_reward("rlcd2")
class RLCD2Reward(RewardFunction):
    """Reward of RLCD²; used through the ``rlcd2`` agent (it awaits the scorer)."""

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        cfg = self.config
        model_path = cfg.get("model_path")
        if not model_path:
            raise ValueError("data_source.reward.model_path is required")
        urls = cfg.get("scorer_urls")
        if not urls:
            raise ValueError("data_source.reward.scorer_urls is required")
        self.decision = DecisionConfig(model_path)
        self.rule = cfg.get("rule", "decision_weighted")
        if self.rule not in RULES:
            raise ValueError(f"reward.rule must be one of {RULES}, got {self.rule!r}")
        self.lam = float(cfg.get("lam", 1e-4))
        self.budget = int(cfg.get("budget", 3072))
        self.empty_tokens = int(cfg.get("empty_tokens", 8))
        # One client for the life of the process: coda has no shutdown hook for reward functions, so its connections
        # are released at exit. scorer_urls: a comma-separated string or a list.
        self.scorer = ScorerClient(urls, self.decision, max_concurrency=int(cfg.get("max_concurrency", 128)))

    def __call__(self, messages, label, context, **kwargs) -> Reward:
        raise NotImplementedError("the rlcd2 reward awaits the frozen scorer; set data_source.agent.name: rlcd2")

    async def score_thought(self, label: dict, text: str, completion_tokens: int, finish_reason: str | None) -> Reward:
        """Reward of one thought. ``text`` is what the policy generated after the forced prefix."""
        ended = "budget" if finish_reason == "length" else "closed"
        tau_tokens = completion_tokens - (1 if ended == "closed" else 0)
        qtype, k = label["type"], int(label["k"])
        t, p0 = label["target"], label["p0"]
        gold = _argmax(t)
        p0_right = _argmax(p0) == gold
        # "metric/<mean,min,max>/<name>" reports <name>_<reduction> for each listed reduction.
        info: dict[str, Any] = {
            "metric/mean/rlcd2_closed": float(ended == "closed"),
            "metric/mean/rlcd2_tau_tokens": float(tau_tokens),
            "metric/mean/rlcd2_over_budget": float(tau_tokens > self.budget),
            "metric/mean/rlcd2_p0_correct": float(p0_right),
            "metric/mean/rlcd2_p0_gold": float(p0[gold]),
            "rlcd2_ended": ended,
        }
        if tau_tokens <= self.empty_tokens:
            # No p1 is read, so the p1 metrics (rlcd2_p1_correct, ...) skip this trajectory, while is_correct=False
            # counts it as wrong in the framework's group_all_correct / group_all_wrong ratios.
            info.update({"metric/mean/rlcd2_empty": 1.0, "metric/mean/rlcd2_gain": 0.0,
                         "metric/mean/rlcd2_len_penalty": 0.0})
            return Reward(final_reward=0.0, is_correct=False, extra_info=info)

        prompt = thought_read_prompt(label["think_base"], text, ended, self.decision.prefix, self.decision.tail_chars)
        t_read = time.perf_counter()
        p1 = await self.scorer.read(prompt, k, qtype, after_thought=True)
        scorer_ms = 1000.0 * (time.perf_counter() - t_read)
        s0 = score(p0, t, qtype, self.rule)
        s1 = score(p1, t, qtype, self.rule)
        penalty = self.lam * max(0, tau_tokens - self.budget)
        mix = self.decision.mix
        mixed = [(1.0 - mix) * a + mix * b for a, b in zip(p0, p1)]
        p1_correct = _argmax(p1) == gold
        tail_chars = self.decision.tail_chars
        tail_cut = bool(tail_chars) and len(thought_text(text, self.decision.prefix)) > tail_chars
        info.update({
            "metric/mean/rlcd2_empty": 0.0,
            "metric/mean/rlcd2_s0": s0,
            "metric/mean/rlcd2_s1": s1,
            "metric/mean/rlcd2_gain": s1 - s0,
            "metric/mean/rlcd2_len_penalty": penalty,
            "metric/mean/rlcd2_p1_correct": float(p1_correct),
            "metric/mean/rlcd2_mix_correct": float(_argmax(mixed) == gold),
            "metric/mean/rlcd2_p1_top": max(p1),
            "metric/mean/rlcd2_p1_gold": p1[gold],
            "metric/mean/rlcd2_gold_shift": p1[gold] - p0[gold],
            "metric/mean/rlcd2_p_shift": 0.5 * sum(abs(a - b) for a, b in zip(p1, p0)),
            "metric/mean/rlcd2_top_flip": float(_argmax(p1) != _argmax(p0)),
            "metric/mean/rlcd2_fix": float(not p0_right and p1_correct),
            "metric/mean/rlcd2_break": float(p0_right and not p1_correct),
            "metric/mean/rlcd2_overconf_wrong": float(max(p1) >= OVERCONFIDENT and not p1_correct),
            "metric/mean/rlcd2_p1_brier": score(p1, t, qtype, "brier"),
            "metric/mean/rlcd2_tail_cut": float(tail_cut),
            "metric/mean/rlcd2_scorer_ms": scorer_ms,
            "rlcd2_p1": p1,
        })
        return Reward(final_reward=s1 - s0 - penalty, is_correct=p1_correct, extra_info=info)


@register_agent("rlcd2")
class RLCD2Agent(BaseAgent):
    """One thought per trajectory: Router call (prefix in the chat template, stop at </think>), then the reward."""

    async def run_trajectory(self, trajectory: dict[str, Any]) -> Reward:
        if not isinstance(self.reward_fn, RLCD2Reward):
            raise TypeError("the rlcd2 agent needs data_source.reward.name: rlcd2")
        result = await self.call_llm(trajectory["prompt"])
        choice = result["choices"][0]
        message = choice.get("message") or {}
        # With the reasoning parser off (recommended) the whole thought is in "content"; with it on, the parser may
        # move it to reasoning_content.
        text = (message.get("reasoning_content") or "") + (message.get("content") or "")
        tokens = int((result.get("usage") or {}).get("completion_tokens") or 0)
        label = trajectory["label"]
        return await self.reward_fn.score_thought(label, text, tokens, choice.get("finish_reason"))
