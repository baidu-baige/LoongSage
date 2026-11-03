"""Frozen probability scorer of RLCD²: letter read-out on a separate, frozen SGLang server.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
from typing import Sequence

import httpx

RELEASE_PREFIX = "Let me work through this carefully, step by step.\n"
CLOSE = "\n</think>\n\n"
CUT = "\n\nI have to decide now with what I have worked out so far.\n</think>\n\n"
TAIL_CHARS = 4000
MIX = 0.75


def probabilities(z: Sequence[float], temperature: float) -> list[float]:
    """softmax(z / T) over the K letter logits z read at the last prompt position."""
    m = max(z)
    e = [math.exp((x - m) / temperature) for x in z]
    s = sum(e)
    return [x / s for x in e]


def thought_text(generated: str, prefix: str = RELEASE_PREFIX) -> str:
    """The whole thought as the read-out sees it before the tail cut: prefix + generated, think tags removed."""
    return (prefix + generated).replace("<think>", "", 1).replace("</think>", "").strip()


def thought_read_prompt(think_base: str, generated: str, ended: str, prefix: str = RELEASE_PREFIX,
                        tail_chars: int = TAIL_CHARS) -> str:
    """The read-out prompt after a thought. ``think_base`` ends with the opened think block (without the prefix);
    ``generated`` is the text the policy wrote after the prefix; ``ended`` is "closed" or "budget"."""
    thought = thought_text(generated, prefix)
    if tail_chars and len(thought) > tail_chars:
        thought = "[...]\n" + thought[-tail_chars:]
    return think_base + thought + (CUT if ended == "budget" else CLOSE)


class DecisionConfig:
    """Read-out settings from a model folder's decision_config.json: letter_token_ids, temperature_by_type, and the
    auto_think overrides of prefix, tail_chars, mix and think_temperature."""

    def __init__(self, model_path: str):
        with open(os.path.join(model_path, "decision_config.json")) as f:
            cfg = json.load(f)
        self.letter_ids: list[int] = list(cfg["letter_token_ids"])
        self.temperature_by_type: dict[str, float] = {k: float(v) for k, v in cfg["temperature_by_type"].items()}
        auto = cfg.get("auto_think") or {}
        self.prefix: str = auto.get("prefix", RELEASE_PREFIX)
        self.tail_chars: int = int(auto.get("tail_chars", TAIL_CHARS))
        self.mix: float = float(auto.get("mix", MIX))     # final = (1 - mix) * p0 + mix * p1
        # The read-out after a thought uses its own temperature, 1.0 for every type unless auto_think.think_temperature
        # overrides it (not temperature_by_type, which is for the one-pass read-out): p1 = softmax(z / think_temperature).
        self.think_temperature: dict[str, float] = {"choice": 1.0, "score": 1.0, "noul": 1.0}
        self.think_temperature.update({k: float(v) for k, v in (auto.get("think_temperature") or {}).items()})

    def temperature(self, question_type: str) -> float:
        return self.temperature_by_type.get(question_type, 1.0)

    def after_thought_temperature(self, question_type: str) -> float:
        return self.think_temperature.get(question_type, 1.0)


class ScorerClient:
    """Async client of one or more frozen SGLang servers (round-robin), reading letter probabilities.
    ``urls`` is a list or a comma-separated string."""

    def __init__(self, urls: Sequence[str] | str, decision: DecisionConfig, max_concurrency: int = 64,
                 timeout: float = 600.0, retries: int = 3):
        if isinstance(urls, str):
            urls = urls.split(",")
        self.urls = [u.strip() for u in urls if u.strip()]
        if not self.urls:
            raise ValueError("at least one scorer url is needed")
        self.decision = decision
        self.max_concurrency = max_concurrency
        self.timeout = timeout
        self.retries = retries
        self._next = 0
        # Created lazily: the client and the semaphore must belong to the event loop that uses them.
        self._loop = None
        self._client: httpx.AsyncClient | None = None
        self._sem: asyncio.Semaphore | None = None

    def _ensure(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            self._loop = loop
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout),
                                             limits=httpx.Limits(max_connections=self.max_concurrency))
            self._sem = asyncio.Semaphore(self.max_concurrency)

    async def letter_logprobs(self, prompt: str, k: int) -> list[float]:
        """log p(letter | prompt) of the first k letters at the next position (full-vocabulary log-softmax)."""
        self._ensure()
        ids = self.decision.letter_ids[:k]
        body = {"text": prompt, "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
                "return_logprob": True, "token_ids_logprob": ids}
        last_exc: Exception | None = None
        for attempt in range(self.retries):
            url = self.urls[self._next % len(self.urls)]
            self._next += 1
            try:
                async with self._sem:
                    r = await self._client.post(f"{url}/generate", json=body)
                r.raise_for_status()
                out = r.json()
                return _parse_letter_logprobs(out, ids)
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                last_exc = exc
                if attempt + 1 < self.retries:
                    await asyncio.sleep(1.0 * (attempt + 1))
        raise RuntimeError(f"scorer request failed after {self.retries} attempts: {last_exc!r}")

    async def read(self, prompt: str, k: int, question_type: str, after_thought: bool = False) -> list[float]:
        """Letter probabilities over the k options: softmax(z / T_type) for the one-pass read-out (p0), softmax(z /
        think_temperature) for the read-out after a thought (p1), as the inference pipeline does."""
        z = await self.letter_logprobs(prompt, k)
        d = self.decision
        t = d.after_thought_temperature(question_type) if after_thought else d.temperature(question_type)
        return probabilities(z, t)

    async def close(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()


def _parse_letter_logprobs(out: dict, ids: Sequence[int]) -> list[float]:
    """Pick the requested token ids from meta_info.output_token_ids_logprobs[0] (entries (logprob, id, text))."""
    if isinstance(out, list):
        out = out[0]
    entries = out["meta_info"]["output_token_ids_logprobs"][0]
    by_id = {int(e[1]): float(e[0]) for e in entries}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise ValueError(f"scorer reply lacks logprobs of token ids {missing}")
    return [by_id[i] for i in ids]
