"""Proper scoring rules of RLCD² (arXiv 2610.17309, §3.1, Eq. 3-5).
"""
from __future__ import annotations

import math
from typing import Sequence

RULES = ("decision_weighted", "brier", "log")
LOG_EPS = 1e-12


def weight(c: float) -> float:
    """w(c): equal mixture of Uniform(0, 1) and Beta(8, 2) densities."""
    return 0.5 + 0.5 * 72.0 * c ** 7 * (1.0 - c)


def big_w(q: float) -> float:
    """W(q) = ∫_0^q w(c) dc."""
    return 0.5 * q + 0.5 * (9.0 * q ** 8 - 8.0 * q ** 9)


def big_c(q: float) -> float:
    """C(q) = ∫_0^q c w(c) dc."""
    return 0.25 * q * q + 0.5 * (8.0 * q ** 9 - 7.2 * q ** 10)


def option_score(q: float, z: float) -> float:
    """s(q, z) for one option with reported probability q and outcome z in [0, 1] (soft outcomes are linear)."""
    q = min(1.0, max(0.0, q))
    return z * big_w(q) - big_c(q)


def decision_weighted(p: Sequence[float], t: Sequence[float]) -> float:
    return sum(option_score(q, z) for q, z in zip(p, t))


def brier(p: Sequence[float], t: Sequence[float]) -> float:
    return -sum((q - z) ** 2 for q, z in zip(p, t))


def log_score(p: Sequence[float], t: Sequence[float]) -> float:
    return sum(z * math.log(max(q, LOG_EPS)) for q, z in zip(p, t) if z > 0)


_NOMINAL = {"decision_weighted": decision_weighted, "brier": brier, "log": log_score}


def _cumulative(x: Sequence[float]) -> list[float]:
    out, acc = [], 0.0
    for v in x[:-1]:
        acc += v
        out.append(min(1.0, max(0.0, acc)))
    return out


def score(p: Sequence[float], t: Sequence[float], question_type: str, rule: str = "decision_weighted") -> float:
    """S(p, t) for one question. ``score`` questions (ordered levels) are scored on their K - 1 cumulative events
    "level <= j", each as a two-option question, and the scores are summed (for Brier this is -2 x RPS)."""
    if rule not in _NOMINAL:
        raise ValueError(f"unknown scoring rule {rule!r}, use one of {RULES}")
    if len(p) != len(t):
        raise ValueError(f"prediction has {len(p)} options, target has {len(t)}")
    fn = _NOMINAL[rule]
    if question_type != "score":
        return fn(p, t)
    return sum(fn((P, 1.0 - P), (T, 1.0 - T)) for P, T in zip(_cumulative(p), _cumulative(t)))
