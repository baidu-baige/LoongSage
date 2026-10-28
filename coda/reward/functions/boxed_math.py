"""Boxed-answer math reward function (e.g. geo3k)."""

from __future__ import annotations

import logging
import re

from coda.reward import register_reward
from coda.reward.base import RewardFunction
from coda.reward.functions.dapo_math_util import (
    last_boxed_only_string,
    normalize_final_answer,
    remove_boxed,
)
from coda.reward.reward import Reward

logger = logging.getLogger(__name__)


def _extract_boxed(text: str) -> str | None:
    """Return the content of the last `\\boxed{...}` in *text*, or None if there is none."""
    boxed = last_boxed_only_string(text)
    return remove_boxed(boxed) if boxed is not None else None


def _unify_fractions(text: str) -> str:
    return text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")


_MAX_SYMBOLIC_LEN = 120  # longer answers are never parsed symbolically (bounds sympy work)
_MAX_CARETS = 2
_SYMBOLIC_CHARS = re.compile(r"[0-9A-Za-z+\-*/().\s]*")
_SMALL_EXPONENT = re.compile(r"-?\d{1,2}|\(\(-?\d{1,2}\)/\(\d{1,2}\)\)")


def _latex_arg(s: str, i: int) -> tuple[str, int] | None:
    """Read one LaTeX argument at *i*: a ``{...}`` group or a single alphanumeric character."""
    while i < len(s) and s[i] == " ":
        i += 1
    if i >= len(s):
        return None
    if s[i] != "{":
        return (s[i], i + 1) if s[i].isalnum() else None
    depth = 0
    for j in range(i, len(s)):
        depth += (s[j] == "{") - (s[j] == "}")
        if depth == 0:
            return s[i + 1 : j], j + 1
    return None


def _latex_to_sympy_src(s: str) -> str | None:
    """Translate the LaTeX subset used by math answers (frac, sqrt, ^, braces) to sympy source, else None."""
    out: list[str] = []
    i = 0
    while i < len(s):
        if s.startswith("\\frac", i):
            num = _latex_arg(s, i + 5)
            den = _latex_arg(s, num[1]) if num else None
            parts = [_latex_to_sympy_src(a[0]) for a in (num, den)] if den else None
            if not parts or None in parts:
                return None
            out.append(f"(({parts[0]})/({parts[1]}))")
            i = den[1]
        elif s.startswith("\\sqrt", i):
            index, j = None, i + 5
            if s[j : j + 1] == "[":
                k = s.find("]", j)
                index = _latex_to_sympy_src(s[j + 1 : k]) if k > 0 else None
                if index is None:
                    return None
                j = k + 1
            arg = _latex_arg(s, j)
            body = _latex_to_sympy_src(arg[0]) if arg else None
            if body is None:
                return None
            out.append(f"sqrt({body})" if index is None else f"root({body},{index})")
            i = arg[1]
        elif s[i] == "^":
            arg = _latex_arg(s, i + 1)
            exp = _latex_to_sympy_src(arg[0]) if arg else None
            # Only small literal exponents, never towers (9^9^9): keeps sympy from computing huge integers.
            if exp is None or not _SMALL_EXPONENT.fullmatch(exp.strip()) or s[arg[1] :].lstrip().startswith("^"):
                return None
            out.append(f"**({exp})")
            i = arg[1]
        elif s[i] in "{}":
            out.append("(" if s[i] == "{" else ")")
            i += 1
        elif s[i] == "\\":
            return None  # unsupported LaTeX command
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def _to_sympy(text: str):
    """Parse a normalized LaTeX answer into an exact sympy expression; None if unsupported or unsafe.

    The model output is untrusted, so parsing is whitelist-based: only digits, single-letter variables,
    `sqrt` / `pi` / `root`, arithmetic operators and small literal exponents reach `parse_expr`.
    """
    from sympy import Float, Function, Integer, Rational, Symbol, root, sqrt
    from sympy import pi as sympy_pi
    from sympy.parsing.sympy_parser import implicit_multiplication, parse_expr, standard_transformations

    for old, new in (("\\left", ""), ("\\right", ""), ("\\cdot", "*"), ("\\times", "*"), ("\\pi", " pi ")):
        text = text.replace(old, new)
    if len(text) > _MAX_SYMBOLIC_LEN or text.count("^") > _MAX_CARETS or re.search(r"\*\s*\*", text):
        return None
    src = _latex_to_sympy_src(text)
    if src is None or not _SYMBOLIC_CHARS.fullmatch(src) or re.search(r"\d{10,}", src):
        return None
    if any(len(w) > 1 and w not in ("sqrt", "pi", "root") for w in re.findall(r"[A-Za-z]+", src)):
        return None
    src = re.sub(r"(?<![A-Za-z])([A-Za-z])(?=\()", r"\1*", src)  # y(2y-1) is a product, not a call
    src = re.sub(r"(\d+)\.(\d+)", r"Rational('\1.\2')", src)  # decimals stay exact
    try:
        return parse_expr(
            src,
            local_dict={"sqrt": sqrt, "pi": sympy_pi, "root": root},
            global_dict={
                "__builtins__": {},
                "Integer": Integer,
                "Float": Float,
                "Rational": Rational,
                "Symbol": Symbol,
                "Function": Function,
            },
            transformations=standard_transformations + (implicit_multiplication,),
        )
    except Exception:
        return None


def _symbolically_equal(pred: str, gt: str) -> bool:
    """Exact algebraic equality of two normalized answers, e.g. `3\\sqrt{15}` == `\\sqrt{135}`.

    Approximations never match: `13.33` != `\\frac{40}{3}`.
    """
    from sympy import simplify

    left, right = _to_sympy(pred), _to_sympy(gt)
    if left is None or right is None:
        return False
    try:
        diff = left - right
        return bool(diff == 0 or simplify(diff) == 0)
    except Exception:
        return False


def _same_answer(pred: str, gt: str) -> bool:
    """Compare two answers after normalization.

    Matches, in order: identical strings, equal plain numbers (270 == 270.0), and exact algebraic
    equivalence via sympy (`3\\sqrt{15}` == `\\sqrt{135}`). `\\dfrac` / `\\tfrac` are treated as `\\frac`,
    since models and labels use them interchangeably.
    """
    pred, gt = (normalize_final_answer(_unify_fractions(s)) for s in (pred, gt))
    if pred == gt:
        return True
    try:
        return float(pred) == float(gt)
    except ValueError:
        return _symbolically_equal(pred, gt)


@register_reward("boxed_math")
class BoxedMathReward(RewardFunction):
    """Binary reward: 1.0 when the last `\\boxed{}` answer matches the label, else 0.0.

    The label is the ground-truth answer string, or a dict holding it under `ground_truth`,
    `value` or `answer`. Rewards are strictly 0/1 on purpose: non-binary format scores make
    `reward - group_mean` non-zero in fp32 even when all samples in a group score the same.
    """

    def __call__(self, messages: list[dict], label, context: dict | None = None, **kwargs) -> Reward:
        if isinstance(label, dict):
            label = next((label[k] for k in ("ground_truth", "value", "answer") if label.get(k) is not None), None)
        if label is None or not str(label).strip():
            return Reward(final_reward=0.0, is_valid=False, is_correct=False)

        response = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "assistant" and m.get("content")),
            "",
        )
        predicted = _extract_boxed(response)
        correct = predicted is not None and _same_answer(predicted, str(label))
        logger.debug("predicted=%s, ground_truth=%s, correct=%s", predicted, label, correct)
        return Reward(final_reward=float(correct), is_correct=correct)
