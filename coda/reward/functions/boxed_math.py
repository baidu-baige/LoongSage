"""Boxed-answer math reward function (e.g. geo3k)."""

from __future__ import annotations

import itertools
import logging
import os
import pickle
import queue
import random
import re
import select
import subprocess
import sys

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


_PROBE_POINTS = 3


def _provably_nonzero(diff) -> bool:
    """True if substituting exact random rationals for the variables gives a nonzero rational.

    That proves the two sides differ, in milliseconds. It never proves equality: a zero, a
    singular point (nan/zoo) or a radical result is inconclusive and returns False. This lets
    wrong answers skip `simplify()`, which is slowest exactly when the difference is nonzero.
    """
    from sympy import Rational

    symbols = sorted(diff.free_symbols, key=str)
    if not symbols:
        return False  # constants (e.g. radicals) need simplify to decide
    rng = random.Random(0)  # fixed points: the same answer always gets the same reward
    for _ in range(_PROBE_POINTS):
        point = {s: Rational(rng.choice((-1, 1)) * rng.randint(1, 17), rng.randint(1, 13)) for s in symbols}
        try:
            value = diff.subs(point)
        except Exception:
            continue
        if value.is_Rational and value != 0:
            return True
    return False


# --- simplify() under a hard timeout -------------------------------------------------------
# simplify() can take tens of seconds and can't be interrupted from a thread, so it runs in
# reusable `python -c` subprocesses that import only sympy; a timeout kills just that worker.
# Not multiprocessing: `spawn` re-imports the trainer's __main__ (torch, ray, ...: ~37s and
# ~1.7GB per worker), and `fork` could inherit a lock held by another AgentFlow thread.

_SIMPLIFY_TIMEOUT_S = 5.0  # hard cap per symbolic comparison; legitimate ones finish well under 1s
_MAX_WORKERS = min(4, os.cpu_count() or 1)
_BOOT_TIMEOUT_S = 60.0  # `import sympy` takes ~0.4s; generous for a loaded host

# Protocol: one pickled request on stdin -> one pickled reply on stdout. The real stdout is
# redirected to stderr so a stray print inside sympy cannot corrupt the reply stream.
_WORKER_SRC = """
import os, pickle, sys
out = os.fdopen(os.dup(1), "wb")
os.dup2(2, 1)
from sympy import simplify
pickle.dump("ready", out); out.flush()
inp = sys.stdin.buffer
while True:
    try:
        expr = pickle.load(inp)
    except EOFError:
        break
    try:
        ok = bool(simplify(expr) == 0)
    except Exception:
        ok = False
    pickle.dump(ok, out); out.flush()
"""


class _SympyWorker:
    """One sympy subprocess and its pipes. Used by one caller at a time."""

    def __init__(self) -> None:
        # -P: don't put the cwd on sys.path, so a stray local sympy.py can't be picked up.
        self.proc = subprocess.Popen(
            [sys.executable, "-P", "-c", _WORKER_SRC],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        if self._recv(_BOOT_TIMEOUT_S) != "ready":
            self.kill()
            raise RuntimeError("sympy worker failed to start")

    def _recv(self, timeout: float):
        """Return the next reply, or None if none arrives within *timeout* seconds."""
        ready, _, _ = select.select([self.proc.stdout], [], [], timeout)
        return pickle.load(self.proc.stdout) if ready else None

    def ask(self, expr, timeout: float):
        self.proc.stdin.write(pickle.dumps(expr))
        self.proc.stdin.flush()
        return self._recv(timeout)

    def kill(self) -> None:
        self.proc.kill()
        self.proc.wait()
        self.proc.stdin.close()
        self.proc.stdout.close()


# One entry per worker slot; None means the slot has no live worker yet. Live workers sort
# before empty slots and the most recently used comes first, so a warm worker is always reused
# before a new one is booted. Idle workers exit on their own when the parent exits, because
# their stdin closes and `pickle.load` hits EOF.
_worker_slots: queue.PriorityQueue[tuple[bool, int, _SympyWorker | None]] = queue.PriorityQueue()
_slot_seq = itertools.count()


def _release_slot(worker: _SympyWorker | None) -> None:
    _worker_slots.put((worker is None, -next(_slot_seq), worker))


for _ in range(_MAX_WORKERS):
    _release_slot(None)


def _simplify_is_zero(expr, timeout: float) -> bool:
    """True iff `simplify(expr) == 0`; False if the comparison times out or errors.

    *timeout* covers only the computation, not the wait for a free worker or a cold worker's
    boot. On timeout the worker is killed and replaced on demand.
    """
    _, _, worker = _worker_slots.get()  # waits while every slot is busy
    result = None
    try:
        if worker is not None and worker.proc.poll() is not None:
            worker.kill()  # died while idle: reap it and start a fresh one
            worker = None
        if worker is None:
            worker = _SympyWorker()
        result = worker.ask(expr, timeout)
    except Exception:
        pass
    finally:
        if not isinstance(result, bool) and worker is not None:
            # Timed out, worker died, or the pipe broke: discard this worker only.
            worker.kill()
            worker = None
        _release_slot(worker)
    return result is True


def _symbolically_equal(pred: str, gt: str) -> bool:
    """Exact algebraic equality of two normalized answers, e.g. `3\\sqrt{15}` == `\\sqrt{135}`.

    Approximations never match: `13.33` != `\\frac{40}{3}`.

    The whitelist parse (`_to_sympy`) runs here in the caller; only an already-validated
    sympy expression is handed to the timeout worker, never an untrusted string.
    """
    left, right = _to_sympy(pred), _to_sympy(gt)
    if left is None or right is None:
        return False
    try:
        diff = left - right
    except Exception:
        return False
    if diff == 0:  # cheap structural check; skips the process hop for identical forms
        return True
    if _provably_nonzero(diff):
        return False
    return _simplify_is_zero(diff, timeout=_SIMPLIFY_TIMEOUT_S)


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
