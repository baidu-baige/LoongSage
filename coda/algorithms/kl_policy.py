"""KL-policy classes for OPD (Online Policy Distillation).

Each KL method is a registered :class:`KLPolicy` subclass that owns the full
lifecycle of its divergence:

* ``collect_teacher``         — how to collect this method's teacher outputs in
  the teacher forward (via :class:`~coda.backends.megatron.kl_ctx.TeacherCtx`
  memoized primitives).
* ``need_teacher_logits``     — whether it needs teacher full logits (the
  hidden-states hook + teacher ``lm_head`` reconstruction, owned by the train
  worker's ``TeacherLMHeads`` singleton).
* ``compute_kl``              — the actual per-token KL (via
  :class:`~coda.backends.megatron.kl_ctx.KLCtx`).

This module is intentionally backend-agnostic: it holds only the KL *math*.
The Megatron/TP plumbing (CP slice / gather / vocab-parallel autograd) lives in
:mod:`coda.backends.megatron.kl_ctx` and
:mod:`coda.backends.megatron.vocab_parallel_kl`, imported lazily inside
``compute_kl`` so importing this module never pulls in the Megatron backend.

The three base classes (:class:`FullVocabPolicy`, :class:`LogProbKLPolicy`,
:class:`TopkVocabPolicy`) are public so users can subclass them to register
their own KL policies.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
from omegaconf import DictConfig

from coda.algorithms.registry import (
    get_kl_policy_cls,
    register_kl_policy,
)

if TYPE_CHECKING:
    from coda.backends.megatron.kl_ctx import KLCtx, TeacherCtx

logger = logging.getLogger(__name__)

__all__ = [
    "build_kl_policies",
    "compute_approx_kl",
    "KLPolicy",
    "FullVocabPolicy",
    "LogProbKLPolicy",
    "TopkVocabPolicy",
    "FullKLPolicy",
    "FullJSDPolicy",
    "K1Policy",
    "K2Policy",
    "K3Policy",
    "TopkKLPolicy",
    "TopkJSDPolicy",
]


def build_kl_policies(config: DictConfig) -> dict[str, "KLPolicy"]:
    """Create stateful policy instances keyed by role (``"pg"`` / ``"gkd"``).

    Meant to be called once at worker init; the returned dict is stored on the
    worker and passed into model functions.
    """
    if not config.opd.get("enable"):
        return {}
    policies: dict[str, KLPolicy] = {}
    if config.opd.pg_ratio > 0 and config.opd.gkd_ratio != 1:
        policies["pg"] = get_kl_policy_cls(str(config.opd.pg_kl_method))(config)
    if config.opd.gkd_ratio > 0:
        policies["gkd"] = get_kl_policy_cls(str(config.opd.gkd_kl_method))(config)
    return policies

@torch.compile(dynamic=True)
def compute_approx_kl(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    kl_type: str,
    importance_ratio: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-token approximate KL between ``log_probs`` and ``ref_log_probs``.

    Standalone scalar estimator for the reference-model KL penalty; operates
    elementwise so it works on either per-trajectory or concatenated tensors.
    Formulas mirror :class:`K1Policy` / :class:`K2Policy` / :class:`K3Policy`.

    Not part of OPD: distinct from the :meth:`KLPolicy.compute_kl` family
    below despite the shared k1/k2/k3 names. ``compute_kl`` is a
    full-distribution student-vs-teacher KL (via ``KLCtx``, topk/full-vocab
    + TP autograd); this is a log-prob scalar approximation against a frozen
    reference model (no ``KLCtx``, no teacher plumbing). The two are
    orthogonal: OPD teacher-KL vs RLHF ref-KL.

    All three estimate the same divergence ``KL(pi || pi_ref)`` under samples
    from ``pi``; they differ in variance and in whether they stay non-negative.

    * ``k1``: ``log_pi - log_pi_ref``. Unbiased, but signed and high-variance.
    * ``k2``: ``0.5 * (log_pi - log_pi_ref)^2``. Always non-negative.
    * ``k3``: ``exp(r) - r - 1`` with ``r = log_pi_ref - log_pi``. Unbiased and
      non-negative; clamped to ``[-10, 10]`` afterwards to bound the ``exp``.

    When ``importance_ratio`` is given (``exp(log_probs - old_log_probs)``), the
    per-token KL is multiplied by it for the unbiased estimate (DeepSeek-V3.2).
    """
    if kl_type == "k1":
        kl = log_probs - ref_log_probs
    elif kl_type == "k2":
        diff = log_probs - ref_log_probs
        kl = 0.5 * diff * diff
    elif kl_type == "k3":
        ratio = ref_log_probs - log_probs
        kl = torch.exp(ratio) - ratio - 1.0
    else:
        raise ValueError(f"unknown kl_type: {kl_type!r}")
    
    if importance_ratio is not None:
        kl = importance_ratio * kl
    if kl_type == "k3":
        kl = torch.clamp(kl, min=-10, max=10)
    return kl



# ════════════════════════════════════════════════════════════════════════
# KLPolicy ABC
# ════════════════════════════════════════════════════════════════════════

class KLPolicy(ABC):
    """Self-contained owner of one KL divergence method.

    Three core interfaces: collect_teacher, need_teacher_logits, compute_kl.
    """

    def __init__(self, config: DictConfig):
        self.config = config

    @abstractmethod
    def collect_teacher(self, ctx: "TeacherCtx") -> dict[str, list[torch.Tensor]]:
        """Collect this method's teacher primitives for one microbatch.

        Returns a ``{field_name: per-trajectory full-sequence tensors}`` dict.
        Each field name (``"teacher_"``-prefixed, owned by the policy) is what
        ``compute_kl`` later reads back via ``ctx.teacher_field(name)`` on the
        :class:`KLCtx` — which re-slices the produced full sequences to the
        student's CP-local chunk. Use the ``TeacherCtx`` convenience primitives
        (:meth:`~TeacherCtx.log_probs`, :meth:`~TeacherCtx.topk`) or
        :meth:`~TeacherCtx.get` to memoize a custom primitive.
        """

    def need_teacher_logits(self) -> bool:
        """Whether this policy needs teacher logits.

        When True, the teacher forward collects hidden states and ``compute_kl``
        can call ``ctx.teacher_logits()`` on the :class:`KLCtx` to get the
        reconstructed (via the teacher lm_head) CP-local teacher logits directly.
        """
        return False

    @abstractmethod
    def compute_kl(
        self, config: DictConfig, ctx: "KLCtx"
    ) -> tuple[list[torch.Tensor], dict[str, list[torch.Tensor]]]:
        """Compute per-token KL (CP-local full-sequence).

        Returns ``(per_token_kl, extra_metrics)``.  ``per_token_kl`` is the
        per-trajectory CP-local full-sequence KL.  Each value in
        ``extra_metrics`` must share that exact shape (per-trajectory CP-local
        full-seq) — the loss caller runs every metric through the same CP
        gather + response-slice + loss-mask-weighted sum as ``per_token_kl``,
        so do NOT pre-reduce to a scalar here.
        """


# ════════════════════════════════════════════════════════════════════════
# full-vocab policies (need teacher_hidden_states + teacher lm_heads)
# ════════════════════════════════════════════════════════════════════════

class FullVocabPolicy(KLPolicy):
    """Shared lifecycle for full-vocabulary KL methods.

    Subclass and implement :meth:`compute_kl` to register a custom full-vocab
    KL divergence.  These methods need reconstructed teacher logits, declared
    purely via :meth:`need_teacher_logits`; the teacher forward then collects
    hidden states and the student side rebuilds logits through the teacher
    ``lm_head`` (:meth:`~coda.backends.megatron.kl_ctx.KLCtx.teacher_logits`).
    That hidden-states transport is a framework implementation detail of "needs
    teacher logits", so this policy owns no teacher field name and produces
    nothing in :meth:`collect_teacher` — consume just calls ``ctx.teacher_logits()``.
    """

    def collect_teacher(self, ctx: "TeacherCtx") -> dict[str, list[torch.Tensor]]:
        return {}

    def need_teacher_logits(self) -> bool:
        return True


@register_kl_policy("full_kl")
class FullKLPolicy(FullVocabPolicy):
    """Full-vocabulary KL over the whole vocabulary.

    In OPD the KL is always the reverse KL ``KL(Q || P)`` with ``Q`` the
    student and ``P`` the teacher;    
    """

    def compute_kl(self, config, ctx):
        from coda.backends.megatron.vocab_parallel_kl import VocabParallelFullKL

        student_logits = ctx.student_logits()
        teacher_logits = ctx.teacher_logits()

        s_cat = torch.cat(student_logits)
        t_cat = torch.cat(teacher_logits)
        lengths = [t.size(0) for t in student_logits]

        # Free teacher logits before the heavy apply.
        del teacher_logits
        ctx.free_teacher_logits()

        per_token_kl_cat = VocabParallelFullKL.apply(s_cat, t_cat)
        del t_cat

        per_token_kl_list = list(per_token_kl_cat.split(lengths))
        return per_token_kl_list, {}


@register_kl_policy("full_jsd")
class FullJSDPolicy(FullVocabPolicy):
    """Jensen-Shannon divergence over the full student and teacher logits."""

    def compute_kl(self, config, ctx):
        from coda.backends.megatron.vocab_parallel_kl import VocabParallelFullJSD

        beta = float(config.opd.get("jsd_beta", 0.5))
        student_logits = ctx.student_logits()
        teacher_logits = ctx.teacher_logits()

        s_cat = torch.cat(student_logits)
        t_cat = torch.cat(teacher_logits)
        lengths = [t.size(0) for t in student_logits]

        del teacher_logits
        ctx.free_teacher_logits()

        per_token_kl_cat = VocabParallelFullJSD.apply(s_cat, t_cat, beta)
        del t_cat

        per_token_kl_list = list(per_token_kl_cat.split(lengths))
        return per_token_kl_list, {}


# ════════════════════════════════════════════════════════════════════════
# per-token log-prob policies (k1/k2/k3 — need teacher_log_probs)
# ════════════════════════════════════════════════════════════════════════

class LogProbKLPolicy(KLPolicy):
    """Shared lifecycle for the scalar-log-prob KL methods (k1/k2/k3).

    Subclass and implement :meth:`compute_kl` to register a custom scalar
    log-prob KL.

    The teacher field name is owned here so produce (:meth:`collect_teacher`)
    and consume (``compute_kl``) cannot drift; it must keep the ``"teacher_"``
    prefix so transport/free-list prefix sweeps pick it up.
    """

    TEACHER_LOG_PROBS = "teacher_log_probs"

    def collect_teacher(self, ctx: "TeacherCtx") -> dict[str, list[torch.Tensor]]:
        return {self.TEACHER_LOG_PROBS: ctx.log_probs()}


@register_kl_policy("k1")
class K1Policy(LogProbKLPolicy):
    """Reverse KL estimator: log π_s(a) - log π_t(a), with a sampled from π_s."""

    def compute_kl(self, config, ctx):
        student_lp = ctx.student_log_probs()
        teacher_lp = ctx.teacher_field(self.TEACHER_LOG_PROBS)

        lengths = [len(s) for s in student_lp]
        s_cat = torch.cat(student_lp)
        t_cat = torch.cat(teacher_lp)

        kl_cat = s_cat - t_cat
        per_token_kl = list(kl_cat.split(lengths))
        return per_token_kl, {}


@register_kl_policy("k2")
class K2Policy(LogProbKLPolicy):
    """Squared-log-ratio (k2) estimator: 0.5 * (log π_s - log π_t)^2."""

    def compute_kl(self, config, ctx):
        student_lp = ctx.student_log_probs()
        teacher_lp = ctx.teacher_field(self.TEACHER_LOG_PROBS)

        lengths = [len(s) for s in student_lp]
        s_cat = torch.cat(student_lp)
        t_cat = torch.cat(teacher_lp)

        diff = s_cat - t_cat
        kl_cat = 0.5 * diff * diff
        per_token_kl = list(kl_cat.split(lengths))
        return per_token_kl, {}


@register_kl_policy("k3")
class K3Policy(LogProbKLPolicy):
    """Reverse KL: exp(log π_t - log π_s) - (log π_t - log π_s) - 1."""

    def compute_kl(self, config, ctx):
        student_lp = ctx.student_log_probs()
        teacher_lp = ctx.teacher_field(self.TEACHER_LOG_PROBS)

        lengths = [len(s) for s in student_lp]
        s_cat = torch.cat(student_lp)
        t_cat = torch.cat(teacher_lp)

        ratio = t_cat - s_cat
        kl_cat = torch.exp(ratio) - ratio - 1.0
        per_token_kl = list(kl_cat.split(lengths))
        return per_token_kl, {}


# ════════════════════════════════════════════════════════════════════════
# top-k policies (need teacher_topk_logprobs / teacher_topk_indices)
# ════════════════════════════════════════════════════════════════════════

class TopkVocabPolicy(KLPolicy):
    """Shared lifecycle for the top-k KL methods.

    Subclass and implement :meth:`compute_kl` to register a custom top-k KL;
    this base collects the teacher's top-k log-probs/indices.

    The teacher field names are owned here so produce (:meth:`collect_teacher`)
    and consume (``compute_kl``) cannot drift; they must keep the ``"teacher_"``
    prefix so transport/free-list prefix sweeps pick them up.
    """

    TEACHER_TOPK_LOGPROBS = "teacher_topk_logprobs"
    TEACHER_TOPK_INDICES = "teacher_topk_indices"

    def collect_teacher(self, ctx: "TeacherCtx") -> dict[str, list[torch.Tensor]]:
        lp, idx = ctx.topk(int(self.config.opd.topk))
        return {self.TEACHER_TOPK_LOGPROBS: lp, self.TEACHER_TOPK_INDICES: idx}


@register_kl_policy("topk_kl")
class TopkKLPolicy(TopkVocabPolicy):
    """Top-K KL on the renormalized top-k support.
    
    In OPD the KL is always the reverse KL ``KL(Q || P)`` with ``Q`` the
    student and ``P`` the teacher;
    """

    def compute_kl(self, config, ctx):
        from coda.backends.megatron.vocab_parallel_kl import VocabParallelTopkKL
        from coda.backends.megatron.logits_utils import compute_topk_overlap

        student_logits = ctx.student_logits()
        t_logps = ctx.teacher_field(self.TEACHER_TOPK_LOGPROBS)
        t_idx = ctx.teacher_field(self.TEACHER_TOPK_INDICES)

        s_cat = torch.cat(student_logits)
        t_logps_cat = torch.cat(t_logps)
        t_idx_cat = torch.cat(t_idx)

        per_token_kl_cat = VocabParallelTopkKL.apply(s_cat, t_logps_cat, t_idx_cat)

        lengths = [t.size(0) for t in student_logits]
        per_token_kl_list = list(per_token_kl_cat.split(lengths))
        return per_token_kl_list, {
            "topk_overlap_ratio": compute_topk_overlap(student_logits, t_idx)
        }


@register_kl_policy("topk_jsd")
class TopkJSDPolicy(TopkVocabPolicy):
    """Jensen-Shannon divergence using top-k teacher data."""

    def compute_kl(self, config, ctx):
        from coda.backends.megatron.vocab_parallel_kl import VocabParallelTopkJSD
        from coda.backends.megatron.logits_utils import compute_topk_overlap

        beta = float(config.opd.get("jsd_beta", 0.5))
        student_logits = ctx.student_logits()
        t_logps = ctx.teacher_field(self.TEACHER_TOPK_LOGPROBS)
        t_idx = ctx.teacher_field(self.TEACHER_TOPK_INDICES)

        s_cat = torch.cat(student_logits)
        t_logps_cat = torch.cat(t_logps)
        t_idx_cat = torch.cat(t_idx)

        per_token_kl_cat = VocabParallelTopkJSD.apply(s_cat, t_logps_cat, t_idx_cat, beta)

        lengths = [t.size(0) for t in student_logits]
        per_token_kl_list = list(per_token_kl_cat.split(lengths))
        return per_token_kl_list, {
            "topk_overlap_ratio": compute_topk_overlap(student_logits, t_idx)
        }
