"""Unit tests for coda/algorithms/kl_policy.py.

Covers (CPU-only, no process group required):
  * Registry round-trip: register / get / duplicate raises / unknown raises.
  * ``need_teacher_logits`` per policy.
  * ``build_kl_policies`` role gating (pg / gkd).
  * Per-policy numeric correctness against an inline reference on fixed inputs:
      - k1/k2/k3: pure log-prob math.
      - full_kl/full_jsd/topk_kl/topk_jsd: the ``VocabParallel*`` autograd
        Function is mocked to a deterministic stub — validates input
        marshalling, per-trajectory splitting, metric, and teacher-logit freeing.
  * Latent-bug fixes: k1/k2/k3 ``loss_masks`` optional (PG forward-only path);
    JSD reads ``config.opd.jsd_beta`` from the full config.
  * ``TeacherCtx`` memoization + immutability (clone protects the base logits so
    call order of log_probs / entropy / topk does not matter).

The math-only policies live in :mod:`coda.algorithms.kl_policy`; the Megatron
context objects (``TeacherCtx`` / ``KLCtx``) and the vocab-parallel autograd
Functions live under :mod:`coda.backends.megatron` and are imported lazily by
the policies, so the ``VocabParallel*`` stubs below patch the
``coda.backends.megatron.vocab_parallel_kl`` module (``vk``).
"""

from unittest.mock import patch, MagicMock

import pytest
import torch
from omegaconf import OmegaConf

# kl_ctx / vocab_parallel_kl import Megatron-Core at module scope even though the
# KL math exercised below is CPU-only, so skip the module instead of failing
# collection where Megatron is not installed.
pytest.importorskip("megatron", reason="Megatron-Core is not installed")

import coda.backends.megatron.kl_ctx as kc
import coda.backends.megatron.vocab_parallel_kl as vk
from coda.algorithms.kl_policy import (
    FullJSDPolicy,
    FullKLPolicy,
    K1Policy,
    K2Policy,
    K3Policy,
    TopkJSDPolicy,
    TopkKLPolicy,
    build_kl_policies,
)
from coda.algorithms.registry import (
    _KL_POLICY_REGISTRY,
    get_kl_policy_cls,
    register_kl_policy,
)
from coda.backends.megatron.kl_ctx import KLCtx, TeacherCtx


# ═══════════════════════════════════════════════════════════════════════════
# Config helpers
# ═══════════════════════════════════════════════════════════════════════════

def make_opd_config(pg_method=None, gkd_method=None, *, enable=True,
                    jsd_beta=0.5, teachers=1):
    """Build a minimal config exercising the active pg/gkd roles."""
    pg_ratio = 0.0 if pg_method is None else 0.5
    gkd_ratio = 0.0 if gkd_method is None else 0.5
    cfg = {
        "opd": {
            "enable": enable,
            "pg_ratio": pg_ratio,
            "gkd_ratio": gkd_ratio,
            "pg_kl_method": pg_method or "k1",
            "gkd_kl_method": gkd_method or "k1",
            "jsd_beta": jsd_beta,
            "teachers": [{"hf_path": f"/tmp/teacher{i}"} for i in range(teachers)],
        }
    }
    return OmegaConf.create(cfg)


# ═══════════════════════════════════════════════════════════════════════════
# Registry round-trip
# ═══════════════════════════════════════════════════════════════════════════

class TestRegistry:
    def test_all_seven_registered(self):
        assert set(_KL_POLICY_REGISTRY.keys()) == {
            "k1", "k2", "k3", "topk_kl", "topk_jsd", "full_kl", "full_jsd",
        }

    def test_get_returns_class(self):
        assert get_kl_policy_cls("k1") is K1Policy
        assert get_kl_policy_cls("full_kl") is FullKLPolicy
        assert get_kl_policy_cls("topk_jsd") is TopkJSDPolicy

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError):
            get_kl_policy_cls("does_not_exist")

    def test_duplicate_registration_raises(self):
        with pytest.raises(ValueError, match="already registered"):
            @register_kl_policy("k1")
            class _Dup(K1Policy):
                pass


# ═══════════════════════════════════════════════════════════════════════════
# need_teacher_logits per policy
# ═══════════════════════════════════════════════════════════════════════════

class TestNeedTeacherLogits:
    def test_full_policies_need_teacher_logits(self):
        cfg = make_opd_config(gkd_method="full_kl")
        assert FullKLPolicy(cfg).need_teacher_logits() is True
        assert FullJSDPolicy(cfg).need_teacher_logits() is True

    def test_other_policies_no_teacher_logits(self):
        cfg = make_opd_config(gkd_method="k1")
        assert K1Policy(cfg).need_teacher_logits() is False
        assert K2Policy(cfg).need_teacher_logits() is False
        assert K3Policy(cfg).need_teacher_logits() is False
        assert TopkKLPolicy(cfg).need_teacher_logits() is False
        assert TopkJSDPolicy(cfg).need_teacher_logits() is False


# ═══════════════════════════════════════════════════════════════════════════
# build_kl_policies role gating
# ═══════════════════════════════════════════════════════════════════════════

class TestActivePolicies:
    def test_pg_only(self):
        policies = build_kl_policies(make_opd_config(pg_method="k1"))
        assert set(policies) == {"pg"}
        assert isinstance(policies["pg"], K1Policy)

    def test_gkd_only(self):
        policies = build_kl_policies(make_opd_config(gkd_method="full_kl"))
        assert set(policies) == {"gkd"}
        assert isinstance(policies["gkd"], FullKLPolicy)

    def test_both_roles(self):
        policies = build_kl_policies(
            make_opd_config(pg_method="k1", gkd_method="topk_jsd"))
        assert set(policies) == {"pg", "gkd"}
        assert isinstance(policies["pg"], K1Policy)
        assert isinstance(policies["gkd"], TopkJSDPolicy)

    def test_disabled_returns_empty(self):
        assert build_kl_policies(
            make_opd_config(pg_method="k1", enable=False)) == {}


# ═══════════════════════════════════════════════════════════════════════════
# Per-policy numeric correctness (k1/k2/k3 — pure log-prob math)
# ═══════════════════════════════════════════════════════════════════════════

def _make_logprob_inputs(seed=0, lengths=(3, 2)):
    """Per-trajectory student/teacher response log-probs + masks."""
    torch.manual_seed(seed)
    student = [torch.randn(n) for n in lengths]
    teacher = [torch.randn(n) for n in lengths]
    masks = [torch.ones(n) for n in lengths]
    masks[0][-1] = 0.0  # exercise masking
    return student, teacher, masks


def _logprob_klctx(student, teacher, masks=None):
    """Build a KLCtx that serves k1/k2/k3 from pre-computed log-probs.

    output_tensor / packed_seq_params are unused by the log-prob policies, so we
    pass ``None`` and inject student/teacher log-probs directly into the cache
    (bypassing the lazy CP-slice path, which needs a process group).
    """
    total = [len(s) for s in student]
    batch = {
        "teacher_log_probs": teacher,
        "total_lengths": total,
        "response_lengths": total,
    }
    if masks is not None:
        batch["loss_masks"] = masks
    ctx = KLCtx(
        batch=batch,
        output_tensor=None,
        packed_seq_params=None,
    )
    ctx._cache["student_log_probs"] = student
    ctx._cache["teacher_log_probs"] = teacher
    return ctx


def _reference_logprob_kl(method, student, teacher):
    """Inline golden reference for the scalar log-prob KL methods (per-token)."""
    s_cat = torch.cat(student)
    t_cat = torch.cat(teacher)
    if method == "k1":
        kl_cat = s_cat - t_cat
    elif method == "k2":
        diff = s_cat - t_cat
        kl_cat = 0.5 * diff * diff
    elif method == "k3":
        ratio = t_cat - s_cat
        kl_cat = torch.exp(ratio) - ratio - 1.0
    else:
        raise ValueError(method)
    lengths = [len(s) for s in student]
    return list(kl_cat.split(lengths))


@pytest.mark.parametrize("method,policy_cls", [
    ("k1", K1Policy),
    ("k2", K2Policy),
    ("k3", K3Policy),
])
class TestLogProbPolicy:
    def test_per_token(self, method, policy_cls):
        student, teacher, masks = _make_logprob_inputs()
        cfg = make_opd_config(gkd_method=method)

        ref_kl = _reference_logprob_kl(method, student, teacher)

        ctx = _logprob_klctx(student, teacher, masks)
        new_kl, new_metrics = policy_cls(cfg).compute_kl(cfg, ctx)

        assert len(new_kl) == len(ref_kl)
        for a, b in zip(new_kl, ref_kl):
            assert torch.allclose(a, b, atol=1e-6)
        assert new_metrics == {}

    def test_loss_masks_optional(self, method, policy_cls):
        """PG forward-only path omits loss_masks; policy must not KeyError."""
        student, teacher, _ = _make_logprob_inputs()
        cfg = make_opd_config(pg_method=method)
        ctx = _logprob_klctx(student, teacher, masks=None)
        new_kl, new_metrics = policy_cls(cfg).compute_kl(cfg, ctx)
        # per-token KL is mask-independent and the policies emit no metric.
        assert len(new_kl) == len(student)
        assert new_metrics == {}


# ═══════════════════════════════════════════════════════════════════════════
# Per-policy behaviour for the VocabParallel-backed methods
# (full_kl/full_jsd/topk_kl/topk_jsd). The autograd Functions need a TP group;
# we stub them to a deterministic op and verify input marshalling /
# per-trajectory split / metric / teacher-logit free against an inline reference.
# ═══════════════════════════════════════════════════════════════════════════

class _StubFull:
    """Stub for VocabParallelFull{RKL,JSD}.apply -> per-token [N]."""
    @staticmethod
    def apply(s_cat, t_cat, *extra):
        # deterministic, depends on both args so marshalling errors surface
        return (s_cat.sum(-1) - t_cat.sum(-1))


class _StubTopk:
    """Stub for VocabParallelTopk{RKL,JSD}.apply -> per-token [N]."""
    @staticmethod
    def apply(s_cat, t_logps_cat, t_idx_cat, *extra):
        return (s_cat.sum(-1) + t_logps_cat.sum(-1) + t_idx_cat.float().sum(-1))


def _logits_list(seed, lengths, vocab=4):
    torch.manual_seed(seed)
    return [torch.randn(n, vocab) for n in lengths]


class TestFullPolicy:
    @pytest.mark.parametrize("method,policy_cls,stub_attr", [
        ("full_kl", FullKLPolicy, "VocabParallelFullKL"),
        ("full_jsd", FullJSDPolicy, "VocabParallelFullJSD"),
    ])
    def test_marshalling_and_split(self, method, policy_cls, stub_attr):
        lengths = (3, 2)
        student = _logits_list(1, lengths)
        teacher = _logits_list(2, lengths)
        cfg = make_opd_config(gkd_method=method, jsd_beta=0.3)

        # Inline reference mirrors _StubFull.apply over the concatenated inputs.
        ref_cat = torch.cat(student).sum(-1) - torch.cat(teacher).sum(-1)
        ref_kl = list(ref_cat.split([s.size(0) for s in student]))

        with patch.object(vk, stub_attr, _StubFull):
            ctx = KLCtx(
                batch={"total_lengths": list(lengths),
                       "response_lengths": list(lengths)},
                output_tensor=None, packed_seq_params=None,
            )
            ctx.student_logits = lambda: [s.clone() for s in student]
            ctx.teacher_logits = lambda: [t.clone() for t in teacher]
            new_kl, new_metrics = policy_cls(cfg).compute_kl(cfg, ctx)

        assert len(new_kl) == len(ref_kl)
        for a, b in zip(new_kl, ref_kl):
            assert torch.allclose(a, b, atol=1e-6)
        # Full policies emit no extra metrics; the loss layer derives kl-mean.
        assert new_metrics == {}

    def test_teacher_logits_freed(self):
        """compute_kl must free the reconstructed teacher logits (memory).

        teacher_logits() reconstructs hidden->logits via the TeacherLMHeads
        singleton (stubbed here) and caches the result; KLCtx then drops the
        consumed ``teacher_hidden_states`` from ``batch``, and free clears the
        ctx cache.
        """
        lengths = (2,)
        student = _logits_list(1, lengths)
        cfg = make_opd_config(gkd_method="full_kl")
        batch = {
            "total_lengths": list(lengths),
            "response_lengths": list(lengths),
            "teacher_hidden_states": [torch.zeros(n, 3) for n in lengths],
            "teacher_idx": [0] * len(lengths),
        }

        class _StubLMHeads:
            """KLCtx applies the per-teacher lm_head directly via ``_by_idx``."""
            def __init__(self):
                self._by_idx = {0: torch.nn.Linear(3, 4)}

        import coda.backends.megatron.teacher_lm_head as tlm
        # slice_cp_with_zigzag resolves mpu from cp_utils, and an uninitialized
        # Megatron parallel state reports world_size 0. This test is about freeing
        # memory, not CP slicing, so declare CP disabled.
        import coda.backends.megatron.cp_utils as cpu
        cp1_mpu = MagicMock()
        cp1_mpu.get_context_parallel_world_size.return_value = 1
        cp1_mpu.get_context_parallel_rank.return_value = 0
        with patch.object(vk, "VocabParallelFullKL", _StubFull), \
             patch.object(cpu, "mpu", cp1_mpu), \
             patch.object(tlm.TeacherLMHeads, "get", return_value=_StubLMHeads()):
            ctx = KLCtx(batch=batch, output_tensor=None,
                        packed_seq_params=None)
            ctx.student_logits = lambda: [s.clone() for s in student]
            FullKLPolicy(cfg).compute_kl(cfg, ctx)
        assert "teacher_logits" not in batch
        assert "teacher_logits" not in ctx._cache
        assert "teacher_hidden_states" not in batch


class TestTeacherLogitsDtype:
    """The lm_head input is cast to the head's dtype before the matmul.

    ``TeacherLMHeads`` states its dtype from config -- fp32 when
    ``trainer.use_fp32_lm_head`` is on -- while the teacher hidden states carry
    whatever dtype the teacher forward ran in (bf16). ``F.linear`` does not
    promote, so without the cast the reconstruction dies with "expected mat1 and
    mat2 to have the same dtype".
    """

    HIDDEN, VOCAB = 3, 4

    def _teacher_logits(self, cp_partition_mode, head_dtype, hidden_dtype):
        lengths = [2, 2]
        head = torch.nn.Linear(self.HIDDEN, self.VOCAB, dtype=head_dtype)

        class _StubLMHeads:
            _by_idx = {0: head}

        batch = {
            "total_lengths": list(lengths),
            "response_lengths": list(lengths),
            "teacher_hidden_states": [
                torch.zeros(n, self.HIDDEN, dtype=hidden_dtype) for n in lengths
            ],
            "teacher_idx": [0] * len(lengths),
        }

        import coda.backends.megatron.teacher_lm_head as tlm
        import coda.backends.megatron.cp_utils as cpu
        cp1_mpu = MagicMock()
        cp1_mpu.get_context_parallel_world_size.return_value = 1
        cp1_mpu.get_context_parallel_rank.return_value = 0
        cp1_mpu.get_tensor_model_parallel_world_size.return_value = 1
        with patch.object(cpu, "mpu", cp1_mpu), \
             patch.object(tlm.TeacherLMHeads, "get", return_value=_StubLMHeads()):
            ctx = KLCtx(batch=batch, output_tensor=None, packed_seq_params=None,
                        cp_partition_mode=cp_partition_mode)
            # Only the contiguous branch consults student_logits (for segment lens).
            ctx.student_logits = lambda: [torch.zeros(n, self.VOCAB) for n in lengths]
            return ctx.teacher_logits()

    @pytest.mark.parametrize("cp_partition_mode", ["zigzag", "contiguous"])
    def test_fp32_head_with_bf16_hidden(self, cp_partition_mode):
        logits = self._teacher_logits(cp_partition_mode, torch.float32, torch.bfloat16)
        assert [tuple(l.shape) for l in logits] == [(2, self.VOCAB)] * 2
        assert all(l.dtype == torch.float32 for l in logits)

    @pytest.mark.parametrize("cp_partition_mode", ["zigzag", "contiguous"])
    def test_matching_dtypes_unchanged(self, cp_partition_mode):
        logits = self._teacher_logits(cp_partition_mode, torch.bfloat16, torch.bfloat16)
        assert all(l.dtype == torch.bfloat16 for l in logits)


class TestTopkPolicy:
    @pytest.mark.parametrize("method,policy_cls,stub_attr", [
        ("topk_kl", TopkKLPolicy, "VocabParallelTopkKL"),
        ("topk_jsd", TopkJSDPolicy, "VocabParallelTopkJSD"),
    ])
    def test_marshalling_and_metric(self, method, policy_cls, stub_attr):
        lengths = (3, 2)
        k = 2
        student = _logits_list(1, lengths)
        torch.manual_seed(7)
        t_logps = [torch.randn(n, k) for n in lengths]
        t_idx = [torch.randint(0, 4, (n, k)) for n in lengths]
        cfg = make_opd_config(gkd_method=method, jsd_beta=0.3)

        # Inline reference mirrors _StubTopk.apply over the concatenated inputs.
        ref_cat = (torch.cat(student).sum(-1)
                   + torch.cat(t_logps).sum(-1)
                   + torch.cat(t_idx).float().sum(-1))
        ref_kl = list(ref_cat.split([s.size(0) for s in student]))

        # compute_topk_overlap needs a TP group; stub it to a per-token marker
        # so we can assert the metric is forwarded in per-token (not scalar) form.
        overlap_stub = [torch.full((n,), 0.5) for n in lengths]

        import coda.backends.megatron.logits_utils as lu
        with patch.object(vk, stub_attr, _StubTopk), \
             patch.object(lu, "compute_topk_overlap", lambda s, t: overlap_stub):
            ctx = KLCtx(batch={"total_lengths": list(lengths),
                               "response_lengths": list(lengths)},
                        output_tensor=None, packed_seq_params=None)
            ctx.student_logits = lambda: [s.clone() for s in student]
            _fields = {
                "teacher_topk_logprobs": [t.clone() for t in t_logps],
                "teacher_topk_indices": [t.clone() for t in t_idx],
            }
            ctx.teacher_field = lambda name: _fields[name]
            new_kl, new_metrics = policy_cls(cfg).compute_kl(cfg, ctx)

        assert len(new_kl) == len(ref_kl)
        for a, b in zip(new_kl, ref_kl):
            assert torch.allclose(a, b, atol=1e-6)
        # topk policies emit a per-token overlap metric (loss layer reduces it).
        assert new_metrics["topk_overlap_ratio"] is overlap_stub


# ═══════════════════════════════════════════════════════════════════════════
# JSD config scope (latent-bug fix): jsd_beta read from full config.opd
# ═══════════════════════════════════════════════════════════════════════════

class TestJSDConfigScope:
    def test_full_jsd_reads_jsd_beta_from_config_opd(self):
        """compute_kl gets the full config; jsd_beta resolves via config.opd."""
        lengths = (2,)
        student = _logits_list(1, lengths)
        teacher = _logits_list(2, lengths)
        captured = {}

        class _BetaCapture:
            @staticmethod
            def apply(s, t, beta):
                captured["beta"] = beta
                return s.sum(-1)

        cfg = make_opd_config(gkd_method="full_jsd", jsd_beta=0.42)
        with patch.object(vk, "VocabParallelFullJSD", _BetaCapture):
            ctx = KLCtx(batch={"total_lengths": list(lengths),
                               "response_lengths": list(lengths)},
                        output_tensor=None, packed_seq_params=None)
            ctx.student_logits = lambda: [s.clone() for s in student]
            ctx.teacher_logits = lambda: [t.clone() for t in teacher]
            FullJSDPolicy(cfg).compute_kl(cfg, ctx)
        assert captured["beta"] == pytest.approx(0.42)


# ═══════════════════════════════════════════════════════════════════════════
# Produce/consume field-name linkage: the keys a policy produces in
# collect_teacher must be exactly the field-name attributes it owns (the same
# attributes its compute_kl consumes via ctx.teacher_field(self.TEACHER_*)), so
# the two sides can never drift, and every name uses the transport prefix.
# ═══════════════════════════════════════════════════════════════════════════

class TestProduceConsumeFieldNames:
    @pytest.mark.parametrize("name", sorted(_KL_POLICY_REGISTRY.keys()))
    def test_collect_keys_match_owned_field_names(self, name):
        policy_cls = get_kl_policy_cls(name)
        cfg = make_opd_config(gkd_method=name)
        cfg.opd.topk = 8  # consumed by TopkKLPolicy.collect_teacher
        policy = policy_cls(cfg)

        # The field names the policy owns (and therefore consumes by the same
        # attribute in compute_kl).
        owned = {
            getattr(policy, attr)
            for attr in dir(policy)
            if attr.startswith("TEACHER_") and isinstance(getattr(policy, attr), str)
        }

        # Stub the TeacherCtx primitives so collect_teacher runs CPU-only.
        ctx = MagicMock()
        ctx.hidden_states.return_value = [torch.zeros(1)]
        ctx.log_probs.return_value = [torch.zeros(1)]
        ctx.topk.return_value = ([torch.zeros(1)], [torch.zeros(1)])

        produced = set(policy.collect_teacher(ctx).keys())
        assert produced == owned
        assert all(k.startswith("teacher_") for k in produced)


# ═══════════════════════════════════════════════════════════════════════════
# TeacherCtx: memoization + immutability (clone protects base logits, so the
# order of log_probs() / entropy() / topk() does not change results).
# ═══════════════════════════════════════════════════════════════════════════

class TestTeacherCtx:
    def test_get_memoizes(self):
        ctx = TeacherCtx(output_tensor=torch.randn(4, 3),
                         total_lengths=[4], response_lengths=[4])
        calls = []

        def compute():
            calls.append(1)
            return object()

        first = ctx.get("k", compute)
        second = ctx.get("k", compute)
        assert first is second
        assert len(calls) == 1

    def test_logits_temperature_scaled_and_cached(self):
        out = torch.randn(4, 3)
        ctx = TeacherCtx(output_tensor=out, total_lengths=[4],
                         response_lengths=[4], temperature=2.0)
        logits = ctx.logits()
        assert torch.allclose(logits, out / 2.0)
        # cached: same object on second call
        assert ctx.logits() is logits

    def test_logits_immutable_under_consumers(self):
        """log_probs()/entropy() clone, so logits() is never mutated and the
        call order does not matter. We stub the underlying primitives to
        *mutate* their input in-place — proving the clone defends the base."""
        out = torch.randn(5, 3)
        baseline = out.clone()

        def mutating_ce(logits, targets, tp_group):
            logits.add_(100.0)  # in-place corruption of the (cloned) input
            return logits.sum(-1)

        def mutating_entropy(total, resp, logits, temperature=1.0, cp_partition_mode="zigzag"):
            logits.mul_(-7.0)  # in-place corruption
            return [logits.sum(-1)]

        cp1_mpu = MagicMock()
        cp1_mpu.get_context_parallel_world_size.return_value = 1
        cp1_mpu.get_tensor_model_parallel_group.return_value = None

        ctx = TeacherCtx(output_tensor=out, total_lengths=[5],
                         response_lengths=[5], packed_targets=torch.zeros(5, dtype=torch.long))
        with patch.object(kc, "fused_vocab_parallel_cross_entropy", mutating_ce), \
             patch.object(kc, "mpu", cp1_mpu), \
             patch.object(kc, "compute_entropy", mutating_entropy):
            lp_first = ctx.log_probs()
            # base logits unchanged despite in-place mutation inside the stub
            assert torch.allclose(ctx.logits(), baseline)
            ent = ctx.entropy()
            assert torch.allclose(ctx.logits(), baseline)
        assert lp_first is not None and ent is not None

    def test_order_independence_logprobs_vs_entropy(self):
        """log_probs() then entropy() == entropy() then log_probs()."""
        out = torch.randn(6, 4)

        def ce(logits, targets, tp_group):
            logits.add_(1.0)
            return logits.sum(-1).clone()

        def ent(total, resp, logits, temperature=1.0, cp_partition_mode="zigzag"):
            logits.mul_(2.0)
            return [logits.sum(-1).clone()]

        cp1_mpu = MagicMock()
        cp1_mpu.get_context_parallel_world_size.return_value = 1
        cp1_mpu.get_tensor_model_parallel_group.return_value = None

        with patch.object(kc, "fused_vocab_parallel_cross_entropy", ce), \
             patch.object(kc, "mpu", cp1_mpu), \
             patch.object(kc, "compute_entropy", ent):
            ctx_a = TeacherCtx(output_tensor=out.clone(), total_lengths=[6],
                               response_lengths=[6],
                               packed_targets=torch.zeros(6, dtype=torch.long))
            a_lp = ctx_a.log_probs()
            a_ent = ctx_a.entropy()

            ctx_b = TeacherCtx(output_tensor=out.clone(), total_lengths=[6],
                               response_lengths=[6],
                               packed_targets=torch.zeros(6, dtype=torch.long))
            b_ent = ctx_b.entropy()
            b_lp = ctx_b.log_probs()

        assert torch.allclose(a_lp[0], b_lp[0], atol=1e-6)
        assert torch.allclose(a_ent[0], b_ent[0], atol=1e-6)





