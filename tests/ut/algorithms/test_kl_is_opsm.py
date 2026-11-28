"""Regression tests for KL k3 numerics, IS correction k3 KL, and OPSM clipfrac."""

import math

import pytest
import torch
from omegaconf import OmegaConf

from coda.algorithms.is_correction import apply_is_correction
from coda.algorithms.kl_policy import compute_approx_kl
from coda.algorithms.off_policy_seq_masking import compute_opsm_mask


def test_compute_approx_kl_k3_extreme_log_ratio_keeps_gradients_finite() -> None:
    """k3 must clamp ref_log_probs - log_probs before exp so backward does not produce NaN."""
    log_probs = torch.tensor([-100.0, -1.0], dtype=torch.float32, requires_grad=True)
    ref_log_probs = torch.tensor([0.0, -0.8], dtype=torch.float32)

    kl = compute_approx_kl(log_probs, ref_log_probs, "k3")
    kl.sum().backward()

    assert torch.isfinite(kl).all()
    assert kl[0].item() == pytest.approx(10.0, abs=1e-6)
    assert log_probs.grad is not None
    assert torch.isfinite(log_probs.grad).all(), f"Non-finite k3 grad: {log_probs.grad}"
    assert log_probs.grad[0].item() == 0.0

    # In-range token 1 must match exact k3 value and gradient.
    r1 = -0.8 - (-1.0)
    expected_kl1 = math.exp(r1) - r1 - 1.0
    expected_grad1 = -(math.exp(r1) - 1.0)
    assert kl[1].item() == pytest.approx(expected_kl1, abs=1e-6)
    assert log_probs.grad[1].item() == pytest.approx(expected_grad1, abs=1e-6)


def test_apply_is_correction_k3_kl_stays_finite_on_masked_and_active_extreme_ratios() -> None:
    """train/is_approx_k3_kl must not become NaN on masked extreme tokens or inf on active ones."""
    cfg = OmegaConf.create({"enable": False})

    # Masked token has old_lp - rollout_lp = 100.0 (exp overflows to inf; inf * 0 -> NaN if unmasked).
    _, _, masked_metrics = apply_is_correction(
        cfg,
        loss_masks=[torch.tensor([1.0, 0.0], dtype=torch.float32)],
        old_log_probs=[torch.tensor([-0.5, -0.5], dtype=torch.float32)],
        rollout_log_probs=[torch.tensor([-0.6, -100.5], dtype=torch.float32)],
    )
    r0 = -0.5 - (-0.6)
    expected_k3 = math.exp(r0) - r0 - 1.0
    assert math.isfinite(masked_metrics["train/is_approx_k3_kl"])
    assert masked_metrics["train/is_approx_k3_kl"] == pytest.approx(expected_k3, abs=1e-6)
    assert masked_metrics["train/is_logprob_abs_diff"] == pytest.approx(abs(r0), abs=1e-6)

    # Active token with extreme log-ratio must stay finite instead of overflowing to inf.
    _, _, active_metrics = apply_is_correction(
        cfg,
        loss_masks=[torch.tensor([1.0, 1.0], dtype=torch.float32)],
        old_log_probs=[torch.tensor([-0.5, -0.5], dtype=torch.float32)],
        rollout_log_probs=[torch.tensor([-0.6, -100.5], dtype=torch.float32)],
    )
    assert math.isfinite(active_metrics["train/is_approx_k3_kl"])


def test_compute_opsm_mask_clipfrac_excludes_zero_loss_mask_tokens() -> None:
    """train/opsm_clipfrac must count only active (loss_mask == 1) dropped tokens."""
    cfg = OmegaConf.create({"algorithm": {"opsm": {"delta": 0.05}}})

    # Sequence 0: 4 response tokens, only 1 active (e.g. 3 tool-observation tokens with mask=0),
    # negative sequence advantage and seq_kl = 1.0 > 0.05 -> dropped (clipfrac contribution = 1.0, not 4.0).
    # Sequence 1: positive advantage -> kept (clipfrac contribution = 0.0).
    opsm_masks, metrics = compute_opsm_mask(
        cfg,
        log_probs=[
            torch.tensor([-2.0, -2.0, -2.0, -2.0], dtype=torch.float32),
            torch.tensor([-2.0, -2.0], dtype=torch.float32),
        ],
        old_log_probs=[
            torch.tensor([-1.0, -1.0, -1.0, -1.0], dtype=torch.float32),
            torch.tensor([-1.0, -1.0], dtype=torch.float32),
        ],
        advantages=[
            torch.tensor([-1.0, -1.0, -1.0, -1.0], dtype=torch.float32),
            torch.tensor([1.0, 1.0], dtype=torch.float32),
        ],
        loss_masks=[
            torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
            torch.tensor([1.0, 1.0], dtype=torch.float32),
        ],
    )

    assert torch.equal(opsm_masks[0], torch.zeros(4))
    assert torch.equal(opsm_masks[1], torch.ones(2))
    assert metrics["train/opsm_clipfrac"] == pytest.approx(1.0, abs=1e-6)
