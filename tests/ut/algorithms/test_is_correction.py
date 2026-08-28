import pytest
import torch
from omegaconf import OmegaConf

from coda.algorithms.is_correction import (
    _compute_correction_weights,
    apply_is_correction,
)
from coda.algorithms.second_moment_trust_policy_optimization import apply_m2po_masking


def test_compute_correction_weights_token_level():
    old_log_probs = [torch.log(torch.tensor([0.25, 0.5], dtype=torch.float32))]
    rollout_log_probs = [torch.log(torch.tensor([0.5, 0.25], dtype=torch.float32))]
    loss_masks = [torch.ones(2, dtype=torch.float32)]

    weights, _ = _compute_correction_weights(old_log_probs, rollout_log_probs, loss_masks, level="token")

    assert len(weights) == 1
    expected = torch.tensor([0.5, 2.0], dtype=torch.float32)
    assert torch.allclose(weights[0], expected)


def test_compute_correction_weights_sequence_level():
    """sequence = exp(Σ Δlogp): two tokens each at ratio 2 give 4.

    The inputs must make Σ Δ != 0, otherwise sequence and geometric both
    collapse to 1.0 and the two implementations become indistinguishable.
    """
    old_log_probs = [torch.log(torch.tensor([0.5, 0.5], dtype=torch.float32))]
    rollout_log_probs = [torch.log(torch.tensor([0.25, 0.25], dtype=torch.float32))]
    loss_masks = [torch.ones(2, dtype=torch.float32)]

    weights, _ = _compute_correction_weights(old_log_probs, rollout_log_probs, loss_masks, level="sequence")

    assert len(weights) == 1
    assert torch.allclose(weights[0], torch.full((2,), 4.0))


def test_compute_correction_weights_geometric_level():
    """geometric = exp(mean Δlogp): the same inputs give 2, not 4."""
    old_log_probs = [torch.log(torch.tensor([0.5, 0.5], dtype=torch.float32))]
    rollout_log_probs = [torch.log(torch.tensor([0.25, 0.25], dtype=torch.float32))]
    loss_masks = [torch.ones(2, dtype=torch.float32)]

    weights, _ = _compute_correction_weights(old_log_probs, rollout_log_probs, loss_masks, level="geometric")

    assert len(weights) == 1
    assert torch.allclose(weights[0], torch.full((2,), 2.0))


def test_compute_correction_weights_sequence_level_ignores_masked_tokens():
    """The masked third token would drag Σ Δ to 0 (weight 1.0) if counted."""
    old_log_probs = [torch.log(torch.tensor([0.5, 0.5, 0.25], dtype=torch.float32))]
    rollout_log_probs = [torch.log(torch.tensor([0.25, 0.25, 1.0], dtype=torch.float32))]
    loss_masks = [torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32)]

    weights, _ = _compute_correction_weights(old_log_probs, rollout_log_probs, loss_masks, level="sequence")

    assert torch.allclose(weights[0], torch.full((3,), 4.0))


def test_compute_correction_weights_geometric_level_ignores_masked_tokens():
    """num_valid must be 2, not 3: exp(2*ln2 / 2) = 2, not exp(2*ln2 / 3)."""
    old_log_probs = [torch.log(torch.tensor([0.5, 0.5, 0.25], dtype=torch.float32))]
    rollout_log_probs = [torch.log(torch.tensor([0.25, 0.25, 1.0], dtype=torch.float32))]
    loss_masks = [torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32)]

    weights, _ = _compute_correction_weights(old_log_probs, rollout_log_probs, loss_masks, level="geometric")

    assert torch.allclose(weights[0], torch.full((3,), 2.0))


def test_compute_correction_weights_invalid_level_raises():
    old_log_probs = [torch.zeros(2)]
    rollout_log_probs = [torch.zeros(2)]
    loss_masks = [torch.ones(2)]

    with pytest.raises(AssertionError, match="Invalid is_correction.level"):
        _compute_correction_weights(old_log_probs, rollout_log_probs, loss_masks, level="invalid")


def test_apply_is_correction_clip_action():
    config = OmegaConf.create(
        {
            "action": "clip",
            "enable": True,
            "level": "token",
            "lower_bound": 0.4,
            "upper_bound": 0.8,
        }
    )
    loss_masks = [torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)]
    old_log_probs = [torch.log(torch.tensor([0.25, 0.5, 1.0], dtype=torch.float32))]
    rollout_log_probs = [torch.zeros(3, dtype=torch.float32)]

    is_weights, corrected_masks, metrics = apply_is_correction(
        config,
        loss_masks,
        old_log_probs,
        rollout_log_probs,
    )

    expected_weights = torch.tensor([0.4, 0.5, 0.8], dtype=torch.float32)
    assert torch.allclose(is_weights[0], expected_weights)
    assert corrected_masks is loss_masks
    assert metrics["train/is_clip_ratio"] == pytest.approx(2.0)


def test_apply_is_correction_mask_action():
    config = OmegaConf.create(
        {
            "action": "mask",
            "enable": True,
            "level": "token",
            "lower_bound": 0.4,
            "upper_bound": 0.8,
        }
    )
    loss_masks = [torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)]
    old_log_probs = [torch.log(torch.tensor([0.25, 0.5, 1.0], dtype=torch.float32))]
    rollout_log_probs = [torch.zeros(3, dtype=torch.float32)]

    is_weights, corrected_masks, metrics = apply_is_correction(
        config,
        loss_masks,
        old_log_probs,
        rollout_log_probs,
    )

    expected_weights = torch.tensor([0.4, 0.5, 0.8], dtype=torch.float32)
    expected_mask = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    assert torch.allclose(is_weights[0], expected_weights)
    assert torch.equal(corrected_masks[0], expected_mask)
    assert metrics["train/is_clip_ratio"] == pytest.approx(2.0)


def test_apply_is_correction_invalid_action_raises():
    config = OmegaConf.create(
        {
            "action": "invalid",
            "enable": True,
            "level": "token",
            "lower_bound": 0.1,
            "upper_bound": 10.0,
        }
    )

    with pytest.raises(AssertionError, match="Invalid is_correction.action"):
        apply_is_correction(
            config,
            loss_masks=[torch.ones(2)],
            old_log_probs=[torch.zeros(2)],
            rollout_log_probs=[torch.zeros(2)],
        )


def test_apply_m2po_masking_keeps_all_when_mean_below_threshold():
    config = OmegaConf.create({"enable": True, "threshold": 10.0})
    loss_masks = [torch.ones(3, dtype=torch.float32)]

    corrected_masks, metrics = apply_m2po_masking(
        config,
        loss_masks,
        old_log_probs=[torch.tensor([3.0, 0.2, 0.1])],
        rollout_log_probs=[torch.zeros(3)],
    )

    assert corrected_masks is loss_masks
    assert torch.equal(corrected_masks[0], loss_masks[0])
    assert metrics["clip_count"] == pytest.approx(0.0)
    assert metrics["m2"] == pytest.approx(9.05, abs=1e-6)


def test_apply_m2po_masking_masks_largest_m2_until_suffix_mean_below_threshold():
    config = OmegaConf.create({"enable": True, "threshold": 0.05})
    loss_masks = [torch.ones(3, dtype=torch.float32)]

    corrected_masks, metrics = apply_m2po_masking(
        config,
        loss_masks,
        old_log_probs=[torch.tensor([3.0, 0.2, 0.1])],
        rollout_log_probs=[torch.zeros(3)],
    )

    expected_mask = torch.tensor([0.0, 1.0, 1.0])
    assert torch.equal(corrected_masks[0], expected_mask)
    assert torch.equal(loss_masks[0], torch.ones(3))
    assert metrics["clip_count"] == pytest.approx(1.0)


def test_apply_m2po_masking_keeps_smallest_m2_when_all_suffixes_exceed_threshold():
    """Match AReaL fallback: mask all but the smallest-M2 valid token."""
    config = OmegaConf.create({"enable": True, "threshold": 0.0})
    loss_masks = [torch.ones(3, dtype=torch.float32)]

    corrected_masks, metrics = apply_m2po_masking(
        config,
        loss_masks,
        old_log_probs=[torch.tensor([3.0, 2.0, 1.0])],
        rollout_log_probs=[torch.zeros(3)],
    )

    assert torch.equal(corrected_masks[0], torch.tensor([0.0, 0.0, 1.0]))
    assert metrics["clip_count"] == pytest.approx(2.0)
    assert metrics["valid_tokens"] == pytest.approx(3.0)


def test_apply_m2po_masking_valid_token_count_ignores_zero_mask_tokens():
    config = OmegaConf.create({"enable": True, "threshold": 10.0})
    loss_masks = [torch.tensor([1.0, 0.0, 1.0])]

    corrected_masks, metrics = apply_m2po_masking(
        config,
        loss_masks,
        old_log_probs=[torch.tensor([3.0, 100.0, 1.0])],
        rollout_log_probs=[torch.zeros(3)],
    )

    assert corrected_masks is loss_masks
    assert metrics["m2"] == pytest.approx(10.0)
    assert metrics["valid_tokens"] == pytest.approx(2.0)


def test_apply_m2po_masking_uses_unclamped_logprob_delta():
    config = OmegaConf.create({"enable": True, "threshold": 400.0})
    loss_masks = [torch.ones(2, dtype=torch.float32)]

    corrected_masks, metrics = apply_m2po_masking(
        config,
        loss_masks,
        old_log_probs=[torch.tensor([30.0, 0.0])],
        rollout_log_probs=[torch.zeros(2)],
    )

    assert torch.equal(corrected_masks[0], torch.tensor([0.0, 1.0]))
    assert metrics["m2"] == pytest.approx(900.0)
    assert metrics["clip_count"] == pytest.approx(1.0)
