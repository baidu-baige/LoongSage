"""Unit tests for coda/backends/megatron/loss.py.

Covers:
  * _aggregate_loss — both aggregation modes, edge cases, gradient flow.
"""

from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf

pytest.importorskip("megatron", reason="Megatron-Core is not installed")

import coda.backends.megatron.loss as _loss_mod
from coda.backends.megatron.loss import (
    _aggregate_loss,
)

_mpu_mock = MagicMock()


@pytest.fixture(autouse=True)
def mock_mpu(monkeypatch):
    """Swap loss.mpu for a controllable mock, restoring it afterwards.

    Rebinding at import time leaked the mock into every later-collected module
    and made results collection-order dependent.
    """
    _mpu_mock.reset_mock()
    _mpu_mock.get_context_parallel_world_size.return_value = 1
    monkeypatch.setattr(_loss_mod, "mpu", _mpu_mock)
    yield


# ═══════════════════════════════════════════════════════════════════════════
# _aggregate_loss
# ═══════════════════════════════════════════════════════════════════════════


class TestAggregateLossTokenMean:
    """Tests for loss_agg_mode='token-mean'."""

    def test_basic(self):
        """Sum of element-wise (mask * value)."""
        values = [torch.tensor([1.0, 2.0, 3.0])]
        masks = [torch.tensor([1.0, 1.0, 0.0])]
        result = _aggregate_loss(values, masks, "token-mean")
        # 1*1 + 2*1 + 3*0 = 3.0
        assert result.item() == pytest.approx(3.0)

    def test_all_masked_out(self):
        """When all masks are zero, result should be 0."""
        values = [torch.tensor([5.0, 10.0])]
        masks = [torch.zeros(2)]
        result = _aggregate_loss(values, masks, "token-mean")
        assert result.item() == pytest.approx(0.0)

    def test_multiple_trajectories(self):
        """Values from multiple trajectories are concatenated then summed."""
        values = [torch.tensor([1.0, 2.0]), torch.tensor([3.0])]
        masks = [torch.tensor([1.0, 1.0]), torch.tensor([1.0])]
        result = _aggregate_loss(values, masks, "token-mean")
        assert result.item() == pytest.approx(6.0)

    def test_gradient_flows(self):
        """Gradient should flow through to the input values."""
        v = torch.tensor([2.0, 3.0], requires_grad=True)
        masks = [torch.tensor([1.0, 0.5])]
        result = _aggregate_loss([v], masks, "token-mean")
        result.backward()
        # d(result)/d(v) = masks
        assert torch.allclose(v.grad, masks[0])


class TestAggregateLossSeqMeanTokenMean:
    """Tests for loss_agg_mode='seq-mean-token-mean'."""

    def test_single_trajectory(self):
        """Per-trajectory mean then sum (single trajectory = just the mean)."""
        values = [torch.tensor([2.0, 4.0])]
        masks = [torch.tensor([1.0, 1.0])]
        result = _aggregate_loss(values, masks, "seq-mean-token-mean")
        # (2*1 + 4*1) / 2 = 3.0
        assert result.item() == pytest.approx(3.0)

    def test_multiple_trajectories_summed(self):
        """Per-trajectory token means are summed, not averaged."""
        values = [torch.tensor([2.0, 4.0]), torch.tensor([10.0])]
        masks = [torch.tensor([1.0, 1.0]), torch.tensor([1.0])]
        result = _aggregate_loss(values, masks, "seq-mean-token-mean")
        # trajectory0: (2+4)/2 = 3.0; trajectory1: 10/1 = 10.0; total = 13.0
        assert result.item() == pytest.approx(13.0)

    def test_partial_mask(self):
        """Only masked-in tokens contribute to the per-trajectory mean."""
        values = [torch.tensor([6.0, 100.0, 4.0])]
        masks = [torch.tensor([1.0, 0.0, 1.0])]
        result = _aggregate_loss(values, masks, "seq-mean-token-mean")
        # (6*1 + 100*0 + 4*1) / 2 = 5.0
        assert result.item() == pytest.approx(5.0)

    def test_all_masked_out_clamps_denominator(self):
        """When all masks are zero, clamp_min(0, 1)=1 prevents division by zero."""
        values = [torch.tensor([5.0, 10.0])]
        masks = [torch.zeros(2)]
        result = _aggregate_loss(values, masks, "seq-mean-token-mean")
        # (0+0) / max(0,1) = 0.0
        assert result.item() == pytest.approx(0.0)

    def test_gradient_flows(self):
        v = torch.tensor([4.0, 6.0], requires_grad=True)
        masks = [torch.tensor([1.0, 1.0])]
        result = _aggregate_loss([v], masks, "seq-mean-token-mean")
        result.backward()
        # d/dv_i = mask_i / mask_sum = 1/2
        assert torch.allclose(v.grad, torch.tensor([0.5, 0.5]))


class TestAggregateLossInvalidMode:
    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown loss_agg_mode"):
            _aggregate_loss([torch.ones(2)], [torch.ones(2)], "bad-mode")


class TestLossFunction:
    def test_rl_path_applies_opsm_after_log_probs_and_preserves_entropy(self, monkeypatch):
        _mpu_mock.get_context_parallel_world_size.return_value = 1

        log_probs = [torch.tensor([0.2, 0.4])]
        entropy = [torch.tensor([0.5, 0.5])]
        policy_loss = [torch.tensor([2.0, 3.0], requires_grad=True)]
        opsm_masks = [torch.tensor([1.0, 0.0])]

        monkeypatch.setattr(_loss_mod, "prepare_packed_seq_params", lambda target_list, **kwargs: (torch.cat(target_list), None))
        monkeypatch.setattr(_loss_mod, "compute_entropy", lambda *args, **kwargs: entropy)
        monkeypatch.setattr(_loss_mod, "compute_log_probs", lambda *args, **kwargs: log_probs)
        monkeypatch.setattr(
            _loss_mod,
            "compute_opsm_mask",
            lambda config, log_probs, old_log_probs, advantages, loss_masks: (
                opsm_masks,
                {"train/opsm_clipfrac": 1.0},
            ),
        )

        def fake_compute_policy_loss(**kwargs):
            assert kwargs["log_prob"] is log_probs
            assert kwargs["raw_loss_masks"] is batch["raw_loss_masks"]
            return policy_loss, {"train/approx_kl": 0.25}

        monkeypatch.setattr(_loss_mod, "compute_policy_loss", fake_compute_policy_loss)

        config = OmegaConf.create(
            {
                "trainer": {
                    "temperature": 1.0,
                    "use_rollout_log_probs": False,
                },
                "algorithm": {
                    "loss_agg_mode": "token-mean",
                    "entropy_coef": 0.1,
                    "opsm": {
                        "enable": True,
                        "delta": 0.1,
                    },
                    "ref_kl": {
                        "enable": False,
                    },
                },
                "opd": {
                    "enable": False,
                },
                # loss_function reads config.megatron.model.cp_partition_mode.
                "megatron": {
                    "model": {
                        "cp_partition_mode": "zigzag",
                    },
                },
            }
        )
        batch = {
            "tokens": [torch.tensor([10, 11, 12])],
            "total_lengths": [3],
            "response_lengths": [2],
            "loss_masks": [torch.ones(2)],
            "raw_loss_masks": [torch.ones(2)],
            "old_log_probs": [torch.zeros(2)],
            "advantages": [torch.ones(2)],
        }

        loss, denominator, metrics = _loss_mod.loss_function(
            config,
            batch,
            packed_seq_params=None,
            gkd_policy=None,
            output_tensor=torch.zeros(1),
        )

        assert loss.item() == pytest.approx(1.9)
        assert denominator.item() == 2
        assert metrics["train/pg_loss"] == pytest.approx(2.0)
        assert metrics["train/entropy"] == pytest.approx(1.0)
        assert metrics["train/loss"] == pytest.approx(1.9)
        assert metrics["train/approx_kl"] == pytest.approx(0.25)
        assert metrics["train/opsm_clipfrac"] == pytest.approx(1.0)
