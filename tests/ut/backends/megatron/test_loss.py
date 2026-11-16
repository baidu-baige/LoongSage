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


# ═══════════════════════════════════════════════════════════════════════════
# PPO critic: response_values / value_loss_function
# ═══════════════════════════════════════════════════════════════════════════

import coda.backends.megatron.cp_utils as _cp_mod


def _value_config(value_clip=0.2, loss_agg_mode="token-mean"):
    return OmegaConf.create({
        "algorithm": {"loss_agg_mode": loss_agg_mode, "ppo": {"value_clip": value_clip}},
        "megatron": {"model": {"cp_partition_mode": "zigzag"}},
    })


@pytest.fixture
def cp1(monkeypatch):
    cp_mpu = MagicMock()
    cp_mpu.get_context_parallel_world_size.return_value = 1
    monkeypatch.setattr(_cp_mod, "mpu", cp_mpu)


class TestResponseValues:
    def test_values_align_with_log_prob_positions(self, cp1):
        """Value at position t-1 belongs to response token t (same slice as log-probs)."""
        # Two packed rows: (prompt 2, response 2) and (prompt 1, response 2).
        out = torch.arange(7, dtype=torch.bfloat16).view(1, 7, 1)
        batch = {"total_lengths": [4, 3], "response_lengths": [2, 2]}
        values = _loss_mod.response_values(_value_config(), batch, out)
        assert [v.tolist() for v in values] == [[1.0, 2.0], [4.0, 5.0]]
        assert all(v.dtype == torch.float32 for v in values)


class TestValueLoss:
    def _batch(self, old, returns, masks=None):
        n = len(old)
        return {
            "total_lengths": [n + 1],
            "response_lengths": [n],
            "loss_masks": [masks if masks is not None else torch.ones(n)],
            "old_values": [old],
            "returns": [returns],
        }

    def _run(self, monkeypatch, vpred, batch, config=None, cp_size=1):
        _mpu_mock.get_context_parallel_world_size.return_value = cp_size
        monkeypatch.setattr(_loss_mod, "response_values", lambda config, batch, out: [vpred])
        return _loss_mod.value_loss_function(
            config or _value_config(), batch, None, None, torch.zeros(1)
        )

    def test_unclipped_is_half_mse(self, monkeypatch):
        vpred = torch.tensor([0.1, 0.5], requires_grad=True)
        batch = self._batch(old=torch.tensor([0.0, 0.4]), returns=torch.tensor([1.0, 0.5]))
        loss, denom, metrics = self._run(monkeypatch, vpred, batch)
        assert loss.item() == pytest.approx(0.5 * 0.81)
        assert denom.item() == 2
        assert metrics["train/value_clipfrac"] == 0.0
        assert metrics["train/value_loss"] == pytest.approx(0.5 * 0.81)
        assert metrics["num_tokens"] == 2 and metrics["num_sequences"] == 1
        loss.backward()
        assert torch.allclose(vpred.grad, torch.tensor([0.1 - 1.0, 0.0]))

    def test_clip_takes_the_pessimistic_branch(self, monkeypatch):
        # vpred moved 0.5 from old towards the return; clipped prediction stays
        # at old + 0.2, whose error is larger, so the clipped loss is used.
        vpred = torch.tensor([0.5], requires_grad=True)
        batch = self._batch(old=torch.tensor([0.0]), returns=torch.tensor([1.0]))
        loss, _, metrics = self._run(monkeypatch, vpred, batch)
        assert loss.item() == pytest.approx(0.5 * 0.8 ** 2)
        assert metrics["train/value_clipfrac"] == 1.0
        loss.backward()
        assert vpred.grad.item() == 0.0  # clipped branch has no gradient

    def test_masked_tokens_are_excluded(self, monkeypatch):
        vpred = torch.tensor([0.0, 5.0])
        batch = self._batch(old=torch.zeros(2), returns=torch.zeros(2), masks=torch.tensor([1.0, 0.0]))
        loss, denom, _ = self._run(monkeypatch, vpred, batch, config=_value_config(value_clip=10.0))
        assert loss.item() == 0.0 and denom.item() == 1

    def test_cp_size_premultiplies_loss(self, monkeypatch):
        vpred = torch.tensor([0.0])
        batch = self._batch(old=torch.zeros(1), returns=torch.ones(1))
        loss, _, metrics = self._run(monkeypatch, vpred, batch, cp_size=2)
        assert metrics["train/value_loss"] == pytest.approx(0.5)
        assert loss.item() == pytest.approx(1.0)
