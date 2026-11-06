"""Unit tests for ``_collect_loss_metrics`` normalization (CPU, no process group)."""
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

pytest.importorskip("megatron", reason="Megatron-Core is not installed")

import coda.backends.megatron.model as model_mod


@pytest.fixture(autouse=True)
def single_rank(monkeypatch):
    mpu = MagicMock()
    mpu.is_pipeline_last_stage.return_value = True
    monkeypatch.setattr(model_mod, "mpu", mpu)
    monkeypatch.setattr(model_mod, "reduce_dict", lambda d, group=None: d)


def collect(loss_agg_mode, losses):
    config = OmegaConf.create({"algorithm": {"loss_agg_mode": loss_agg_mode}})
    return model_mod._collect_loss_metrics(config, losses, grad_norm=1.5)


@pytest.mark.parametrize("loss_key", ["train/loss", "train/value_loss"])
def test_aggregated_losses_follow_seq_mean(loss_key):
    losses = [
        {"num_tokens": 6, "num_sequences": 2, loss_key: 4.0, "train/value_clipfrac": 3.0},
        {"num_tokens": 2, "num_sequences": 2, loss_key: 2.0, "train/value_clipfrac": 1.0},
    ]
    out = collect("seq-mean-token-mean", losses)
    assert out[loss_key] == pytest.approx(6.0 / 4)           # per sequence
    assert out["train/value_clipfrac"] == pytest.approx(4.0 / 8)  # per token
    assert out["train/grad_norm"] == 1.5


@pytest.mark.parametrize("loss_key", ["train/loss", "train/value_loss"])
def test_token_mean_divides_everything_by_tokens(loss_key):
    out = collect("token-mean", [{"num_tokens": 4, "num_sequences": 2, loss_key: 2.0}])
    assert out[loss_key] == pytest.approx(0.5)
