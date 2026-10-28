import pytest
import torch

from omegaconf import OmegaConf

import coda.algorithms.advantage as adv_mod
from coda.algorithms.advantage import (
    _group_normalize,
    compute_advantages_and_returns,
    gae_advantage,
    masked_whiten,
)


# ---------------------------------------------------------------------------
# _group_normalize — group_mean (divide_std=False)
# ---------------------------------------------------------------------------

def test_group_mean_basic():
    """Subtract group mean; no std division."""
    rewards = torch.tensor([1.0, 3.0, 2.0, 4.0])
    prompt_ids = ["p0", "p0", "p1", "p1"]
    result = _group_normalize(rewards, prompt_ids, divide_std=False)
    # p0: mean=2, centered=[-1, 1]; p1: mean=3, centered=[-1, 1]
    assert torch.allclose(torch.tensor(result), torch.tensor([-1.0, 1.0, -1.0, 1.0]))


def test_group_mean_single_trajectory():
    """group_size=1 with mean: advantage should be 0 (reward - itself)."""
    rewards = torch.tensor([1.0, 2.0, 3.0])
    prompt_ids = ["p0", "p1", "p2"]
    result = _group_normalize(rewards, prompt_ids, divide_std=False)
    assert torch.allclose(torch.tensor(result), torch.tensor([0.0, 0.0, 0.0]))


# ---------------------------------------------------------------------------
# _group_normalize — group_zscore (divide_std=True)
# ---------------------------------------------------------------------------

def test_group_zscore_basic():
    """Standard z-score normalisation with group_size=2."""
    rewards = torch.tensor([1.0, 3.0, 2.0, 4.0])
    prompt_ids = ["p0", "p0", "p1", "p1"]
    result = _group_normalize(rewards, prompt_ids, divide_std=True)
    # std (unbiased) of [1,3] = sqrt(2), so z = ±1/sqrt(2) ≈ 0.7071
    expected = torch.tensor([-0.7071, 0.7071, -0.7071, 0.7071])
    assert torch.allclose(torch.tensor(result), expected, atol=1e-4)


def test_group_zscore_single_trajectory_no_nan():
    """group_size=1 with zscore: must not produce NaN.

    Regression test for the bug where torch.std(unbiased=True) with n=1
    returns NaN, propagating into advantages and crashing training.
    The fix skips std normalization when group_size < 2, falling back to
    group_mean semantics (advantage = 0).
    """
    rewards = torch.tensor([1.0, 2.0, 3.0])
    prompt_ids = ["p0", "p1", "p2"]
    result = _group_normalize(rewards, prompt_ids, divide_std=True)

    result_tensor = torch.tensor(result)
    assert not torch.any(torch.isnan(result_tensor)), "advantage contains NaN"
    # With a single sample, mean == reward, so centered == 0
    assert torch.allclose(result_tensor, torch.tensor([0.0, 0.0, 0.0]))


def test_group_zscore_single_trajectory_mixed_groups():
    """Mix of group_size=1 and group_size>1 should not contaminate the latter."""
    # p0 has 1 sample, p1 has 2 — but _group_normalize asserts equal sizes,
    # so test with all groups of size 1 instead.
    rewards = torch.tensor([5.0, 10.0])
    prompt_ids = ["p0", "p1"]
    result = _group_normalize(rewards, prompt_ids, divide_std=True)
    result_tensor = torch.tensor(result)
    assert not torch.any(torch.isnan(result_tensor))
    assert torch.allclose(result_tensor, torch.tensor([0.0, 0.0]))


def test_group_zscore_identical_rewards():
    """When all rewards in a group are identical, std=0, centered=0."""
    rewards = torch.tensor([2.0, 2.0, 2.0, 2.0])
    prompt_ids = ["p0", "p0", "p1", "p1"]
    result = _group_normalize(rewards, prompt_ids, divide_std=True)
    result_tensor = torch.tensor(result)
    assert not torch.any(torch.isnan(result_tensor))
    # centered = 0 regardless of std because (x - mean) = 0
    assert torch.allclose(result_tensor, torch.tensor([0.0, 0.0, 0.0, 0.0]))


def test_group_zscore_unequal_group_sizes_raises():
    """Unequal group sizes should hit the assert."""
    rewards = torch.tensor([1.0, 2.0, 3.0])
    prompt_ids = ["p0", "p0", "p1"]  # p0 has 2, p1 has 1
    with pytest.raises(AssertionError):
        _group_normalize(rewards, prompt_ids, divide_std=True)


# ---------------------------------------------------------------------------
# masked_whiten — step-wide whitening over DP (all-reduce mocked: one rank)
# ---------------------------------------------------------------------------

@pytest.fixture
def single_rank(monkeypatch):
    """No process group: all_reduce is the identity, as on a 1-rank DP group."""
    # The code under test imports Megatron's parallel_state lazily.
    mpu = pytest.importorskip(
        "megatron.core.parallel_state", reason="Megatron-Core is not installed"
    )

    monkeypatch.setattr(adv_mod.dist, "all_reduce", lambda tensor, **kwargs: None)
    monkeypatch.setattr(mpu, "get_data_parallel_group", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")


def test_masked_whiten_uses_masked_stats_only(single_rank):
    advs = [torch.tensor([1.0, 2.0, 100.0]), torch.tensor([3.0])]
    masks = [torch.tensor([1.0, 1.0, 0.0]), torch.tensor([1.0])]
    out = masked_whiten(advs, masks)

    valid = torch.tensor([1.0, 2.0, 3.0])
    expected = (valid - valid.mean()) / valid.std(unbiased=True)
    assert torch.allclose(torch.cat([out[0][:2], out[1]]), expected, atol=1e-5)
    assert [t.shape for t in out] == [(3,), (1,)]
    # The masked-out token is transformed too (it never reaches the loss).
    assert out[0][2] == pytest.approx((100.0 - 2.0) / 1.0, rel=1e-5)


def test_masked_whiten_all_masked_raises(single_rank):
    with pytest.raises(ValueError, match="no unmasked token"):
        masked_whiten([torch.tensor([1.0, 2.0])], [torch.zeros(2)])


def _algo_config(**overrides):
    cfg = {
        "advantage_estimator": "grpo",
        "advantage_norm_mode": "none",
        "use_masked_whiten": False,
        "ppo": {"gamma": 1.0, "lam": 1.0},
    }
    cfg.update(overrides)
    return OmegaConf.create(cfg)


def _grpo_rollout():
    return {
        "rewards": [1.0, 0.0, 2.0],
        "prompt_id": ["p0", "p1", "p2"],
        "dp_local_traj_idx": [0, 1, 2],
        "response_lengths": [2, 1, 3],
        "raw_loss_masks": [torch.ones(2), torch.ones(1), torch.tensor([1.0, 1.0, 0.0])],
    }


def test_grpo_returns_equal_advantages(single_rank):
    advs, rets, metrics = compute_advantages_and_returns(_algo_config(), _grpo_rollout())
    assert rets is advs
    assert [a.tolist() for a in advs] == [[1.0, 1.0], [0.0], [2.0, 2.0, 2.0]]
    assert metrics == {"train/adv_zero_ratio": pytest.approx(1 / 3)}  # no critic metrics without old_values


def test_whiten_runs_after_estimator_on_raw_masks(single_rank):
    advs, rets, _ = compute_advantages_and_returns(_algo_config(use_masked_whiten=True), _grpo_rollout())
    valid = torch.tensor([1.0, 1.0, 0.0, 2.0, 2.0])  # last token masked out of the stats
    mean, std = valid.mean(), valid.std(unbiased=True)
    assert torch.allclose(advs[0], (torch.tensor([1.0, 1.0]) - mean) / std, atol=1e-5)
    # returns keep the un-whitened values.
    assert [r.tolist() for r in rets] == [[1.0, 1.0], [0.0], [2.0, 2.0, 2.0]]


def test_unknown_norm_mode_raises():
    with pytest.raises(AssertionError, match="advantage_norm_mode"):
        adv_mod.grpo_advantage(_algo_config(advantage_norm_mode="batch_zscore"), _grpo_rollout())


# ---------------------------------------------------------------------------
# gae estimator — runs on the critic over its old_values
# ---------------------------------------------------------------------------

def _gae_rollout(traj_idx=(0, 1)):
    return {
        "dp_local_traj_idx": list(traj_idx),
        "response_lengths": [2, 3],
        "token_rewards": [torch.tensor([0.0, 1.0]), torch.tensor([0.0, 0.0, -1.0])],
        "old_values": [torch.tensor([0.5, 0.5]), torch.tensor([0.1, 0.2, 0.3])],
        "raw_loss_masks": [torch.ones(2), torch.ones(3)],
    }


def test_gae_estimator_returns_advantages_and_returns(single_rank):
    advs, rets, _ = compute_advantages_and_returns(_algo_config(advantage_estimator="gae"), _gae_rollout())
    # gamma = lam = 1: A_t = sum of future rewards - V(s_t); R_t = sum of future rewards.
    assert torch.allclose(advs[0], torch.tensor([0.5, 0.5]))
    assert torch.allclose(rets[0], torch.tensor([1.0, 1.0]))
    assert torch.allclose(advs[1], torch.tensor([-1.1, -1.2, -1.3]))
    assert torch.allclose(rets[1], torch.tensor([-1.0, -1.0, -1.0]))


def test_gae_metrics_include_masked_value_and_returns_means(single_rank):
    rollout = _gae_rollout()
    rollout["raw_loss_masks"] = [torch.tensor([1.0, 0.0]), torch.ones(3)]
    _, _, metrics = compute_advantages_and_returns(_algo_config(advantage_estimator="gae"), rollout)
    # Masked tokens: values [0.5, 0.1, 0.2, 0.3]; the masked-out token (and its reward)
    # is not a GAE step, so the masked returns are [0.0, -1.0, -1.0, -1.0].
    assert metrics["critic/value_mean"] == pytest.approx(1.1 / 4)
    assert metrics["critic/returns_mean"] == pytest.approx(-3.0 / 4)
    assert metrics["train/adv_zero_ratio"] == 0.0


def test_gae_estimator_rejects_multi_segment_trajectories():
    with pytest.raises(ValueError, match="one Segment per trajectory"):
        gae_advantage(_algo_config(advantage_estimator="gae"), _gae_rollout(traj_idx=(0, 0)))
