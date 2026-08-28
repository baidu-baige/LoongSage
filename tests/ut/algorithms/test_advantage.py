import pytest
import torch

from coda.algorithms.advantage import _group_normalize, _parse_advantage_norm


# ---------------------------------------------------------------------------
# _parse_advantage_norm
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "norm, expected",
    [
        ("group_mean", ("group", False)),
        ("group_zscore", ("group", True)),
        ("batch_mean", ("batch", False)),
        ("batch_zscore", ("batch", True)),
    ],
)
def test_parse_advantage_norm(norm, expected):
    assert _parse_advantage_norm(norm) == expected


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
