"""Unit tests for the SFT loss / advantage plugins.

Run: python -m pytest tests/ut/algorithms/test_sft.py -v
"""

import pytest
import torch

import coda.algorithms  # noqa: F401  (registers built-in algorithms)
from coda.algorithms.registry import get_advantage, get_policy_loss


def test_sft_loss_is_negative_log_prob():
    log_prob = [torch.tensor([-0.5, -1.0]), torch.tensor([-2.0])]
    old = [torch.zeros(2), torch.zeros(1)]
    adv = [torch.full((2,), 3.0), torch.full((1,), -3.0)]
    masks = [torch.ones(2), torch.ones(1)]

    per_token, metrics = get_policy_loss("sft")(None, old, log_prob, adv, masks)

    assert metrics == {}
    for got, lp in zip(per_token, log_prob):
        assert torch.equal(got, -lp)


def test_sft_loss_ignores_masked_positions_regardless_of_value():
    """The mask, not the loss, decides which tokens count; masked NLL may be any value."""
    log_prob = [torch.tensor([-0.5, -99.0, -1.5])]
    masks = [torch.tensor([1.0, 0.0, 1.0])]

    per_token, _ = get_policy_loss("sft")(None, None, log_prob, None, masks)
    masked = (torch.cat(masks) * torch.cat(per_token)).sum()

    # The -99.0 position is masked out, so it never reaches the aggregated loss.
    assert torch.isclose(masked, torch.tensor(0.5 + 1.5))


def test_sft_loss_keeps_gradient():
    log_prob = [torch.tensor([-0.5, -1.0], requires_grad=True)]
    per_token, _ = get_policy_loss("sft")(None, None, log_prob, None, [torch.ones(2)])
    per_token[0].sum().backward()
    assert torch.equal(log_prob[0].grad, torch.tensor([-1.0, -1.0]))


# ---------------------------------------------------------------------------
# Integration with the Megatron loss aggregation the SFT path actually uses.
# ---------------------------------------------------------------------------

def _aggregate_loss():
    """Import the real reducer; skip when Megatron-Core is unavailable."""
    pytest.importorskip("megatron", reason="Megatron-Core is not installed")
    from coda.backends.megatron.loss import _aggregate_loss as fn
    return fn


def test_sft_loss_token_mean_sums_masked_nll():
    """token-mean returns the summed masked NLL (Megatron divides by num_tokens externally)."""
    aggregate = _aggregate_loss()
    log_prob = [torch.tensor([-0.5, -1.0, -1.5]), torch.tensor([-2.0, -4.0])]
    masks = [torch.tensor([1.0, 0.0, 1.0]), torch.tensor([0.0, 1.0])]

    per_token, _ = get_policy_loss("sft")(None, None, log_prob, None, masks)
    summed = aggregate(per_token, masks, "token-mean")

    assert torch.isclose(summed, torch.tensor(0.5 + 1.5 + 4.0))


def test_sft_loss_seq_mean_averages_within_each_sequence():
    aggregate = _aggregate_loss()
    log_prob = [torch.tensor([-0.5, -1.0, -1.5]), torch.tensor([-2.0, -4.0])]
    masks = [torch.tensor([1.0, 0.0, 1.0]), torch.tensor([0.0, 1.0])]

    per_token, _ = get_policy_loss("sft")(None, None, log_prob, None, masks)
    summed = aggregate(per_token, masks, "seq-mean-token-mean")

    # Per-sequence token mean, summed: (0.5 + 1.5) / 2 + 4.0 / 1.
    assert torch.isclose(summed, torch.tensor((0.5 + 1.5) / 2 + 4.0))


def test_sft_loss_zero_weight_row_contributes_nothing():
    """A fully-masked SFTSampler fallback row (mask == [0]) adds 0 to the batch loss."""
    aggregate = _aggregate_loss()
    log_prob = [torch.tensor([-1.0, -2.0]), torch.tensor([-9.0])]
    masks = [torch.tensor([1.0, 1.0]), torch.tensor([0.0])]

    per_token, _ = get_policy_loss("sft")(None, None, log_prob, None, masks)

    assert torch.isclose(aggregate(per_token, masks, "token-mean"), torch.tensor(3.0))
    assert torch.isclose(aggregate(per_token, masks, "seq-mean-token-mean"), torch.tensor(1.5))


def test_sft_advantage_is_ones_per_segment():
    rollout_data = {
        "response_lengths": [3, 1],
        "loss_masks": [torch.ones(3, dtype=torch.long), torch.zeros(1, dtype=torch.long)],
    }
    advantages = get_advantage("sft")(None, rollout_data)
    assert [a.tolist() for a in advantages] == [[1.0, 1.0, 1.0], [1.0]]
    assert all(a.dtype == torch.float32 for a in advantages)


def test_sft_advantage_empty_batch_returns_empty():
    """An empty batch leaves device as None; the comprehension must still return []."""
    assert get_advantage("sft")(None, {"response_lengths": [], "loss_masks": []}) == []
