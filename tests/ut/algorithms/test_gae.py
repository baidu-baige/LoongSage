"""Unit tests for the GAE recursion of ``gae_advantage`` (CPU only)."""
import pytest
import torch
from omegaconf import OmegaConf

from coda.algorithms.advantage import gae_advantage


def compute_gae(rewards, values, masks, gamma, lam):
    """Run ``gae_advantage`` on single-Segment rows."""
    config = OmegaConf.create({"ppo": {"gamma": gamma, "lam": lam}})
    rollout_data = {
        "dp_local_traj_idx": list(range(len(values))),
        "response_lengths": [v.numel() for v in values],
        "token_rewards": rewards,
        "old_values": values,
        "raw_loss_masks": masks,
    }
    return gae_advantage(config, rollout_data)


def reference_gae(rewards, values, mask, gamma, lam):
    """Step-by-step GAE over the mask == 1 tokens only, written independently.

    Compacts each row to its valid tokens, runs the textbook backward
    recursion, then scatters back (mask == 0 tokens take the advantage of the
    next valid token, i.e. the carried running value).
    """
    valid = [i for i in range(len(mask)) if mask[i] > 0]
    adv_valid = [0.0] * len(valid)
    # Only valid tokens carry rewards here (rewards on masked tokens do not occur).
    gae, next_v = 0.0, 0.0
    for k in reversed(range(len(valid))):
        t = valid[k]
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        adv_valid[k] = gae
        next_v = values[t]
    out = [0.0] * len(mask)
    carry = 0.0
    k = len(valid) - 1
    for t in reversed(range(len(mask))):
        if mask[t] > 0:
            carry = adv_valid[k]
            k -= 1
        out[t] = carry
    return torch.tensor(out)


def _rows(seed, lengths, mask_holes=False):
    g = torch.Generator().manual_seed(seed)
    rewards, values, masks = [], [], []
    for n in lengths:
        r = torch.zeros(n)
        r[-1] = torch.randn(1, generator=g).item()
        rewards.append(r)
        values.append(torch.randn(n, generator=g))
        m = torch.ones(n)
        if mask_holes and n > 3:
            m[1:3] = 0.0  # e.g. tool output in the middle of the response
        masks.append(m)
    return rewards, values, masks


@pytest.mark.parametrize("gamma, lam", [(1.0, 1.0), (1.0, 0.95), (0.99, 0.9), (0.9, 0.0)])
@pytest.mark.parametrize("mask_holes", [False, True])
def test_matches_reference(gamma, lam, mask_holes):
    rewards, values, masks = _rows(0, [1, 4, 7, 12], mask_holes)
    advs, rets = compute_gae(rewards, values, masks, gamma, lam)
    for r, v, m, a, ret in zip(rewards, values, masks, advs, rets):
        expected = reference_gae(r.tolist(), v.tolist(), m.tolist(), gamma, lam)
        valid = m > 0
        assert torch.allclose(a[valid], expected[valid], atol=1e-5)
        assert torch.allclose(ret, a + v, atol=1e-6)
        assert a.shape == v.shape and a.dtype == torch.float32


def test_masked_tokens_are_skipped_not_steps():
    """A mask == 0 token between two steps must not discount or bootstrap."""
    rewards = [torch.tensor([0.0, 0.0, 1.0])]
    values = [torch.tensor([0.2, 9.0, 0.4])]  # the masked value must be ignored
    masks = [torch.tensor([1.0, 0.0, 1.0])]
    advs, _ = compute_gae(rewards, values, masks, gamma=0.5, lam=1.0)
    # t=2: 1 - 0.4 = 0.6 ; t=0: delta = 0 + 0.5*0.4 - 0.2 = 0 ; A = 0 + 0.5*0.6 = 0.3
    assert advs[0][2].item() == pytest.approx(0.6)
    assert advs[0][0].item() == pytest.approx(0.3)


def test_lam_zero_is_one_step_td():
    rewards = [torch.tensor([0.0, 0.0, 1.0])]
    values = [torch.tensor([0.1, 0.2, 0.3])]
    advs, _ = compute_gae(rewards, values, [torch.ones(3)], gamma=1.0, lam=0.0)
    assert torch.allclose(advs[0], torch.tensor([0.2 - 0.1, 0.3 - 0.2, 1.0 - 0.3]))
