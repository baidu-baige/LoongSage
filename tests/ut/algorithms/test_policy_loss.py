"""Unit tests for the REINFORCE policy loss."""

import torch
from omegaconf import OmegaConf

from coda.algorithms.policy_loss import compute_policy_loss


def _config():
    return OmegaConf.create({"policy_loss": "reinforce"})


def test_reinforce_loss_is_negative_advantage_times_log_prob():
    log_prob = [torch.tensor([-0.5, -1.0]), torch.tensor([-2.0])]
    advantages = [torch.tensor([1.0, 1.0]), torch.tensor([-0.5])]
    old = [lp.clone() for lp in log_prob]
    masks = [torch.ones(2), torch.ones(1)]

    loss, _ = compute_policy_loss(_config(), old, log_prob, advantages, loss_masks=masks)

    assert torch.allclose(loss[0], torch.tensor([0.5, 1.0]))
    assert torch.allclose(loss[1], torch.tensor([-1.0]))


def test_reinforce_gradient_is_advantage_weighted_score_without_clipping():
    # A ratio far outside any PPO clip range: the gradient must still be -A.
    log_prob = torch.tensor([-0.1, -3.0], requires_grad=True)
    old = [torch.tensor([-3.0, -0.1])]
    advantages = [torch.tensor([2.0, -1.0])]

    loss, _ = compute_policy_loss(_config(), old, [log_prob], advantages, loss_masks=[torch.ones(2)])
    loss[0].sum().backward()

    assert torch.allclose(log_prob.grad, torch.tensor([-2.0, 1.0]))


def test_reinforce_approx_kl_uses_masked_old_minus_new():
    log_prob = [torch.tensor([-1.0, -1.0])]
    old = [torch.tensor([-0.5, -3.0])]
    masks = [torch.tensor([1.0, 0.0])]

    _, metrics = compute_policy_loss(_config(), old, log_prob, [torch.zeros(2)], loss_masks=masks)

    assert metrics == {"train/approx_kl": 0.5}
