# Custom RL Algorithm Development Guide

This document explains how to add a custom advantage estimator or policy loss in LoongSage. The algorithm extension is likewise **"registry + config-driven"**: write a function, add a decorator, and it can be referenced via `algorithm.advantage_estimator` or `algorithm.policy_loss`, with no changes to any scheduling code. For the built-in algorithms, off-policy protection, and the execution order of the whole pipeline, see [Training Algorithms](./training-algorithms.md).

## 1. Registration Mechanism

Advantage and policy loss are registered through the decorators in [registry.py](../../coda/algorithms/registry.py); the configuration items look up implementations by name. The recommended workflow is:

1. Create a new module under [coda/custom/](../../coda/custom/) and register a function with `@register_advantage("name")` or `@register_policy_loss("name")`;
2. Modify the config file: point `algorithm.advantage_estimator` or `algorithm.policy_loss` to the registered name. If custom hyperparameters are needed, they can be written directly inside the `algorithm` section and read through `config`.

Modules under `coda/custom/` are imported automatically so that `@register_*` executes, with no manual registration needed; see [Custom Extensions](./custom-extensions.md). A module placed inside the `coda/algorithms/` package is discovered as well, through the `pkgutil.walk_packages` call in [algorithms/\_\_init\_\_.py](../../coda/algorithms/__init__.py) — that is where the built-in algorithms live.

## 2. Custom Advantage

The function signature is `(config, rollout_data) -> list[Tensor]`. The fields available in `rollout_data` include `rewards` (the scalar reward of each trajectory), `prompt_id` (used for within-group normalization), `response_lengths`, and so on; the return value is a list of per-trajectory advantage tensors, each of shape `(response_len_i,)`.

```python
import torch

from coda.algorithms.registry import register_advantage

@register_advantage("reward_centered")
def reward_centered(config, rollout_data):
    rewards = torch.tensor(rollout_data["rewards"], dtype=torch.float32)
    lengths = rollout_data["response_lengths"]
    adv = rewards - rewards.mean()
    return list(torch.repeat_interleave(adv, torch.tensor(lengths)).split(lengths))
```

## 3. Custom GRPO-like Loss

The function signature is `(config, old_log_prob, log_prob, advantages, loss_masks, **kwargs) -> (per_token_loss, metrics)`. All four inputs are per-trajectory tensor lists (of shape `(response_len_i,)`), where `log_prob` carries gradients; `kwargs` additionally provides `raw_loss_masks` (the original mask before IS/M2PO modification). Two contracts must be respected in the implementation:

- Return a **per-token** list of loss tensors, and do not aggregate inside the function — the OPSM mask, IS weights, and aggregation are all applied uniformly by the framework afterwards;
- The monitored values in `metrics` should be loss-mask weighted **sums**; the framework will uniformly divide by the global token count after the cross-DP all-reduce to obtain the mean.

Vocabulary-parallel reduction, CP gathering, and prompt slicing are all completed before the call, so an implementation never needs to be aware of TP / CP or issue collective communication itself; communication is batched per packed micro-batch, so sequence-level algorithms (such as GSPO) add no extra communication.

```python
import torch

from coda.algorithms.registry import register_policy_loss

@register_policy_loss("vanilla_pg")
def vanilla_pg(config, old_log_prob, log_prob, advantages, loss_masks, **kwargs):
    lengths = [len(t) for t in log_prob]
    ratio = torch.exp(torch.cat(log_prob) - torch.cat(old_log_prob))
    per_token_loss = -(ratio * torch.cat(advantages))
    approx_kl = ((torch.cat(old_log_prob) - torch.cat(log_prob)) * torch.cat(loss_masks)).sum()
    return list(per_token_loss.split(lengths)), {"approx_kl": approx_kl.item()}
```
