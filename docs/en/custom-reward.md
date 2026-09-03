# Custom Reward Function Development Guide

This document explains how to add a reward function in LoongSage to score trajectories produced by rollout. Reward extensions are **registry- and config-driven**: inherit the base class, implement `__call__`, add a registration decorator, and reference the implementation through `data_source.reward.name` without changing framework scheduling code.

An agent executes a trajectory and produces conversation history, while a reward computes a score from that history and the label. Without an agent, the framework calls the reward directly. When an agent is configured, the reward is passed to it as `reward_fn`. Prefer an existing implementation such as `gsm8k`, `r2e_gym`, or `bcp` when it already matches the task.

## 1. Development Steps

The reward base class and built-in implementations live in [base.py](../../coda/reward/base.py) and the [reward functions directory](../../coda/reward/functions/). Follow these steps to add a reward:

1. Inherit from `RewardFunction` and implement `__call__(messages, label, **kwargs) -> Reward`. Read reward-specific options from `self.config`.
2. Return a [Reward](../../coda/reward/reward.py) with at least `final_reward`. Set `is_valid=False` when scoring is impossible, and set `is_correct` explicitly when a positive score does not mean a correct answer.
3. Register the class with `@register_reward("your-name")` and place it in [coda/custom/](../../coda/custom/); LoongSage discovers it automatically, see [Custom Extensions](./custom-extensions.md).

## 2. Minimal Example

The following example implements case-insensitive exact matching:

```python
# coda/custom/exact_match.py
from omegaconf import DictConfig

from coda.reward import register_reward
from coda.reward.base import RewardFunction
from coda.reward.reward import Reward


@register_reward("exact-match")
class ExactMatchReward(RewardFunction):
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self.case_sensitive = bool(self.config.get("case_sensitive", False))

    def __call__(self, messages: list[dict], label, **kwargs) -> Reward:
        expected = label.get("answer") if isinstance(label, dict) else label
        if expected is None:
            return Reward(final_reward=0.0, is_valid=False, is_correct=False)

        predicted = next(
            (
                message["content"]
                for message in reversed(messages)
                if message.get("role") == "assistant" and message.get("content")
            ),
            "",
        )
        predicted = str(predicted).strip()
        expected = str(expected).strip()
        if not self.case_sensitive:
            predicted, expected = predicted.lower(), expected.lower()

        correct = predicted == expected
        return Reward(final_reward=float(correct), is_correct=correct)
```

For more involved implementations, see answer parsing in [gsm8k.py](../../coda/reward/functions/gsm8k.py), process reward in [bcp.py](../../coda/reward/functions/bcp.py), and sandbox-based scoring in [r2e_gym.py](../../coda/reward/functions/r2e_gym.py).

## 3. Config Enablement

Once registered, reference the reward by name in the data source's `reward` block. The full block is passed to the reward constructor:

```yaml
data_source:
  reward:
    name: exact-match       # ← the @register_reward registered name
    case_sensitive: false   # ignore letter case during exact matching
```

With multiple data sources, each `data_sources[i]` may select a different reward. An empty `name` disables reward construction: execution without an agent then produces the default invalid zero reward, while a configured agent does not receive `reward_fn`.
