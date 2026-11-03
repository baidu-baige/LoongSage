# Custom Reward Function Development Guide

This document explains how to add a reward function in LoongSage to score trajectories produced by rollout. Reward extensions are **registry- and config-driven**: inherit the base class, implement `__call__`, add a registration decorator, and reference the implementation through `data_source.reward.name` without changing framework scheduling code.

An agent executes a trajectory and produces conversation history, while a reward computes a score from that history and the label. Without an agent, the framework calls the reward directly. When an agent is configured, the reward is passed to it as `reward_fn`. Prefer an existing implementation such as `gsm8k`, `r2e_gym`, or `bcp` when it already matches the task.

## 1. Development Steps

The reward base class and built-in implementations live in [base.py](../../coda/reward/base.py) and the [reward functions directory](../../coda/reward/functions/). Follow these steps to add a reward:

1. Inherit from `RewardFunction` and implement `__call__(messages, label, context, **kwargs) -> Reward`. `context` is the trajectory dump in single-turn mode and agent-supplied metadata in multi-turn mode. Read reward-specific options from `self.config`.
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

    def __call__(self, messages: list[dict], label, context: dict, **kwargs) -> Reward:
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

For more involved implementations, see answer parsing in [gsm8k.py](../../coda/reward/functions/gsm8k.py), process reward in [bcp.py](../../coda/reward/functions/bcp.py), and sandbox-based scoring in [r2e_gym.py](../../coda/reward/functions/r2e_gym.py). A sandbox-backed reward may define `prepare_sandbox(client, sandbox_id, metadata)` to initialize trusted state before the agent runs and return a sandbox state for its later `__call__`. AgentFlow treats that dict as opaque, retains it with the sandbox on partial resume, and discards it at terminal state.

## 3. Config Enablement

Once registered, reference the reward by name in the data source's `reward` block. The full block is passed to the reward constructor:

```yaml
data_source:
  reward:
    name: exact-match       # ← the @register_reward registered name
    case_sensitive: false   # ignore letter case during exact matching
```

With multiple data sources, each `data_sources[i]` may select a different reward.

## 4. Group Reward (`group_rm`)

Some rewards need every response of a prompt at once, for example an LLM judge that ranks them or a score normalized within the group. Implement `async def group_rm(messages, label, contexts, **kwargs) -> list[Reward]` and set `reward.group_rm: true`:

```python
@register_reward("length-rank")
class LengthRankReward(RewardFunction):
    async def group_rm(self, messages: list[list[dict]], label, contexts: list[dict], **kwargs) -> list[Reward]:
        lengths = [len(m[-1].get("content") or "") for m in messages]
        order = sorted(range(len(lengths)), key=lengths.__getitem__)
        rank = {idx: pos for pos, idx in enumerate(order)}
        return [Reward(final_reward=rank[i] / max(len(lengths) - 1, 1)) for i in range(len(lengths))]
```

```yaml
data_source:
  reward:
    name: length-rank
    group_rm: true
```

- `messages[i]` and `contexts[i]` (`Trajectory.model_dump()`) belong to the i-th trajectory; `label` is shared by the group. Return one `Reward` per trajectory, in the same order.
- `group_rm` must be `async def`. It is awaited on the AgentFlow event loop, so blocking calls there stall every rollout; wrap blocking or CPU-heavy work in `await asyncio.to_thread(...)`.
- It runs once all trajectories of the group have finished, in both single-turn and multi-turn mode. Until then the group is not handed to the trainer. If `__call__` is also implemented, it runs per trajectory first and `group_rm` overrides its result; per-trajectory rewards are visible in `contexts[i]["reward"]`.
- An exception or a malformed return value marks the whole group FAILED, so the `status` filter drops it.
- Groups containing an aborted or failed trajectory are not scored. After partial rollout resumes them, the completed group is scored once; a group already scored before the abort is not scored again.
- With `agentflow.dump_trajectory_path` set, each trajectory's JSON is still dumped after `__call__`, so its `reward` is the `__call__` result (the placeholder 0 for group_rm-only rewards). The `group_rm` result is appended to `group_rm.jsonl` in the same directory, one line per group with `prompt_id`, `status` (`ok`/`failed`), `error`, and each trajectory's `trajectory_id`, `attempt_id`, `reward`, `is_correct`, `is_valid` and `extra_info`; join it with the JSON files on `trajectory_id`.
- Limitations: sandboxes are already released when `group_rm` runs.
