# 自定义 Reward 函数开发指南

本文说明如何在 LoongSage 中新增一个 reward 函数，为 rollout 产出的 trajectory 打分。reward 扩展是 **“注册表 + 配置驱动”**：继承基类、实现 `__call__`、添加注册装饰器，即可通过 `data_source.reward.name` 引用，无需修改框架调度逻辑。

agent 负责执行轨迹并产出对话历史，reward 负责根据对话历史和 label 计算分数。未配置 agent 时，框架直接调用 reward；配置了 agent 时，reward 会作为 `reward_fn` 传给 agent。若已有实现（如 `gsm8k`、`r2e_gym`、`bcp`）满足需求，应优先直接复用。

## 1. 开发步骤

reward 基类和内置实现分别位于 [base.py](../../coda/reward/base.py) 与 [reward 函数目录](../../coda/reward/functions/)。新增 reward 时，按以下步骤操作：

1. 继承 `RewardFunction`，实现 `__call__(messages, label, context, **kwargs) -> Reward`。单轮模式下 `context` 是 trajectory dump，多轮模式下是 agent 传入的 metadata。reward 专属配置从 `self.config` 读取。
2. 返回 [Reward](../../coda/reward/reward.py)，至少填写 `final_reward`。无法打分时设置 `is_valid=False`；正分不代表答案正确时显式填写 `is_correct`。
3. 使用 `@register_reward("your-name")` 注册，并把实现放到 [coda/custom/](../../coda/custom/) 下，LoongSage 会自动发现，详见[自定义扩展](./custom-extensions.md)。

## 2. 最小示例

以下示例实现一个大小写不敏感的精确匹配 reward：

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

更复杂的实现可参考 [gsm8k.py](../../coda/reward/functions/gsm8k.py) 的答案解析、[bcp.py](../../coda/reward/functions/bcp.py) 的过程奖励，以及 [r2e_gym.py](../../coda/reward/functions/r2e_gym.py) 的 sandbox 判分流程。依赖 sandbox 的 reward 可以定义 `prepare_sandbox(client, sandbox_id, metadata)`，在 agent 运行前初始化可信状态，并返回供后续 `__call__` 使用的 sandbox 状态字典。AgentFlow 不解释该 dict，只在 partial resume 时随 sandbox 保留，终态时丢弃。

## 3. 配置启用

注册后，在数据源的 `reward` 配置块中按注册名引用；其余字段会作为完整配置传给 reward 构造函数：

```yaml
data_source:
  reward:
    name: exact-match       # ← @register_reward 的注册名
    case_sensitive: false   # 精确匹配时忽略大小写
```

多数据源配置下，每个 `data_sources[i]` 可以选择不同的 reward。

## 4. 组级打分（`group_rm`）

有些 reward 需要同时看到一个 prompt 的全部回答，例如由 LLM judge 排序，或在组内做归一化。实现 `async def group_rm(messages, label, contexts, **kwargs) -> list[Reward]`，并设置 `reward.group_rm: true`：

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

- `messages[i]` 和 `contexts[i]`（即 `Trajectory.model_dump()`）对应第 i 条 trajectory，`label` 由整组共享。按相同顺序为每条 trajectory 返回一个 `Reward`。
- `group_rm` 必须是 `async def`。它在 AgentFlow 的 event loop 上被 await，函数内的阻塞调用会卡住所有 rollout；阻塞或 CPU 密集的逻辑请用 `await asyncio.to_thread(...)` 包起来。
- 单轮和多轮模式下，都在整组 trajectory 结束后调用一次；打分完成前，该组不会交给 trainer。如果同时实现了 `__call__`，会先逐条执行 `__call__`，再由 `group_rm` 覆盖结果；逐条的分数可以在 `contexts[i]["reward"]` 中读到。
- 抛异常或返回值不合法时，整组标记为 FAILED，由 `status` filter 丢弃。
- 含 aborted 或 failed trajectory 的组不打分。partial rollout 续跑后，组内全部完成时打分一次；abort 前已经打过分的组不会重复打分。
- 配置了 `agentflow.dump_trajectory_path` 时，每条 trajectory 的 JSON 仍在 `__call__` 之后 dump，其中的 `reward` 是 `__call__` 的结果（只实现 `group_rm` 时为占位的 0）。`group_rm` 的结果另外追加到同目录的 `group_rm.jsonl`，每组一行，包含 `prompt_id`、`status`（`ok`/`failed`）、`error`，以及每条 trajectory 的 `trajectory_id`、`attempt_id`、`reward`、`is_correct`、`is_valid`、`extra_info`；按 `trajectory_id` 与 JSON 文件对应。
- 限制：`group_rm` 执行时 sandbox 已经释放。
