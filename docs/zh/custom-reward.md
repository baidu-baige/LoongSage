# 自定义 Reward 函数开发指南

本文说明如何在 LoongSage 中新增一个 reward 函数，为 rollout 产出的 trajectory 打分。reward 扩展是 **“注册表 + 配置驱动”**：继承基类、实现 `__call__`、添加注册装饰器，即可通过 `data_source.reward.name` 引用，无需修改框架调度逻辑。

agent 负责执行轨迹并产出对话历史，reward 负责根据对话历史和 label 计算分数。未配置 agent 时，框架直接调用 reward；配置了 agent 时，reward 会作为 `reward_fn` 传给 agent。若已有实现（如 `gsm8k`、`r2e_gym`、`bcp`）满足需求，应优先直接复用。

## 1. 开发步骤

reward 基类和内置实现分别位于 [base.py](../../coda/reward/base.py) 与 [reward 函数目录](../../coda/reward/functions/)。新增 reward 时，按以下步骤操作：

1. 继承 `RewardFunction`，实现 `__call__(messages, label, **kwargs) -> Reward`。reward 专属配置从 `self.config` 读取。
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

更复杂的实现可参考 [gsm8k.py](../../coda/reward/functions/gsm8k.py) 的答案解析、[bcp.py](../../coda/reward/functions/bcp.py) 的过程奖励，以及 [r2e_gym.py](../../coda/reward/functions/r2e_gym.py) 的 sandbox 判分流程。

## 3. 配置启用

注册后，在数据源的 `reward` 配置块中按注册名引用；其余字段会作为完整配置传给 reward 构造函数：

```yaml
data_source:
  reward:
    name: exact-match       # ← @register_reward 的注册名
    case_sensitive: false   # 精确匹配时忽略大小写
```

多数据源配置下，每个 `data_sources[i]` 可以选择不同的 reward。`name` 为空时不创建 reward：未配置 agent 时框架返回默认无效的零分，配置了 agent 时则不会注入 `reward_fn`。
