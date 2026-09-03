# 自定义RL算法开发指南

本文说明如何在 LoongSage 中新增自定义的 advantage 估计与 policy loss。算法扩展同样是 **"注册表 + 配置驱动"**：写一个函数、加一个装饰器，即可通过 `algorithm.advantage_estimator` 或 `algorithm.policy_loss` 引用，无需修改任何调度代码。内置算法、离策略保护与整条链路的执行顺序见[训练算法](./training-algorithms.md)。

## 1. 注册机制

advantage 与 policy loss 通过 [registry.py](../../coda/algorithms/registry.py) 中的装饰器注册，配置项按名字查找实现。推荐流程：

1. 在 [coda/custom/](../../coda/custom/) 下新建模块，用 `@register_advantage("name")` 或 `@register_policy_loss("name")` 注册函数；
2. config文件修改：通过 `algorithm.advantage_estimator` 或 `algorithm.policy_loss` 指向注册名。如需自定义超参，可直接写在 `algorithm` 节内，并通过 `config` 读取。

`coda/custom/` 下的模块会被自动 import 并执行 `@register_*`，无需手动登记，详见[自定义扩展](./custom-extensions.md)。放在 `coda/algorithms/` 包内同样会被 [algorithms/\_\_init\_\_.py](../../coda/algorithms/__init__.py) 的 `pkgutil.walk_packages` 发现，内置算法就在那里。

## 2. 自定义 advantage

函数签名为 `(config, rollout_data) -> list[Tensor]`。`rollout_data` 中可用的字段包括 `rewards`（每条 trajectory 的标量 reward）、`prompt_id`（组内归一化用）、`response_lengths` 等；返回值为逐 trajectory 的 advantage 张量列表，每个张量形状为 `(response_len_i,)`。

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

## 3. 自定义类 GRPO loss

函数签名为 `(config, old_log_prob, log_prob, advantages, loss_masks, **kwargs) -> (per_token_loss, metrics)`。四个输入均为逐 trajectory 的张量列表（形状 `(response_len_i,)`），其中 `log_prob` 带梯度；`kwargs` 中额外提供 `raw_loss_masks`（IS/M2PO 修改前的原始 mask）。实现时须遵守两条契约：

- 返回**逐 token** 的 loss 张量列表，不要在函数内聚合——OPSM 掩码、IS 权重与聚合均由框架在其后统一施加；
- `metrics` 中的监控值应为 loss-mask 加权**求和**，框架会在跨 DP all-reduce 后统一除以全局 token 数得到均值。

词表并行归约、CP 聚合与 prompt 裁剪均已在调用前完成，实现中无需感知 TP / CP，也无需自行通信；通信按 packed micro-batch 批量进行，序列级算法（如 GSPO）不产生额外通信。

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
