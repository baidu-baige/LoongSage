# 自定义KL算法开发指南

本文说明如何在在线蒸馏（OPD）中新增一个 KL / 散度算法。KL 扩展同样是 **"注册表 + 配置驱动"**：继承对应基类、加一个装饰器，即可通过 `opd.pg_kl_method` / `opd.gkd_kl_method` 引用。内置的 `k1`/`k2`/`k3`、`topk_*`、`full_*` 方法及其适用场景见[在线蒸馏 (On-Policy Distillation)](./on-policy-distillation.md)。

## 1. 注册一个 KL 方法

KL 数学层与后端解耦：TP/CP 通信收敛在 `coda.backends.megatron.kl_ctx`，策略类只写散度公式。新增一个 KL 方法只需继承对应基类并注册：

```python
from coda.algorithms.kl_policy import TopkVocabPolicy
from coda.algorithms.registry import register_kl_policy

@register_kl_policy("my_kl")
class MyKL(TopkVocabPolicy):        # 或 LogProbKLPolicy / FullVocabPolicy
    def compute_kl(self, config, ctx):
        ...                        # 返回 (per_token_kl, extra_metrics)
```

模块放在 [coda/custom/](../../coda/custom/) 下即会被自动发现并完成注册（见[自定义扩展](./custom-extensions.md)），之后即可在配置中引用 `my_kl`。

## 2. 三类基类

按所需的教师数据选择基类：

- `LogProbKLPolicy`：需要教师 log-prob（k1/k2/k3 一类）。
- `TopkVocabPolicy`：需要教师 top-k log-prob 与索引。
- `FullVocabPolicy`：声明 `need_teacher_logits()`，通过 `ctx.teacher_logits()` 取重建后的全词表 logits。

## 3. 配置引用

注册后即可在 `opd.pg_kl_method` / `opd.gkd_kl_method` 中按名字引用：

```yaml
opd:
  enable: true
  gkd_ratio: 1
  gkd_kl_method: my_kl      # ← @register_kl_policy 的注册名
  topk: 64                  # top-k 类方法需要
```

注意 PG 与 GKD 对 KL 的用法不同——PG 只用 KL 的**值**（需有符号无偏），GKD 只用其**梯度**——自定义方法应明确自己适用于哪一侧，详见[在线蒸馏](./on-policy-distillation.md)的 KL 方法一节。
