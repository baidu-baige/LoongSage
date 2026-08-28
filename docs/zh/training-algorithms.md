# 训练算法

框架目前仅支持**类 GRPO**（critic-free）算法族：advantage 直接由 reward 在组内或批内归一化得到，不训练 value model，因此暂不支持 PPO 等需要 critic 的算法。整条算法链路被拆分为若干独立组件——advantage 估计、policy loss、离策略保护（IS correction、M2PO，即 Second-Moment Trust Policy Optimization、OPSM，即 Off-Policy Sequence Masking）、正则项（entropy / ref KL）和 loss 聚合——所有开关集中在配置的 `algorithm` 节，其中 advantage 与 policy loss 支持注册自定义实现，详见[自定义RL算法开发指南](./custom-algorithm.md)。

一个 mini-batch 内各组件的执行顺序为：（可选）重算 `old_log_probs` →（可选）ref model 前向 →（可选）M2PO 掩码 →（可选）IS 权重计算 → advantage 计算 → actor 前向得到 `log_probs` →（可选）OPSM 掩码 → policy loss →（可选）乘 OPSM / IS 权重 → 聚合，（可选）再叠加 entropy 与 ref KL 项。

## 1.内置算法

### 1.1.Advantage 估计

`algorithm.advantage_estimator` 当前内置 `grpo`（默认）。它把每条 trajectory 的标量 reward 归一化为 advantage，并广播到该 trajectory 的每个 response token。归一化方式由 `algorithm.advantage_norm_mode` 控制：

| 取值 | 含义 |
| --- | --- |
| `group_zscore`（默认） | 同一 prompt 组内 z-score：`(x - mean) / std` |
| `group_mean` | 组内去均值：`x - mean` |
| `batch_zscore` | 跨所有 DP rank 的全批 z-score |
| `batch_mean` | 跨所有 DP rank 的全批去均值 |
| `none` | 直接使用原始 reward |

`batch_*` 模式会在 DP 组内 all-reduce 统计量，因此不同 DP rank 上的样本共享同一组 mean/std；`group_*` 模式要求每个 prompt 组的 trajectory 数量一致（即 `num_trajectories_per_prompt`）。

### 1.2.Policy loss

`algorithm.policy_loss` 内置两种类 GRPO 的 clipped surrogate loss：

| 取值 | 重要性比率 | 说明 |
| --- | --- | --- |
| `grpo`（默认） | token 级 `exp(logπ - logπ_old)` | 非对称裁剪（`clip_ratio_low` / `clip_ratio_high`，默认 0.2 / 0.28，即 DAPO 的 clip-higher）；对负 advantage 额外做 dual-clip（`clip_ratio_c`，默认 10.0） |
| `gspo` | 序列级 `exp(mean(Δlogp))` | 序列级比率、token 级梯度路由，见 [GSPO](https://arxiv.org/pdf/2507.18071)；仅使用 `clip_ratio_low` / `clip_ratio_high` |

行为策略 `logπ_old` 默认由训练引擎在每个 step 开始时重算；设置 `trainer.use_rollout_log_probs: true` 可改用推理引擎返回的 `rollout_log_probs`（此时不能启用 IS correction 和 M2PO，见下节）。

### 1.3.离策略保护

以下机制均针对训练-推理不一致（`π_old` 与 `π_rollout` 的偏差）以及异步 / partial rollout 引入的离策略数据，可组合开启；但 IS correction、M2PO 与 `trainer.use_rollout_log_probs: true` 不能同时使用：

| 机制 | 配置 | 原理 | 约束 |
| --- | --- | --- | --- |
| IS correction | `algorithm.is_correction.enable` | 按 `π_old / π_rollout` 对逐 token loss 加权，权重越界时裁剪或屏蔽 | 与 `trainer.use_rollout_log_probs: true` 互斥 |
| M2PO | `algorithm.m2po.enable` | 在整批范围内屏蔽 `(log(π_old/π_rollout))²` 最大的 token，直至剩余 token 的平均二阶矩降到 `threshold`（默认 0.04）以下 | 同上 |
| OPSM | `algorithm.opsm.enable` | 当序列满足 `advantage < 0` 且序列级 KL 超过 `delta`（默认 0.1）时，丢弃该序列的梯度（分母保持不变） | 无 |

IS correction 的行为由两个维度组合而成：

- `level`：权重粒度，`token`（逐 token 比率）、`sequence`（整条序列概率比）或 `geometric`（token 比率的几何平均）；
- `action`：越界处理，`clip` 将权重钳位到 `[lower_bound, upper_bound]`，`mask` 额外将越界 token（或整条序列）从 loss 的分子与分母中同时剔除。

在非 pure-GKD、且训练侧计算了 `old_log_probs` 时，框架会上报训推不一致指标 `train/is_approx_k3_kl`；`train/is_clip_ratio` 和 `train/is_nan_ratio` 仅在 IS correction 启用时上报。

### 1.4.正则项

- `algorithm.entropy_coef`（默认 0.0）：非零时在 loss 中减去 `entropy_coef × entropy`。
- `algorithm.ref_kl.enable`：叠加参考模型 KL 惩罚 `coef × KL(π_θ ‖ π_ref)`。`kl_type` 支持 `k1|k2|k3`；`use_unbiased_kl: true` 时逐 token KL 会乘以重要性比率（DeepSeek-V3.2 做法）；`update_interval > 0` 时每 N 步用当前 actor 刷新参考模型。启用时需提供 `ref_dist_ckpt_path`（指向一个具体的 `train_step_N/dist_ckpt` 目录）或 `ref_hf_model_path`。

### 1.5.配置示例

```yaml
algorithm:
  advantage_estimator: grpo
  advantage_norm_mode: group_zscore
  policy_loss: grpo
  loss_agg_mode: token-mean
  clip_ratio_low: 0.2
  clip_ratio_high: 0.28
  is_correction:
    enable: true
    action: clip
    level: token
    lower_bound: 0.5
    upper_bound: 2.0
  opsm:
    enable: true
    delta: 0.1
```

## 2.无偏 Loss 聚合

`algorithm.loss_agg_mode` 决定一个 mini-batch 内逐 token loss 的聚合方式：

- `token-mean`（默认）：mini-batch 内全部有效 token 的均值，每个 token 等权；
- `seq-mean-token-mean`：先对每条 trajectory 内取 token 均值，再对 trajectory 取均值，每条序列等权。

框架采用 Megatron 新版的 per-token loss 聚合协议（强制 `calculate_per_token_loss=True`）：loss 函数向 Megatron 返回的是当前 micro-batch 的 **loss 求和与分母**（token 数或序列数），而非 micro-batch 内的均值。反向阶段各 micro-batch 的梯度直接累加，`finalize_model_grads` 将分母跨全部 micro-batch 与 DP rank 求和后，对梯度做一次性缩放：

```text
g = Σ(minibatch 内全部有效 token 的梯度) / Σ(minibatch 内全部有效 token 数)
```

因此 mini-batch 内的聚合是**无偏**的：梯度与 micro-batch 的切分方式（包括 `use_dynamic_batch_size` 的变长切分）、DP rank 间的样本长度分布均无关。相比之下，传统"每个 micro-batch 先取均值、再对 micro-batch 平均"的做法会赋予短 micro-batch 中的 token 更大权重，在变长序列的 RL 训练中引入长度相关偏差。`seq-mean-token-mean` 同理，只是全局分母换为序列数。

监控指标遵循同一约定：loss 函数上报求和值，框架在 DP all-reduce 后统一除以全局分母，因此日志中的 `pg_loss`、`approx_kl` 等均为全局均值。
