# 训推一致性

训推不一致指同一条 trajectory 在推理引擎（SGLang）与训练引擎（Megatron）下计算出的 token 概率不同，即 `π_rollout ≠ π_train`。偏差来源可分为四类：一是训练侧使用的 token 串本身可能与推理侧实际生成的不同——轨迹若以文本回传再重新编码，tokenizer 往返不可逆会造成 token 边界错位，此时两侧比较的已不是同一条序列；二是两侧数值精度可能不统一，例如推理侧启用量化或低精度 KV cache，而训练侧为 bf16；三是即使精度完全一致，两个引擎的算子实现也不相同——kernel 选择、并行切分与浮点累加顺序的差异都会引入微小的数值扰动；四是这些扰动会被模型结构放大——对 MoE 模型，router 的 top-k 是离散决策，微小的 logits 差即可翻转专家选择、走出完全不同的计算路径，使 MoE 成为不一致的重灾区。此外，异步与 partial rollout 场景还会叠加权重版本差异。偏差累积会使 on-policy 假设失效、重要性比率失真，严重时导致训练崩溃。

框架分三层应对：

1. **源头对齐**：TITO 全链路 token 传递、R3 路由回放、FP32 LM head，直接缩小 `π_train` 与 `π_rollout` 的差距；
2. **算法层修正**：IS correction / M2PO / OPSM，对残余偏差加权或屏蔽；
3. **持续监控**：`train/is_approx_k3_kl` 等指标量化两个引擎间的偏差。

## 1.TITO（Token-In-Token-Out）

最上游的一致性问题不是概率，而是**两侧拿到的 token 串是否相同**。

若以文本采集轨迹，训练侧需重新编码，而 tokenizer 的 encode/decode 并非严格互逆（空白合并、特殊 token 显隐、多字节字符被切在 token 边界等），重编码结果可能与引擎实际生成的序列错位；一旦错位，logprob 与 token 不再对应，训练算的是另一条序列的概率，且多轮场景下逐轮累积。

LoongSage 默认全链路只传 token，无需开关：Router 中间件把智能体的 OpenAI chat 请求改写为 SGLang 原生 `/generate`，以 `input_ids` 下发并强制 `return_logprob=true`（DeepSeek-V4 系列还会额外置 `skip_special_tokens=false`）；响应侧直接从 `output_token_logprobs` 取 token id 与 logprob；多轮续接只增量编码新增消息，已生成部分永不重编码；think 裁剪、长度截断等改写也都在 token 序列上完成，掩码与 logprob 同步切片。

由此训练与推理使用的 token 序列完全一致，token ↔ logprob ↔ 掩码严格对齐，这也是下一节 R3 路由张量能逐 token 回放的前提。代价是 AgentFlow 需持有与推理侧一致的分词器（由 TokenizerManager 统一提供），智能体侧仍只面对标准 OpenAI 接口，无需感知 token。

## 2.Rollout Routing Replay（R3）

R3（[arXiv:2510.11370](https://arxiv.org/abs/2510.11370)）针对 MoE 模型：记录推理引擎生成每个 token 时各 MoE 层实际选中的专家，训练前向直接回放同一路由。这样可以消除专家选择不一致；回放只固定 top-k 专家选择，router 权重仍参与梯度计算。对 MoE 模型的 RL 训练建议默认开启。

开启方式为 `trainer.use_rollout_routing_replay: true`，整条链路随之自动打通：

1. SGLang 以 `enable_return_routed_experts` 启动，每次生成随响应返回逐 token、逐 MoE 层的专家索引；
2. Router 中间件逐轮增量收集并拼接成整条 trajectory 的路由张量——多轮对话、partial rollout 续跑、think 块裁剪等改写 token 序列的操作都会同步维护该张量；
3. 训练侧强制 `megatron.model.moe_enable_routing_replay=true`：在重算 `old_log_probs` 的 forward-only 与训练 forward/backward 中，将路由张量按 CP/TP 切分对齐后写入各 MoE 层的 Megatron `RouterReplay` 实例，替换 router 的 top-k 输出。

注意事项：

- 仅对 MoE 模型有意义，且需要 SGLang 支持返回 routed experts（≥ 0.5.14）；
- 若某 DP shard 中存在缺少路由记录的 trajectory（如从旧数据恢复），该 shard 的路由张量整体不会下发到训练侧（日志以 info 级记录省略的条数），此时开启 R3 的训练步会因拿不到回放数据而失败；因此启用 R3 时需保证参与训练的数据都带有路由记录；
- 与 partial rollout 兼容：跨权重版本续跑的 trajectory 回放的是各段生成时的真实路由，可参考 [bcp_h20_1node.yaml](../../conf/qwen3_30b_a3b/bcp_h20_1node.yaml)。

## 3.FP32 LM head

`trainer.use_fp32_lm_head: true` 会将输出层（LM head）的权重与计算保持在 FP32（等价于 `megatron.keep_fp32_weights` 配置 `output_layer`）。bf16 下 logits 的舍入误差经 softmax 后直接进入 log_probs，可能成为 dense 模型训推概率偏差的重要来源；以 FP32 计算 LM head 可以降低这部分误差，代价是输出层额外的显存与计算。

## 4.算法层修正

源头对齐无法完全消除偏差（精度与算子实现的差异仍在），也不覆盖异步 / partial rollout 引入的权重版本差；残余部分由算法层处理，详见[训练算法](./training-algorithms.md)第 1.3 节：

- **IS correction**：按 `π_old / π_rollout` 对逐 token loss 加权，将残余偏差显式折算进梯度，权重越界时裁剪或屏蔽；
- **M2PO / OPSM**：不加权，而是屏蔽偏差最严重的 token / 序列，防止极端样本主导更新。

另一条路线是 `trainer.use_rollout_log_probs: true`：直接以 `π_rollout` 作为行为策略参与 policy loss，让比率 `π_θ / π_rollout` 隐式吸收训推差，省去 `old_log_probs` 重算；该开关与 IS correction、M2PO 互斥。

## 5.监控指标

在非 pure-GKD、且训练侧计算了 `old_log_probs` 时，框架总会上报以下指标；其中 clip/nan 指标仅在 IS correction 启用时出现：

| 指标 | 含义 | 解读 |
| --- | --- | --- |
| `train/is_approx_k3_kl` | `π_old` 与 `π_rollout` 之间的 k3 KL 估计 | 持续增大说明训推偏差在恶化，应检查 R3 / 精度设置，或加强算法层修正 |
| `train/is_clip_ratio`、`train/is_nan_ratio` | IS 权重越界与 log-ratio 钳位的 token 占比 | 偏高说明 bounds 过紧或偏差过大 |
| `rollout/partial_ratio`、`rollout/partial_span_max` | trajectory 跨越权重版本的比例与最大跨度 | 异步 / partial 场景的离策略程度，见[全异步模式](./fully-async-mode.md) |

实践上，MoE 模型建议 R3 + IS correction 组合作为基线。
