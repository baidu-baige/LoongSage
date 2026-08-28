# 全异步模式

全异步模式将 rollout（采样）与训练部署在相互独立的机器资源上。rollout 在后台持续生成轨迹，训练流程从缓冲区取得一个 mini-batch 后即可立即进行训练。相较于默认的“整批 rollout 完成后再整批训练”，该模式能够重叠采样与训练，降低两类 GPU 的空闲时间；对于原本 colocate 模式下采样与训练耗时相近的任务，收益尤其明显。

需要注意的是，“全异步”并不意味着训练、权重更新和 checkpoint 保存之间完全没有同步边界。每个训练 step 结束时，系统仍会暂停 rollout，清理或等待未完成的请求，依据 `trainer.save_freq` 决定是否保存 checkpoint，将新权重同步至推理引擎，然后恢复后台生产。`stale_steps` 用于扩大当前未消费的 trajectory group 数量上限。设置此值越大，轨迹离策略的可能程度越大。由于 rollout 与训练使用的权重可能不同，建议结合 IS correction、M2PO 或 OPSM 等离策略保护机制使用。

建议以 [conf/qwen3_30b_a3b/dapo_h20_1node.yaml](../../conf/qwen3_30b_a3b/dapo_h20_1node.yaml) 作为配置起点。运行前请替换其中的 `hf_model_path`、`prompt_data_path`、`checkpoint_path`、tracking 地址以及 GPU 拓扑，并把 `colocate` 改为 `false`（该示例默认是 colocate 模式）：

```bash
python -m coda.controller.trainer --config-name qwen3_30b_a3b/dapo_h20_1node \
  colocate=false fully_async.enable=true
```

## 1.关键参数

下表汇总全异步开关、直接参与异步调度的参数，以及会影响 step 边界清理行为的相关参数。下文统一使用以下符号：`B` 表示 `data_sources[0].num_prompts_per_step`，`N` 表示 `data_sources[0].num_trajectories_per_prompt`，`M` 表示 `trainer.mini_batch_size`。

| 参数 | 默认值 | 说明 | 约束或建议 |
| --- | ---: | --- | --- |
| `fully_async.enable` | `false` | 是否启用全异步训练流程。启用后会创建后台 rollout producer、collector 和 `PipelineBuffer`。 | 必须同时设置 `colocate: false`。 |
| `fully_async.sliding_window` | `no-window` | 控制 prompt group 的派发与收集策略。可选 `no-window`、`window-gated`、`windowed-fifo`。 | 具体选择方式见第 3 节。 |
| `fully_async.stale_steps` | `0` | 以训练 step 的倍数表示额外的流水线容量；三种内置策略都会使用该值。 | 必须 `>= 0`，可以是小数；最终容量通过 `int(...)` 向下取整。 |
| `colocate` | `true` | rollout 与 trainer 是否复用同一组 GPU。 | 全异步必须设置为 `false`，因此集群需要同时容纳 trainer GPU 与 rollout GPU。 |
| `run_mode` | `default` | 运行入口模式。 | 全异步仅支持 `default`，不支持 `train-only` 或 `rollout-only`。 |
| `data_sources` | `[${data_source}]` | 实际参与运行的数据源列表。 | 当前全异步仅支持一个数据源。 |
| `data_source.num_prompts_per_step` | `64` | 每个训练 step 计划消费的 prompt group 数量 `B`，也是滑动窗口的基准批量。每个 group 包含 `N` 条 trajectory。 | `B × N` 必须能被 `M` 整除，否则会报错。 |
| `trainer.mini_batch_size` | `64` | 每次从 `PipelineBuffer` 取出 `M // N` 个完整 group，然后发起一次训练。 | 建议满足 `M % N == 0` 且 `(B × N) % M == 0`。 |
| `trainer.num_nodes`、`trainer.num_gpus_per_node` | `1`、`8` | 定义常驻的训练 GPU 池。 | 训练 GPU 总数为两者的乘积，且不与 rollout 复用。 |
| `rollout.num_gpus_per_node`、`rollout.sglang_replicas.*` | 见默认配置 | 定义常驻的 SGLang GPU 池、replica 数量以及每个 replica 的 GPU 数。 | rollout 总 GPU 数按各 replica 类型的 `num_nodes × rollout.num_gpus_per_node` 求和。 |
| `rollout.sampler.num_oversample` | `0` | 同步动态采样使用的额外派发量。 | 全异步必须为 `0`；额外的在途容量统一由 `stale_steps` 控制。 |
| `rollout.partial` | `false` | 是否在 step 边界中止正在生成的请求。设置为 `true` 时取消请求，并将完整 group 放回 datasource buffer；设置为 `false` 时等待所有在途请求完成后再回收。 | 对于长轨迹或多轮 agent，通常建议设置为 `true`，以减少长尾轨迹造成的流水线气泡。 |
| `rollout.mask_offpolicy_in_partial_rollout` | `false` | 恢复 partial trajectory 时，将旧版本已生成 response token 的 `loss_mask` 置为 0。 | 只能在 `rollout.partial: true` 时开启。 |
| `rollout.sglang_args.max_running_requests`、`agentflow.router.max_connections` | 后者为 `512` | 分别限制推理服务和 Router 的并发能力。 | 属于性能调优项，应结合 `B × N` 的最大 trajectory 规模以及每条 trajectory 的多轮请求数设置。 |
| `algorithm.m2po.*` | 默认关闭 | 根据 `old_log_probs - rollout_log_probs` 的二阶矩，过滤偏差最大的 token。 | 不能与 `trainer.use_rollout_log_probs: true` 同时使用。 |
| `algorithm.opsm.*` | 默认关闭 | 当 advantage 为负且 sequence KL 超过阈值时，屏蔽该序列的梯度。 | 可与其他离策略保护机制组合使用；上述示例配置默认只开启了 IS correction，M2PO 与 OPSM 需自行按需开启。 |

`fully_async.stale_steps` 的容量含义如下：

```text
capacity = int(B * (1 + stale_steps))
```

例如，当 `B=64`、`stale_steps=1.0` 时，容量为 128 个 prompt group，即最多覆盖约 2 个训练 step 的 prompt 数量。每个 group 包含的 trajectory 数量仍由 `N` 决定。增大该参数通常可以提高流水线填充度，但也会增加策略陈旧的风险。

## 2.原理

全异步模式由三个并发实体组成：

1. **rollout producer**：运行于独立线程及其事件循环中，从 datasource 读取 prompt group，并按照滑动窗口策略将其派发给 AgentFlow ，由AgentFlow对接推理引擎。
2. **collector**：运行于另一个后台线程中。每条 terminal trajectory 完成（成功、失败或被取消）后，先进入 `TrajQueue`；collector 等待同一 prompt 的 `N` 条 trajectory 凑齐，经策略检查后执行过滤和统计，再将有效的完整 group 写入 `PipelineBuffer`。
3. **trainer consumer**：运行于主线程中。每次从 `PipelineBuffer` 取出 `M // N` 个 group，整理为训练数据并执行一次 optimizer update；一个外层 step 共执行 `(B × N) // M` 次训练调用。

两级 queue 将“trajectory 完成”和“训练消费”解耦：

- `TrajQueue`：AgentFlow → collector。按 `prompt_id` 聚合单条 trajectory，只有凑齐一个 group 后才能出队。
- `PipelineBuffer`：collector → trainer。保存已完成且通过过滤的 `TrajectoryGroup`，并依据 `prompt_id` 中的 epoch/prompt 序号按优先级消费。

![全异步模式架构图](../_static/image/fully-async-architecture.svg)

一个训练 step 的同步关系如下：

1. producer 与 collector 持续运行；trainer 根据需要等待并消费已完成的 group，随后执行若干 mini-batch 更新。
2. trainer 完成 `(B × N) // M` 次取数和训练后调用 `pause()`。在推荐的整除配置下，此时恰好消费 `B × N` 条 trajectory。producer 停止派发新请求，并根据 `rollout.partial` 取消或等待未完成的请求；collector 已写入 `PipelineBuffer` 的完整 group 不会被清空，可供下一 step 使用。
3. 清理完成后，系统汇总本 step 的 rollout 指标，依据 `trainer.save_freq` 保存 checkpoint（保存时会额外记录 `PipelineBuffer` 中尚未消费的 prompt，以便恢复后重新生成），并同步最新模型权重。SGLang 会在更新前刷新旧权重对应的 prefix/KV cache。
4. `resume()` 将 rollout step 加一；同步完成的新权重也会以递增的 `weight_version` 暴露给 SGLang，producer 随后继续生产。SGLang 返回的版本会记录在每个生成 token 以及 trajectory 的首尾版本字段中，用于监控和离策略修正。

训练侧在同一 step 内可能执行多次 optimizer update。第一次计算 `old_log_probs` 时，系统会保存该 step 的 `old_actor` 快照；后续 mini-batch 临时切回该快照计算行为策略概率，再切回持续更新后的 actor，从而避免同一 step 内 old policy 随 mini-batch 漂移。

## 3.滑动窗口策略

三种策略均以 prompt group 为计数单位。记：

- `R`：已派发、尚未被 collector 收集的 group 数；
- `Q`：`PipelineBuffer` 中已完成、尚未被 trainer 消费的 group 数；
- `C = int(B × (1 + stale_steps))`。

| 策略 | 派发约束 | 收集约束 | 主要取舍 |
| --- | --- | --- | --- |
| `no-window` | `R + Q <= C` | 任意已完成 group | 吞吐优先，完成顺序偏差最大 |
| `window-gated` | `R + Q <= C`，且下一派发序号与最老未收集序号的差小于 `C` | 任意已完成 group | 从源头限制慢样本被跨越的距离 |
| `windowed-fifo` | `R + Q <= C` | 只收集当前最老未完成序号起的前 `B` 个序号 | 派发积极，但训练样本顺序更接近 FIFO |

三种内置策略之外，还可以注册自己的策略，见[自定义滑动窗口策略开发指南](./custom-sliding-window.md)。

### 3.1.no-window

`no-window` 仅控制总容量：

```text
dispatch_count = max(0, C - R - Q)
```

任何率先完成的完整 group 都可以进入 `PipelineBuffer`。因此，慢请求不会阻塞后续 prompt，通常能够获得最高的 GPU 利用率；但训练数据会更偏向容易完成或长度较短的样本，对原始数据顺序的保持也最弱。

适用场景：优先追求吞吐量、任务长度较为均匀、对数据顺序不敏感，或已启用较强离策略修正的任务。在该策略下，`stale_steps` 直接表示额外的全局在途/缓冲容量。

### 3.2.window-gated

`window-gated` 为每个派发的 group 分配单调递增的序号，并同时施加以下两项约束：

```text
next_seq - oldest_uncollected_seq < C
R + Q <= C
```

如果窗口中最早的 prompt 迟迟未完成，即使后续 prompt 已完成并被消费，也无法继续无限派发新的 prompt。只有最早未完成序号前移后，窗口才能继续滑动；收集端仍允许窗口内任意已完成的 group 进入 `PipelineBuffer`。

适用场景：需要限制请求被后续数据跨越的最大距离，降低完成时间差异造成的采样偏置，同时允许窗口内乱序完成。其代价是单个极慢的 group 可能降低 rollout 利用率。在该策略下，`stale_steps` 会直接扩大“最早未完成序号到下一个待派发序号”之间的窗口。

### 3.3.windowed-fifo

`windowed-fifo` 的派发容量与 `no-window` 相同，即 `R + Q <= C`；两者的区别在于收集端。假设当前最早未完成序号为 `min_seq`，只有满足下式的完整 group 才能从 `TrajQueue` 进入 `PipelineBuffer`：

```text
seq - min_seq < B
```

因此，该策略并不是严格的逐条 FIFO，而是**以一个训练 step 的 prompt 数 `B` 作为收集窗口**：窗口内仍允许乱序完成；窗口外的 group 即使率先完成，也会保留在 `TrajQueue` 中，直到前部窗口向前滑动。

适用场景：希望训练顺序更接近数据集原始顺序，或减少短样本优先进入训练所引入的偏差，同时不希望像 `window-gated` 一样直接限制后续请求的派发。`stale_steps` 仅扩大总派发容量 `C`；收集窗口始终为 `B`，不会随 `stale_steps` 改变。需要注意的是，窗口外已完成的 group 会暂存在 `TrajQueue` 中，仍然计入 `R` 和总容量，同时会占用 queue 内存，并可能造成队首阻塞。

## 4.统计指标

全异步模式会在每个 step 的 `pause()` 完成后，上报以下新增或需要重点关注的指标：

| 指标 | 含义 | 解读 |
| --- | --- | --- |
| `rollout/pipeline_buf_size` | step 边界清理完成时，`PipelineBuffer` 中尚未被 trainer 消费的完整 prompt group 数量。 | 如果长期为 0 且 `timing/rollout` 较高，说明 trainer 经常等待数据；如果持续较大，则说明 rollout 生产领先，策略陈旧性与内存占用可能上升。 |
| `rollout/filter_drop` | 本 step 内由 collector 经 `rollout.filter` 丢弃的 prompt group 数量（同步 dynamic 采样与全异步统一为该指标）。 | 如果持续偏高，应检查 reward/status 过滤条件以及数据质量。 |
| `rollout/partial_ratio` | 本 step 中由 trainer 消费的 trajectory 里，首个与最后一个生成响应所使用的 `weight_version` 不同的比例。 | 用于衡量单条 trajectory 是否跨越权重更新边界；值为 0 并不等同于完全 on-policy，仅表示首尾版本未发生变化。 |
| `rollout/partial_span_max` | 本 step 已消费 trajectory 中，`end_rollout_weight_version - start_rollout_weight_version` 的最大值。 | 大于 1 表示至少存在一条轨迹跨越了多个权重版本，需要检查 partial 恢复和离策略修正。 |
| `rollout/partial_restored_count` | step 清理时，被放回 datasource buffer 且属于完整 prompt group 的 trajectory 数量；仅在存在待回收数据时上报。 | “完整”表示同一 prompt 已凑齐 `N` 条 trajectory，并不代表每条 trajectory 都已完成生成。该值除以 `N` 即为恢复的 group 数量。 |
| `rollout/partial_dropped_incomplete_count` | step 清理时，由于未凑齐 `N` 条而无法恢复并被丢弃的 trajectory 数量；仅在存在待回收数据时上报。 | 数值偏高说明 step 边界经常打断同一 prompt 的部分 trajectory。 |
| `timing/pause_delay` | trainer 发出 pause 后，等待 producer 取消或等待请求、清空 `TrajQueue` 并重置策略所花费的总时间。 | 当 `rollout.partial: false` 时，该指标还包含等待全部在途请求完成的时间，数值可能较大。 |
| `perf/wait_ratio` | trainer **非训练时间**占「等待 + 训练」时间之和的比例，即 `wait / (wait + timing/train)`。其中 `wait` 通过 `inverse_timer` 在训练区间之外累加（取数、处理 trajectory、teacher、flush、pause 等待等），仅用于派生该比例、不单独上报。 | 越高说明训练越常被 rollout 生产饿等，可考虑提升 rollout 并发或放宽 staleness。 |

此外，在全异步模式下，`timing/rollout`、`timing/process_traj`、`timing/teacher`（如启用）和 `timing/train` 表示一个外层 step 内所有 mini-batch 调用的**累计耗时**；`timing/step`、`timing/save_ckpt` 和 `timing/update_weights` 仍按外层 step 记录。

与 colocate/同步模式相比，以下指标的统计方式或含义有所不同：

- `rollout/partial_ratio` 与 `rollout/partial_span_max`：同步模式基于当前整批 accepted groups 统计；全异步模式则基于该 step 从 `PipelineBuffer` 中实际取出的 trajectory 统计。
- `train/loss`、`train/pg_loss`、`train/entropy`、`train/grad_norm`、`train/approx_kl`、`train/clip_ratio`、`train/dual_clip_ratio`、`train/nan_ratio` 以及 `train/is_*` 等训练指标：全异步模式下，一个 step 会训练多个 mini-batch，因此 worker 会先缓存每次上报的数值，并在 step 结束时按命名规则聚合——`timing/*` 取总和、以 `_max` 结尾取最大、以 `_min` 结尾取最小、其余取算术平均；`perf/train_memory_allocated_max` 和 `perf/train_memory_reserved_max` 因此取最大值。同步/colocate 模式通常直接上报单次训练调用的结果。
- `timing/rollout`：在全异步模式下，该指标主要表示 trainer 等待 `PipelineBuffer` 并取数的累计墙钟时间，而不是后台推理引擎完成整批生成的耗时。因此，应结合 buffer size 和 pause delay 综合分析。
- 全异步模式不会执行 colocate 模式中的 `offload_rollout`、`onload_train`、`offload_train`、`onload_rollout_weights` 和 `onload_rollout_kv` 阶段，因此不会产生相应的 timing 指标。全异步模式在独立 GPU 上常驻模型，并在 step 边界直接更新 rollout 权重。

## 5.测试结果

### 5.1.实验设置

* Machine: 2 × H20（共 16 GPU）
* Model: Qwen3-30B-A3B（bf16，TP=4，PP=2，EP=4）
* Algorithm: GRPO + IS correction + M2PO + OPSM
* Dataset: dapo-math-17K
* Rollout length: `max_response_length = 20K` tokens
* Engine: SGLang + Megatron
* `num_prompts_per_step = 64`、`num_trajectories_per_prompt = 8`、`mini_batch_size = 128`
* Total steps: 100

**资源分配**：

* `colocate` 模式：16 GPU 全部同时承载 trainer 与 rollout。
* `fully_async` 模式：`trainer.num_nodes = 1`（8 GPU）+ `rollout.sglang_replicas.regular.num_nodes = 1`（8 GPU，`num_gpus_per_replica = 4`，共 2 个 replica）。

所有全异步实验均设置 `rollout.partial = true` 与 `rollout.mask_offpolicy_in_partial_rollout = true`。由于 step 0 需要额外完成 SGLang 侧的首轮 prefix cache 预热、KV cache 初始化以及全异步 buffer 首次填充，其耗时明显高于稳态；因此下文同时给出完整 100 step 与去掉 step 0 后的 99 step两种口径的加速比。

### 5.2.全异步 vs colocate

| config | resource | avg step (100 steps) | speedup | avg step (excl. step 0) | speedup (excl. step 0) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| colocate | 16 | 580.79 | 1.00x | 578.63 | 1.00x |
| colocate + partial | 16 | 489.63 | 1.19x | 486.34 | 1.19x |
| fully_async (no-window, stale = 1.0) | 8:8 | 454.50 | 1.28x | 451.74 | 1.28x |
| fully_async (windowed-fifo, stale = 1.0) | 8:8 | 434.76 | 1.34x | 424.30 | 1.36x |
| fully_async (windowed-fifo, stale = 2.0) | 8:8 | 428.76 | 1.36x | 414.58 | 1.40x |

在最佳配置（`windowed-fifo`, `stale_steps = 2.0`）下，端到端 100 step 平均每步耗时由 580.79s 降至 428.76s（1.36x）；如果剔除首个 step 的冷启动开销，稳态加速比可达 1.40x。相较于同样启用 partial rollout 的 colocate baseline，稳态每步平均耗时缩短约 15%。

### 5.3.滑动窗口策略消融

固定 `stale_steps = 1.0`、8:8 资源分配、`rollout.partial = true`，切换 `sliding_window` 并与 colocate 对齐比较：

|         config          | avg step (100) | speedup | avg step (excl. step 0) | speedup (excl. step 0) | pipeline_buf_size | partial_ratio | partial_rollout_restored |
|:-----------------------:|:--------------:|:-------:|:-----------------------:|:----------------------:|:-----------------:|:-------------:|:------------------------:|
|   colocate (baseline)   |     580.79     |  1.00x  |          578.63         |          1.00x         |         -         |     0.000     |             -            |
|      no-window          |     454.50     |  1.28x  |          451.74         |          1.28x         |         27        |     0.834     |            808           |
|     windowed-fifo       |     434.76     |  1.34x  |          424.30         |          1.36x         |         13        |     0.852     |            920           |

* 两种滑动窗口策略相对 colocate 均取得 1.28x–1.36x 的加速；`windowed-fifo` 比 `no-window` 再快约 4%（100 step 口径）到 6%（去掉 step 0 口径）。
* `windowed-fifo` 的 `pipeline_buf_size` 显著更小（13 vs 27）：收集窗口约束把已完成但序号靠前的 group 留在 `TrajQueue` 中，trainer 消费与 rollout 生产更同步；`no-window` 允许快样本无限制入队，缓冲更容易被完成早的短样本填满。
* 两者 `partial_ratio` 与 `max_partial_span` 相近，partial 恢复的整体规模差异不大；`windowed-fifo` 因 buffer 更贴近满载，`partial_rollout_restored` 略高一些。
* 建议默认使用 `windowed-fifo`；`no-window` 适合样本长度较均匀、对采样顺序不敏感或已启用较强离策略修正的任务。

### 5.4.stale_steps 消融

固定 `sliding_window = windowed-fifo`、8:8 资源分配、`rollout.partial = true`，扫描 `stale_steps` 并与 colocate 对齐比较：

|      config       | pipeline capacity `int(B × (1 + stale))` | avg step (100) | speedup | avg step (excl. step 0) | speedup (excl. step 0) | pipeline_buf_size | partial_rollout_restored |
|:-----------------:|:----------------------------------------:|:--------------:|:-------:|:-----------------------:|:----------------------:|:-----------------:|:------------------------:|
| colocate (baseline) |                     -                  |     580.79     |  1.00x  |          578.63         |          1.00x         |         -         |             -            |
|   stale = 1.0     |                    128                   |     434.76     |  1.34x  |          424.30         |          1.36x         |         13        |            920           |
|   stale = 2.0     |                    192                   |     428.76     |  1.36x  |          414.58         |          1.40x         |         18        |           1392           |

* `stale_steps` 由 1.0 提升到 2.0，稳态口径（去掉 step 0）加速比从 1.36x 提升至 1.40x，稳态 avg step 从 424.30s 降至 414.58s，缩减约 2.3%。相对 colocate 的整体收益主要来自"生成与训练重叠"，`stale_steps` 主要提供额外的 buffer 深度以吸收 rollout 抖动。
* 100 step 口径下的加速比（1.34x → 1.36x）小于稳态口径（1.36x → 1.40x）：更深的 buffer 需要更长时间填充，`stale = 2.0` 的 step 0 耗时高达 1832.71s，明显高于 `stale = 1.0` 的 1470.11s，抵消了后续 step 的稳态收益。训练步数越多，端到端加速比会越接近稳态口径。
* `partial_rollout_restored` 从 920 提升至 1392，说明更宽的 staleness 让更多轨迹跨越权重更新恢复。相应的离策略偏差需要通过 partial loss mask（`rollout.mask_offpolicy_in_partial_rollout`）以及 IS correction / M2PO / OPSM 组合控制，不建议在缺少这些保护机制时盲目放大。

### 5.5.小结

* 在长响应（20K）、rollout 存在长尾的任务上，全异步模式相较 colocate 端到端 100 step 训练时间缩短约 21%–26%（1.28x–1.36x）；剔除首个 step 冷启动后，稳态加速比可达 1.28x–1.40x。相对同样启用 partial rollout 的 colocate baseline，稳态每步平均耗时仍能再降 12%–15%。
* `stale_steps` 越大 rollout 与训练的重叠越充分，稳态 step 时间越短；但初次填充 buffer 的开销也越大，需要结合总训练步数评估摊销收益，并配合离策略保护机制使用。`stale_steps` 达到一定数值后，sglang在此并发数下达到吞吐瓶颈，性能提升转化率下降，不宜再加。
* 滑动窗口策略中，`windowed-fifo` 在吞吐与样本顺序之间取得较好折中，是默认推荐；`no-window` 更强调吞吐，`window-gated` 从派发端严格限制窗口距离，视任务特性选择。
