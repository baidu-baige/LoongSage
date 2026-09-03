# 默认配置参数手册

本文档对 [conf/default.yaml](../../conf/default.yaml) 的所有配置项做逐项说明，供用户在编写自己的 experiment yaml（通过 Hydra `defaults: [default]` 继承）时查阅。

- **不要直接修改 `default.yaml`**：它是全局默认值，用户配置应通过 override 覆盖。
- `???` 表示 Hydra 的强制项（missing），必须由用户在自己的 yaml 或命令行 override 中显式赋值。
- 表格中「默认」列以 `default.yaml` 的当前值为准；「约束」列汇总代码里会实际校验或彼此耦合的部分，非风格建议。
- 本文档以顶层字段为章节，每章内附一张参数表，最后一章 [11. 常见组合与约束速查](#11-常见组合与约束速查) 汇总跨字段的依赖关系。

## 1. 顶层运行参数

控制整体运行模式、随机性、checkpoint 与日志的开关。

| 参数 | 默认 | 说明 | 约束 |
| --- | --- | --- | --- |
| `run_mode` | `default` | 运行入口模式。`default` 同时启用 rollout + train；`train-only` 仅训练（消费 `rollout_data_path` 中已有的数据）；`rollout-only` 仅采样落盘。 | 全异步模式（`fully_async.enable=true`）仅支持 `default`。 |
| `seed` | `42` | 全局随机种子，作用于数据 shuffle、采样、初始化等。 | — |
| `colocate` | `true` | rollout 与 trainer 是否共用同一批 GPU（推理/训练分时复用）。 | 全异步必须为 `false`。`true` 时 rollout 总卡数与 teacher 总卡数必须 ≤ trainer 卡数，详见 [11.10](#1110-colocate-的资源关系)。 |
| `checkpoint_path` | `???` | 训练模型的 checkpoint **基目录**：既是训练过程中保存 megatron 分布式 checkpoint 的输出目录，也是续训的读入目录；续训的 step 由其中的 `latest_checkpointed_iteration.txt` 决定。 | 必填。 |
| `hf_model_path` | `???` | HuggingFace 格式的初始权重路径，用于初始化 actor 与推理引擎。 | 必填；tokenizer 也从此路径加载。 |
| `rollout_data_path` | `./rollout_data` | `rollout-only` 落盘目录，或 `train-only` 读入目录。 | 仅 `run_mode != default` 时生效。 |
| `ref_dist_ckpt_path` | `null` | 参考模型权重来自 megatron 分布式 checkpoint（优先使用）。指向**一个具体的 `dist_ckpt` 目录**，如 `/path/to/run/train_step_100/dist_ckpt`，不是基目录。 | 启用 `algorithm.ref_kl.enable=true` 时，须与 `ref_hf_model_path` 至少提供其一。启动时即校验，路径不可用会直接报错，不回退到 `ref_hf_model_path`。 |
| `ref_hf_model_path` | `null` | 参考模型的 HF 权重目录。在 `ref_dist_ckpt_path` 留空时使用 —— 是并列的另一个来源，而非配错时的兜底。 | 同上。 |
| `total_steps` | `3000` | 训练总步数。 | — |
| `log_level` | `INFO` | 全局日志级别。 | 取值 `DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL`。 |

## 2. 全异步（fully_async）

以下开关用于启用「rollout 与训练并发运行」的全异步流水线。详细原理与调优建议见 [全异步模式](fully-async-mode.md)。

| 参数 | 默认 | 说明 | 约束 |
| --- | --- | --- | --- |
| `fully_async.enable` | `false` | 是否启用全异步流水线（rollout 与 trainer 分离 GPU）。 | 需同时 `colocate: false`、`run_mode: default`、`rollout.sampler.num_oversample: 0`。 |
| `fully_async.sliding_window` | `no-window` | prompt group 派发与收集策略。可选 `no-window` / `window-gated` / `windowed-fifo`。 | 语义见 fully-async-mode 文档「滑动窗口策略」一节。 |
| `fully_async.stale_steps` | `0` | 以训练 step 为单位的额外流水线容量：`capacity = int(B × (1 + stale_steps))`，其中 `B = num_prompts_per_step`。 | `>= 0`，可取小数；越大越离策略，建议配合 IS correction / M2PO / OPSM 使用。 |

## 3. 数据源（data_source / data_sources）

`data_source` 定义单个数据源的默认字段，`data_sources` 是实际运行时使用的数据源列表。列表中每个元素都会继承 `data_source` 的默认值，并允许在自身内做局部覆盖。当前全异步只支持一个数据源。

### 3.1 dataset 子节

| 参数 | 默认 | 说明 | 约束 |
| --- | --- | --- | --- |
| `dataset.prompt_data_path` | `???` | prompt 数据集路径，支持 `.jsonl` / `.parquet`；可加行切片后缀 `data.jsonl@[0:1000]`。 | 必填。 |
| `dataset.eval_prompt_data_path` | `null` | 评测集路径，格式同上。 | `null` 表示禁用 eval；同时需 `rollout.eval.interval > 0` 才会真正运行 eval。 |
| `dataset.max_prompt_len` | `null` | 单个 prompt 允许的最大字符数，超长样本会在加载时丢弃。 | `null` 关闭过滤；这里是**字符**长度，不是 token 长度。 |
| `dataset.input_key` | `???` | prompt 文本或 message 列的列名。 | 必填。 |
| `dataset.label_key` | `???` | 参考答案 / label 列的列名。 | 必填；不需要时可显式设为 `null`。 |
| `dataset.metadata_key` | `metadata` | 每行 metadata dict 的列名。列值会原样交给 agent，即 `trajectory["metadata"]`。schema 不固定，可放任意样本级信息（如 `{"difficulty": "hard"}`）。 | 列值须为 dict；缺省时视为空 dict。 |
| `dataset.shuffle` | `true` | 是否对训练集做 shuffle。 | eval 集始终不 shuffle。 |
| `dataset.data_pre_processor` | `null` | 原始样本预处理器名，在构建 messages 前对每条原始数据生效（用 `@register_data_pre_processor` 注册；见 [data_pre_processor.py](../../coda/data_factory/data_pre_processor.py)）。内置：`gsm8k`。 | `null` 表示不做预处理。 |
| `dataset.buffer_replay_strategy` | `null` | Buffer 回放策略名（用 `@register_buffer_replay_strategy` 注册；见 [data_source.py](../../coda/data_factory/data_source.py)）。 | 仅 `RolloutDataSourceWithBuffer` 生效；`null` 等价于 `fifo`。 |

### 3.2 agent / reward / 其他

| 参数 | 默认 | 说明 | 约束 |
| --- | --- | --- | --- |
| `agent.name` | `null` | Agent 实现名。`null` 表示 single-turn（无 agent），multi-turn 场景必填。 | multi-turn agent 需要通过 registry 注册；示例见 [自定义 Agent 开发指南](custom-agent.md)。 |
| `reward.name` | `???` | Reward function 名，必须在 registry 中注册；见 [自定义 Reward 函数开发指南](custom-reward.md)。 | 必填；`reward` 节允许其它自定义字段透传给 reward 构造函数。 |
| `teacher_name` | `""` | 对该数据源使用的 teacher 名（配合 OPD）。 | 空串表示不使用 teacher；需与 `opd.teachers[].name` 匹配。 |
| `max_response_len_per_trajectory` | `32768` | 单次 attempt 中「LLM 回复 + tool response」占用的 token 上限。single-turn 时作为请求 `max_tokens`；multi-turn 时下发给 agent，agent 自己决定每次调用的 `max_tokens`。 | 需要小于等于推理引擎允许的 `max_total_tokens - prompt_len`。 |
| `num_prompts_per_step` | `64` | 每个训练 step 消耗的 prompt group 数量 `B`。 | 需满足 `B × N` 能被 `trainer.mini_batch_size` 整除，`N = num_trajectories_per_prompt`；还需 `B % (dp_size × num_mini_batch) == 0`，详见 [11.9](#119-dp-与-batch-尺寸)。 |
| `num_trajectories_per_prompt` | `8` | 每个 prompt 采样出的 trajectory 数 `N`，即 GRPO 的 group size。 | 组内 advantage 归一化要求同一组内数量一致。 |
| `completion_params` | `{}` | 透传给推理引擎的采样参数字典（如 `top_p`、`top_k`、`temperature` 等）。 | 空字典表示走推理引擎默认值；该字典在请求体中**最后展开**，因此其中的 `temperature` 会覆盖 `trainer.temperature`。 |

`data_sources` 默认展开为 `[${data_source}]`，即整个训练只用一个数据源；如需多数据源，在自己的 experiment yaml 中显式写成 list 即可（各元素会**独立**继承 `data_source` 的默认值）。

## 4. Rollout（推理与采样）

控制 SGLang 推理集群、采样器、过滤器与 eval 的行为。

### 4.1 顶层与 SGLang

| 参数 | 默认 | 说明 | 约束 |
| --- | --- | --- | --- |
| `rollout.partial` | `false` | step 边界是否中止在途请求（保留已生成 token，下一 step 从中断处继续）。 | 长轨迹 / 多轮 agent 场景推荐 `true`；`mask_offpolicy_in_partial_rollout` 依赖此项开启。 |
| `rollout.mask_offpolicy_in_partial_rollout` | `false` | 恢复 partial trajectory 时，把「旧 weight 生成的 response token」的 `loss_mask` 置 0。 | 仅 `rollout.partial: true` 时可开启。 |
| `rollout.use_fault_tolerance` | `false` | 是否为每个 SGLang replica group 启动 `RolloutHealthMonitor` 后台线程，检测异常引擎并在训练前调用 `recover_faulty_engines()` 尝试恢复。 | 面向**推理引擎**级别的健康监测与恢复，与下面的 `retry_limit`（trajectory 级重试）无耦合，可独立开关。 |
| `rollout.retry_limit` | `3` | 单条 trajectory 在 AgentFlow 内的最大 attempt 次数（含首次尝试）：当一次 attempt 抛错并被标记 `FAILED` 时，AgentFlow 会重新生成新的 attempt，直到成功或耗尽 `retry_limit`。 | 该项一直生效，不依赖 `use_fault_tolerance`；attempts 全部失败后 trajectory 才最终判为 FAILED，并可能被 `rollout.filter.status` 过滤掉整组。 |
| `rollout.backend` | `sglang` | 推理后端类型。 | 当前仅支持 `sglang`。 |
| `rollout.num_gpus_per_node` | `8` | 单个 rollout 节点的 GPU 数量。 | 与 `sglang_replicas[*].num_nodes` 相乘得到该 replica 的总 GPU 数。 |
| `rollout.env_vars` | `{}` | 注入到 rollout worker 的环境变量。 | — |
| `rollout.sglang_args` | 见下 | 透传给 SGLang `ServerArgs` 的字典。 | 字段名与 SGLang 版本保持一致；下表仅列默认覆盖过的字段。 |

`rollout.sglang_args` 默认覆盖：

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `mem_fraction_static` | `0.8` | SGLang 为 KV cache 预留的显存占比。 |
| `disable_cuda_graph` | `true` | 关闭 CUDA graph（便于权重热更新与调试）。 |
| `disable_custom_all_reduce` | `true` | 关闭自定义 all-reduce（与训练侧通信兼容性有关）。 |
| `load_format` | `dummy` | 权重加载方式，`dummy` 表示先建结构后热更新；正常离线加载可设 `auto`。 |

### 4.2 sglang_replicas（PD 拆分）

`sglang_replicas` 描述推理集群按角色的切分，每个角色都可以独立占用 GPU。默认只启用 `regular` 一个 replica group（相当于不做 PD 拆分）。

> **注意**：完整的 PD 分离功能暂不支持。当前请保持 `prefill.num_nodes` 与 `decode.num_nodes` 为 `0`，只使用 `regular`。

| 参数 | 默认 | 说明 | 约束 |
| --- | --- | --- | --- |
| `sglang_replicas.regular.num_nodes` | `1` | 「统一 prefill+decode」replica 数量（以 replica 为单位，每个 replica 独占 `num_gpus_per_replica` 张卡）。 | 与 `prefill` / `decode` 二者互斥使用。 |
| `sglang_replicas.regular.num_gpus_per_replica` | `8` | 单个 regular replica 使用的 GPU 数。 | 需能被 `rollout.num_gpus_per_node` 整除或跨节点排布。 |
| `sglang_replicas.prefill.num_nodes` | `0` | Prefill 专用 replica 数量。 | 完整 PD 分离暂不支持，保持 `0`。 |
| `sglang_replicas.prefill.num_gpus_per_replica` | `8` | 单个 prefill replica 的 GPU 数。 |  |
| `sglang_replicas.decode.num_nodes` | `0` | Decode 专用 replica 数量。 |  |
| `sglang_replicas.decode.num_gpus_per_replica` | `8` | 单个 decode replica 的 GPU 数。 |  |
| `sglang_replicas.decode.sglang_args` | `{disable_cuda_graph: false}` | 为 decode replica 覆盖 `rollout.sglang_args`（decode 通常需要开启 CUDA graph）。 | 只影响 decode replica。 |

### 4.3 sampler / filter / eval

| 参数 | 默认 | 说明 | 约束 |
| --- | --- | --- | --- |
| `rollout.sampler.name` | `dynamic` | 采样器实现，目前仅支持 `dynamic`（动态采样，允许 oversample+refill）。 | 全异步时 `num_oversample` 必须为 0，见下。 |
| `rollout.sampler.num_oversample` | `0` | 同步动态采样阶段额外派发的 prompt group 数。 | 全异步必须为 `0`；额外容量交由 `fully_async.stale_steps` 控制。 |
| `rollout.sampler.refill_ratio` | `2` | 从 buffer 补齐 prompt 时的放大系数：`refill_num = need × refill_ratio`。 | `dynamic` 采样器专用。 |
| `rollout.sampler.max_refill_count` | `256` | 单次 refill 的 prompt 上限。 | `dynamic` 采样器专用。 |
| `rollout.sampler.timeout` | `7200` | 单次 rollout 的最长等待秒数，超时会中止当前 rollout 并触发下一次。 | 单位：秒。 |
| `rollout.filter.status` | 启用（空 dict） | 组内出现任何 FAILED trajectory 则整组丢弃。 | 显式设 `false` 可关闭；键名相同即覆盖，新键名会追加。 |
| `rollout.filter.reward` | 关闭 | 组内 reward 全相同（无对比信号）则丢弃。 | 设 `reward: {}` 可启用；`reward: false` 显式关闭。 |
| `rollout.eval.interval` | `-1` | 每隔 N 步跑一次 eval，`<=0` 关闭。 | 同时需要 `data_source.dataset.eval_prompt_data_path` 非空。 |
| `rollout.eval.temperature` | `null` | eval 使用的采样温度。 | `null` 表示复用 `trainer.temperature`。 |

## 5. AgentFlow

Agent 侧的路由、tokenizer 与 sandbox。多轮 agent 通过 `AgentFlow.router` 转发到 SGLang，同时管理 tokenizer 与代码 sandbox。

### 5.1 router

| 参数 | 默认 | 说明 | 约束 |
| --- | --- | --- | --- |
| `agentflow.dump_trajectory_path` | `""` | 非空时把每条 trajectory dump 到该目录，用于事后回放/分析。 | — |
| `agentflow.router.ip` | `""` | Router 监听 IP。为空时运行期自动解析本机路由 IP。 | 通常无需手动配置。 |
| `agentflow.router.port` | `0` | Router 监听端口。`0` 表示运行期自动分配。 | — |
| `agentflow.router.accumulate_reasoning` | `true` | 多轮对话中把 reasoning tokens 一并累积到上下文。 | 关闭时思考内容只在当轮生效。 |
| `agentflow.router.rollout_worker_load_threshold` | `32` | 单 rollout worker 的负载阈值，超过后 router 会做负载均衡调度。 | 该值越小分发越均衡，但调度开销越高。 |
| `agentflow.router.proxy_timeout_seconds` | `1800` | Router → SGLang 单请求超时秒数。 | 需要 ≥ 单次 attempt 的最长生成时间。 |
| `agentflow.router.abort_timeout_seconds` | `600` | step 边界 abort 未完成请求的等待秒数。 | 与 `rollout.partial` 配套。 |
| `agentflow.router.max_connections` | `512` | Router 允许的最大并发连接数。 | 与 `sglang_args.max_running_requests` 配套调优。 |
| `agentflow.router.middleware.parser` | — | 中间件解析器占位；用户可扩展自己的解析器。 | 默认为空。 |

### 5.2 tokenizer

| 参数 | 默认 | 说明 | 约束 |
| --- | --- | --- | --- |
| `agentflow.tokenizer.custom_chat_template_path` | `null` | 自定义 chat template 文件路径（Jinja2）。相对路径按 `conf/` 目录解析（如自行添加的 `chat_template/my_model.jinja`），绝对路径原样使用。 | 想控制思考模式请用 `generation_prompt_kwargs`，不要用字符串替换。 |
| `agentflow.tokenizer.generation_prompt_kwargs` | `{}` | 透传给 tokenizer `apply_chat_template` 的关键字参数。 | 常见用法：开启think `enable_thinking: true`。 |
| `agentflow.tokenizer.manager.mode` | `thread` | Tokenizer 并发模式。 | 目前主要使用 `thread`。 |
| `agentflow.tokenizer.manager.num_workers` | `8` | 并发 tokenize 的 worker 数。 | 与 rollout 并发和 CPU 数相关。 |

### 5.3 sandbox

| 参数 | 默认 | 说明 | 约束 |
| --- | --- | --- | --- |
| `agentflow.sandbox.type` | `k8s` | 沙箱类型。 | 当前主要支持 `k8s`。 |
| `agentflow.sandbox.command_exec_timeout_seconds` | `600` | 沙箱内单次命令执行超时（秒）。 | — |
| `agentflow.sandbox.sandbox_creation_timeout_seconds` | `600` | 沙箱创建超时（秒）。 | — |
| `agentflow.sandbox.working_dir` | `/rl-sandbox` | 沙箱内工作目录。 | — |
| `agentflow.sandbox.kubeconfig` | `k8s/kubeconfig.yaml` | k8s 集群 kubeconfig 路径。 | 相对路径按项目 `conf/` 目录解析，绝对路径原样使用。 |
| `agentflow.sandbox.pod_manifest_path` | `k8s/pod_manifest.yaml` | 沙箱 pod 模板路径。 | 相对路径按项目 `conf/` 目录解析，绝对路径原样使用。 |

## 6. Tracking（追踪与实验记录）

| 参数 | 默认 | 说明 | 约束 |
| --- | --- | --- | --- |
| `tracking.project_name` | `default` | 实验项目名（例如上报到 MLflow 时的 project）。 | — |
| `tracking.experiment_name` | `default` | 具体实验名。 | — |
| `tracking.tracking_backend` | `console` | metrics 上报后端，例如 `console` / `mlflow` / `wandb` 自定义 backend。 | — |
| `tracking.mlflow_tracking_uri` | `""` | MLflow server URI。 | `tracking_backend: mlflow` 时必填。 |

## 7. Megatron 训练后端

`megatron` 节控制 actor 模型的并行策略、优化器与学习率调度。字段命名与 Megatron-LM 上游保持一致。

### 7.1 model（并行与精度）

| 参数 | 默认 | 说明 | 约束 |
| --- | --- | --- | --- |
| `megatron.model.bf16` | `true` | 使用 bf16 训练。 | 与 `fp16` 互斥。 |
| `megatron.model.fp16` | `false` | 使用 fp16 训练。 | 与 `bf16` 互斥。 |
| `megatron.model.fp8` | `null` | FP8 训练配方名，`null` 关闭。 | 需 Hopper 及以上；与 `fp8_recipe` / `fp8_param` 一起使用。 |
| `megatron.model.fp8_recipe` | `null` | FP8 recipe 详细配置。 | `fp8` 启用时才生效。 |
| `megatron.model.fp8_param` | `false` | 是否把参数也存成 FP8。 | 显存收益 vs. 精度取舍。 |
| `megatron.model.tensor_model_parallel_size` | `1` | Tensor Parallel 大小（TP）。 | 训练总卡数需能被 `TP × PP × CP` 整除，启动即校验，详见 [11.9](#119-dp-与-batch-尺寸)。 |
| `megatron.model.pipeline_model_parallel_size` | `1` | Pipeline Parallel 大小（PP）。 | 同上，参与 `TP × PP × CP`。 |
| `megatron.model.virtual_pipeline_model_parallel_size` | `null` | Virtual PP（VPP）大小，用于减小 PP bubble。 | 需 PP > 1。 |
| `megatron.model.context_parallel_size` | `1` | Context Parallel 大小（CP，序列切分）。 | 与 `cp_partition_mode` 配套；参与 `TP × PP × CP`。 |
| `megatron.model.cp_partition_mode` | `zigzag` | CP 切分方式，`zigzag` 等。 | 仅 CP > 1 时生效。 |
| `megatron.model.expert_model_parallel_size` | `1` | MoE 的 Expert Parallel（EP）。 | 非 MoE 模型保持 1。要求训练总卡数能被 `ETP × EP × PP` 整除。 |
| `megatron.model.expert_tensor_parallel_size` | `null` | MoE 专家部分的 TP 大小。 | `null` 表示与 `tensor_model_parallel_size` 相同，该值即上式中的 `ETP`。 |
| `megatron.model.overlap_p2p_comm` | `false` | PP 阶段 P2P 通信是否与计算 overlap。 | PP > 1 时才有收益。 |
| `megatron.model.moe_grouped_gemm` | `true` | MoE 使用 grouped GEMM 加速。 | MoE 模型开启。 |
| `megatron.model.moe_shared_expert_overlap` | `false` | MoE shared expert 计算与 dispatch overlap。 | 已知问题：部分 MoE 模型上会出现 NaN，未验证前保持关闭（RCA 文档 `moe-shared-expert-overlap-nan-rca` 待补充）。 |

注释掉的 `recompute_*` 三项（`granularity` / `method` / `num_layers`）用于激活值重算，需要时在自己的 yaml 中打开即可。

> **说明**：`megatron.model` 透明支持 Megatron `TransformerConfig` 的**所有**字段，上表仅列常用项。大部分与模型结构强绑定的参数（层数、hidden size、num heads、rotary、norm 类型等）已由 megatron-bridge 根据 `hf_model_path` 自动推断填充，用户无需手动声明；只需在 yaml 中显式覆盖并行策略、精度、重算、通信 overlap 等**训练侧**开关即可。


### 7.2 ddp_config / optimizer / scheduler

| 参数 | 默认 | 说明 | 约束 |
| --- | --- | --- | --- |
| `megatron.ddp_config.use_distributed_optimizer` | `true` | 使用 Megatron 分布式优化器（ZeRO-1 类切分）。 | 与 `optimizer_sharding_type` 联动。 |
| `megatron.ddp_config.overlap_param_gather` | `false` | 参数 all-gather 与前向 overlap。 | 有一定显存代价。 |
| `megatron.ddp_config.overlap_grad_reduce` | `false` | grad reduce-scatter 与反向 overlap。 | — |
| `megatron.ddp_config.grad_reduce_in_fp32` | `true` | 梯度 reduce 时升到 fp32，数值更稳。 | 保持默认可获得更好精度。 |
| `megatron.optimizer.lr` | `1.0e-6` | 峰值学习率。 | 常见 RLHF 起点。 |
| `megatron.optimizer.weight_decay` | `0.01` | AdamW weight decay。 | — |
| `megatron.optimizer.optimizer_cpu_offload` | `false` | 是否把优化器状态 offload 到 CPU。 | 大模型显存不够时可开启。 |
| `megatron.optimizer.optimizer_offload_fraction` | `1.0` | offload 的比例，`1.0` = 全部。 | 仅 `optimizer_cpu_offload: true` 时生效。 |
| `megatron.scheduler.lr_warmup_steps` | `0` | warmup 步数。 | — |
| `megatron.scheduler.lr_decay_steps` | `1` | 学习率衰减总步数。 | 与 `lr_decay_style` 组合决定曲线。 |
| `megatron.scheduler.lr_decay_style` | `constant` | 学习率衰减方式，如 `constant` / `linear` / `cosine`。 | — |
| `megatron.scheduler.wd_incr_steps` | `0` | weight decay 递增步数。 | 一般保持 0。 |
| `megatron.scheduler.wd_incr_style` | `constant` | weight decay 变化曲线。 | — |
| `megatron.keep_fp32_weights` | `{}` | 指定「保留 FP32 主权重」的参数子串（模糊匹配），值为 bool，表示对应层输出是否也保持 FP32。 | 示例：`output_layer: true`。 |
| `megatron.optimizer_sharding_type` | `dp_reshardable` | 优化器状态切分方式，可选 `dp_reshardable` / `fully_reshardable`。 | 与 checkpoint 兼容性、DP 变化时的重切分能力有关，详见 [模型加载与保存](model-checkpointing.md)。 |

> **说明**：三个子节都以 `**kwargs` 形式透传到 Megatron 上游对应结构，表中仅列常用项，上游其它字段可直接在 yaml 中同名添加：
> - `megatron.ddp_config` → `megatron.core.distributed.DistributedDataParallelConfig`
> - `megatron.optimizer`  → `megatron.core.optimizer.OptimizerConfig`（`torch.dtype` 字段会自动做字符串→dtype 转换）
> - `megatron.scheduler`  → `megatron.core.optimizer_param_scheduler.OptimizerParamScheduler` 构造参数；其中 `max_lr` / `min_lr` / `init_lr` / `start_wd` / `end_wd` 未显式设置时会自动从 `megatron.optimizer.lr` / `min_lr` / `weight_decay` 兜底。


## 8. OPD（On-Policy Distillation）

OPD（On-Policy Distillation）在 RL 流程中额外挂载若干 teacher 模型，把 teacher 的分布作为软目标，用可配置的 KL 项与 policy gradient 项混合，形成 `L = (1-gkd_ratio) × [A - pg_ratio × KL_token] + gkd_ratio × L_GKD`。详见 [在线蒸馏 (On-Policy Distillation)](on-policy-distillation.md)。

### 8.1 顶层

| 参数 | 默认 | 说明 | 约束 |
| --- | --- | --- | --- |
| `opd.enable` | `false` | 是否启用 OPD。 | 启用后需要提供 `teachers`。 |
| `opd.pg_ratio` | `0.0` | 混合公式中 policy gradient 分支的 KL 系数。 | 与 `gkd_ratio` 至少一个 `> 0`。 |
| `opd.gkd_ratio` | `0.0` | 混合公式中 GKD 分支的权重。 | `<= 1`；与 `pg_ratio` 至少一个 `> 0`；`pg_ratio > 0` 与 `gkd_ratio == 1` 互斥。 |
| `opd.pg_kl_method` | `k1` | PG 分支 KL 估计方法。 | 取值 `k1` / `topk_kl` / `full_kl` / `topk_jsd` / `full_jsd`。 |
| `opd.gkd_kl_method` | `topk_kl` | GKD 分支 KL 估计方法。 | 取值 `k2` / `k3` / `topk_kl` / `full_kl` / `topk_jsd` / `full_jsd`。 |
| `opd.topk` | `256` | `topk_*` 类方法保留的 top-K logits 数量。 | 仅 top-k 方法生效。 |
| `opd.teacher_nodes` | `1` | teacher 集群节点数。 | 所有 teacher 共用该资源池。`teacher_nodes × teacher_gpus_per_node` 必须能被 `opd.model` 的 `TP × PP × CP` 整除，详见 [11.11](#1111-opd-teacher-并行度)。 |
| `opd.teacher_gpus_per_node` | `8` | 单节点 teacher GPU 数。 | 同上；由此得出的 `teacher_dp` 需与 `train_dp` 互为整数倍。 |
| `opd.teachers` | `[]` | teacher 列表，每项包含 `name`、`hf_path` 与可选的 `dist_ckpt_path`。 | 启用 OPD 时必填；`name` 与 data source 的 `teacher_name` 对应。`hf_path` 始终必填 —— 它提供模型结构。`dist_ckpt_path` 让权重从 Megatron dist checkpoint 加载而不读 `hf_path` 的 safetensors；与顶层 `checkpoint_path` 不同，它指向**一个具体的 `dist_ckpt` 目录**。启动时即校验，路径不可用会直接启动失败，不回退到 `hf_path`。 |

### 8.2 opd.model / memory_pool

`opd.model` 字段语义与 `megatron.model` 完全一致（`bf16` / `fp16` / `fp8*` / TP / PP / VPP / CP / EP / ETP / `overlap_p2p_comm` / `moe_grouped_gemm`），控制 teacher 侧的 megatron 前向。

| 参数 | 默认 | 说明 | 约束 |
| --- | --- | --- | --- |
| `opd.memory_pool.backend` | `null` | teacher 显存池后端；`null` 关闭。 | 主要用于 colocate teacher 时的显存复用。 |
| `opd.env_vars` | `{}` | teacher worker 的环境变量。 | — |

## 9. Algorithm

`algorithm` 节聚合了 advantage 估计、policy loss、离策略保护、正则项与 loss 聚合，是训练算法配置的核心入口。详见 [训练算法](training-algorithms.md)。

### 9.1 主要开关

| 参数 | 默认 | 说明 | 约束 |
| --- | --- | --- | --- |
| `algorithm.advantage_estimator` | `grpo` | advantage 估计方法名。 | 通过 registry 注册；当前内置 `grpo`。 |
| `algorithm.advantage_norm_mode` | `group_zscore` | advantage 归一化模式。 | `none` / `group_mean` / `group_zscore` / `batch_mean` / `batch_zscore`；`group_*` 要求同组 trajectory 数一致。 |
| `algorithm.policy_loss` | `grpo` | policy loss 名。 | 内置 `grpo` / `gspo`；GSPO 见 [论文](https://arxiv.org/pdf/2507.18071)。 |
| `algorithm.loss_agg_mode` | `token-mean` | loss 聚合方式。 | `token-mean` / `seq-mean-token-mean`。 |
| `algorithm.entropy_coef` | `0.0` | entropy 正则系数。 | 非零时 loss 中减去 `entropy_coef × entropy`。 |
| `algorithm.clip_ratio_low` | `0.2` | GRPO / GSPO 的下侧 clip。 | 与 `clip_ratio_high` 组合成非对称裁剪（DAPO clip-higher）。 |
| `algorithm.clip_ratio_high` | `0.28` | 上侧 clip。 | 同上。 |
| `algorithm.clip_ratio_c` | `10.0` | 负 advantage 的 dual-clip 上界。 | 仅 `policy_loss: grpo` 生效。 |

### 9.2 is_correction（重要性采样修正）

| 参数 | 默认 | 说明 | 约束 |
| --- | --- | --- | --- |
| `algorithm.is_correction.enable` | `false` | 是否开启 IS correction。 | 与 `trainer.use_rollout_log_probs: true` 互斥。 |
| `algorithm.is_correction.action` | `clip` | 越界处理方式，`clip`（钳位）或 `mask`（分子分母同时剔除）。 | — |
| `algorithm.is_correction.level` | `token` | 权重粒度：`token` / `sequence` / `geometric`。 | — |
| `algorithm.is_correction.lower_bound` | `???` | IS 权重下界。 | 必填；token 级建议 `0.5`，geometric 建议 `0.9999`。 |
| `algorithm.is_correction.upper_bound` | `???` | IS 权重上界。 | 必填；token 级建议 `2.0`，geometric 建议 `1.0001`。 |

### 9.3 opsm / m2po

| 参数 | 默认 | 说明 | 约束 |
| --- | --- | --- | --- |
| `algorithm.opsm.enable` | `false` | 开启 Off-Policy Sequence Masking：当 `advantage < 0` 且序列级 KL > `delta` 时，丢弃该序列的梯度。 | 可与 IS correction / M2PO 组合。 |
| `algorithm.opsm.delta` | `0.1` | OPSM 的序列级 KL 阈值。 | — |
| `algorithm.m2po.enable` | `false` | 开启 M2PO（Second-Moment Trust Policy Optimization）：按 `(log(π_old/π_rollout))²` 屏蔽偏差最大的 token，直到剩余 token 的二阶矩降到 `threshold` 以下。 | 与 `trainer.use_rollout_log_probs: true` 互斥。 |
| `algorithm.m2po.threshold` | `0.04` | M2PO 的二阶矩阈值。 | — |

### 9.4 ref_kl（参考模型 KL）

| 参数 | 默认 | 说明 | 约束 |
| --- | --- | --- | --- |
| `algorithm.ref_kl.enable` | `false` | 是否叠加 `coef × KL(π_θ ‖ π_ref)`。 | 启用时必须提供 `ref_dist_ckpt_path` 或 `ref_hf_model_path`。 |
| `algorithm.ref_kl.coef` | `0.001` | KL 项系数。 | — |
| `algorithm.ref_kl.kl_type` | `k3` | KL 估计器：`k1` / `k2` / `k3`。 | — |
| `algorithm.ref_kl.use_unbiased_kl` | `false` | 逐 token KL 乘以 `exp(log_probs - old_log_probs)`（DeepSeek-V3.2 做法）。 | — |
| `algorithm.ref_kl.update_interval` | `-1` | `<=0` 表示 ref 保持冻结；`>0` 表示每 N 步用当前 actor 覆盖 ref。 | 需要 `enable: true`。 |

## 10. Trainer

`trainer` 节控制训练执行侧的 batch 组织、精度、超时与 checkpoint 保存。

| 参数 | 默认 | 说明 | 约束 |
| --- | --- | --- | --- |
| `trainer.backend` | `megatron` | 训练后端。 | 当前仅支持 `megatron`。 |
| `trainer.num_nodes` | `1` | 训练节点数。 | 与 `num_gpus_per_node` 相乘得到训练 GPU 总数；`dp_size = num_nodes × num_gpus_per_node / (TP × PP × CP)`（**不含 EP**），且总卡数必须整除该乘积，启动即校验。详见 [11.9](#119-dp-与-batch-尺寸)。 |
| `trainer.num_gpus_per_node` | `8` | 单节点训练 GPU 数。 | 同上，参与 `dp_size` 计算。 |
| `trainer.use_rollout_log_probs` | `false` | 使用推理引擎返回的 `rollout_log_probs` 替代训练侧重算的 `old_log_probs`。 | 与 IS correction、M2PO 互斥。 |
| `trainer.use_rollout_routing_replay` | `false` | 使用推理端记录的 MoE routing 结果做 replay。 | 用于对齐推理/训练 MoE 路由。 |
| `trainer.use_fp32_lm_head` | `false` | LM head 是否使用 FP32 计算。 | 数值敏感场景可开启。 |
| `trainer.temperature` | `1.0` | 采样温度。eval 轮次在 `rollout.eval.temperature` 非 `null` 时改用后者。 | 该值会被 `data_source.completion_params.temperature` 覆盖（后者在请求体中最后展开）。 |
| `trainer.mini_batch_size` | `64` | 每次 optimizer step 消费的 trajectory 数 `M`。`num_mini_batch = (B × N) / M`。 | `(B × N) % M == 0`；且每个数据源需满足 `num_prompts_per_step % (dp_size × num_mini_batch) == 0`。两者都启动即校验，满足后 `M % dp_size == 0` 自动成立。详见 [11.9](#119-dp-与-batch-尺寸)。 |
| `trainer.micro_batch_size` | `8` | 每张 GPU 单次前向的样本数。 | 仅当 `use_dynamic_batch_size: false` 时要求 `mini_batch_size % micro_batch_size == 0`；开启动态 batch 后本项失效。 |
| `trainer.max_tokens_per_gpu` | `16440` | 动态 batch 时单 GPU 最大 token 数。 | 仅 `use_dynamic_batch_size: true` 时生效。 |
| `trainer.use_dynamic_batch_size` | `false` | 是否按 token 数动态打包 micro-batch。 | 开启后 `micro_batch_size` 失效。 |
| `trainer.deterministic_mode` | `false` | 打开位精确复现：强制 NCCL / TransformerEngine / cuBLAS 使用确定性 kernel。 | 会明显变慢；仅用于调试和精度对齐。 |
| `trainer.nccl_timeout_minutes` | `null` | NCCL 通信超时（分钟），`null` 使用框架默认。 | — |
| `trainer.gloo_timeout_minutes` | `null` | GLOO 通信超时（分钟），`null` 使用框架默认。 | 大集群跨机 CPU 通信建议显式设长。 |
| `trainer.save_freq` | `-1` | 每 N 个训练 step 保存一次 checkpoint，`<=0` 关闭定期保存。 | — |
| `trainer.async_save` | `true` | 异步保存 checkpoint（保存线程与训练主循环并发）。 | 见 [模型加载与保存](model-checkpointing.md)。 |
| `trainer.save_checkpoint` | `true` | 是否保存 megatron 分布式 checkpoint。 | 关闭后仅在 `save_hf: true` 时能恢复。 |
| `trainer.save_hf` | `false` | 是否额外导出 HuggingFace 格式权重到 `train_step_{step}/hf_model`。 | 便于下游推理直接加载。 |
| `trainer.env_vars` | `{}` | 训练 worker 环境变量。 | — |

## 11. 常见组合与约束速查

下面这些跨字段的依赖在实际配置时最容易踩坑，单独列出以便查阅。

### 11.1 参考模型（ref model）

- 只要 `algorithm.ref_kl.enable=true`，就**必须**提供 `ref_dist_ckpt_path` 或 `ref_hf_model_path` 之一；两者同时提供时优先使用 megatron dist checkpoint (`ref_dist_ckpt_path`)。
- `ref_dist_ckpt_path` 指向一个具体的 `dist_ckpt` 目录（`<run>/train_step_<N>/dist_ckpt`），不是基目录。启动时即校验，配了但不可用时不会回退到 `ref_hf_model_path`。
- `algorithm.ref_kl.update_interval > 0` 表示每 N 步用 actor 权重刷新 ref；等价于「移动 ref」的 KL 惩罚。`<= 0` 保持 ref 冻结。

### 11.2 训练/推理 log-prob 与离策略保护

- `trainer.use_rollout_log_probs: true` 会用推理端 `rollout_log_probs` 替代训练侧重算的 `old_log_probs`；此时 **不能** 同时启用：
  - `algorithm.is_correction.enable: true`
  - `algorithm.m2po.enable: true`
- `algorithm.opsm.*` 与上述三者可自由组合。

### 11.3 全异步模式

- 启用 `fully_async.enable: true` 时必须同时：
  - `colocate: false`
  - `run_mode: default`
  - `rollout.sampler.num_oversample: 0`（额外容量由 `fully_async.stale_steps` 提供）
- 目前 `data_sources` 只允许一个数据源。
- 需满足 `B × N` 能被 `trainer.mini_batch_size` 整除，且强烈建议 `M % N == 0`（否则拿不齐完整 group）。其中 `B = num_prompts_per_step`、`N = num_trajectories_per_prompt`、`M = mini_batch_size`。

### 11.4 Partial rollout

- `rollout.mask_offpolicy_in_partial_rollout: true` 仅在 `rollout.partial: true` 时才有意义（否则没有恢复的 partial trajectory 需要 mask）。
- 长轨迹或多轮 agent 场景推荐 `rollout.partial: true`，可显著缩短 step 边界的长尾等待。

### 11.5 SGLang PD 拆分

- **完整的 PD 分离功能暂不支持**，当前请保持 `regular` 单形态，即 `sglang_replicas.prefill.num_nodes` 与 `decode.num_nodes` 均为 `0`。

### 11.6 Eval

- 只有当 `rollout.eval.interval > 0` **且** `data_source.dataset.eval_prompt_data_path` 非空时，eval 才会真正运行。
- `rollout.eval.temperature: null` 表示 eval 复用 `trainer.temperature`。

### 11.7 精度组合

- `megatron.model.bf16` 与 `fp16` 互斥，二选一。
- `megatron.model.fp8` 系列开启后仍需保留 `bf16` 或 `fp16` 之一作为主精度；`fp8_param: true` 会进一步把权重也存成 FP8。
- `megatron.ddp_config.grad_reduce_in_fp32: true` 是数值稳定性推荐值，一般不要关闭。

### 11.8 Checkpoint 保存

- `trainer.save_freq <= 0` 意味着关闭定期保存；`trainer.save_checkpoint` 和 `trainer.save_hf` 至少保留一个为 `true`，否则完全不落盘（除非只做一次性实验）。
- `trainer.async_save: true` 需要额外内存承载后台保存队列，详见 [模型加载与保存](model-checkpointing.md)。

### 11.9 DP 与 batch 尺寸

这一组是最容易踩的约束，四条按顺序成链：

1. **DP 数的定义**：`dp_size = trainer.num_nodes × trainer.num_gpus_per_node / (TP × PP × CP)`。注意 **EP 不参与**，MoE 的 expert 侧另有一套 DP。
2. **训练总卡数必须整除 `TP × PP × CP`**，否则 `dp_size` 无意义。启动即校验，报错形如
   `trainer GPUs (8) must be divisible by TP*PP*CP (3)`。
3. **`(B × N) % M == 0`**，其中 `B = num_prompts_per_step`、`N = num_trajectories_per_prompt`、`M = trainer.mini_batch_size`；多数据源时 `B × N` 取所有数据源之和。由此得到 `num_mini_batch = (B × N) / M`。
4. **每个数据源的 `B % (dp_size × num_mini_batch) == 0`**。这一条最常被忽略，因为它把 batch 配置和 GPU 并行度耦合在了一起。报错形如
   `data_sources[0] num_prompts_per_step (64) is not divisible by (dp_size * num_mini_batch) = (8 * 2) = 16`。

`M % dp_size == 0` 不需要单独配置：由第 4 条可得 `B_i = dp_size × num_mini_batch × m_i`，代入 `M = (B × N) / num_mini_batch` 即得 `M` 是 `dp_size` 的整数倍，自动成立。

另外，`micro_batch_size` 只在 `use_dynamic_batch_size: false` 时参与约束（要求 `M % micro_batch_size == 0`）；开启动态 batch 后按 token 数打包，本项失效。

### 11.10 colocate 的资源关系

`colocate: true` 时 placement group 只按 **trainer 的卡数**申请，rollout 与 teacher 复用同一批卡，因此：

- rollout 总卡数（`Σ sglang_replicas[*].num_nodes × rollout.num_gpus_per_node`）必须 ≤ trainer 总卡数；
- 启用 OPD 时 teacher 总卡数（`opd.teacher_nodes × opd.teacher_gpus_per_node`）也必须 ≤ trainer 总卡数。

`colocate: false` 时三者的卡数是相加关系，Ray 集群需能同时提供 trainer + rollout + teacher 的全部 GPU。

### 11.11 OPD teacher 并行度

启用 OPD 时，teacher 侧有四条整除关系，全部启动即校验：

- `opd.teacher_nodes × opd.teacher_gpus_per_node > 0`；
- 该乘积必须能被 `opd.model` 的 `TP × PP × CP` 整除，由此得出 `teacher_dp`；
- `teacher_dp` 与 `len(opd.teachers)` 必须**互为整数倍**（谁除谁都行，但不能互不整除）；
- `teacher_dp` 与 `train_dp`（即 [11.9](#119-dp-与-batch-尺寸) 的 `dp_size`）必须**互为整数倍**。

---

## 相关文档

- [训练算法](training-algorithms.md)：算法组件与执行顺序
- [全异步模式](fully-async-mode.md)：全异步流水线原理与调优
- [模型加载与保存](model-checkpointing.md)：checkpoint 与异步保存
- [自定义扩展](custom-extensions.md)：所有扩展点的统一存放目录
- [自定义 Agent 开发指南](custom-agent.md) / [自定义 Reward 函数开发指南](custom-reward.md) / [自定义 Sandbox 开发指南](custom-sandbox.md)：三大扩展点
- [自定义RL算法开发指南](custom-algorithm.md) / [自定义KL算法开发指南](custom-kl.md) / [自定义滑动窗口策略开发指南](custom-sliding-window.md)：算法侧扩展点
- [在线蒸馏 (On-Policy Distillation)](on-policy-distillation.md)：OPD 详细用法
- [训推一致性](train-inference-consistency.md)：训练-推理一致性对齐








