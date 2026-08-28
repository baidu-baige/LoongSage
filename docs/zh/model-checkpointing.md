# 模型加载与保存

本文档介绍 LoongSage 在训练中如何加载初始权重、保存 checkpoint、以及导出 HuggingFace 权重。

## 1. 目录结构

LoongSage 以训练模型的 `checkpoint_path` 为根目录组织一次训练的所有产物：

```text
checkpoint_path/
├── latest_checkpointed_iteration.txt   # tracker 文件，记录最近一次成功保存的 step
├── train_step_10/
│   ├── dist_ckpt/                       # 分布式 checkpoint（模型 + 优化器 + 调度器 + RNG）
│   │                                    # 也就是 ref_dist_ckpt_path /
│   │                                    # opd.teachers[].dist_ckpt_path 要填的值
│   ├── hf_model/                        # HuggingFace 格式导出（仅当开启 hf 导出时）
│   └── data_source/                     # 数据源游标 + 未消费 prompt 缓冲快照，用于断点续训
│       └── global_dataset_state_dict_ds{ds_idx}.pt
├── train_step_20/
│   └── ...

rollout_data/                           # rollout-only / train-only 模式的轨迹数据，独立目录，由 rollout_data_path 配置
└── step_{step}.pt
```

tracker 文件**只有训练模型**在从自己的 `checkpoint_path` 续训时才会读：只有它记录的 step 才被视为有效可续训的 checkpoint。异步保存下 tracker 在完整落盘后才更新，因此从 tracker 读到的 step 一定完整可用。只读的权重来源 —— 参考模型与 OPD teacher —— 不看这个文件，step 由你自己挑。

## 2. 关键配置

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `checkpoint_path` | 必填 | 训练模型的 checkpoint **基目录**（既是训练输出也是续训入口，step 由 tracker 文件给出）。 |
| `hf_model_path` | 必填 | HuggingFace 模型路径。两种加载方式下都需要，用于提供模型结构信息。 |
| `rollout_data_path` | `./rollout_data` | rollout-only / train-only 模式的轨迹数据落盘目录。 |
| `trainer.save_freq` | `-1` | 保存频率（step）。`<= 0` 表示不保存。 |
| `trainer.async_save` | `true` | 是否使用异步保存，避免训练主流程被磁盘 IO 阻塞。 |
| `trainer.save_checkpoint` | `true` | 是否保存分布式 checkpoint（用于断点续训）。 |
| `trainer.save_hf` | `false` | 是否额外导出 HuggingFace 格式权重。 |
| `megatron.optimizer_sharding_type` | `dp_reshardable` | 分布式优化器分片格式，详见 [4.4](#44-优化器分片格式megatronoptimizer_sharding_type)。 |
| `ref_dist_ckpt_path` / `ref_hf_model_path` | `null` | 参考模型路径（用于 ref-KL），二选一。与训练模型的 `checkpoint_path` 语义不同：`ref_dist_ckpt_path` 直接指定某一个 `train_step_N/dist_ckpt` 目录 —— 不读 tracker 文件，也不扫描最新 step。 |
| `opd.teachers[].dist_ckpt_path` | `null` | OPD 中单个 teacher 的权重来源；仍需同时配 `hf_path`。语义同 `ref_dist_ckpt_path`：指定一个具体的 `dist_ckpt` 目录，而不是基目录。 |

## 3. 模型加载

LoongSage 支持两种加载方式，训练启动时会自动选择：

- **从 checkpoint 格式加载**：`checkpoint_path` 下存在有效 checkpoint 时自动走这条路径，恢复模型权重、优化器、学习率调度器、RNG 等完整训练状态。若续训时改动了 PP/TP size，RNG 会跳过恢复并打 warning，其他状态仍会正确加载。
- **从 HF 格式加载**：`checkpoint_path` 下没有可用 checkpoint 时，从 `hf_model_path` 加载 HuggingFace 权重作为初始参数。这是全新训练的默认入口。

> `hf_model_path` 在两种加载方式下都必须提供 —— 即使从 checkpoint 恢复，LoongSage 也需要读取 HF 配置来确定模型结构。

数据源续训时按存储的 `prompt_data_path` 字段匹配当前 datasource（而非按索引硬对应），因此续训时可以重新排序数据源而不会加载到错误的状态。

## 4. 模型保存

### 4.1 保存时机

每 `trainer.save_freq` 个 step 自动保存一次。此外，训练最后一步（`total_steps`）无论是否对齐 `save_freq` 都会强制保存一次，确保训练结束后总有一个完整可用的 checkpoint。保存成功后 tracker 文件更新，指向该 step。

### 4.2 同步 vs 异步保存

- **异步（`async_save: true`，默认）**：训练主流程不被磁盘 IO 阻塞，下一 step 可立即开始训练。同一时刻最多只有一个异步保存在进行——若上一次异步写盘尚未完成，新的保存会阻塞等待其结束后再调度。每个 train step 开始时会非阻塞地尝试 finalize 上一次异步保存（更新 tracker 文件），训练结束后会阻塞 flush 确保最后一次保存完整落盘。
- **同步（`async_save: false`）**：训练暂停，等 checkpoint 完全落盘后再继续。适合对 GPU 显存或 `/dev/shm` 敏感、希望立刻释放保存缓冲的场景。

### 4.3 保存格式：checkpoint 与 HuggingFace

由 `trainer.save_checkpoint` 和 `trainer.save_hf` 两个开关控制，可自由组合；两者都关闭时保存会被跳过。

- **分布式 checkpoint（`save_checkpoint`，默认开启）**：写入 `train_step_{step}/dist_ckpt/`，按 rank 分片存储（Megatron-Core `TorchDist` 格式），含模型 + 优化器 + 调度器 + RNG。是**断点续训的唯一来源**，但与并行拓扑（TP/PP/EP/DP）强绑定，不能直接用于推理部署。在 fully_async 模式下，保存时还会通过 `snapshot_pipeline_buf()` 捕获 pipeline 中尚未消费的在途数据，确保 resume 后不丢失。
- **HuggingFace 格式（`save_hf`，默认关闭）**：写入 `train_step_{step}/hf_model/`，跨 rank 聚合成完整 `safetensors` 权重 + config + tokenizer。方便离线评估、部署到 vLLM/SGLang 等推理框架、模型分享。**只有权重，不能用于续训**；导出需跨 rank 聚合，比 checkpoint 保存更耗时。

### 4.4 优化器分片格式：`megatron.optimizer_sharding_type`

开启分布式优化器（`megatron.ddp_config.use_distributed_optimizer=true`，默认开启）时，控制 checkpoint 里优化器状态的落盘形态：

| 取值 | 布局 | 保存 | 续训灵活度 |
| --- | --- | --- | --- |
| `dp_reshardable`（默认） | 按分布式优化器内部 bucket 布局 | 完全并行，无跨 rank 通信 | 仅 **DP 维度**可重分片 |
| `fully_reshardable` | 聚合后按模型参数原始形状展开 | 需 DP 维度 gather | **TP/PP/EP/DP 全维度**可重分片 |

训练全程拓扑不变就用默认值；需要切换 TP/PP/EP 拓扑续训时改为 `fully_reshardable`，代价是保存更慢、峰值内存更高。未开启分布式优化器时该字段被忽略。
