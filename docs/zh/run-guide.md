# 运行指南
## 多机训练准备

多机训练需要先在所有参与训练的机器上启动 Ray。[`examples/start_ray_cluster.sh`](../../examples/start_ray_cluster.sh) 以**主节点 IP** 为参数，脚本会自动识别当前节点是 head 还是 worker，所以主节点和每台工作节点执行的是同一条命令：

```bash
# 每台参与训练的机器都在仓库根目录执行，<master-ip> 替换成主节点 IP
bash examples/start_ray_cluster.sh <master-ip>
```

所有节点加入后，在主节点确认集群状态：

```bash
ray status
```

几点说明：

- 脚本默认按 8 卡节点启动，GB200 等 4 卡节点用 `NUM_GPUS=4` 覆盖。
- 多机训练时，请确保每个节点能以相同路径访问模型和数据集，可在各节点分别下载，或统一挂载共享存储。下文以 `/root/...` 为例，修改目录时请同步更新相关命令。
- 单机训练（`colocate` 单节点预设）不需要 Ray 集群，可直接执行下面的任务。

## 示例任务

### 任务一：使用 Qwen3-30B-A3B 跑 DAPO

下载模型与数据集：

```bash
hf download Qwen/Qwen3-30B-A3B --local-dir /root/Qwen3-30B-A3B
hf download Haitao999/DAPO-Math-17k-unique --repo-type=dataset --local-dir /root/DAPO-Math-17k-unique
```

启动单机 8 卡预设 [`qwen3_30b_a3b/dapo_h20_1node`](../../conf/qwen3_30b_a3b/dapo_h20_1node.yaml)：

```bash
bash examples/start.sh qwen3_30b_a3b/dapo_h20_1node \
  hf_model_path=/root/Qwen3-30B-A3B \
  data_source.dataset.prompt_data_path=/root/DAPO-Math-17k-unique \
  checkpoint_path=/root/ckpt/math_expert \
  trainer.save_freq=10
```

`trainer.save_freq` 默认为 `-1`，即完全不保存 checkpoint，这里显式打开。产出的 checkpoint 位于 `/root/ckpt/math_expert/train_step_<N>/dist_ckpt`。

### 任务二：运行 BCP

先安装 FAISS（其余依赖训练镜像已内置），再下载 Embedding 模型、BrowseComp-Plus 的 prompt 数据、语料和 dense 索引：

```bash
python3 -m pip install faiss-cpu

hf download Qwen/Qwen3-Embedding-8B --local-dir /root/Qwen3-Embedding-8B
hf download Tevatron/browsecomp-plus --repo-type=dataset --include "data/*.parquet" --local-dir /root/browsecomp-plus/raw
hf download Tevatron/browsecomp-plus-corpus --repo-type=dataset --local-dir /root/browsecomp-plus/corpus
hf download Tevatron/browsecomp-plus-indexes --repo-type=dataset --local-dir /root/browsecomp-plus/indexes
```

BCP 依赖一个检索服务。先用官方 dense shard 和语料一次性构建 cache，再启动服务，下例的 `--gpu_id 7` 会占用该卡的部分显存：

```bash
python3 examples/bcp/build_dense_cache.py \
  --index-path "/root/browsecomp-plus/indexes/qwen3-embedding-8b/corpus.shard*.pkl" \
  --corpus-path /root/browsecomp-plus/corpus/data \
  --output /root/browsecomp-plus/browsecomp_dense_cache.pkl

./examples/bcp/run_retrieval_server.sh \
  --data_dir /root/browsecomp-plus/corpus/data \
  --model /root/Qwen3-Embedding-8B \
  --dense_cache /root/browsecomp-plus/browsecomp_dense_cache.pkl \
  --gpu_id 7 \
  --port 9000
```

可通过健康检查和一次真实查询验证服务：

```bash
curl -sS http://127.0.0.1:9000/health
curl -sS http://127.0.0.1:9000/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"queries":["Tribeca Festival Gotham Week"],"topk":3,"return_scores":true}'
```

检索服务就绪后，启动单机 8 卡预设 [`qwen3_30b_a3b/bcp_h20_1node`](../../conf/qwen3_30b_a3b/bcp_h20_1node.yaml)：

```bash
bash examples/start.sh qwen3_30b_a3b/bcp_h20_1node \
  hf_model_path=/root/Qwen3-30B-A3B \
  data_source.dataset.prompt_data_path=/root/browsecomp-plus/raw \
  checkpoint_path=/root/ckpt/bcp_expert \
  trainer.save_freq=10
```

若检索服务不在默认的 `http://127.0.0.1:9000`，用 `data_source.agent.retrieval_service_url=...` 覆盖。只有检索服务和训练运行在同一个 Pod 时才能使用 localhost。产出的 checkpoint 位于 `/root/ckpt/bcp_expert/train_step_<N>/dist_ckpt`。

### 任务三：运行 MOPD 多教师蒸馏

把任务一和任务二训练出的两个专家当教师：任务一的 checkpoint 作为 MATH 教师，任务二的作为 BCP 教师，并为两个数据源产生的轨迹分别配置对应教师（数据源上的 `data_sources[].teacher_name` 按名字匹配 `opd.teachers[].name`），对同一个学生模型做在线蒸馏。

BCP 数据源仍需检索服务在运行：

```bash
./examples/bcp/run_retrieval_server.sh \
  --data_dir /root/browsecomp-plus/corpus/data \
  --model /root/Qwen3-Embedding-8B \
  --dense_cache /root/browsecomp-plus/browsecomp_dense_cache.pkl \
  --gpu_id 7 \
  --port 9000
```

从两次训练里各挑一个 step，填入对应的 `dist_ckpt` 目录，启动 [`qwen3_30b_a3b/mopd_h20_1node`](../../conf/qwen3_30b_a3b/mopd_h20_1node.yaml)：

```bash
bash examples/start.sh qwen3_30b_a3b/mopd_h20_1node \
  hf_model_path=/root/Qwen3-30B-A3B \
  checkpoint_path=/root/ckpt/mopd \
  data_sources.0.dataset.prompt_data_path=/root/browsecomp-plus/raw \
  data_sources.1.dataset.prompt_data_path=/root/DAPO-Math-17k-unique \
  opd.teachers.0.hf_path=/root/Qwen3-30B-A3B \
  opd.teachers.0.dist_ckpt_path=/root/ckpt/bcp_expert/train_step_100/dist_ckpt \
  opd.teachers.1.hf_path=/root/Qwen3-30B-A3B \
  opd.teachers.1.dist_ckpt_path=/root/ckpt/math_expert/train_step_100/dist_ckpt
```

opd.teachers的 `hf_path` 始终必填，但只用于按其 `config.json` 建 Megatron 模型结构；权重来自 `dist_ckpt_path`，两者必须描述同一套架构。教师并行度、KL 方法等参数见[在线策略蒸馏](on-policy-distillation.md)。

### 任务四：使用 DeepSeek-V4-Flash 跑 SWE

先下载 FP8 模型和数据集，再将模型转换为训练所需的 BF16 权重：

```bash
hf download sgl-project/DeepSeek-V4-Flash-FP8 --local-dir /root/DeepSeek-V4-Flash-FP8
hf download R2E-Gym/R2E-Gym-Subset --repo-type=dataset --local-dir /root/R2E-Gym-Subset

python examples/convert_dsv4_fp8_to_bf16.py \
  --input-fp8-hf-path /root/DeepSeek-V4-Flash-FP8 \
  --output-bf16-hf-path /root/DeepSeek-V4-Flash-BF16
```

转换完成后，准备 SWE rollout 使用的 K8s 沙箱所需的 kubeconfig。可将 kubeconfig 放到 `k8s/kubeconfig.yaml`（`agentflow.sandbox.kubeconfig` 的默认路径），或在启动命令中覆盖。然后启动 8 机 H20 预设 [`dsv4_flash_bf16/swe_h20_8node`](../../conf/dsv4_flash_bf16/swe_h20_8node.yaml)：先在所有节点拉起 Ray 集群，再仅从 head 节点提交训练任务。

```bash
# 在每台机器上执行同一条命令
bash examples/start_ray_cluster.sh <master-ip>

# 集群就绪后，仅在 head 节点执行
bash examples/start.sh dsv4_flash_bf16/swe_h20_8node \
  checkpoint_path=/root/ckpt/dsv4_swe \
  hf_model_path=/root/DeepSeek-V4-Flash-BF16 \
  data_source.dataset.prompt_data_path=/root/R2E-Gym-Subset \
  agentflow.sandbox.kubeconfig=/path/to/kubeconfig.yaml
```

`checkpoint_path` 与 `hf_model_path` 需放在所有节点都能通过相同路径访问的存储上。

### 任务五：运行 OpenCode 黑盒智能体

下载模型和已经准备好的 R2E-Gym 数据集。数据中的镜像已预装 OpenCode，无需再做数据转换。

```bash
hf download Qwen/Qwen3-Coder-30B-A3B-Instruct --local-dir /root/Qwen3-Coder-30B-A3B-Instruct
hf download LoongSage/R2E-Gym R2E_Gym_Subset_opencode.parquet \
  --repo-type=dataset --local-dir /root/R2E-Gym-OpenCode
```

准备 K8s 沙箱 kubeconfig，可将其放到 `k8s/kubeconfig.yaml`，或在启动命令中覆盖路径。然后在四台 H20 机器上执行同一条 Ray 启动命令：

```bash
bash examples/start_ray_cluster.sh <master-ip>
```

所有节点加入后，在 head 节点启动 [`qwen3_coder_30b_a3b/opencode_h20_4node`](../../conf/qwen3_coder_30b_a3b/opencode_h20_4node.yaml)：

```bash
bash examples/start.sh qwen3_coder_30b_a3b/opencode_h20_4node \
  checkpoint_path=/root/ckpt/opencode \
  hf_model_path=/root/Qwen3-Coder-30B-A3B-Instruct \
  data_source.dataset.prompt_data_path=/root/R2E-Gym-OpenCode/R2E_Gym_Subset_opencode.parquet \
  agentflow.sandbox.kubeconfig=/path/to/kubeconfig.yaml
```

提示：各预设继承 [`default.yaml`](../../conf/default.yaml)，[`conf/qwen3_30b_a3b/`](../../conf/qwen3_30b_a3b/) 这类子目录是特定模型与机型的预设。更多可用配置见 [`conf/`](../../conf/)。

## 运行监控

### 查看运行状态与监控

训练在后台运行，日志写入 `log/trainer_<时间戳>.log`（`start.sh` 启动时会打印该文件名）：

```bash
tail -f log/trainer_*.log
```

`tracking.tracking_backend` 默认取 `console`，指标直接打印到日志，重点观察：

| 指标 | 含义 |
| --- | --- |
| `rollout/completed_count` | 本 step 收齐的轨迹数，应等于 `num_prompts_per_step × num_trajectories_per_prompt` |
| `rollout/reward_mean` | 平均奖励 |
| `train/pg_loss`、`train/entropy` | 训练 loss、熵 |
| `timing/*` | 各阶段耗时 |

多机的集群状态用 `ray status` 查看。需要集中可视化时可改用 MLflow 等后端：`tracking.tracking_backend=mlflow tracking.mlflow_tracking_uri=http://...`。

### 停止训练
在启动训练的节点上执行 `pkill -f coda.controller.trainer`（多机时每台节点都要做），之后可用 `ray stop` 清理 Ray 集群。

## 常用 Hydra 配置

顶层运行参数（`run_mode`、`total_steps`、`checkpoint_path`、`rollout_data_path` 等）与所有
字段的逐项说明见 [配置参数参考 - 顶层运行参数](config-reference.md#1-顶层运行参数)。

### 数据源

本节涉及的配置字段：

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `data_source` / `data_sources` | `[${data_source}]` | 单个数据源模板 / 实际使用的数据源列表 |
| `data_source.dataset.prompt_data_path` | 必填 | prompt 数据文件或目录，支持 `@[start:stop]` 切片 |
| `data_source.dataset.input_key` / `label_key` | 必填 | 数据集的 prompt / 标签列名 |
| `data_source.num_prompts_per_step` | `64` | 每 step 每个数据源派发的 prompt 组数 |
| `data_source.num_trajectories_per_prompt` | `8` | 每个 prompt 组采样的轨迹条数 |
| `data_source.agent.name` | `null`（单轮） | Agent 实现名，多轮场景必须配置 |
| `data_source.reward.name` | 必填 | 奖励函数名（参数见各 reward 插件） |
| `data_source.max_response_len_per_trajectory` | `32768` | 每条轨迹响应区 token 上限 |

`data_source` 是单个数据源的默认值模板，`data_sources` 是实际使用的列表，默认展开为
`[${data_source}]`（即单数据源）。单数据源用 `data_source.` 前缀覆盖；多数据源用
`data_sources.<下标>.` 前缀或显式写列表。两种渠道等价：命令行 `key=value`（追加在
`start.sh` 配置名之后）或 experiment yaml。

**单数据源 · 命令行**

```bash
bash examples/start.sh qwen3_30b_a3b/dapo_h20_1node \
  hf_model_path=/root/Qwen3-30B-A3B \
  data_source.dataset.prompt_data_path=/root/DAPO-Math-17k-unique \
  data_source.num_prompts_per_step=64 \
  data_source.num_trajectories_per_prompt=8
```

**单数据源 · yaml**

```yaml
defaults:
  - /qwen3_30b_a3b/dapo_h20_1node
  - _self_

data_source:
  dataset:
    prompt_data_path: /root/DAPO-Math-17k-unique
    input_key: prompt
    label_key: reward_model
  num_prompts_per_step: 64
  num_trajectories_per_prompt: 8
```

**多数据源 · yaml**（把 `data_sources` 显式写成列表，每个元素可独立配置 prompt 数据、
采样规模、agent 与 reward，未写字段继承 `data_source` 默认值）：

```yaml
defaults:
  - /qwen3_30b_a3b/dapo_h20_1node
  - _self_

data_sources:
  - dataset:
      prompt_data_path: /root/browsecomp-plus/prompts/browsecomp_plus.parquet
      input_key: prompt
      label_key: answer
    num_prompts_per_step: 32
    num_trajectories_per_prompt: 8
    agent:
      name: bcp
      retrieval_service_url: http://127.0.0.1:9000
      max_turns: 15
    reward:
      name: bcp
    max_response_len_per_trajectory: 10240
  - dataset:
      prompt_data_path: /root/DAPO-Math-17k-unique
      input_key: prompt
      label_key: reward_model
    num_prompts_per_step: 32
    num_trajectories_per_prompt: 8
    reward:
      name: dapo_math
      overlong_penalty_length: 4096
    max_response_len_per_trajectory: 20480
```

**多数据源 · 命令行**（按下标覆盖列表元素，未覆盖的字段沿用 `data_source` 默认值）：

```bash
bash examples/start.sh qwen3_30b_a3b/dapo_h20_1node \
  hf_model_path=/root/Qwen3-30B-A3B \
  checkpoint_path=/root/ckpt/multi \
  data_sources.0.dataset.prompt_data_path=/root/browsecomp-plus/prompts/browsecomp_plus.parquet \
  data_sources.1.dataset.prompt_data_path=/root/DAPO-Math-17k-unique
```
注意：全异步（`fully_async.enable=true`）只支持一个数据源；修改 batch 规模等相关约束见
[默认配置参数手册 - 常见组合与约束速查](config-reference.md#11-常见组合与约束速查)。

### 指标追踪

`tracking` 决定指标上报位置。本节涉及的配置项：

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `tracking.tracking_backend` | `console` | 指标后端：`console` / `mlflow` / `wandb`，也支持列表 |
| `tracking.mlflow_tracking_uri` | `""` | MLflow server 地址 |
| `tracking.project_name` / `experiment_name` | `default` | MLflow 实验 / run 分组名 |
| `tracking.wandb_args.*` | — | W&B 参数，原样透传 `wandb.Settings()` |

默认 `tracking_backend=console` 只把指标输入
`log/trainer_*.log` 文件，需要集中可视化时换成 `mlflow` 或 `wandb`。

**上报到 MLflow**

```bash
bash examples/start.sh qwen3_30b_a3b/dapo_h20_1node \
  hf_model_path=/root/Qwen3-30B-A3B \
  data_source.dataset.prompt_data_path=/root/DAPO-Math-17k-unique \
  tracking.tracking_backend=mlflow \
  tracking.mlflow_tracking_uri=http://<host>:<port>/ \
  tracking.project_name=math-rl \
  tracking.experiment_name=dapo-30b-a3b-lr1e6
```

`project_name` 对应 MLflow 的 experiment，`experiment_name` 对应该 experiment 下的 run 名，
两者默认都是 `default`，多人共用 server 时务必显式区分。`mlflow_tracking_uri` 默认空字符串串，
选了 `mlflow` 却不填会连不上 server。

**上报到 W&B**

```bash
bash examples/start.sh qwen3_30b_a3b/dapo_h20_1node \
  hf_model_path=/root/Qwen3-30B-A3B \
  data_source.dataset.prompt_data_path=/root/DAPO-Math-17k-unique \
  tracking.tracking_backend=wandb \
  tracking.wandb_args.base_url=https://<wandb-host> \
  tracking.wandb_args.api_key=<key>
```

`wandb_args` 下的键大多原样透传给 `wandb.Settings()`。`mode` 只有两个取值：`shared`（默认）
实时上报，允许多个进程写同一个 run，需要 `base_url` + `api_key`，且 W&B SDK ≥ 0.19.9、
W&B Server ≥ 0.68；`offline` 只落本地磁盘。

### Checkpoint 与断点续训

本节涉及的配置项：

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `checkpoint_path` | 必填 | 训练模型 checkpoint 基目录（输出 + 续训入口） |
| `rollout_data_path` | `./rollout_data` | rollout-only / train-only 的轨迹交接目录 |
| `trainer.save_freq` | `-1` | checkpoint 保存频率（step）；`<= 0` 不保存 |
| `trainer.save_hf` | `false` | 是否额外导出 HuggingFace 格式权重 |
| `megatron.optimizer_sharding_type` | `dp_reshardable` | 优化器分片格式（决定续训能否改拓扑） |


`trainer.save_freq` 默认为 `-1`（不保存）。开启后（如任务一、任务二设为 10），产物按 `checkpoint_path` 组织（详见 [模型加载与保存](model-checkpointing.md)）：

```text
checkpoint_path/
├── latest_checkpointed_iteration.txt   # tracker 文件：记录最近成功保存的 step
├── train_step_10/
│   ├── dist_ckpt/                       # 分布式 checkpoint（模型 + 优化器 + 调度器 + RNG）
│   │                                    # 也是 MOPD 中 opd.teachers[].dist_ckpt_path 要填的值
│   ├── hf_model/                        # HuggingFace 格式导出（trainer.save_hf=true 时）
│   └── data_source/                     # 数据源游标 + 缓冲快照，用于断点续训
└── ...
```

几个要点：

- **续训**：用相同的 `checkpoint_path` 再次执行 `start.sh` 即自动从 tracker 记录的 step 恢复完整状态（优化器、lr 调度器、RNG、数据源游标），无需额外参数。
- **导出 HF 权重**：`trainer.save_hf=true` 时每 `save_freq` 步额外写出 `train_step_N/hf_model/safetensors`，可直接用于离线评估或部署 sglang/vLLM；但只有权重，不能用来续训。
- **改拓扑续训**：默认 `optimizer_sharding_type=dp_reshardable` 只支持沿 DP 维度重新分片；需要切换 TP/PP/EP 拓扑续训时改为 `fully_reshardable`（保存更慢、峰值内存更高）。

## 预设配置总览

除上文任务用到的预设外，`conf/` 下还有以下开箱即用预设，均可用 「`hf_model_path` + `checkpoint_path` + `data_source.dataset.prompt_data_path`」 三件套启动（`data_source.dataset.prompt_data_path` 指向文件或目录均可，可加 `@[start:stop]` 切片试跑）：

| 目录 | 预设 | 模型 | 任务/算法 | 资源 |
| --- | --- | --- | --- | --- |
| `conf/qwen3_4b/` | `dapo_h800_1node` | Qwen3-4B | DAPO 数学 | 单机 8xH800 |
| `conf/qwen3_30b_a3b/` | `dapo_h20_1node` | Qwen3-30B-A3B | DAPO 数学（任务一）| 单机 8xH20 |
| | `gsm8k_h20_1node` | Qwen3-30B-A3B | GSM8K 数学 | 单机 8xH20 |
| | `bcp_h20_1node` | Qwen3-30B-A3B | BCP 检索（任务二）| 单机 8xH20 |
| | `mopd_h20_1node` | Qwen3-30B-A3B | MOPD 多教师蒸馏（任务三）| 单机 8xH20 |
| `conf/qwen3_coder_30b_a3b/` | `mini_swe_h20_4node` | Qwen3-Coder-30B-A3B | mini-SWE（R2E-Gym）| 4 节点 xH20 |
| | `opencode_h20_4node` | Qwen3-Coder-30B-A3B | OpenCode（R2E-Gym）（任务五）| 4 节点 xH20 |
| `conf/dsv4_flash_bf16/` | `swe_h20_8node` / `swe_gb200_8node` | DeepSeek-V4-Flash-BF16 | SWE（任务四）| 8xH20 / 8xGB200 |
| | `dapo_h20_6node` / `dapo_gb200_8node` | DeepSeek-V4-Flash-BF16 | DAPO 数学 | 6xH20 / 8xGB200 |


## 相关文档

- 从零跑通第一个 step（环境、数据、验证）→ [快速开始](quick-start.md)
- 配置项逐项说明与整除约束 → [配置参数参考](config-reference.md)
- 训练算法（DAPO/GRPO/GSPO 等）→ [训练算法](training-algorithms.md)
- 模型加载、保存、导出 → [模型加载与保存](model-checkpointing.md)
- 在线策略蒸馏（MOPD）→ [在线策略蒸馏](on-policy-distillation.md)
- 定制 Agent / Reward / 沙箱 → [自定义 Agent](custom-agent.md) / [自定义 Reward](custom-reward.md) / [自定义沙箱](custom-sandbox.md)
