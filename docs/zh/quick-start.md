# 快速开始

本文按「环境 → 模型和数据 → 启动训练 → 验证结果」的顺序给出最快的一条路径：在单机 8 卡
（H800）上用 `conf/qwen3_4b/dapo_h800_1node` 预设（Qwen3-4B + DAPO）跑通第一个训练 step。
多机等其他启动方式和运行模式见 [运行指南](run-guide.md)，配置项的逐项说明见
[默认配置参数手册](config-reference.md)。

## 1. 环境准备

由于 LoongSage 可能会包含针对 sglang/megatron 的特定版本和临时补丁（patch）。为避免潜在的环境配置问题，强烈建议用户使用我们提供的最新 Docker 镜像。

### 拉取并启动 Docker 容器

```bash
# 拉取最新镜像
docker pull loongsage/loongsage:latest

docker run --rm --gpus all --ipc=host \
  -it loongsage/loongsage:latest /bin/bash
```

## 2. 准备模型和数据

### 下载模型 与 数据集

本预设使用 Qwen3-4B，直接从 Hugging Face 下载权重：

```bash
hf download Qwen/Qwen3-4B --local-dir /root/Qwen3-4B
```

本预设是 DAPO 数学任务，使用 DAPO-Math-17k 训练集：

```bash
hf download Haitao999/DAPO-Math-17k-unique --repo-type=dataset \
  --local-dir /root/DAPO-Math-17k-unique
```

## 3. 启动训练

### 启动

LoongSage 提供两种等价的启动方式：

**方式一：填好配置文件，用 `examples/start.sh` 启动（第一次跑推荐这个）。** 把两个 `???` 直接填到
`conf/qwen3_4b/dapo_h800_1node.yaml` 里。

填必填项

```yaml
hf_model_path: /root/Qwen3-4B
data_source:
  dataset:
    prompt_data_path: /root/DAPO-Math-17k-unique/<文件>.parquet
```

运行

```bash
bash examples/start.sh qwen3_4b/dapo_h800_1node
```

**方式二：不改配置文件，在命令行上覆盖。** `examples/start.sh` 会把配置名之后的参数原样透传给 Hydra：

```bash
bash examples/start.sh qwen3_4b/dapo_h800_1node \
  hf_model_path=/root/Qwen3-4B \
  data_source.dataset.prompt_data_path=/root/DAPO-Math-17k-unique
```

## 4. 验证结果

`tracking.tracking_backend` 默认取 `console`，指标直接写入日志，重点观察以下指标：

| 指标 | 含义 |
| --- | --- |
| `rollout/completed_count` | 本 step 收齐的轨迹数，应等于 `num_prompts_per_step × num_trajectories_per_prompt` |
| `rollout/reward_mean` | 平均奖励 |
| `train/is_approx_k3_kl` | 当前策略与采样策略的近似 KL |
| `train/pg_loss`、`train/entropy` | 训练loss、熵 |
| `timing/*` | 各阶段耗时 |

## 5. 下一步

- 启动方式（含多机）、运行模式、多数据源、评测 → [运行指南](run-guide.md)
- 配置项逐项说明 → [默认配置参数手册](config-reference.md)
- 自定义 Agent / Reward / 沙箱 → [自定义 Agent](custom-agent.md)、
  [自定义 Reward](custom-reward.md)、[自定义沙箱](custom-sandbox.md)