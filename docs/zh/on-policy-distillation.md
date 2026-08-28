# 在线蒸馏 (On-Policy Distillation)

在线蒸馏 (OPD) 让学生模型在自己的 rollout 数据上训练，同时匹配教师模型的 token 级分布。LoongSage 的 OPD 与 advantage estimator 正交，可以叠加在任意 estimator（GRPO、GSPO 等）之上，也可以作为纯蒸馏损失单独使用。同时，LoongSage 支持多老师蒸馏。

LoongSage 将 OPD 拆成两个可独立开关、可加权混合的角色：

- **PG (policy-gradient penalty)**：把 token 级 KL 作为惩罚项减到 advantage 上，与 RL 目标一起做策略梯度。
- **GKD (generalized knowledge distillation)**：把 token 级 KL 直接作为监督损失，不依赖 reward。

## 关键参数

| 参数 | 说明 |
|------|------|
| `opd.enable` | 启用在线蒸馏。 |
| `opd.pg_ratio` | PG 惩罚系数 $\lambda_{pg}$。`>0` 时把 KL 减到 advantage 上。 |
| `opd.gkd_ratio` | GKD 损失权重 $\lambda_{gkd}\in[0,1]$。`=1` 为纯蒸馏（跳过 RL 前向与 advantage）。 |
| `opd.pg_kl_method` | PG 使用的 KL 方法：`k1` / `topk_kl` / `topk_jsd` / `full_kl` / `full_jsd`。 |
| `opd.gkd_kl_method` | GKD 使用的 KL 方法：`k2` / `k3` / `topk_kl` / `topk_jsd` / `full_kl` / `full_jsd`。 |
| `opd.topk` | top-k 方法保留的教师词表大小。 |
| `opd.teachers` | 教师列表，每项含 `name`、`hf_path` 与可选的 `dist_ckpt_path`（一个具体的 `train_step_N/dist_ckpt` 目录）。单/多老师均在此配置。 |
| `opd.teacher_nodes` / `opd.teacher_gpus_per_node` | 教师池的节点数与每节点 GPU 数。 |
| `opd.model` | 教师的并行度（TP / PP / CP / EP）等模型配置。 |
| `data_sources[].teacher_name` | 该数据源使用的教师名（对应 `opd.teachers[].name`）。OPD 开启时必填。 |

> `pg_ratio` 与 `gkd_ratio` 不能同时为 0；`pg_ratio>0` 与 `gkd_ratio==1` 互斥。

## 原理

PG 在原始 advantage 上减去 token 级 KL 惩罚后再做策略梯度：

$$
\hat{A}_t = A_t - \lambda_{pg} \cdot D_{\text{KL}}(\pi_{\text{student}} \,\|\, \pi_{\text{teacher}})_t
$$

GKD 则作为独立损失，可与 RL 目标加权混合：

$$
L=(1-\lambda_{gkd})L_{RL}+\lambda_{gkd}L_{GKD}
$$

因此可以只用 PG（RL + KL 惩罚）、只用 GKD（纯蒸馏），或两者按比例混合。

## KL 方法

框架中的 KL 均为 reverse KL，即 $D_{\text{KL}}(\pi_{student}\|\pi_{teacher})$；JSD 是同一接口提供的对称散度选项。所有方法都是注册在 `coda.algorithms.kl_policy` 里的 `KLPolicy` 子类，按需要的教师数据分为三类：

| 方法 | 类型 | 教师数据 |
|------|------|----------|
| `k1` / `k2` / `k3` | reverse KL 的 token 级估计 | 教师在学生 rollout 采样 token 上的 log-prob（标量） |
| `topk_kl` / `topk_jsd` | top-k 词表近似 | 教师 top-k 的 log-prob 与索引 |
| `full_kl` / `full_jsd` | 全词表 | 教师 hidden state（用于重建全词表 logits） |

`topk_kl` 在教师 top-k 支持上对双方分布重新归一化后计算 reverse KL；`topk_jsd` 还对非 top-k 的 residual mass 做近似处理。它们比只用采样 token 的 log-prob 携带更多教师信息，但不是全词表精确散度；只有 `full_*` 方法使用完整词表。

`k1`/`k2`/`k3` 是同一 reverse KL 的三种 token 级估计，但 PG 与 GKD 用法不同：**PG** 把 KL 作为 detached 惩罚值减到 advantage 上，只用其**值**，需要有符号无偏的 per-token 对数比，故用 `k1`（`k2`/`k3` 恒非负，会扭曲惩罚信号）；**GKD** 把 KL 作为可微损失、只用其**梯度**，而 `k1` 的梯度在 on-policy 下期望为零、无法蒸馏，故用 `k2`/`k3`（`k3` 恒非负、方差低，默认推荐）。

新增自定义 KL / 散度算法见[自定义KL算法开发指南](./custom-kl.md)。

### 全词表的显存优化

全词表 KL 需要教师的完整 logits，其规模为 `[seq × vocab]`，直接传输或落盘会占用巨量显存。LoongSage 改为只传输紧凑的教师 **hidden state** `[seq × hidden]`，在学生侧用一份 TP 切分的教师 `lm_head` 重建 logits。重建按 microbatch 粒度进行、每个 microbatch 只算一次并在用完后立即释放（`TeacherCtx` / `KLCtx` 记忆化），从而在不牺牲全词表精度的前提下大幅降低显存与传输开销。log-prob 与 top-k 方法同样在每个 microbatch 只前向计算一次教师量并复用。

## 教师编排：TeacherManager

为了统一支持**单老师 / 多老师**、**同模型 / 异模型**蒸馏，LoongSage 用 `TeacherManager` 抽象出教师模型与 GPU 资源的管理，与训练侧解耦：

- **资源分组**：按 `teacher_nodes × teacher_gpus_per_node` 组成教师池，单个教师组的 world size = `dp_per_teacher × TP × PP × CP`，教师按数量划分到各组。
- **同模型多老师**：教师 GPU 资源足够时（教师 DP 数 ≥ 教师数）每个教师独占一组、常驻 GPU，不需要切换；资源不足时多个教师落到同一组内，复用同一份模型结构，非激活教师的权重以 CPU pinned 内存备份，前向时按需拷回显存。
- **异模型蒸馏**：不同组可各自加载不同架构/权重的教师，在不同 GPU 上并存。
- **数据路由**：`data_sources[].teacher_name` 把每条 rollout 映射到对应教师；`compute_teacher` 先按 `teacher_idx` 分桶，**每个教师对自己的数据只前向一次**（当前激活的教师优先处理，避免多余的权重切换）。
- **colocate**：与 rollout / 训练分时复用 GPU 时，`TeacherManager` 负责教师的 onload / offload。

## 数据重排

教师池的数据并行度（teacher_dp）与训练侧（train_dp）通常不同，LoongSage 通过一次**双向重排**衔接两者，全程用 `train_dp_ranks` + `seq_index` 记录每条轨迹的来源：

1. **按 train dp 切分**：`_fetch_rollout_data` 依据 `train_dp == / > / < teacher_dp` 三种情形，把训练分片映射到教师 DP。
2. **按 teacher 分组前向**：按 `teacher_idx` 分桶，各教师并行前向。
3. **按 train dp 归并**：教师输出按 `train_dp_ranks` 分发回各训练 rank，训练侧再按 `seq_index` 归并（`merge_rollout_batch`），恢复与原始轨迹的对齐。

此外，数据切分（`split_traj_group_by_dp`）保证：每个 DP rank 拿到的各 data source 比例一致，且**每个 minibatch 内不同 data source 的比例与整体一致**，避免多数据源混训时的分布偏差。

## 配置示例

**单老师**：
```yaml
data_sources:
  - dataset: { prompt_data_path: /path/train.parquet }
    teacher_name: "math_expert"

opd:
  enable: true
  pg_ratio: 0
  gkd_ratio: 1              # 纯蒸馏
  gkd_kl_method: full_kl
  teacher_nodes: 1
  teacher_gpus_per_node: 8
  teachers:
    - name: "math_expert"
      hf_path: /path/OpenThinker3-7B
  model: { tensor_model_parallel_size: 4, pipeline_model_parallel_size: 2 }
```

**教师权重来自 Megatron dist checkpoint**：配上 `dist_ckpt_path`，即可直接把一次 LoongSage
训练的产出当教师用，无需先导出成 HF 格式。`hf_path` 仍然必填 —— bridge 要靠它的
`config.json` 建 Megatron 模型结构，而 `dist_ckpt` 目录里没有这个文件 —— 但不会读它的
权重。两者必须描述同一套架构；不一致会在加载时显式报错，不会静默。

```yaml
  teachers:
    - name: "math_expert"
      hf_path: /path/OpenThinker3-7B      
      # 提供权重：一个具体的 dist_ckpt 目录，step 由用户自己挑。
      dist_ckpt_path: /path/run/train_step_100/dist_ckpt
```

`dist_ckpt_path` 指向的模型，其 TP/PP/EP 可以与 `opd.model` 不同，权重会在加载时重新切分。
若配了路径但它不是可用的 dist checkpoint，会直接启动失败而不是回退到 `hf_path`。用 `full_kl` /
`full_jsd` 时，student 侧的教师 `lm_head` 也从同一份 checkpoint 读取 —— 同样地，
`dist_ckpt_path` 不可用时它会报错，而不会改读 `hf_path`。

**多老师（不同数据源路由到不同教师）**：
```yaml
data_sources:
  - dataset: { prompt_data_path: /path/math.parquet }
    teacher_name: "math_expert"
  - dataset: { prompt_data_path: /path/code.parquet }
    teacher_name: "code_expert"

opd:
  enable: true
  pg_ratio: 0.5            # PG + GKD 混合
  gkd_ratio: 0.5
  pg_kl_method: k1
  gkd_kl_method: topk_kl
  topk: 64
  teachers:
    - name: "math_expert"
      hf_path: /path/math_model
    - name: "code_expert"
      hf_path: /path/code_model
```

## 运行示例

OPD 通过 `--config-name` 指定配置启动，示例配置见 [conf/qwen3_30b_a3b/mopd_h20_1node.yaml](../../conf/qwen3_30b_a3b/mopd_h20_1node.yaml)：

```bash
python -m coda.controller.trainer --config-name qwen3_30b_a3b/mopd_h20_1node
```
