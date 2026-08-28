# On-Policy Distillation

On-policy distillation (OPD) lets the student model train on its own rollout data while matching the token-level distribution of the teacher model. LoongSage's OPD is orthogonal to the advantage estimator: it can be stacked on top of any estimator (GRPO, GSPO, etc.), or used standalone as a pure distillation loss. LoongSage also supports multi-teacher distillation.

LoongSage splits OPD into two roles that can be enabled independently and mixed with weights:

- **PG (policy-gradient penalty)**: subtracts the token-level KL as a penalty term from the advantage, and performs the policy gradient together with the RL objective.
- **GKD (generalized knowledge distillation)**: uses the token-level KL directly as a supervised loss, without depending on reward.

## Key Parameters

| Parameter | Description |
|------|------|
| `opd.enable` | Enables on-policy distillation. |
| `opd.pg_ratio` | PG penalty coefficient $\lambda_{pg}$. When `>0`, the KL is subtracted from the advantage. |
| `opd.gkd_ratio` | GKD loss weight $\lambda_{gkd}\in[0,1]$. `=1` means pure distillation (the RL forward pass and advantage estimation are skipped). |
| `opd.pg_kl_method` | The KL method used by PG: `k1` / `topk_kl` / `topk_jsd` / `full_kl` / `full_jsd`. |
| `opd.gkd_kl_method` | The KL method used by GKD: `k2` / `k3` / `topk_kl` / `topk_jsd` / `full_kl` / `full_jsd`. |
| `opd.topk` | The teacher vocabulary size kept by the top-k methods. |
| `opd.teachers` | The teacher list; each entry contains `name`, `hf_path` and an optional `dist_ckpt_path` (one concrete `train_step_N/dist_ckpt` dir). Both single-teacher and multi-teacher setups are configured here. |
| `opd.teacher_nodes` / `opd.teacher_gpus_per_node` | The number of nodes in the teacher pool and the number of GPUs per node. |
| `opd.model` | Model configuration of the teacher, such as its parallelism degrees (TP / PP / CP / EP). |
| `data_sources[].teacher_name` | The teacher name used by this data source (corresponding to `opd.teachers[].name`). Required when OPD is enabled. |

> `pg_ratio` and `gkd_ratio` cannot both be 0; `pg_ratio>0` and `gkd_ratio==1` are mutually exclusive.

## Principles

PG subtracts the token-level KL penalty from the original advantage before performing the policy gradient:

$$
\hat{A}_t = A_t - \lambda_{pg} \cdot D_{\text{KL}}(\pi_{\text{student}} \,\|\, \pi_{\text{teacher}})_t
$$

GKD instead acts as an independent loss, and can be mixed with the RL objective by weight:

$$
L=(1-\lambda_{gkd})L_{RL}+\lambda_{gkd}L_{GKD}
$$

Therefore one can use PG only (RL + KL penalty), GKD only (pure distillation), or mix the two by ratio.

## KL Methods

All KL in the framework is reverse KL, that is $D_{\text{KL}}(\pi_{student}\|\pi_{teacher})$; JSD is the symmetric divergence option provided by the same interface. All methods are `KLPolicy` subclasses registered in `coda.algorithms.kl_policy`, and fall into three categories according to the teacher data they require:

| Method | Type | Teacher Data |
|------|------|----------|
| `k1` / `k2` / `k3` | Token-level estimates of reverse KL | The teacher's log-prob on the tokens sampled in the student rollout (a scalar) |
| `topk_kl` / `topk_jsd` | Top-k vocabulary approximation | The teacher's top-k log-probs and indices |
| `full_kl` / `full_jsd` | Full vocabulary | The teacher's hidden states (used to reconstruct full-vocabulary logits) |

`topk_kl` renormalizes both distributions over the teacher's top-k support before computing the reverse KL; `topk_jsd` additionally applies an approximation to the residual mass outside the top-k. They carry more teacher information than using only the log-prob of the sampled token, but they are not exact full-vocabulary divergences; only the `full_*` methods use the complete vocabulary.

`k1`/`k2`/`k3` are three token-level estimates of the same reverse KL, but PG and GKD use them differently: **PG** subtracts the KL from the advantage as a detached penalty value and only uses its **value**, which requires a signed unbiased per-token log-ratio, hence `k1` (`k2`/`k3` are always non-negative and would distort the penalty signal); **GKD** uses the KL as a differentiable loss and only uses its **gradient**, and the gradient of `k1` has zero expectation on-policy and therefore cannot distill, hence `k2`/`k3` (`k3` is always non-negative and has low variance, and is the recommended default).

To add a custom KL / divergence algorithm, see the [Custom KL Algorithm Development Guide](./custom-kl.md).

### Memory Optimization for Full Vocabulary

Full-vocabulary KL requires the teacher's complete logits, whose size is `[seq × vocab]`; transferring or dumping them directly would consume an enormous amount of memory. LoongSage instead transfers only the compact teacher **hidden state** `[seq × hidden]`, and reconstructs the logits on the student side with a TP-sharded copy of the teacher's `lm_head`. Reconstruction is done at microbatch granularity, computed only once per microbatch and released immediately after use (`TeacherCtx` / `KLCtx` memoization), which greatly reduces memory and transfer overhead without sacrificing full-vocabulary precision. The log-prob and top-k methods likewise forward the teacher only once per microbatch and reuse the result.

## Teacher Orchestration: TeacherManager

To uniformly support **single-teacher / multi-teacher** and **same-model / different-model** distillation, LoongSage uses `TeacherManager` to abstract the management of teacher models and GPU resources, decoupled from the training side:

- **Resource grouping**: the teacher pool is formed by `teacher_nodes × teacher_gpus_per_node`; the world size of a single teacher group is `dp_per_teacher × TP × PP × CP`, and teachers are divided into groups by count.
- **Multiple teachers of the same model**: when there are enough teacher GPUs (teacher DP count ≥ teacher count) each teacher gets its own group and stays resident on GPU, so no switching is needed; when resources are short several teachers land in the same group and reuse a single model structure, with the inactive teachers' weights backed up in CPU pinned memory and copied back to GPU on demand.
- **Different-model distillation**: different groups can each load teachers with different architectures/weights, coexisting on different GPUs.
- **Data routing**: `data_sources[].teacher_name` maps each rollout to the corresponding teacher; `compute_teacher` first buckets by `teacher_idx`, and **each teacher forwards its own data only once** (the currently active teacher is processed first, avoiding redundant weight switching).
- **colocate**: when GPUs are time-shared with rollout / training, `TeacherManager` is responsible for onloading / offloading the teachers.

## Data Rearrangement

The data parallel degree of the teacher pool (teacher_dp) usually differs from that of the training side (train_dp). LoongSage bridges the two with a **bidirectional rearrangement**, using `train_dp_ranks` + `seq_index` throughout to record the origin of each trajectory:

1. **Split by train dp**: `_fetch_rollout_data` maps training shards to teacher DP according to the three cases `train_dp == / > / < teacher_dp`.
2. **Grouped forward by teacher**: bucket by `teacher_idx`, and each teacher forwards in parallel.
3. **Merge by train dp**: teacher outputs are distributed back to each training rank according to `train_dp_ranks`, and the training side then merges by `seq_index` (`merge_rollout_batch`), restoring alignment with the original trajectories.

In addition, the data splitting (`split_traj_group_by_dp`) guarantees that each DP rank receives the same proportion of each data source, and that **the proportions of the different data sources within each minibatch match the overall proportions**, avoiding distribution bias when training on a mix of multiple data sources.

## Configuration Examples

**Single teacher**:
```yaml
data_sources:
  - dataset: { prompt_data_path: /path/train.parquet }
    teacher_name: "math_expert"

opd:
  enable: true
  pg_ratio: 0
  gkd_ratio: 1              # pure distillation
  gkd_kl_method: full_kl
  teacher_nodes: 1
  teacher_gpus_per_node: 8
  teachers:
    - name: "math_expert"
      hf_path: /path/OpenThinker3-7B
  model: { tensor_model_parallel_size: 4, pipeline_model_parallel_size: 2 }
```

**Teacher weights from a Megatron dist checkpoint**: set `dist_ckpt_path` to use a
LoongSage training run's output directly, without exporting it to HF format first.
`hf_path` stays required — the bridge builds the Megatron model from its
`config.json`, which a `dist_ckpt` directory does not contain — but its weights are
not read. The two must describe the same architecture; a mismatch surfaces as a
load error, not silently.

```yaml
  teachers:
    - name: "math_expert"
      hf_path: /path/OpenThinker3-7B      
      # weights: one concrete dist_ckpt dir, you pick the step.
      dist_ckpt_path: /path/run/train_step_100/dist_ckpt
```

`dist_ckpt_path` may point at a model whose TP/PP/EP differs from `opd.model`: the
weights are re-sharded on load. If the path is set but is not a usable dist
checkpoint, startup fails rather than falling back to `hf_path`. With `full_kl` /
`full_jsd` the student-side teacher `lm_head` is read from the same checkpoint —
likewise, it raises on an unusable `dist_ckpt_path` instead of reading `hf_path`.

**Multiple teachers (different data sources routed to different teachers)**:
```yaml
data_sources:
  - dataset: { prompt_data_path: /path/math.parquet }
    teacher_name: "math_expert"
  - dataset: { prompt_data_path: /path/code.parquet }
    teacher_name: "code_expert"

opd:
  enable: true
  pg_ratio: 0.5            # mix of PG + GKD
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

## Running Example

OPD is launched by specifying the configuration with `--config-name`; see [conf/qwen3_30b_a3b/mopd_h20_1node.yaml](../../conf/qwen3_30b_a3b/mopd_h20_1node.yaml) for an example config:

```bash
python -m coda.controller.trainer --config-name qwen3_30b_a3b/mopd_h20_1node
```
