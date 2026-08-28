# Run Guide
## Multi-Node Training Setup

Multi-node training requires Ray to run on every participating machine.
[`examples/start_ray_cluster.sh`](../../examples/start_ray_cluster.sh) takes the **head-node IP**
as its argument and auto-detects head vs. worker from the local node, so the head node and every
worker node run the same command:

```bash
# Run from the repository root on every participating machine, <master-ip> = head node IP
bash examples/start_ray_cluster.sh <master-ip>
```

After all nodes have joined, verify the cluster from the head node:

```bash
ray status
```

Notes:

- Nodes default to 8 GPUs; for 4-GPU nodes such as GB200, override with `NUM_GPUS=4`.
- For multi-node training, ensure every node can reach the models and datasets at identical
  paths — download them on each node or mount shared storage. The commands below use
  `/root/...`; update the paths consistently when you use different directories.
- Single-node training (`colocate` presets) does not need a Ray cluster; go straight to the
  tasks below.

## Example tasks

### Task 1: Run DAPO with Qwen3-30B-A3B

Download the model and dataset:

```bash
hf download Qwen/Qwen3-30B-A3B --local-dir /root/Qwen3-30B-A3B
hf download Haitao999/DAPO-Math-17k-unique --repo-type=dataset --local-dir /root/DAPO-Math-17k-unique
```

Launch the single-node 8-GPU preset [`qwen3_30b_a3b/dapo_h20_1node`](../../conf/qwen3_30b_a3b/dapo_h20_1node.yaml):

```bash
bash examples/start.sh qwen3_30b_a3b/dapo_h20_1node \
  hf_model_path=/root/Qwen3-30B-A3B \
  data_source.dataset.prompt_data_path=/root/DAPO-Math-17k-unique \
  checkpoint_path=/root/ckpt/math_expert \
  trainer.save_freq=10
```

`trainer.save_freq` defaults to `-1`, which saves no checkpoint at all, so it is enabled explicitly here. The checkpoints land in `/root/ckpt/math_expert/train_step_<N>/dist_ckpt`.

### Task 2: Run BCP

Install FAISS (the other dependencies ship with the training image), then download the embedding model, the BrowseComp-Plus prompts, corpus, and dense indexes:

```bash
python3 -m pip install faiss-cpu

hf download Qwen/Qwen3-Embedding-8B --local-dir /root/Qwen3-Embedding-8B
hf download Tevatron/browsecomp-plus --repo-type=dataset --include "data/*.parquet" --local-dir /root/browsecomp-plus/raw
hf download Tevatron/browsecomp-plus-corpus --repo-type=dataset --local-dir /root/browsecomp-plus/corpus
hf download Tevatron/browsecomp-plus-indexes --repo-type=dataset --local-dir /root/browsecomp-plus/indexes
```

BCP depends on a retrieval service. Build its cache once from the official dense shards and the corpus, then start the service. `--gpu_id 7` below uses part of that GPU's memory.

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

Validate the service with a health check and one real query:

```bash
curl -sS http://127.0.0.1:9000/health
curl -sS http://127.0.0.1:9000/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"queries":["Tribeca Festival Gotham Week"],"topk":3,"return_scores":true}'
```

Once it is ready, launch the single-node 8-GPU preset [`qwen3_30b_a3b/bcp_h20_1node`](../../conf/qwen3_30b_a3b/bcp_h20_1node.yaml):

```bash
bash examples/start.sh qwen3_30b_a3b/bcp_h20_1node \
  hf_model_path=/root/Qwen3-30B-A3B \
  data_source.dataset.prompt_data_path=/root/browsecomp-plus/raw \
  checkpoint_path=/root/ckpt/bcp_expert \
  trainer.save_freq=10
```

If the retrieval service is not on the default `http://127.0.0.1:9000`, override `data_source.agent.retrieval_service_url=...`. A localhost URL works only when the retrieval service and training run in the same pod. The checkpoints land in `/root/ckpt/bcp_expert/train_step_<N>/dist_ckpt`.

### Task 3: Run MOPD Multi-Teacher Distillation

Use the two experts trained in Task 1 and Task 2 as teachers: the Task 1 checkpoint as the MATH teacher and the Task 2 checkpoint as the BCP teacher. Configure the matching teacher for the trajectories produced by each of the two data sources (a data source's `data_sources[].teacher_name` is matched by name against `opd.teachers[].name`) and distill into a single student model.

The BCP data source still needs the retrieval service running:

```bash
./examples/bcp/run_retrieval_server.sh \
  --data_dir /root/browsecomp-plus/corpus/data \
  --model /root/Qwen3-Embedding-8B \
  --dense_cache /root/browsecomp-plus/browsecomp_dense_cache.pkl \
  --gpu_id 7 \
  --port 9000
```

Pick a step from each run, point at the corresponding `dist_ckpt` directories, and launch [`qwen3_30b_a3b/mopd_h20_1node`](../../conf/qwen3_30b_a3b/mopd_h20_1node.yaml):

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

The `hf_path` of `opd.teachers` is always required, but only to build the Megatron model structure from its `config.json`; the weights come from `dist_ckpt_path`, and the two must describe the same architecture. For teacher parallelism, KL methods, and other knobs, see [On-Policy Distillation](on-policy-distillation.md).

### Task 4: Run SWE with DeepSeek-V4-Flash

Download the FP8 model and dataset first, then convert the model to BF16 weights for training:

```bash
hf download sgl-project/DeepSeek-V4-Flash-FP8 --local-dir /root/DeepSeek-V4-Flash-FP8
hf download R2E-Gym/R2E-Gym-Subset --repo-type=dataset --local-dir /root/R2E-Gym-Subset

python examples/convert_dsv4_fp8_to_bf16.py \
  --input-fp8-hf-path /root/DeepSeek-V4-Flash-FP8 \
  --output-bf16-hf-path /root/DeepSeek-V4-Flash-BF16
```

After conversion, prepare the kubeconfig required by the K8s sandbox used for SWE rollouts. Place the kubeconfig at `k8s/kubeconfig.yaml` (the default `agentflow.sandbox.kubeconfig` path), or override it on the command line. Then launch the 8-node H20 preset [`dsv4_flash_bf16/swe_h20_8node`](../../conf/dsv4_flash_bf16/swe_h20_8node.yaml): bring up the Ray cluster on every node and submit the training job from the head node only.

```bash
# Run the same command on every machine
bash examples/start_ray_cluster.sh <master-ip>

# Once the cluster is ready, run this on the head node only
bash examples/start.sh dsv4_flash_bf16/swe_h20_8node \
  checkpoint_path=/root/ckpt/dsv4_swe \
  hf_model_path=/root/DeepSeek-V4-Flash-BF16 \
  data_source.dataset.prompt_data_path=/root/R2E-Gym-Subset \
  agentflow.sandbox.kubeconfig=/path/to/kubeconfig.yaml
```

`checkpoint_path` and `hf_model_path` must live on storage that every node can reach through the same path.

### Task 5: Run the OpenCode Black-Box Agent

Download the model and the prepared R2E-Gym dataset. The dataset already points to OpenCode-enabled sandbox images, so no additional data conversion is required.

```bash
hf download Qwen/Qwen3-Coder-30B-A3B-Instruct --local-dir /root/Qwen3-Coder-30B-A3B-Instruct
hf download LoongSage/R2E-Gym R2E_Gym_Subset_opencode.parquet \
  --repo-type=dataset --local-dir /root/R2E-Gym-OpenCode
```

Prepare the K8s sandbox kubeconfig, placing it at `k8s/kubeconfig.yaml` or overriding its path in the launch command. Then run the same Ray command on all four H20 nodes:

```bash
bash examples/start_ray_cluster.sh <master-ip>
```

Once all nodes have joined, launch [`qwen3_coder_30b_a3b/opencode_h20_4node`](../../conf/qwen3_coder_30b_a3b/opencode_h20_4node.yaml) from the head node:

```bash
bash examples/start.sh qwen3_coder_30b_a3b/opencode_h20_4node \
  checkpoint_path=/root/ckpt/opencode \
  hf_model_path=/root/Qwen3-Coder-30B-A3B-Instruct \
  data_source.dataset.prompt_data_path=/root/R2E-Gym-OpenCode/R2E_Gym_Subset_opencode.parquet \
  agentflow.sandbox.kubeconfig=/path/to/kubeconfig.yaml
```

Tip: Each preset inherits from [`conf/default.yaml`](../../conf/default.yaml), and subdirectories such as [`conf/qwen3_30b_a3b/`](../../conf/qwen3_30b_a3b/) hold presets for a specific model and machine type. For more available configurations, see [`conf/`](../../conf/).

## Monitoring

### Checking run status and metrics

Training runs in the background and writes its log to `log/trainer_<timestamp>.log` (`start.sh`
prints the file name on launch):

```bash
tail -f log/trainer_*.log
```

`tracking.tracking_backend` defaults to `console`, with metrics printed straight to the log.
Watch for:

| Metric | Meaning |
| --- | --- |
| `rollout/completed_count` | trajectories collected this step; equals `num_prompts_per_step × num_trajectories_per_prompt` |
| `rollout/reward_mean` | mean reward |
| `train/pg_loss`, `train/entropy` | training loss and entropy |
| `timing/*` | per-phase timings |

Check cluster status with `ray status`. For centralized visualization, switch to MLflow and
friends: `tracking.tracking_backend=mlflow tracking.mlflow_tracking_uri=http://...`.

### Stopping training

Run `pkill -f coda.controller.trainer` on the node that launched the training (on every node
for multi-node), then `ray stop` cleans up the Ray cluster.

## Common Hydra configuration

### Data sources

Fields covered in this section:

| Config | Default | Description |
| --- | --- | --- |
| `data_source` / `data_sources` | `[${data_source}]` | single-source template / the list actually used |
| `data_source.dataset.prompt_data_path` | required | prompt data file or directory; `@[start:stop]` slicing supported |
| `data_source.dataset.input_key` / `label_key` | required | prompt / label column names |
| `data_source.num_prompts_per_step` | `64` | prompt groups dispatched per data source per step |
| `data_source.num_trajectories_per_prompt` | `8` | trajectories sampled per prompt group |
| `data_source.agent.name` | `null` (single-turn) | agent implementation; required for multi-turn |
| `data_source.reward.name` | required | reward function (parameters per reward plugin) |
| `data_source.max_response_len_per_trajectory` | `32768` | response-area token cap per trajectory |

`data_source` is the default-value template for one source, and `data_sources` is the list
actually used, expanding to `[${data_source}]` (a single source) by default. Override a single
source with the `data_source.` prefix; for multiple sources use `data_sources.<index>.` or an
explicit list. Both channels are equivalent: `key=value` on the command line (after the config
name of `start.sh`) or in an experiment yaml.

**Single source, on the command line**

```bash
bash examples/start.sh qwen3_30b_a3b/dapo_h20_1node \
  hf_model_path=/root/Qwen3-30B-A3B \
  data_source.dataset.prompt_data_path=/root/DAPO-Math-17k-unique \
  data_source.num_prompts_per_step=64 \
  data_source.num_trajectories_per_prompt=8
```

**Single source, in yaml**

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

**Multiple sources, in yaml** (write `data_sources` as an explicit list; each element can
configure its own prompt data, sampling scale, agent, and reward; fields left unset inherit the
`data_source` defaults):

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

**Multiple sources, on the command line** (override list elements by index; fields not
overridden keep their `data_source` defaults):

```bash
bash examples/start.sh qwen3_30b_a3b/dapo_h20_1node \
  hf_model_path=/root/Qwen3-30B-A3B \
  checkpoint_path=/root/ckpt/multi \
  data_sources.0.dataset.prompt_data_path=/root/browsecomp-plus/prompts/browsecomp_plus.parquet \
  data_sources.1.dataset.prompt_data_path=/root/DAPO-Math-17k-unique
```

### Metrics tracking

`tracking` decides where metrics go. Fields covered by this section:

| Config | Default | Description |
| --- | --- | --- |
| `tracking.tracking_backend` | `console` | metric backend: `console` / `mlflow` / `wandb`, lists supported |
| `tracking.mlflow_tracking_uri` | `""` | MLflow server address |
| `tracking.project_name` / `experiment_name` | `default` | MLflow experiment / run grouping |
| `tracking.wandb_args.*` | — | passed through verbatim to `wandb.Settings()` |

The default `tracking_backend=console` only prints into
`log/trainer_*.log` — fine for a quick look, but curves cannot be compared across runs; switch
to `mlflow` or `wandb` for centralized visualization. Common settings that do not depend on the
specific backend:

- `tracking.tracking_backend` also accepts a list, reporting to several backends at once:
  `'tracking.tracking_backend=[console,mlflow]'` — quote the whole argument, otherwise the
  shell swallows the square brackets. Valid values are `console` / `mlflow` / `wandb`;
  anything else fails at startup.

**Reporting to MLflow**

```bash
bash examples/start.sh qwen3_30b_a3b/dapo_h20_1node \
  hf_model_path=/root/Qwen3-30B-A3B \
  data_source.dataset.prompt_data_path=/root/DAPO-Math-17k-unique \
  tracking.tracking_backend=mlflow \
  tracking.mlflow_tracking_uri=http://<host>:<port>/ \
  tracking.project_name=math-rl \
  tracking.experiment_name=dapo-30b-a3b-lr1e6
```

`project_name` maps to the MLflow experiment, `experiment_name` to the run name under it;
both default to `default` — set them explicitly when sharing a server. `mlflow_tracking_uri`
defaults to an empty string — picking `mlflow` without filling it in will not connect to a server.

**Reporting to W&B**

```bash
bash examples/start.sh qwen3_30b_a3b/dapo_h20_1node \
  hf_model_path=/root/Qwen3-30B-A3B \
  data_source.dataset.prompt_data_path=/root/DAPO-Math-17k-unique \
  tracking.tracking_backend=wandb \
  tracking.wandb_args.base_url=https://<wandb-host> \
  tracking.wandb_args.api_key=<key>
```

`wandb_args` keys are mostly passed through to `wandb.Settings()`. `mode` takes exactly two
values: `shared` (default) reports in real time and lets several processes write the same run —
needs `base_url` + `api_key`, W&B SDK ≥ 0.19.9 and W&B Server ≥ 0.68; `offline` only writes
locally, suitable for clusters without external network.
### Checkpoints and resume

Fields covered by this section:

| Config | Default | Description |
| --- | --- | --- |
| `checkpoint_path` | required | checkpoint base dir (output + resume entry) |
| `rollout_data_path` | `./rollout_data` | trajectory hand-off dir for rollout-only / train-only |
| `trainer.save_freq` | `-1` | checkpoint frequency (steps); `<= 0` disables saving |
| `trainer.save_hf` | `false` | additionally export HuggingFace-format weights |
| `megatron.optimizer_sharding_type` | `dp_reshardable` | optimizer sharding format (decides reshard topology) |


`trainer.save_freq` defaults to `-1` (no checkpoints at all). When enabled (e.g. set to 10 in
Tasks 1 and 2), artifacts are laid out under `checkpoint_path` (see
[Model Checkpointing](model-checkpointing.md) for the full picture):

```text
checkpoint_path/
├── latest_checkpointed_iteration.txt   # tracker file: last successfully saved step
├── train_step_10/
│   ├── dist_ckpt/                       # distributed checkpoint (model + optimizer + scheduler + RNG)
│   │                                    # also what opd.teachers[].dist_ckpt_path points at in MOPD
│   ├── hf_model/                        # HuggingFace export (when trainer.save_hf=true)
│   └── data_source/                     # data-source cursor + buffer snapshot, for resume
└── ...
```

Key points:

- **Resume**: running `start.sh` again with the same `checkpoint_path` restores the full state
  (optimizer, LR scheduler, RNG, data-source cursors) from the step recorded by the tracker file,
  with no extra arguments. The final training step is always saved once even if it does not align
  with `save_freq`, so a finished run always leaves a complete checkpoint.
- **HF export**: with `trainer.save_hf=true`, `train_step_N/hf_model/safetensors` is written
  every `save_freq` steps — directly usable for offline evaluation or serving with sglang/vLLM,
  but weights only, not resumable.
- **Resume with different topology**: the default `optimizer_sharding_type=dp_reshardable`
  only supports resharding along the DP dimension; switch to `fully_reshardable` (slower saves,
  higher peak memory) to change TP/PP/EP topology on resume.

## Preset overview

Besides the presets used in the tasks above, `conf/` ships the following ready-to-run presets
(all launched with the triple `hf_model_path` + `checkpoint_path` +
`data_source.dataset.prompt_data_path`; the prompt path may be a file or a directory, and
`@[start:stop]` slices the first N rows for a quick run):

| Directory | Preset | Model | Task / algorithm | Resources |
| --- | --- | --- | --- | --- |
| `conf/qwen3_4b/` | `dapo_h800_1node` | Qwen3-4B | DAPO math | 1 node x 8 H800 |
| `conf/qwen3_30b_a3b/` | `dapo_h20_1node` | Qwen3-30B-A3B | DAPO math (Task 1) | 1 node x 8 H20 |
| | `gsm8k_h20_1node` | Qwen3-30B-A3B | GSM8K math | 1 node x 8 H20 |
| | `bcp_h20_1node` | Qwen3-30B-A3B | BCP retrieval (Task 2) | 1 node x 8 H20 |
| | `mopd_h20_1node` | Qwen3-30B-A3B | MOPD distillation (Task 3) | 1 node x 8 H20 |
| `conf/qwen3_coder_30b_a3b/` | `mini_swe_h20_4node` | Qwen3-Coder-30B-A3B | mini-SWE (R2E-Gym) | 4 nodes × H20 |
| | `opencode_h20_4node` | Qwen3-Coder-30B-A3B | OpenCode (R2E-Gym) (Task 5) | 4 nodes × H20 |
| `conf/dsv4_flash_bf16/` | `swe_h20_8node` / `swe_gb200_8node` | DeepSeek-V4-Flash-BF16 | SWE (Task 4) | 8×H20 / 8×GB200 |
| | `dapo_h20_6node` / `dapo_gb200_8node` | DeepSeek-V4-Flash-BF16 | DAPO math | 6×H20 / 8×GB200 |

## Related documents

- First run end-to-end (environment, data, verification) → [Quick Start](quick-start.md)
- Field-by-field reference and divisibility constraints → [Config Reference](config-reference.md)
- Training algorithms (PG, GRPO, GSPO, ...) → [Training Algorithms](training-algorithms.md)
- Checkpoint, load and export → [Model Checkpointing](model-checkpointing.md)
- Online policy distillation (MOPD) → [Online Policy Distillation](on-policy-distillation.md)
- Custom Agent / Reward / Sandbox → [Custom Agent](custom-agent.md) / [Custom Reward](custom-reward.md) / [Custom Sandbox](custom-sandbox.md)