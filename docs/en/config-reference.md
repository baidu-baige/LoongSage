# Config Reference

This document describes every field in [conf/default.yaml](../../conf/default.yaml). Use it as a reference when writing your own experiment yaml (via Hydra `defaults: [default]`).

- **Do not edit `default.yaml` directly** — it holds global defaults. Put user overrides in a separate yaml.
- `???` marks a Hydra *missing* value: it must be set explicitly in your yaml or on the command line.
- The "Default" column reflects the current values in `default.yaml`. The "Constraints" column summarizes rules that the code actually enforces or that couple fields together, not style suggestions.
- The document is organized by top-level section, one parameter table per section. Section [11. Cross-field constraints cheatsheet](#11-cross-field-constraints-cheatsheet) collects the dependencies that span multiple sections.

## 1. Top-level runtime

Global run mode, randomness, checkpoint paths and logging.

| Field | Default | Description | Constraints |
| --- | --- | --- | --- |
| `run_mode` | `default` | Entry mode. `default` runs rollout + train; `train-only` trains from existing data under `rollout_data_path`; `rollout-only` only samples and dumps to disk. | Fully-async mode (`fully_async.enable=true`) only supports `default`. |
| `seed` | `42` | Global random seed for data shuffle, sampling and init. | — |
| `colocate` | `true` | Whether rollout and trainer share the same GPUs (time-multiplexed). | Must be `false` for fully-async mode. When `true`, total rollout GPUs and total teacher GPUs must be ≤ trainer GPUs — see [11.10](#1110-colocate-resource-relationships). |
| `checkpoint_path` | `???` | Training model checkpoint **base** directory: both the output dir for checkpoints saved during training and the resume input. The step to resume from is found via `latest_checkpointed_iteration.txt` inside it. | Required. |
| `hf_model_path` | `???` | HuggingFace-format initial weight path; used to initialize both actor and the inference engine. | Required; the tokenizer is also loaded from here. |
| `rollout_data_path` | `./rollout_data` | Output dir for `rollout-only`, input dir for `train-only`. | Only used when `run_mode != default`. |
| `ref_dist_ckpt_path` | `null` | Reference model weights from a megatron dist checkpoint (preferred). Names **one concrete `dist_ckpt` dir**, e.g. `/path/to/run/train_step_100/dist_ckpt`, not a base dir. | When `algorithm.ref_kl.enable=true`, at least one of this / `ref_hf_model_path` must be set. Validated at startup; an unusable path fails the run instead of falling back to `ref_hf_model_path`. |
| `ref_hf_model_path` | `null` | Reference model HF weight dir. Used when `ref_dist_ckpt_path` is unset — an alternative source, not a fallback for a broken one. | Same as above. |
| `total_steps` | `3000` | Total training steps. | — |
| `log_level` | `INFO` | Global log level. | One of `DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL`. |

## 2. Fully-async (`fully_async`)

Switches that turn on the "rollout runs concurrently with training" pipeline. See [Fully Async Mode](fully-async-mode.md) for the full design and tuning guide.

| Field | Default | Description | Constraints |
| --- | --- | --- | --- |
| `fully_async.enable` | `false` | Enable the fully-async pipeline (rollout and trainer on disjoint GPU pools). | Requires `colocate: false`, `run_mode: default`, `rollout.sampler.num_oversample: 0`. |
| `fully_async.sliding_window` | `no-window` | Dispatch / collection strategy for prompt groups. One of `no-window` / `window-gated` / `windowed-fifo`. | See the "sliding-window strategies" section of fully-async-mode.md. |
| `fully_async.stale_steps` | `0` | Extra pipeline capacity, in units of training step: `capacity = int(B × (1 + stale_steps))`, where `B = num_prompts_per_step`. | `>= 0`, may be fractional. Larger values are more off-policy; pair with IS correction / M2PO / OPSM. |

## 3. Data sources (`data_source` / `data_sources`)

`data_source` defines the default fields for a single data source; `data_sources` is the actual list used at runtime. Each list element inherits from `data_source` and may override individual fields. Fully-async mode currently supports exactly one data source.

### 3.1 `dataset` sub-node

| Field | Default | Description | Constraints |
| --- | --- | --- | --- |
| `dataset.prompt_data_path` | `???` | Prompt dataset path. Supports `.jsonl` / `.parquet`, plus an optional row-slice suffix like `data.jsonl@[0:1000]`. | Required. |
| `dataset.eval_prompt_data_path` | `null` | Eval split path, same format as above. | `null` disables eval; also requires `rollout.eval.interval > 0` for eval to actually run. |
| `dataset.max_prompt_len` | `null` | Maximum characters allowed per prompt. Rows exceeding this are dropped at load time. | `null` disables the filter. This is a **character** length, not tokens. |
| `dataset.input_key` | `???` | Column holding the prompt text or message list. | Required. |
| `dataset.label_key` | `???` | Column holding the ground-truth label. | Required; set to `null` explicitly if unused. |
| `dataset.metadata_key` | `metadata` | Column holding per-row metadata dicts. Passed to the agent unchanged as `trajectory["metadata"]`. Schema is free-form — put whatever per-sample info your agent/reward need (e.g. `{"difficulty": "hard"}`). | Column value must be a dict; absent means an empty dict. |
| `dataset.shuffle` | `true` | Whether to shuffle the training split. | The eval split is never shuffled. |
| `dataset.data_pre_processor` | `null` | Name of a raw-record pre-processor applied before messages are built (register with `@register_data_pre_processor`; see [data_pre_processor.py](../../coda/data_factory/data_pre_processor.py)). Built-in: `gsm8k`. | `null` disables pre-processing. |
| `dataset.buffer_replay_strategy` | `null` | Name of the buffer-replay-strategy function (register with `@register_buffer_replay_strategy`; see [data_source.py](../../coda/data_factory/data_source.py)). | Only used by `RolloutDataSourceWithBuffer`; `null` is equivalent to `fifo`. |

### 3.2 agent / reward / others

| Field | Default | Description | Constraints |
| --- | --- | --- | --- |
| `agent.name` | `null` | Agent implementation name. `null` means single-turn (no agent); required for multi-turn. | Multi-turn agents must be registered via the registry; see [Custom Agent Development Guide](custom-agent.md). |
| `reward.name` | `???` | Reward function name; must be registered in the registry — see [Custom Reward Function Development Guide](custom-reward.md). | Required. Other fields under `reward` are forwarded verbatim to the reward constructor. |
| `teacher_name` | `""` | Teacher name used by this data source (paired with OPD). | Empty string means no teacher; must match some `opd.teachers[].name`. |
| `max_response_len_per_trajectory` | `32768` | Cap on response-area tokens (LLM replies + tool responses) per attempt. Single-turn uses this as request `max_tokens`; multi-turn agents receive this and pick their own per-call `max_tokens`. | Must be `<= max_total_tokens - prompt_len` on the inference engine. |
| `num_prompts_per_step` | `64` | Number of prompt groups `B` consumed per training step. | `B × N` must be divisible by `trainer.mini_batch_size`, where `N = num_trajectories_per_prompt`; additionally `B % (dp_size × num_mini_batch) == 0` — see [11.9](#119-dp-and-batch-sizes). |
| `num_trajectories_per_prompt` | `8` | Trajectories per prompt `N`, i.e. the GRPO group size. | Group-level advantage normalization requires equal group sizes. |
| `completion_params` | `{}` | Sampling params forwarded to the inference engine (e.g. `top_p`, `top_k`, `temperature`). | Empty dict means use the engine defaults. This dict is spread **last** into the request body, so a `temperature` here overrides `trainer.temperature`. |

`data_sources` defaults to `[${data_source}]`, i.e. a single data source. For multiple data sources, spell out the list explicitly in your experiment yaml — each element **independently** inherits from `data_source`.

## 4. Rollout (inference & sampling)

Controls the SGLang inference cluster, the sampler, filters and eval.

### 4.1 Top level and SGLang

| Field | Default | Description | Constraints |
| --- | --- | --- | --- |
| `rollout.partial` | `false` | Whether to abort in-flight requests at step boundaries (preserve already-generated tokens; resume from the same point next step). | Recommended `true` for long trajectories / multi-turn agents. `mask_offpolicy_in_partial_rollout` depends on this. |
| `rollout.mask_offpolicy_in_partial_rollout` | `false` | When restoring a partial trajectory, zero out `loss_mask` for response tokens generated under the previous weights. | Only valid when `rollout.partial: true`. |
| `rollout.use_fault_tolerance` | `false` | Whether to launch a `RolloutHealthMonitor` background thread per SGLang replica group to detect faulty engines and call `recover_faulty_engines()` before training starts. | Engine-level health monitoring and recovery. Independent of `retry_limit` (trajectory-level retry) — they can be toggled separately. |
| `rollout.retry_limit` | `3` | Maximum number of attempts per trajectory inside AgentFlow (including the first attempt): if an attempt throws and is marked `FAILED`, AgentFlow starts a new attempt until success or the limit is exhausted. | Always in effect, independent of `use_fault_tolerance`. Only after all attempts fail is the trajectory finally FAILED and possibly filtered out by `rollout.filter.status`. |
| `rollout.backend` | `sglang` | Inference backend. | Currently only `sglang` is supported. |
| `rollout.num_gpus_per_node` | `8` | GPUs per rollout node. | Multiplied by `sglang_replicas[*].num_nodes` to get each replica's total GPUs. |
| `rollout.env_vars` | `{}` | Environment variables injected into rollout workers. | — |
| `rollout.sglang_args` | See below | Dict forwarded to SGLang `ServerArgs`. | Field names track the SGLang version. The table below only lists fields that `default.yaml` overrides. |

`rollout.sglang_args` overrides:

| Field | Default | Description |
| --- | --- | --- |
| `mem_fraction_static` | `0.8` | Fraction of GPU memory reserved by SGLang for KV cache. |
| `disable_cuda_graph` | `true` | Disable CUDA graph (helps with hot weight updates and debugging). |
| `disable_custom_all_reduce` | `true` | Disable custom all-reduce (compatibility with the training-side comms). |
| `load_format` | `dummy` | Weight loading strategy. `dummy` builds the model shell and relies on hot updates; use `auto` for a normal offline load. |

### 4.2 `sglang_replicas` (PD disaggregation)

`sglang_replicas` describes how the inference cluster is split by role; each role can own dedicated GPUs. By default only the `regular` group is enabled (i.e. no PD split).

> **Note**: Full PD disaggregation is not supported yet. For now keep `prefill.num_nodes` and `decode.num_nodes` at `0` and use `regular` only.

| Field | Default | Description | Constraints |
| --- | --- | --- | --- |
| `sglang_replicas.regular.num_nodes` | `1` | Number of "unified prefill+decode" replicas (each replica owns `num_gpus_per_replica` GPUs). | Mutually exclusive with `prefill` / `decode`. |
| `sglang_replicas.regular.num_gpus_per_replica` | `8` | GPUs per regular replica. | Must divide evenly against `rollout.num_gpus_per_node` or span nodes cleanly. |
| `sglang_replicas.prefill.num_nodes` | `0` | Number of dedicated prefill replicas. | Full PD disaggregation is unsupported; keep at `0`. |
| `sglang_replicas.prefill.num_gpus_per_replica` | `8` | GPUs per prefill replica. |  |
| `sglang_replicas.decode.num_nodes` | `0` | Number of dedicated decode replicas. |  |
| `sglang_replicas.decode.num_gpus_per_replica` | `8` | GPUs per decode replica. |  |
| `sglang_replicas.decode.sglang_args` | `{disable_cuda_graph: false}` | Overrides `rollout.sglang_args` for the decode replica (decode usually wants CUDA graph on). | Only affects the decode replica. |

### 4.3 sampler / filter / eval

| Field | Default | Description | Constraints |
| --- | --- | --- | --- |
| `rollout.sampler.name` | `dynamic` | Sampler implementation. Only `dynamic` (allows oversample + refill) is supported. | For fully-async mode `num_oversample` must be 0 (see below). |
| `rollout.sampler.num_oversample` | `0` | Extra prompt groups dispatched during synchronous dynamic sampling. | Must be `0` in fully-async mode; extra capacity is provided by `fully_async.stale_steps`. |
| `rollout.sampler.refill_ratio` | `2` | Amplification factor when refilling prompts from the buffer: `refill_num = need × refill_ratio`. | `dynamic` sampler only. |
| `rollout.sampler.max_refill_count` | `256` | Maximum prompts per refill call. | `dynamic` sampler only. |
| `rollout.sampler.timeout` | `7200` | Max seconds to wait for one rollout. On timeout, the current rollout is aborted and the next one starts. | Unit: seconds. |
| `rollout.filter.status` | enabled (empty dict) | Drop the entire group if it contains any FAILED trajectory. | Set to `false` to disable. Keys with the same name overwrite; new keys append. |
| `rollout.filter.reward` | disabled | Drop groups with identical rewards across all trajectories (no contrastive signal). | Set `reward: {}` to enable; `reward: false` to explicitly disable. |
| `rollout.eval.interval` | `-1` | Run eval every N steps; `<=0` disables it. | Also requires `data_source.dataset.eval_prompt_data_path` to be non-null. |
| `rollout.eval.temperature` | `null` | Sampling temperature used during eval. | `null` reuses `trainer.temperature`. |

## 5. AgentFlow

Agent-side router, tokenizer and sandbox. Multi-turn agents forward requests to SGLang through `AgentFlow.router`, which also manages the tokenizer and code sandbox.

### 5.1 router

| Field | Default | Description | Constraints |
| --- | --- | --- | --- |
| `agentflow.dump_trajectory_path` | `""` | When non-empty, dump every trajectory to this directory for later replay / analysis. | — |
| `agentflow.router.ip` | `""` | Router listen IP. Empty means auto-resolve to the local routable IP at runtime. | Usually no manual configuration needed. |
| `agentflow.router.port` | `0` | Router listen port. `0` means pick a free port at runtime. | — |
| `agentflow.router.accumulate_reasoning` | `true` | Accumulate reasoning tokens into the context across multi-turn dialogues. | If disabled, reasoning content only affects the current turn. |
| `agentflow.router.rollout_worker_load_threshold` | `32` | Per-worker load threshold; above it the router load-balances. | Smaller values balance more aggressively at higher scheduling overhead. |
| `agentflow.router.proxy_timeout_seconds` | `1800` | Router → SGLang per-request timeout in seconds. | Must be `>=` the longest single-attempt generation time. |
| `agentflow.router.abort_timeout_seconds` | `600` | Seconds to wait when aborting in-flight requests at a step boundary. | Paired with `rollout.partial`. |
| `agentflow.router.max_connections` | `512` | Max concurrent connections accepted by the router. | Tune together with `sglang_args.max_running_requests`. |
| `agentflow.router.middleware.parser` | — | Placeholder for a middleware parser; users can extend with their own. | Empty by default. |

### 5.2 tokenizer

| Field | Default | Description | Constraints |
| --- | --- | --- | --- |
| `agentflow.tokenizer.custom_chat_template_path` | `null` | Path to a Jinja2 chat-template file. Relative paths resolve against `conf/` (e.g. `chat_template/my_model.jinja`, a file you add yourself); absolute paths are used as-is. | To control thinking mode, use `generation_prompt_kwargs` — do NOT rely on string substitution. |
| `agentflow.tokenizer.generation_prompt_kwargs` | `{}` | kwargs forwarded to the tokenizer's `apply_chat_template`. | Common use: enable thinking with `enable_thinking: true`. |
| `agentflow.tokenizer.manager.mode` | `thread` | Tokenizer concurrency mode. | Currently `thread` is the primary mode. |
| `agentflow.tokenizer.manager.num_workers` | `8` | Number of concurrent tokenize workers. | Tune with rollout concurrency and CPU count. |

### 5.3 sandbox

| Field | Default | Description | Constraints |
| --- | --- | --- | --- |
| `agentflow.sandbox.type` | `k8s` | Sandbox type. | `k8s` is the primary supported backend. |
| `agentflow.sandbox.command_exec_timeout_seconds` | `600` | Timeout for a single command inside the sandbox (seconds). | — |
| `agentflow.sandbox.sandbox_creation_timeout_seconds` | `600` | Sandbox creation timeout (seconds). | — |
| `agentflow.sandbox.working_dir` | `/rl-sandbox` | Working directory inside the sandbox. | — |
| `agentflow.sandbox.kubeconfig` | `k8s/kubeconfig.yaml` | Path to the k8s kubeconfig. | Relative paths resolve against the project `conf/` directory; absolute paths are used as-is. |
| `agentflow.sandbox.pod_manifest_path` | `k8s/pod_manifest.yaml` | Path to the sandbox pod template. | Relative paths resolve against the project `conf/` directory; absolute paths are used as-is. |

## 6. Tracking

| Field | Default | Description | Constraints |
| --- | --- | --- | --- |
| `tracking.project_name` | `default` | Project name for experiments (e.g. MLflow project). | — |
| `tracking.experiment_name` | `default` | Experiment name. | — |
| `tracking.tracking_backend` | `console` | Metrics backend, e.g. `console` / `mlflow` / `wandb` / custom. | — |
| `tracking.mlflow_tracking_uri` | `""` | MLflow server URI. | Required when `tracking_backend: mlflow`. |

## 7. Megatron training backend

The `megatron` node controls parallelism, optimizer and LR scheduling for the actor. Field names follow upstream Megatron-LM.

### 7.1 model (parallelism and precision)

| Field | Default | Description | Constraints |
| --- | --- | --- | --- |
| `megatron.model.bf16` | `true` | Train in bf16. | Mutually exclusive with `fp16`. |
| `megatron.model.fp16` | `false` | Train in fp16. | Mutually exclusive with `bf16`. |
| `megatron.model.fp8` | `null` | FP8 training recipe; `null` disables it. | Requires Hopper+; used together with `fp8_recipe` / `fp8_param`. |
| `megatron.model.fp8_recipe` | `null` | Detailed FP8 recipe. | Only used when `fp8` is enabled. |
| `megatron.model.fp8_param` | `false` | Store parameters in FP8 as well. | Memory savings vs. precision trade-off. |
| `megatron.model.tensor_model_parallel_size` | `1` | Tensor parallel size (TP). | Total trainer GPUs must be divisible by `TP × PP × CP`; validated at startup — see [11.9](#119-dp-and-batch-sizes). |
| `megatron.model.pipeline_model_parallel_size` | `1` | Pipeline parallel size (PP). | Same as above; participates in `TP × PP × CP`. |
| `megatron.model.virtual_pipeline_model_parallel_size` | `null` | Virtual PP (VPP) size, used to shrink PP bubbles. | Requires PP > 1. |
| `megatron.model.context_parallel_size` | `1` | Context parallel size (CP, sequence sharding). | Paired with `cp_partition_mode`; participates in `TP × PP × CP`. |
| `megatron.model.cp_partition_mode` | `zigzag` | CP partitioning strategy (e.g. `zigzag`). | Only used when CP > 1. |
| `megatron.model.expert_model_parallel_size` | `1` | Expert parallel size (EP) for MoE. | Keep at 1 for non-MoE models. Total trainer GPUs must be divisible by `ETP × EP × PP`. |
| `megatron.model.expert_tensor_parallel_size` | `null` | TP size for the MoE experts. | `null` means "same as `tensor_model_parallel_size`"; that value is the `ETP` in the formula above. |
| `megatron.model.overlap_p2p_comm` | `false` | Overlap PP P2P communication with computation. | Only worthwhile when PP > 1. |
| `megatron.model.moe_grouped_gemm` | `true` | Use grouped GEMM to accelerate MoE. | Enable for MoE models. |
| `megatron.model.moe_shared_expert_overlap` | `false` | Overlap MoE shared-expert compute with dispatch. | Known issue: can produce NaN on some MoE models, keep it off unless verified (RCA doc `moe-shared-expert-overlap-nan-rca` pending). |

The commented-out `recompute_*` triplet (`granularity` / `method` / `num_layers`) enables activation recomputation — enable it in your own yaml when needed.

> **Note**: `megatron.model` transparently supports **every** field of Megatron's `TransformerConfig`; the table above only lists commonly tuned entries. Most model-structural parameters (number of layers, hidden size, num heads, rotary, norm type, etc.) are auto-populated by megatron-bridge based on `hf_model_path` — users do not need to declare them. In yaml you only override **training-side** switches such as parallelism, precision, recomputation and communication overlap.

### 7.2 ddp_config / optimizer / scheduler

| Field | Default | Description | Constraints |
| --- | --- | --- | --- |
| `megatron.ddp_config.use_distributed_optimizer` | `true` | Use the Megatron distributed optimizer (ZeRO-1-style sharding). | Interacts with `optimizer_sharding_type`. |
| `megatron.ddp_config.overlap_param_gather` | `false` | Overlap param all-gather with forward. | Has a memory cost. |
| `megatron.ddp_config.overlap_grad_reduce` | `false` | Overlap grad reduce-scatter with backward. | — |
| `megatron.ddp_config.grad_reduce_in_fp32` | `true` | Reduce gradients in fp32 for numerical stability. | Keep the default for best precision. |
| `megatron.optimizer.lr` | `1.0e-6` | Peak learning rate. | Common RLHF starting point. |
| `megatron.optimizer.weight_decay` | `0.01` | AdamW weight decay. | — |
| `megatron.optimizer.optimizer_cpu_offload` | `false` | Offload optimizer state to CPU. | Useful when GPU memory is tight for large models. |
| `megatron.optimizer.optimizer_offload_fraction` | `1.0` | Fraction of optimizer state to offload; `1.0` = all. | Only used when `optimizer_cpu_offload: true`. |
| `megatron.scheduler.lr_warmup_steps` | `0` | Warmup steps. | — |
| `megatron.scheduler.lr_decay_steps` | `1` | Total LR decay steps. | Combined with `lr_decay_style` to shape the curve. |
| `megatron.scheduler.lr_decay_style` | `constant` | LR decay style, e.g. `constant` / `linear` / `cosine`. | — |
| `megatron.scheduler.wd_incr_steps` | `0` | Weight-decay ramp steps. | Usually left at 0. |
| `megatron.scheduler.wd_incr_style` | `constant` | Weight-decay ramp style. | — |
| `megatron.keep_fp32_weights` | `{}` | Parameter-name substrings (fuzzy match) whose FP32 master weights are kept; the value (bool) indicates whether that layer's output should also stay FP32. | Example: `output_layer: true`. |
| `megatron.optimizer_sharding_type` | `dp_reshardable` | Optimizer-state sharding scheme: `dp_reshardable` / `fully_reshardable`. | Controls checkpoint compatibility and DP re-sharding — see [Model Loading and Saving](model-checkpointing.md). |

> **Note**: all three sub-nodes are forwarded as `**kwargs` to the corresponding upstream Megatron structure. The table only lists commonly tuned entries; any other upstream field can be added by name in yaml:
> - `megatron.ddp_config` → `megatron.core.distributed.DistributedDataParallelConfig`
> - `megatron.optimizer`  → `megatron.core.optimizer.OptimizerConfig` (`torch.dtype` fields are auto-converted from strings)
> - `megatron.scheduler`  → constructor kwargs of `megatron.core.optimizer_param_scheduler.OptimizerParamScheduler`. If `max_lr` / `min_lr` / `init_lr` / `start_wd` / `end_wd` are not set explicitly, they default from `megatron.optimizer.lr` / `min_lr` / `weight_decay`.

## 8. OPD (On-Policy Distillation)

OPD (On-Policy Distillation) attaches one or more teacher models to the RL loop and mixes a configurable KL term with policy gradient, yielding `L = (1-gkd_ratio) × [A - pg_ratio × KL_token] + gkd_ratio × L_GKD`. See [On-Policy Distillation](on-policy-distillation.md) for details.

### 8.1 Top-level

| Field | Default | Description | Constraints |
| --- | --- | --- | --- |
| `opd.enable` | `false` | Enable OPD. | Requires a non-empty `teachers` list. |
| `opd.pg_ratio` | `0.0` | KL coefficient in the policy-gradient branch of the mixture. | No upper bound in code; at least one of this / `gkd_ratio` must be `> 0`. |
| `opd.gkd_ratio` | `0.0` | Weight on the GKD branch of the mixture. | Must be `<= 1`; at least one of this / `pg_ratio` must be `> 0`; `pg_ratio > 0` and `gkd_ratio == 1` are mutually exclusive. |
| `opd.pg_kl_method` | `k1` | KL estimator for the PG branch. | One of `k1` / `topk_kl` / `full_kl` / `topk_jsd` / `full_jsd`. |
| `opd.gkd_kl_method` | `topk_kl` | KL estimator for the GKD branch. | One of `k2` / `k3` / `topk_kl` / `full_kl` / `topk_jsd` / `full_jsd`. |
| `opd.topk` | `256` | Top-K logits retained by `topk_*` methods. | Only relevant to top-k methods. |
| `opd.teacher_nodes` | `1` | Number of teacher nodes. | Shared across all teachers. `teacher_nodes × teacher_gpus_per_node` must be divisible by `opd.model`'s `TP × PP × CP` — see [11.11](#1111-opd-teacher-parallelism). |
| `opd.teacher_gpus_per_node` | `8` | GPUs per teacher node. | Same as above; the resulting `teacher_dp` must be mutually divisible with `train_dp`. |
| `opd.teachers` | `[]` | Teacher list; each entry has `name`, `hf_path` and an optional `dist_ckpt_path`. | Required when OPD is enabled; `name` must match the data source's `teacher_name`. `hf_path` is always required — it supplies the model structure. `dist_ckpt_path` loads the weights from a Megatron dist checkpoint instead of `hf_path`'s safetensors; it names **one concrete `dist_ckpt` dir**, unlike the top-level `checkpoint_path`. Validated at startup; an unusable path fails the run instead of falling back to `hf_path`. |

### 8.2 `opd.model` / `memory_pool`

`opd.model` mirrors `megatron.model` field-for-field (`bf16` / `fp16` / `fp8*` / TP / PP / VPP / CP / EP / ETP / `overlap_p2p_comm` / `moe_grouped_gemm`) and drives the teacher-side megatron forward pass.

| Field | Default | Description | Constraints |
| --- | --- | --- | --- |
| `opd.memory_pool.backend` | `null` | Teacher memory-pool backend; `null` disables it. | Mainly used when co-locating teachers to reuse GPU memory. |
| `opd.env_vars` | `{}` | Environment variables for teacher workers. | — |

## 9. Algorithm

The `algorithm` node bundles advantage estimation, policy loss, off-policy safeguards, regularizers and loss aggregation — it is the main entry point for training algorithm configuration. See [Training Algorithms](training-algorithms.md).

### 9.1 Main switches

| Field | Default | Description | Constraints |
| --- | --- | --- | --- |
| `algorithm.advantage_estimator` | `grpo` | Advantage estimator name. | Registered via the registry; built-in: `grpo`. |
| `algorithm.advantage_norm_mode` | `group_zscore` | Advantage normalization mode. | `none` / `group_mean` / `group_zscore` / `batch_mean` / `batch_zscore`; `group_*` requires equal group sizes. |
| `algorithm.policy_loss` | `grpo` | Policy loss name. | Built-in: `grpo` / `gspo`; see [GSPO paper](https://arxiv.org/pdf/2507.18071). |
| `algorithm.loss_agg_mode` | `token-mean` | Loss aggregation mode. | `token-mean` / `seq-mean-token-mean`. |
| `algorithm.entropy_coef` | `0.0` | Entropy regularization coefficient. | When non-zero, subtracts `entropy_coef × entropy` from the loss. |
| `algorithm.clip_ratio_low` | `0.2` | Lower clip ratio for GRPO / GSPO. | Combined with `clip_ratio_high` for asymmetric clipping (DAPO clip-higher). |
| `algorithm.clip_ratio_high` | `0.28` | Upper clip ratio. | Same as above. |
| `algorithm.clip_ratio_c` | `10.0` | Dual-clip upper bound for negative advantages. | Only active for `policy_loss: grpo`. |

### 9.2 `is_correction` (importance-sampling correction)

| Field | Default | Description | Constraints |
| --- | --- | --- | --- |
| `algorithm.is_correction.enable` | `false` | Enable IS correction. | Mutually exclusive with `trainer.use_rollout_log_probs: true`. |
| `algorithm.is_correction.action` | `clip` | Out-of-bound handling: `clip` (clamp) or `mask` (drop from both numerator and denominator). | — |
| `algorithm.is_correction.level` | `token` | Weight granularity: `token` / `sequence` / `geometric`. | — |
| `algorithm.is_correction.lower_bound` | `???` | IS weight lower bound. | Required; suggested `0.5` for token-level, `0.9999` for geometric. |
| `algorithm.is_correction.upper_bound` | `???` | IS weight upper bound. | Required; suggested `2.0` for token-level, `1.0001` for geometric. |

### 9.3 `opsm` / `m2po`

| Field | Default | Description | Constraints |
| --- | --- | --- | --- |
| `algorithm.opsm.enable` | `false` | Enable Off-Policy Sequence Masking: drop a sequence's gradient when `advantage < 0` and its sequence-level KL exceeds `delta`. | Composable with IS correction / M2PO. |
| `algorithm.opsm.delta` | `0.1` | OPSM sequence-level KL threshold. | — |
| `algorithm.m2po.enable` | `false` | Enable M2PO (Second-Moment Trust Policy Optimization): mask the tokens with the largest `(log(π_old/π_rollout))²` until the second moment of the remaining tokens is below `threshold`. | Mutually exclusive with `trainer.use_rollout_log_probs: true`. |
| `algorithm.m2po.threshold` | `0.04` | M2PO second-moment threshold. | — |

### 9.4 `ref_kl` (reference-model KL)

| Field | Default | Description | Constraints |
| --- | --- | --- | --- |
| `algorithm.ref_kl.enable` | `false` | Add `coef × KL(π_θ ‖ π_ref)` to the loss. | Requires `ref_dist_ckpt_path` or `ref_hf_model_path`. |
| `algorithm.ref_kl.coef` | `0.001` | KL coefficient. | — |
| `algorithm.ref_kl.kl_type` | `k3` | KL estimator: `k1` / `k2` / `k3`. | — |
| `algorithm.ref_kl.use_unbiased_kl` | `false` | Multiply per-token KL by `exp(log_probs - old_log_probs)` (DeepSeek-V3.2 style). | — |
| `algorithm.ref_kl.update_interval` | `-1` | `<=0` keeps ref frozen; `>0` refreshes ref from the current actor every N steps. | Requires `enable: true`. |

## 10. Trainer

The `trainer` node controls batch organization, precision, timeouts and checkpoint saving on the training side.

| Field | Default | Description | Constraints |
| --- | --- | --- | --- |
| `trainer.backend` | `megatron` | Training backend. | Only `megatron` is supported today. |
| `trainer.num_nodes` | `1` | Number of training nodes. | Multiplied by `num_gpus_per_node` for the total training GPU count; `dp_size = num_nodes × num_gpus_per_node / (TP × PP × CP)` (**EP excluded**), and the total must divide that product evenly — validated at startup. See [11.9](#119-dp-and-batch-sizes). |
| `trainer.num_gpus_per_node` | `8` | GPUs per training node. | Same as above; participates in `dp_size`. |
| `trainer.use_rollout_log_probs` | `false` | Use `rollout_log_probs` returned by the inference engine instead of re-computing `old_log_probs` on the training side. | Mutually exclusive with IS correction and M2PO. |
| `trainer.use_rollout_routing_replay` | `false` | Replay MoE routing recorded on the inference side. | Used to align rollout / training MoE routing. |
| `trainer.use_fp32_lm_head` | `false` | Compute the LM head in FP32. | Enable in precision-sensitive settings. |
| `trainer.temperature` | `1.0` | Sampling temperature. Eval rounds use `rollout.eval.temperature` instead when it is not `null`. | This value is overridden by `data_source.completion_params.temperature`, which is spread last into the request body. |
| `trainer.mini_batch_size` | `64` | Trajectories `M` consumed per optimizer step. `num_mini_batch = (B × N) / M`. | `(B × N) % M == 0`; and every data source must satisfy `num_prompts_per_step % (dp_size × num_mini_batch) == 0`. Both are validated at startup, after which `M % dp_size == 0` holds automatically. See [11.9](#119-dp-and-batch-sizes). |
| `trainer.micro_batch_size` | `8` | Samples per forward per GPU. | Only when `use_dynamic_batch_size: false` is `mini_batch_size % micro_batch_size == 0` required; with dynamic batching this field is ignored. |
| `trainer.max_tokens_per_gpu` | `16440` | Per-GPU token cap for dynamic batching. | Only used when `use_dynamic_batch_size: true`. |
| `trainer.use_dynamic_batch_size` | `false` | Pack micro-batches by token count. | When enabled, `micro_batch_size` is ignored. |
| `trainer.deterministic_mode` | `false` | Bit-exact reproducibility: force deterministic NCCL / TransformerEngine / cuBLAS kernels. | Noticeably slower; use only for debugging / precision alignment. |
| `trainer.nccl_timeout_minutes` | `null` | NCCL comms timeout in minutes; `null` uses the framework default. | — |
| `trainer.gloo_timeout_minutes` | `null` | GLOO comms timeout in minutes; `null` uses the framework default. | Consider setting explicitly for large-scale cross-node CPU comms. |
| `trainer.save_freq` | `-1` | Save a checkpoint every N training steps; `<=0` disables periodic saves. | — |
| `trainer.async_save` | `true` | Save checkpoints asynchronously (saver thread runs concurrently with training). | See [Model Loading and Saving](model-checkpointing.md). |
| `trainer.save_checkpoint` | `true` | Whether to save the megatron distributed checkpoint. | If disabled, recovery only works when `save_hf: true`. |
| `trainer.save_hf` | `false` | Also export HuggingFace-format weights to `train_step_{step}/hf_model`. | Convenient for downstream inference. |
| `trainer.env_vars` | `{}` | Environment variables for training workers. | — |

## 11. Cross-field constraints cheatsheet

The dependencies below span multiple sections and are the most common pitfalls when writing an experiment yaml. The three most frequent are [11.9 DP and batch sizes](#119-dp-and-batch-sizes), [11.10 colocate resource relationships](#1110-colocate-resource-relationships) and [11.11 OPD teacher parallelism](#1111-opd-teacher-parallelism).

### 11.1 Reference model

- When `algorithm.ref_kl.enable=true`, **at least one** of `ref_dist_ckpt_path` / `ref_hf_model_path` must be provided. If both are set, the megatron dist checkpoint (`ref_dist_ckpt_path`) wins.
- `ref_dist_ckpt_path` names one concrete `dist_ckpt` dir (`<run>/train_step_<N>/dist_ckpt`), not a base dir. It is validated at startup, and a set-but-unusable path never falls back to `ref_hf_model_path`.
- `algorithm.ref_kl.update_interval > 0` refreshes the ref from the actor every N steps — a "moving ref" KL penalty. `<= 0` keeps the ref frozen.

### 11.2 Train/rollout log-prob and off-policy safeguards

- `trainer.use_rollout_log_probs: true` uses the inference engine's `rollout_log_probs` in place of the training-side `old_log_probs` recomputation. When enabled, you **cannot** simultaneously turn on:
  - `algorithm.is_correction.enable: true`
  - `algorithm.m2po.enable: true`
- `algorithm.opsm.*` is compatible with all of the above.

### 11.3 Fully-async mode

- To enable `fully_async.enable: true` you must also set:
  - `colocate: false`
  - `run_mode: default`
  - `rollout.sampler.num_oversample: 0` (extra capacity is provided by `fully_async.stale_steps`)
- Only a single entry in `data_sources` is supported today.
- `B × N` must be divisible by `trainer.mini_batch_size`; `M % N == 0` is strongly recommended (otherwise you cannot form complete groups). Here `B = num_prompts_per_step`, `N = num_trajectories_per_prompt`, `M = mini_batch_size`.

### 11.4 Partial rollout

- `rollout.mask_offpolicy_in_partial_rollout: true` is only meaningful when `rollout.partial: true` (otherwise there are no resumed partial trajectories to mask).
- Long trajectories or multi-turn agents typically benefit from `rollout.partial: true`, which cuts the long-tail wait at step boundaries.

### 11.5 SGLang PD disaggregation

- **Full PD disaggregation is not supported yet.** For now keep the single `regular` shape, i.e. both `sglang_replicas.prefill.num_nodes` and `decode.num_nodes` at `0`.

### 11.6 Eval

- Eval only actually runs when `rollout.eval.interval > 0` **and** `data_source.dataset.eval_prompt_data_path` is non-null.
- `rollout.eval.temperature: null` means eval reuses `trainer.temperature`.

### 11.7 Precision combinations

- `megatron.model.bf16` and `fp16` are mutually exclusive — pick one.
- Even with `megatron.model.fp8` enabled, keep `bf16` or `fp16` as the primary precision; `fp8_param: true` additionally stores weights in FP8.
- `megatron.ddp_config.grad_reduce_in_fp32: true` is the numerically stable recommendation — usually keep it on.

### 11.8 Checkpoint saving

- `trainer.save_freq <= 0` disables periodic saves. Keep at least one of `trainer.save_checkpoint` / `trainer.save_hf` set to `true`, otherwise nothing is written (unless the run is one-shot).
- `trainer.async_save: true` requires extra memory for the background save queue — see [Model Loading and Saving](model-checkpointing.md).

### 11.9 DP and batch sizes

This is the easiest group to get wrong. The four rules form a chain:

1. **Definition of DP size**: `dp_size = trainer.num_nodes × trainer.num_gpus_per_node / (TP × PP × CP)`. Note that **EP does not participate**; the MoE expert side has its own DP.
2. **Total trainer GPUs must be divisible by `TP × PP × CP`**, otherwise `dp_size` is meaningless. Validated at startup; the error reads
   `trainer GPUs (8) must be divisible by TP*PP*CP (3)`.
3. **`(B × N) % M == 0`**, where `B = num_prompts_per_step`, `N = num_trajectories_per_prompt` and `M = trainer.mini_batch_size`; with multiple data sources, `B × N` is summed over all of them. This yields `num_mini_batch = (B × N) / M`.
4. **`B % (dp_size × num_mini_batch) == 0` for every data source.** This one is most often missed because it couples batch configuration to GPU parallelism. The error reads
   `data_sources[0] num_prompts_per_step (64) is not divisible by (dp_size * num_mini_batch) = (8 * 2) = 16`.

`M % dp_size == 0` needs no separate configuration: rule 4 gives `B_i = dp_size × num_mini_batch × m_i`, and substituting into `M = (B × N) / num_mini_batch` shows `M` is always a multiple of `dp_size`.

Separately, `micro_batch_size` only participates when `use_dynamic_batch_size: false` (requiring `M % micro_batch_size == 0`); with dynamic batching, micro-batches are packed by token count and this field is ignored.

### 11.10 colocate resource relationships

With `colocate: true` the placement group is sized from the **trainer's GPU count** only, and rollout and teacher reuse those same GPUs. Therefore:

- total rollout GPUs (`Σ sglang_replicas[*].num_nodes × rollout.num_gpus_per_node`) must be ≤ total trainer GPUs;
- with OPD enabled, total teacher GPUs (`opd.teacher_nodes × opd.teacher_gpus_per_node`) must also be ≤ total trainer GPUs.

With `colocate: false` the three counts are additive: the Ray cluster must supply trainer + rollout + teacher GPUs simultaneously.

### 11.11 OPD teacher parallelism

With OPD enabled there are four divisibility rules on the teacher side, all validated at startup:

- `opd.teacher_nodes × opd.teacher_gpus_per_node > 0`;
- that product must be divisible by `opd.model`'s `TP × PP × CP`, which yields `teacher_dp`;
- `teacher_dp` and `len(opd.teachers)` must be **mutually divisible** (either may divide the other, but they cannot be coprime-ish);
- `teacher_dp` and `train_dp` (the `dp_size` from [11.9](#119-dp-and-batch-sizes)) must be **mutually divisible**.

---

## Related docs

- [Training Algorithms](training-algorithms.md): algorithm components and execution order
- [Fully Async Mode](fully-async-mode.md): fully-async pipeline design and tuning
- [Model Loading and Saving](model-checkpointing.md): checkpoints and async saving
- [Custom Extensions](custom-extensions.md): one directory for every extension point
- [Custom Agent Development Guide](custom-agent.md) / [Custom Reward Function Development Guide](custom-reward.md) / [Custom Sandbox Development Guide](custom-sandbox.md): the three main extension points
- [Custom RL Algorithm Development Guide](custom-algorithm.md) / [Custom KL Algorithm Development Guide](custom-kl.md) / [Custom Sliding Window Strategy Development Guide](custom-sliding-window.md): algorithm-side extension points
- [On-Policy Distillation](on-policy-distillation.md): OPD in detail
- [Train-Inference Consistency](train-inference-consistency.md): training / inference alignment







