# Model Loading and Saving

This document describes how LoongSage loads initial weights during training, saves checkpoints, and exports HuggingFace weights.

## 1. Directory Layout

LoongSage organizes all artifacts of a training run under the training model's `checkpoint_path` as the root directory:

```text
checkpoint_path/
├── latest_checkpointed_iteration.txt   # tracker file recording the most recent successfully saved step
├── train_step_10/
│   ├── dist_ckpt/                       # distributed checkpoint (model + optimizer + scheduler + RNG)
│   │                                    # also the value you pass to ref_dist_ckpt_path /
│   │                                    # opd.teachers[].dist_ckpt_path
│   ├── hf_model/                        # HuggingFace-format export (only when hf export is enabled)
│   └── data_source/                     # data source cursors + unconsumed prompt buffer snapshot, used for resuming
│       └── global_dataset_state_dict_ds{ds_idx}.pt
├── train_step_20/
│   └── ...

rollout_data/                           # trajectory data for rollout-only / train-only modes, a separate directory configured by rollout_data_path
└── step_{step}.pt
```

The tracker file is read **only by the training model** when resuming from its own `checkpoint_path`: only the step it records is treated as a valid, resumable checkpoint. Under async save, the tracker is updated only after the write has fully landed on disk, so the step read from the tracker is guaranteed to be complete and usable. The read-only weight sources — the reference model and the OPD teachers — do not consult it; you pick the step yourself.

## 2. Key Parameters

| Parameter | Default | Description |
| --- | ---: | --- |
| `checkpoint_path` | required | Training model checkpoint **base** directory (training output and resume input; the step comes from the tracker file). |
| `hf_model_path` | required | HuggingFace model path. Required by both loading paths, as it provides the model structure information. |
| `rollout_data_path` | `./rollout_data` | Directory where trajectory data is written in rollout-only / train-only modes. |
| `trainer.save_freq` | `-1` | Save frequency (in steps). `<= 0` means no saving. |
| `trainer.async_save` | `true` | Whether to use async save, avoiding blocking the main training loop on disk IO. |
| `trainer.save_checkpoint` | `true` | Whether to save the distributed checkpoint (used for resuming training). |
| `trainer.save_hf` | `false` | Whether to additionally export HuggingFace-format weights. |
| `megatron.optimizer_sharding_type` | `dp_reshardable` | Distributed optimizer sharding format, see [4.4](#44-optimizer-sharding-format-megatronoptimizer_sharding_type). |
| `ref_dist_ckpt_path` / `ref_hf_model_path` | `null` | Reference model path (used for ref-KL); pick one of the two. Unlike the training model's `checkpoint_path`, `ref_dist_ckpt_path` names one concrete `train_step_N/dist_ckpt` dir — no tracker file, no latest-step scan. |
| `opd.teachers[].dist_ckpt_path` | `null` | Per-teacher weight source for OPD; `hf_path` is still required alongside it. Same semantics as `ref_dist_ckpt_path`: one concrete `dist_ckpt` dir, not a base dir. |

## 3. Model Loading

LoongSage supports two loading paths and picks one automatically at training startup:

- **Load from checkpoint format**: taken automatically when a valid checkpoint exists under `checkpoint_path`, restoring the complete training state including model weights, optimizer, LR scheduler, and RNG. If the PP/TP size changed when resuming, RNG restoration is skipped with a warning, while the remaining state is still loaded correctly.
- **Load from HF format**: when no usable checkpoint exists under `checkpoint_path`, HuggingFace weights are loaded from `hf_model_path` as the initial parameters. This is the default entry point for a brand-new training run.

> `hf_model_path` must be provided under both loading paths — even when restoring from a checkpoint, LoongSage needs to read the HF config to determine the model structure.

When resuming data sources, the current datasource is matched by the stored `prompt_data_path` field (rather than by positional index), so data sources can be reordered on resume without loading the wrong state.

## 4. Model Saving

### 4.1 When Saving Happens

A save is triggered automatically every `trainer.save_freq` steps. In addition, the final training step (`total_steps`) always forces a save regardless of whether it aligns with `save_freq`, guaranteeing that a complete, usable checkpoint always exists once training finishes. After a successful save, the tracker file is updated to point at that step.

### 4.2 Sync vs Async Save

- **Async (`async_save: true`, default)**: the main training loop is not blocked on disk IO, and the next step can start training immediately. At most one async save is in flight at any moment — if the previous async write has not finished, the new save blocks until it completes before being scheduled. At the beginning of each train step, LoongSage makes a non-blocking attempt to finalize the previous async save (updating the tracker file), and at the end of training it performs a blocking flush to ensure the last save fully lands on disk.
- **Sync (`async_save: false`)**: training pauses and resumes only after the checkpoint has been fully written. Suitable for setups sensitive to GPU memory or `/dev/shm` usage, where the save buffers should be released immediately.

### 4.3 Save Formats: Checkpoint and HuggingFace

Controlled by the two switches `trainer.save_checkpoint` and `trainer.save_hf`, which can be combined freely; when both are disabled, saving is skipped.

- **Distributed checkpoint (`save_checkpoint`, enabled by default)**: written to `train_step_{step}/dist_ckpt/`, stored as per-rank shards (Megatron-Core `TorchDist` format), containing model + optimizer + scheduler + RNG. This is the **only source for resuming training**, but it is tightly coupled to the parallel topology (TP/PP/EP/DP) and cannot be used directly for inference deployment. In fully_async mode, the save additionally captures in-flight data still unconsumed in the pipeline via `snapshot_pipeline_buf()`, ensuring nothing is lost after resume.
- **HuggingFace format (`save_hf`, disabled by default)**: written to `train_step_{step}/hf_model/`, aggregated across ranks into complete `safetensors` weights + config + tokenizer. Convenient for offline evaluation, deployment to inference frameworks such as vLLM/SGLang, and model sharing. It contains **weights only and cannot be used to resume training**; the export requires cross-rank aggregation and is therefore slower than a checkpoint save.

### 4.4 Optimizer Sharding Format: `megatron.optimizer_sharding_type`

When the distributed optimizer is enabled (`megatron.ddp_config.use_distributed_optimizer=true`, enabled by default), this controls the on-disk form of the optimizer state inside the checkpoint:

| Value | Layout | Saving | Resume Flexibility |
| --- | --- | --- | --- |
| `dp_reshardable` (default) | Follows the distributed optimizer's internal bucket layout | Fully parallel, no cross-rank communication | Reshardable along the **DP dimension** only |
| `fully_reshardable` | Aggregated and unflattened into the original model parameter shapes | Requires a gather along the DP dimension | Reshardable along **all of TP/PP/EP/DP** |

Use the default when the topology stays unchanged for the whole training run; switch to `fully_reshardable` when you need to resume with a different TP/PP/EP topology, at the cost of slower saves and higher peak memory. The field is ignored when the distributed optimizer is not enabled.
