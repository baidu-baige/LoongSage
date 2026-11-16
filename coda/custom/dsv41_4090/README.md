# DeepSeek V4.1 single-GPU streaming training

The controller, rollout/reward pipeline and training algorithm use LoongSage's standard
implementations. `patch.py` selects the lightweight subprocess worker and disk-based weight
sync only when `dsv41_4090.stream` is enabled.

The subprocess builds a meta-initialized model with a shared layer-weight pool and LoRA,
then wraps it in standard Megatron DDP. Frozen parameters have no gradient buffers.
It calls the stock `MegatronActorWorker.train()` and `_build_optimizer()` methods:

- `get_rollout_data` preserves segment metadata and routed experts;
- standard forward-only evaluation computes old-policy log probabilities;
- configured advantage estimators and policy losses use the existing registries;
- standard IS correction applies the configured mask/clip mode and bounds;
- `train_minibatch` uses Megatron's forward/backward scheduler and R3 hooks;
- standard optimizer and learning-rate scheduler state are saved between subprocesses.

Only model initialization, process lifetime, metric transport and LoRA export are specialized.
The LoRA forward includes a detached quantization-residual correction to approximate the
FP8 merged weights served by SGLang. This is not the general `trainer.qat` implementation.

## Configuration

Use `conf/gsm8k_stream.yaml` in this directory (see its header for preparation commands).
R3 is enabled by default; both SGLang and Megatron use the pruned expert IDs. Recorded routes
must accompany the batch: old trajectory-only dumps cannot be used for R3 validation.

The current memory layout supports one training GPU, adapter-only training, and no critic,
OPD or reference-policy snapshot. Those unsupported modes are rejected rather than ignored.
The recipe disables distributed optimizer sharding/overlap and uses a conservative packed
microbatch token budget. The standard algorithm still processes all rows, including zero-advantage
groups; it no longer skips those groups as the initial feasibility prototype did.

## Checkpoints

Each `stream_state` directory contains the full `rollout_batch.pt`, resolved `config.yaml`,
`adapter.pt`, standard Megatron optimizer/scheduler state in `optimizer.pt`, and metrics.
Optimizer state is tagged `standard_actor_v1`; the prototype's torch AdamW checkpoints are
incompatible. Start a new work directory and matching pristine serving checkpoint when
migrating from that implementation.

Model files and previous experiment directories are not modified by isolated validation runs.

## Packed microbatch budget

On the same 64-trajectory GSM8K batch (R3 enabled), sequential GPU-0 runs from
the same initial state measured the following. Timings include model construction,
old-logprob evaluation, training and merge; device memory was sampled with nvidia-smi.

| Token budget | Microbatches per forward/train pass | Training seconds | PyTorch peak allocated GiB | Device peak MiB |
| --- | --- | --- | --- | --- |
| 512 | 7 | 1209 | 10.98 | 14710 |
| 1024 | 4 | 672 | 12.96 | 16804 |
| 2048 | 2 | 362 | 17.74 | 21874 |

All runs completed with finite loss/gradients. Changing packing changes numerical
rounding and can change which tokens cross the configured IS mask thresholds; these
are performance tests, not evidence of identical optimization trajectories.

The default is **1024**, leaving room for the colocated SGLang process after offload.
Two real controller steps with this budget completed in **20.2 and 22.0 minutes**,
including rollout, with training subprocess times of 692 and 684 seconds. R3, optimizer
restoration and the intermediate 48-tensor serving reload succeeded.
2048 is faster in isolation but its device peak leaves only about 1.1 GiB on this
23028-MiB device, before accounting for a resident offloaded rollout process; it has
not been validated in colocated mode. Re-measure for longer prompts or different hardware.
