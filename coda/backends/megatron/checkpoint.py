"""Checkpoint manager for Megatron-Core distributed checkpointing."""

import os
import re
import random
import logging
import time
import numpy as np
import torch
import torch.distributed as dist
import transformer_engine as te
from torch.distributed.checkpoint import FileSystemReader

from megatron.core import dist_checkpointing, parallel_state as mpu
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
    FullyParallelSaveStrategyWrapper,
)
from megatron.core.dist_checkpointing.serialization import (
    TorchDistSaveShardedStrategy,
    TorchDistLoadShardedStrategy,
)
from megatron.core.dist_checkpointing.mapping import ShardedObject
from megatron.core.tensor_parallel import get_cuda_rng_tracker
from nvidia_resiliency_ext.checkpointing.async_ckpt.core import AsyncCallsQueue

CPU_SHM_MODE = True
# cpu_shm_mode=True: preload copies GPU tensors into CPU shared memory in the training
# process before handing off to the async worker, avoiding CUDA IPC / pinned-memory fork
# corruption. Must match TorchDistSaveShardedStrategy(cpu_shm_mode=True) below.
async_calls = AsyncCallsQueue(cpu_shm_mode=CPU_SHM_MODE)
from megatron.core.utils import unwrap_model

logger = logging.getLogger(__name__)


def save_checkpoint(model, optimizer, scheduler, ckpt_path, async_save=False,
                    finalize_fn=None, use_distributed_optimizer=False, optimizer_sharding_type=None):
    """Save model, optimizer, scheduler and RNG states to a distributed checkpoint."""
    start_time = time.time()
    if async_save:
        async_calls.maybe_finalize_async_calls(blocking=True)
    # Unwrap DDP module wrapper if present
    model = unwrap_model(model)
    metadata = _build_sharded_state_dict_metadata(use_distributed_optimizer, optimizer_sharding_type)
    sharded_state_dict = _build_model_state_dict(model, metadata=metadata)
    # Optimizer sharded state dict (with metadata for MCore >= 0.14)
    sharded_state_dict["optimizer"] = optimizer.sharded_state_dict(sharded_state_dict, metadata=metadata)
    sharded_state_dict["scheduler"] = scheduler.state_dict()
    sharded_state_dict["rng_state"] = _get_rng_state()
    state_dict_time = time.time()
    async_save_request = _save_dist_checkpointing(
        sharded_state_dict, ckpt_path, async_save=async_save, content_metadata=metadata,
    )
    save_ckpt_time = time.time()
    if async_save:
        if finalize_fn:
            async_save_request.add_finalize_fn(finalize_fn)
        async_calls.schedule_async_request(async_save_request)
        _release_staging_tensors(async_save_request)
    else:
        # Wait for distributed save to complete across all ranks
        dist.barrier()
        if finalize_fn:
            finalize_fn()
            # Ensure rank 0's tracker file write is visible to all ranks
            dist.barrier()
    logger.info(f"finish async save request, rank={dist.get_rank()}, async_save={async_save}, "
            f"optimizer_sharding_type={optimizer_sharding_type}, "
            f"generate state dict time {state_dict_time - start_time:.2f} sec,"
            f"generate request time {save_ckpt_time - state_dict_time:.2f} sec,"
            f"schedule reqeust time {time.time() - save_ckpt_time:.2f} sec,"
            f"total time {time.time() - start_time:.2f} sec")


def _release_staging_tensors(async_request):
    """Clear the training-side references to save staging tensors after scheduling.

    The scheduled request's ``preload_fn`` is
    ``partial(preload_tensors, (checkpoint_dir, data_to_pass), True)`` where
    ``data_to_pass = (identifier, (sep_hint, cached_tensor_data,
    uncached_tensor_data, byte_io_data, thread_count, storage_plan))``.
    ``cached_tensor_data`` holds either the /dev/shm staging tensors (cpu_shm_mode)
    or the GPU tensor references (IPC mode); either way it is pinned on the training
    side until the request is finalized, which coda does lazily at the next save.
    The persistent worker already has its own handles (preload_q.join() has returned,
    so D2H is complete) and the finalize path (retrieve_write_results) does not read
    these tensors, so clearing the lists in place is safe. It lets /dev/shm (cpu_shm)
    or GPU memory (IPC) be reclaimed as soon as the worker finishes writing instead
    of at the next (lazy) finalize.
    """
    preload_fn = getattr(async_request, "preload_fn", None)
    args = getattr(preload_fn, "args", None)
    if not args:
        return
    try:
        data_to_pass = args[0][1]
        structure = data_to_pass[1]
        cached_tensor_data, uncached_tensor_data = structure[1], structure[2]
    except (IndexError, TypeError):
        # Layout changed in an upstream version; skip rather than crash the save.
        logger.warning("Could not locate save staging tensors to release; skipping.")
        return
    released, released_bytes = 0, 0
    for data in (cached_tensor_data, uncached_tensor_data):
        if isinstance(data, tuple) and len(data) == 2 and isinstance(data[1], list):
            for t in data[1]:
                if isinstance(t, torch.Tensor):
                    released += 1
                    released_bytes += t.nbytes
            data[1].clear()
    logger.info(
        f"Released {released} staging tensors ({released_bytes / (1024**3):.2f} GB) "
        f"after scheduling async save, rank={dist.get_rank()}"
    )


@torch.no_grad()
def load_checkpoint(model, optimizer, scheduler, ckpt_dir):
    """Load checkpoint and restore model, optimizer, scheduler and RNG states."""
    logger.info(f"Loading checkpoint from {ckpt_dir}")

    # Register optimizer classes for safe deserialization (PyTorch 2.6+ defaults
    # torch.load to weights_only=True). Must be done before load_content_metadata,
    # which reads common.pt containing the optimizer object.
    torch.serialization.add_safe_globals([torch.optim.AdamW])
    torch.serialization.add_safe_globals([te.pytorch.optimizers.fused_adam.FusedAdam])

    metadata = dist_checkpointing.load_content_metadata(checkpoint_dir=ckpt_dir)
    assert metadata is not None, "Checkpoint metadata should not be None"

    model = unwrap_model(model)
    # Build template sharded_state_dict with same key structure as save
    sharded_state_dict = _build_model_state_dict(model, metadata=metadata)

    sharded_state_dict["optimizer"] = optimizer.sharded_state_dict(
        sharded_state_dict, is_loading=True, metadata=metadata
    )
    sharded_state_dict["scheduler"] = scheduler.state_dict()

    # Check if parallel topology changed by inspecting checkpoint's rng_state shard keys
    load_rng = _can_load_rng_state(ckpt_dir)
    if load_rng:
        sharded_state_dict["rng_state"] = _get_rng_state()

    # Load from disk
    loaded_state_dict = _load_dist_checkpointing(sharded_state_dict, ckpt_dir)

    # Restore model, optimizer, scheduler and RNG states
    for i, m in enumerate(model):
        model_key = f"model{i}" if len(model) > 1 else "model"
        m.load_state_dict(loaded_state_dict[model_key])
    optimizer.load_state_dict(loaded_state_dict["optimizer"])
    scheduler.load_state_dict(loaded_state_dict["scheduler"])
    if load_rng:
        _load_rng_states(loaded_state_dict["rng_state"])
    else:
        logger.warning(
            "Parallel topology changed (PP or TP size mismatch), skipping RNG state restoration. "
            "Training will continue with fresh RNG states."
        )

    logger.info(f"Successfully loaded checkpoint from {ckpt_dir}")

@torch.no_grad()
def load_model_weights(model, ckpt_dir):
    """Load only model weights from a distributed checkpoint.

    Unlike :func:`load_checkpoint`, this skips optimizer / scheduler / RNG
    state (analogous to Megatron's ``no_load_optim`` / ``no_load_rng``). Used to
    load a frozen reference model into the live GPU model without disturbing the
    student's optimizer state.
    """
    logger.info(f"Loading model weights from {ckpt_dir}")

    # Same registration as load_checkpoint: load_content_metadata reads common.pt,
    # which holds the optimizer object, and PyTorch 2.6+ defaults torch.load to
    # weights_only=True. Needed here too because add_safe_globals is process-global
    # and callers such as the teacher worker never go through load_checkpoint.
    torch.serialization.add_safe_globals([torch.optim.AdamW])
    torch.serialization.add_safe_globals([te.pytorch.optimizers.fused_adam.FusedAdam])

    metadata = dist_checkpointing.load_content_metadata(checkpoint_dir=ckpt_dir)
    assert metadata is not None, "Checkpoint metadata should not be None"

    model = unwrap_model(model)
    sharded_state_dict = _build_model_state_dict(model, metadata=metadata)

    loaded_state_dict = _load_dist_checkpointing(sharded_state_dict, ckpt_dir)

    for i, m in enumerate(model):
        model_key = f"model{i}" if len(model) > 1 else "model"
        m.load_state_dict(loaded_state_dict[model_key])

    logger.info(f"Successfully loaded model weights from {ckpt_dir}")


@torch.no_grad()
def load_tensor_from_checkpoint(ckpt_dir, candidate_keys):
    """Read one whole (unsharded) tensor out of a Megatron dist checkpoint.

    Tries ``candidate_keys`` in order and returns the first one present, so
    callers can express fallbacks (e.g. Megatron ties ``output_layer.weight`` to
    ``embedding.word_embeddings.weight`` and then only stores the latter).

    Collective-free, so it is safe to call from a subset of ranks (e.g. the last
    pipeline stage only): the plain TorchDistLoadShardedStrategy reads with
    no_dist=True, and validate_access_integrity=False skips the WORLD
    all_gather_object -- every rank claims the whole tensor, so that check would
    fail anyway. Same rationale as Megatron's own load_plain_tensors, which we
    avoid here because it materializes the entire checkpoint on every caller.

    Returns ``(matched_key, cpu_tensor)`` in the checkpoint's stored dtype.
    """
    torch.serialization.add_safe_globals([torch.optim.AdamW])
    torch.serialization.add_safe_globals([te.pytorch.optimizers.fused_adam.FusedAdam])

    strategy = TorchDistLoadShardedStrategy()
    available = dist_checkpointing.load_tensors_metadata(
        ckpt_dir, sharded_strategy=strategy
    )
    key = next((k for k in candidate_keys if k in available), None)
    if key is None:
        raise KeyError(
            f"None of {list(candidate_keys)} found in {ckpt_dir}; the checkpoint "
            f"holds {len(available)} tensors, e.g. {list(available)[:10]}"
        )

    loaded = dist_checkpointing.load(
        {key: available[key]},
        ckpt_dir,
        sharded_strategy=strategy,
        validate_access_integrity=False,
    )
    tensor = loaded[key]
    logger.info(
        f"Loaded '{key}' {tuple(tensor.shape)} {tensor.dtype} from {ckpt_dir}"
    )
    return key, tensor


def _build_sharded_state_dict_metadata(use_distributed_optimizer=False, optimizer_sharding_type=None):
    """Build metadata dict for sharded_state_dict versioning."""
    metadata = {}
    if use_distributed_optimizer:
        metadata["distrib_optim_sharding_type"] = optimizer_sharding_type

    metadata["singleton_local_shards"] = False
    metadata["chained_optim_avoid_prefix"] = True
    return metadata


def _build_model_state_dict(model, metadata):
    """Build model sharded_state_dict, handling VPP chunks.

    ``model`` must already be DDP-unwrapped by the caller.
    """
    state_dict = {}
    for i, m in enumerate(model):
        model_key = f"model{i}" if len(model) > 1 else "model"
        state_dict[model_key] = m.sharded_state_dict(metadata=metadata)
    return state_dict


def _can_load_rng_state(ckpt_dir):
    """Check if rng_state in the checkpoint matches current parallel topology.

    ShardedObject keys follow the pattern 'rng_state/shard_{pp_rank}.{tp_rank}_{pp_size}.{tp_size}'.
    If the saved PP/TP size differs from the current topology, the shard keys won't match
    and loading will fail. In that case, we skip rng_state restoration.
    """
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    tp_size = mpu.get_tensor_model_parallel_world_size()

    try:
        reader = FileSystemReader(ckpt_dir)
        ckpt_metadata = reader.read_metadata()
        for key in ckpt_metadata.state_dict_metadata.keys():
            match = re.match(r"rng_state/shard_(\d+)\.(\d+)_(\d+)\.(\d+)", key)
            if match:
                saved_pp_size = int(match.group(3))
                saved_tp_size = int(match.group(4))
                if saved_pp_size != pp_size or saved_tp_size != tp_size:
                    logger.info(
                        f"RNG state topology mismatch: checkpoint has PP={saved_pp_size}/TP={saved_tp_size}, "
                        f"current is PP={pp_size}/TP={tp_size}"
                    )
                    return False
                return True
    except Exception as e:
        logger.warning(f"Failed to read checkpoint metadata for RNG state check: {e}")
        return False

    # No rng_state key found in metadata, cannot load
    logger.info("No rng_state found in checkpoint metadata")
    return False


def _get_rng_state():
    """Capture all RNG states and wrap as ShardedObject for distributed checkpoint."""
    rng_state = {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.random.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(),
        "rng_tracker_states": get_cuda_rng_tracker().get_states(),
    }

    # Wrap as ShardedObject for distributed checkpoint
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    tp_size = mpu.get_tensor_model_parallel_world_size()

    return ShardedObject(
        "rng_state",
        rng_state,
        (pp_size, tp_size),
        (pp_rank, tp_rank),
        replica_id=mpu.get_data_parallel_rank(with_context_parallel=True),
    )


def _load_rng_states(rng_state):
    """Restore all RNG states from a checkpoint snapshot."""
    random.setstate(rng_state["python_rng_state"])
    np.random.set_state(rng_state["numpy_rng_state"])
    torch.random.set_rng_state(rng_state["torch_rng_state"])
    torch.cuda.set_rng_state(rng_state["cuda_rng_state"])
    get_cuda_rng_tracker().set_states(rng_state["rng_tracker_states"])
    logger.info("Restored all RNG states from checkpoint.")


def _load_dist_checkpointing(sharded_state_dict, ckpt_dir):
    """Load a distributed checkpoint using FullyParallel strategy."""
    load_strategy = FullyParallelLoadStrategyWrapper(
        TorchDistLoadShardedStrategy(),
        mpu.get_data_parallel_group(with_context_parallel=True),
    )
    return dist_checkpointing.load(
        sharded_state_dict, ckpt_dir, sharded_strategy=load_strategy
    )


def _save_dist_checkpointing(sharded_state_dict, ckpt_path, async_save=False, content_metadata=None):
    """Save a distributed checkpoint using FullyParallel strategy."""
    os.makedirs(ckpt_path, exist_ok=True)
    save_strategy = FullyParallelSaveStrategyWrapper(
        TorchDistSaveShardedStrategy(cpu_shm_mode=CPU_SHM_MODE),
        mpu.get_data_parallel_group(with_context_parallel=True),
    )
    return dist_checkpointing.save(sharded_state_dict, ckpt_path,
        sharded_strategy=save_strategy,
        async_sharded_save=async_save,
        validate_access_integrity=True,
        content_metadata=content_metadata
    )
