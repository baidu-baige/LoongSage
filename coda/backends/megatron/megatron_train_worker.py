"""Megatron train worker base — per-GPU Ray actor backed by Megatron-Core.

``MegatronTrainWorker`` owns everything a trainable Megatron model needs no
matter what it is trained for: process groups, model build via
Megatron-bridge, optimizer / LR scheduler, distributed checkpoint save/load and
GPU <-> CPU offload. Role-specific behavior lives in subclasses:

* :class:`~coda.backends.megatron.megatron_actor_worker.MegatronActorWorker` —
  the policy (log-probs, ref model, advantages, policy loss, weight sync to
  rollout).
* :class:`~coda.backends.megatron.megatron_critic_worker.MegatronCriticWorker` —
  the value model used by PPO.

Subclasses extend through plain overrides and ``super()``. The base exposes two
extension points only because those differences sit in the middle of
:meth:`MegatronTrainWorker.init`: :meth:`_configure_provider` (runs before DDP
wrapping) and the :attr:`CKPT_DIR_NAME` class attribute.
"""

import random
import logging
from abc import abstractmethod
from typing import override, Optional, get_type_hints
from enum import Enum
from functools import partial

import numpy as np
import torch
from omegaconf import OmegaConf

from megatron.core import parallel_state
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
from megatron.core.distributed import DistributedDataParallelConfig, finalize_model_grads
import coda.backends.megatron.monkey_patch  # noqa: F401  # apply patches before any Megatron usage
from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig as MegatronOptimizerConfig
from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.utils import get_model_config
from megatron.core.distributed import DistributedDataParallel as DDP
from coda.backends.megatron.checkpoint import async_calls
from megatron.bridge import AutoBridge
from megatron.bridge.models import GPTModelProvider

from coda.backends.train_worker import TrainWorker
from coda.utils.memory_utils import print_memory, clear_memory
from coda.utils.checkpoint_utils import (
    DIST_CKPT_DIR_NAME, find_latest_ckpt_path, update_latest, get_ckpt_dir,
)
from coda.backends.megatron.checkpoint import load_checkpoint, save_checkpoint
from coda.utils.types import to_torch_dtype
from coda.utils.tracking import configure_tracking
from coda.backends.megatron.mixed_precision import KeepFP32Module

logger = logging.getLogger(__name__)


class MegatronTrainWorker(TrainWorker):
    """Megatron-Core base for trainable per-GPU Ray actors (actor / critic)."""

    # Sub-directory of ``train_step_N/`` holding this model's dist checkpoint.
    CKPT_DIR_NAME = DIST_CKPT_DIR_NAME

    @classmethod
    def validate_config(cls, config):
        """Validate or fill the Megatron config shared by every trainable model.

        Idempotent: the actor and the critic worker both run it on the same
        driver config.
        """

        # Forced parameters
        config.megatron.model.variable_seq_lengths = True
        config.megatron.model.moe_token_dispatcher_type = "alltoall"
        config.megatron.model.calculate_per_token_loss = True
        config.megatron.model.moe_router_load_balancing_type = "none"

        # Derived parameters
        if config.megatron.model.bf16:
            config.megatron.model.params_dtype = "torch.bfloat16"
        elif config.megatron.model.fp16:
            config.megatron.model.params_dtype = "torch.float16"
        config.megatron.model.sequence_parallel = int(config.megatron.model.tensor_model_parallel_size) > 1
        if config.megatron.model.sequence_parallel:
            config.trainer.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        if config.megatron.model.virtual_pipeline_model_parallel_size is not None \
            and config.megatron.model.virtual_pipeline_model_parallel_size > 1:
            config.megatron.model.overlap_p2p_comm = True
        config.megatron.model.batch_p2p_comm = not config.megatron.model.overlap_p2p_comm

        if config.trainer.deterministic_mode:
            config.megatron.model.deterministic_mode = True
            config.megatron.model.cross_entropy_loss_fusion = False
            config.trainer.env_vars["NCCL_ALGO"] = "Ring"
            config.trainer.env_vars["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
            config.trainer.env_vars["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            config.trainer.env_vars["FLASH_ATTENTION_DETERMINISTIC"] = "1"

        if config.megatron.ddp_config.overlap_param_gather:
            config.megatron.ddp_config.setdefault("align_param_gather", True)
        config.megatron.ddp_config.fp8_param_gather = config.megatron.model.fp8_param

        cls._finalize_optim_sched(config.megatron.optimizer, config.megatron.scheduler, config)

        # Parameter validation    TODO: validate data iterator parameters in trainer
        if (
            not config.trainer.use_dynamic_batch_size
            and config.trainer.mini_batch_size % config.trainer.micro_batch_size != 0
        ):
            raise ValueError(
                f"When use_dynamic_batch_size is disabled, mini_batch_size ({config.trainer.mini_batch_size}) "
                f"must be divisible by micro_batch_size ({config.trainer.micro_batch_size})."
            )

        return config

    @staticmethod
    def _finalize_optim_sched(opt_cfg, sched_cfg, config):
        """Fill the derived fields of one optimizer / LR-scheduler config pair.

        Everything lr-relative is derived from ``opt_cfg`` itself (only when not
        set explicitly), so a model with its own lr gets its own schedule.
        """
        if opt_cfg.optimizer_cpu_offload:
            opt_cfg.use_precision_aware_optimizer = True
            opt_cfg.overlap_cpu_optimizer_d2h_h2d = True
        if opt_cfg.chunked_optimizer_state_offload:
            opt_cfg.use_precision_aware_optimizer = True

        opt_cfg.setdefault("min_lr", opt_cfg.lr * 0.1)
        opt_cfg.use_distributed_optimizer = config.megatron.ddp_config.use_distributed_optimizer
        opt_cfg.fp16 = config.megatron.model.fp16
        opt_cfg.bf16 = config.megatron.model.bf16
        opt_cfg.fp8_recipe = config.megatron.model.fp8_recipe
        if "params_dtype" in config.megatron.model:
            opt_cfg.params_dtype = config.megatron.model.params_dtype
        opt_cfg.overlap_param_gather = config.megatron.ddp_config.overlap_param_gather

        sched_cfg.setdefault("max_lr", opt_cfg.lr)
        sched_cfg.setdefault("min_lr", opt_cfg.min_lr)
        sched_cfg.setdefault("init_lr", opt_cfg.min_lr)
        sched_cfg.setdefault("start_wd", opt_cfg.weight_decay)
        sched_cfg.setdefault("end_wd", opt_cfg.weight_decay)

    @override
    def init(self, config, master_addr, master_port):
        """Initialize the Megatron model, optimizer and LR scheduler.

        Steps:
          1. Call super().init() — creates NCCL + Gloo process groups
          2. Initialize parallel communication groups (TP/PP/VPP/CP/EP/ETP)
          3. Initialize random seeds for reproducibility
          4. Detect checkpoint availability
          5. Load model via Megatron-bridge (HF weights or structure only)
          6. Create optimizer
          7. Create learning rate scheduler
          8. Load checkpoint and restore model/optimizer/scheduler/rng state if one exists

        Subclasses extend this with ``super().init(...)``: rewrite their own
        process-local ``config`` before it, build role-specific state after it.
        """
        # Step 1: base class init (env vars, NCCL, Gloo)
        super().init(config, master_addr, master_port)
        configure_tracking(config)

        # Step 2: parallel groups
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=config.megatron.model.tensor_model_parallel_size,
            pipeline_model_parallel_size=config.megatron.model.pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size=config.megatron.model.virtual_pipeline_model_parallel_size,
            context_parallel_size=config.megatron.model.context_parallel_size,
            expert_model_parallel_size=config.megatron.model.expert_model_parallel_size,
            expert_tensor_parallel_size=config.megatron.model.expert_tensor_parallel_size,
        )
        logger.info(
            f"[Rank {self.rank}] 5D parallel groups initialized: "
            f"TP={parallel_state.get_tensor_model_parallel_world_size()}, "
            f"PP={parallel_state.get_pipeline_model_parallel_world_size()}, "
            f"EP={parallel_state.get_expert_model_parallel_world_size()}, "
            f"CP={parallel_state.get_context_parallel_world_size()}, "
            f"ETP={parallel_state.get_expert_tensor_parallel_world_size()}, "
            f"VP={parallel_state.get_virtual_pipeline_model_parallel_world_size()}"
        )

        # Step 3: random seeds
        seed = config.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model_parallel_cuda_manual_seed(seed)

        # The megatron.bridge path skips Megatron's validate_args, which is the
        # only place that flips the global torch deterministic flag when
        # deterministic_mode is set. Do it here so it takes effect (requires
        # CUBLAS_WORKSPACE_CONFIG, injected via the worker runtime_env).
        if config.trainer.deterministic_mode:
            torch.use_deterministic_algorithms(True)
            # use_deterministic_algorithms does NOT force cuDNN algo selection to be
            # stable; benchmark=True lets cuDNN pick different kernels run-to-run.
            # Pin both so the training forward is bit-reproducible.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Step 4: determine checkpoint availability
        ckpt_dir = find_latest_ckpt_path(config.checkpoint_path, name=self.CKPT_DIR_NAME)
        has_checkpoint = ckpt_dir is not None
        logger.info(
            f"[Rank {self.rank}] Checkpoint "
            f"{'found at ' + ckpt_dir if has_checkpoint else 'not found, loading HF weights'}"
        )

        # Step 5: load model via Megatron-bridge
        # If checkpoint exists: structure only (weights restored from ckpt later)
        # Otherwise: load HF weights into model
        self.bridge = AutoBridge.from_hf_pretrained(config.hf_model_path)
        provider = self.bridge.to_megatron_provider(load_weights=not has_checkpoint)

        model_config_dict = OmegaConf.to_container(config.megatron.model)
        hints = get_type_hints(GPTModelProvider)
        overrides = {}
        for k, v in model_config_dict.items():
            # Convert string dtype like "torch.bfloat16" to actual torch.dtype
            hint = hints.get(k)
            if hint is torch.dtype or hint is Optional[torch.dtype]:
                v = to_torch_dtype(v)
            elif isinstance(hint, type) and issubclass(hint, Enum) and isinstance(v, str):
                # e.g. attention_backend: flash -> AttnBackend.flash. Megatron's enums
                # are plain Enum (no str mixin), so a raw string silently compares
                # unequal to every member instead of failing loudly.
                v = hint[v]
            overrides[k] = v
        # Apply overrides and finalize via the bridge helper so provider paths
        # that need post-override setup (e.g. DeepSeek-V4 hash MoE auto-setting
        # pipeline_model_parallel_layout when PP > 1) run before finalize().
        provider.apply_overrides_and_finalize(overrides=overrides)
        self._configure_provider(provider)
        megatron_ddp_config = DistributedDataParallelConfig(
            **OmegaConf.to_container(config.megatron.ddp_config)
        )
        self.model = provider.provide_distributed_model(
            ddp_config=megatron_ddp_config,
            wrap_with_ddp=True,
            mixed_precision_wrapper=partial(KeepFP32Module, config.megatron.keep_fp32_weights)
        )
        logger.info(f"[Rank {self.rank}] Model created via Megatron-bridge")

        # Step 6: create optimizer
        optimizer_config_dict = OmegaConf.to_container(config.megatron.optimizer)
        hints = get_type_hints(MegatronOptimizerConfig)
        for k, v in optimizer_config_dict.items():
            hint = hints.get(k)
            if hint is torch.dtype or hint is Optional[torch.dtype]:
                    optimizer_config_dict[k] = to_torch_dtype(v)
        megatron_opt_config = MegatronOptimizerConfig(**optimizer_config_dict)
        self.optimizer = get_megatron_optimizer(megatron_opt_config, self.model)
        self._register_training_hooks()
        logger.info(f"[Rank {self.rank}] Optimizer created")

        # Step 7: create learning rate scheduler
        scheduler_config = OmegaConf.to_container(config.megatron.scheduler)
        self.scheduler = OptimizerParamScheduler(self.optimizer, **scheduler_config)
        logger.info(f"[Rank {self.rank}] LR scheduler created")

        # Step 8: load checkpoint if exists
        if has_checkpoint:
            load_checkpoint(self.model, self.optimizer, self.scheduler, ckpt_dir)
            logger.info(f"[Rank {self.rank}] Checkpoint loaded from {ckpt_dir}")
        self.offloaded = False
        self.params_offloaded = False

    def _configure_provider(self, provider) -> None:
        """Hook to adjust the finalized model provider before DDP wrapping.

        Called between ``apply_overrides_and_finalize`` and
        ``provide_distributed_model``, i.e. before the DDP grad buffers and the
        optimizer are built from the model's parameters. No-op by default.
        """

    def _register_training_hooks(self):
        "following https://github.com/NVIDIA/Megatron-LM/blob/core_v0.16.0/megatron/training/training.py#L2546-L2565"
        optimizer = self.optimizer

        def finalize_model_grads_with_state_reload(*args, **kwargs):
            # Restore chunked-offloaded optimizer state (fp32 masters + first
            # state chunk) here so the async H2D overlaps gradient finalization,
            # ``optimizer.step()`` also prefetches internally, so this is purely a perf overlap;
            optimizer.prefetch_optimizer_state_for_gradient_finalization()
            return finalize_model_grads(*args, **kwargs)

        for model_chunk in self.model:
            assert isinstance(model_chunk, DDP)
            config = get_model_config(model_chunk)
            ddp_config = model_chunk.ddp_config
            config.grad_scale_func = self.optimizer.scale_loss
            config.finalize_model_grads_func = finalize_model_grads_with_state_reload

            if ddp_config.overlap_grad_reduce:
                config.no_sync_func = [m.no_sync for m in self.model]
                config.grad_sync_func = [m.start_grad_sync for m in self.model]
                if len(self.model) == 1:
                    config.no_sync_func = config.no_sync_func[0]
                    config.grad_sync_func = config.grad_sync_func[0]
            if ddp_config.overlap_param_gather and ddp_config.align_param_gather:
                config.param_sync_func = [m.start_param_sync for m in self.model]
                if len(self.model) == 1:
                    config.param_sync_func = config.param_sync_func[0]


    @abstractmethod
    def train(self, step, rollout_data_ref):
        """Run one training step on this rank's rollout shard."""

    @override
    @torch.no_grad()
    def save_model(self, step, checkpoint=True, hf=False):
        """Save the distributed checkpoint under ``train_step_N/<CKPT_DIR_NAME>``.

        Passing ``step=None`` skips scheduling a new save and instead
        drains the pending async-save queue so the last ``finalize_fn``
        (which writes the tracker recording the last step) runs before
        return. HF export is model-specific and handled by subclasses.

        Args:
            step: Current training step; ``None`` means flush only.
            checkpoint: If True, save the distributed checkpoint.
            hf: Ignored here; see subclasses.
        """
        if step is None:
            async_calls.maybe_finalize_async_calls(blocking=True)
            return

        if checkpoint:
            self._save_dist_checkpoint(step, async_save=self.config.trainer.async_save)

    def _save_dist_checkpoint(self, step, async_save: bool, update_tracker: bool = True):
        """Save model/optimizer/scheduler to ``train_step_{step}/<CKPT_DIR_NAME>``.

        When ``update_tracker`` is set, the shared tracker file is written to
        ``step`` once the save has completed.
        """
        ckpt_dir = get_ckpt_dir(self.config.checkpoint_path, step, name=self.CKPT_DIR_NAME)

        def _on_save_complete():
            if update_tracker:
                update_latest(self.config.checkpoint_path, step)
            logger.info(f"[Rank {self.rank}] Checkpoint saved for step={step} at {ckpt_dir}")

        save_checkpoint(
            self.model, self.optimizer, self.scheduler,
            ckpt_dir, async_save=async_save,
            finalize_fn=_on_save_complete,
            use_distributed_optimizer=self.config.megatron.ddp_config.use_distributed_optimizer,
            optimizer_sharding_type=self.config.megatron.optimizer_sharding_type,
        )

    @override
    @torch.no_grad()
    def onload(self):
        """Onload: move from CPU memory to GPU memory (CPU RAM -> VRAM)"""
        print_memory("before onload")
        self._onload_model()
        self._move_optimizer(torch.cuda.current_device())
        clear_memory()
        print_memory("after onload")
        self.offloaded = False
        self.params_offloaded = False

    @override
    @torch.no_grad()
    def offload(self, move_params: bool = False):
        """Offload: move from GPU memory to CPU memory (VRAM -> CPU RAM)

        Safe to call again on an already-offloaded worker: ``move_params=True``
        additionally releases parameters that were kept on GPU, while
        ``move_params=False`` never brings offloaded parameters back.
        """
        print_memory(f"before offload (move_params={move_params})")
        self._offload_model(move_params=move_params, move_grads=True)
        self._move_optimizer("cpu")
        clear_memory()
        print_memory(f"after offload (move_params={move_params})")
        self.params_offloaded = move_params or (self.offloaded and self.params_offloaded)
        self.offloaded = True

    def _offload_model(self, move_params: bool = True, move_grads: bool = True):
        """Offload model parameters and gradients to CPU."""
        for model in self.model:
            for buffer in model.buffers + model.expert_parallel_buffers:
                buffer.offload_to_cpu(move_params, move_grads)


    def _onload_model(self, move_params: bool = True, move_grads: bool = True):
        """Onload model parameters and gradients from CPU."""
        for model in self.model:
            for buffer in model.buffers + model.expert_parallel_buffers:
                buffer.reload_from_cpu(move_params, move_grads)


    def _move_optimizer(self, device):
        """Move GPU-resident optimizer state (master params + Adam moments) to ``device``.

        Two inner-optimizer layouts are supported:

        * Classic ``DistributedOptimizer``: fp32 master params live in
          ``shard_fp32_from_float16_groups`` and Adam moments in
          ``optimizer.optimizer.state``.
        * precision-aware + ``optimizer_cpu_offload``: the inner optimizer is a
          ``HybridDeviceOptimizer`` (HDO).  Here ``shard_fp32_from_float16_groups``
          is filled with ``None`` (masters are held by HDO), so the classic path
          would crash on ``None.data``.

        HDO partitioning (``optimizer_offload_fraction``):
          - GPU-home params: fp32 master in ``param_to_fp32_param`` (keys NOT in
            ``gpu_params_map_cpu_copy``) and moments in ``gpu_optimizer.state``.
          - CPU-home params (fraction > 0): master + moments are permanently on
            CPU for their ``cpu_optimizers`` and must NOT be touched, otherwise
            onload would wrongly drag them onto GPU.
          So we only relocate the GPU-home half.

        Native-fp32 shards (e.g. the ``use_fp32_lm_head`` output layer) are NOT
        in ``param_to_fp32_param`` — they are *views* into model params that
        ``offload()`` keeps on GPU.  Reassigning their ``.data`` would break the
        view aliasing (the optimizer update would stop writing back to the model
        param), so they are deliberately left untouched; only their moments move.
        Identity is preserved (``t.data = t.data.to(...)``) so HDO's internal
        param maps stay valid across the move.

        With ``chunked_optimizer_state_offload`` the offloader owns its state:
        ``step()`` leaves masters on GPU, so the D2H below hands them back to its
        pinned CPU buffers (H2D is implicit in the next ``step()``; no onload
        branch). Moments are offloader-owned and skipped below.
        """
        if device == "cpu":
            # Hand masters back to the offloader's pinned CPU buffers; they are
            # invisible to the loops below (shard_fp32_from_float16_groups is None).
            self.optimizer.offload_optimizer_state_for_forward()

        def _move(t):
            if t is not None:
                t.data = t.data.to(device, non_blocking=True)

        def _move_moments(state, offloader=None):
            for param, value in state.items():
                if offloader is not None and offloader.is_param_offloaded(param):
                    continue
                if "exp_avg" in value:
                    value["exp_avg"] = value["exp_avg"].to(device, non_blocking=True)
                if "exp_avg_sq" in value:
                    value["exp_avg_sq"] = value["exp_avg_sq"].to(device, non_blocking=True)

        for optimizer in self.optimizer.chained_optimizers:
            inner = optimizer.optimizer
            if isinstance(inner, HybridDeviceOptimizer):
                # Only GPU-home fp32 masters; CPU-home masters stay on CPU.
                for orig_param, fp32_param in inner.param_to_fp32_param.items():
                    if orig_param not in inner.gpu_params_map_cpu_copy:
                        _move(fp32_param)
                # GPU-home Adam moments live in gpu_optimizer.state;
                # cpu_optimizers' moments are left on CPU.
                if inner.gpu_optimizer is not None:
                    _move_moments(inner.gpu_optimizer.state)
            else:
                for group in optimizer.shard_fp32_from_float16_groups:
                    for param in group:
                        _move(param)
                _move_moments(
                    optimizer.optimizer.state, optimizer._optimizer_state_offloader
                )


    @staticmethod
    def _reorder_to_rows(collected: dict[str, list], data_iterator, n_rows: int) -> dict[str, list]:
        """Undo the dynamic-batch permutation / -1 padding of forward-only outputs.

        ``forward_only`` returns values in micro-batch order; this maps them
        back so ``collected[key][i]`` aligns with Segment row ``i`` of the
        rollout batch.
        """
        if not collected:
            return collected
        order = sum(data_iterator[0].micro_batch_indices, [])
        for key in collected:
            reordered: list = [None] * n_rows
            for value, idx in zip(collected[key], order):
                if idx >= 0:
                    reordered[idx] = value
            collected[key] = reordered
        return collected
