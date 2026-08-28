"""Megatron train worker — per-GPU Ray actor backed by Megatron-Core."""

import random
import logging
from typing import override, Optional, get_type_hints
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
from coda.utils.channel_helper import ChannelMeta, create_sender_channel

from coda.backends.train_worker import TrainWorker
from coda.utils.tensor_backuper import TensorBackuper
from coda.utils.memory_utils import print_memory, clear_memory
from coda.utils.checkpoint_utils import (
    find_latest_ckpt_path, resolve_dist_ckpt_dir, update_latest, get_ckpt_dir, get_hf_dir
)
from coda.backends.megatron.checkpoint import load_checkpoint, save_checkpoint, load_model_weights
from coda.utils.types import RolloutBatch, to_torch_dtype
from coda.algorithms.advantage import compute_advantages
from coda.algorithms.is_correction import apply_is_correction
from coda.algorithms.second_moment_trust_policy_optimization import apply_m2po_masking
from coda.backends.megatron.data import (
    get_rollout_data,
    get_data_iterator,
)
from coda.backends.megatron.model import forward_only, train_minibatch, is_main_rank, reduce_dict
from coda.utils.tracking import (
    time_tracker, configure_tracking, track,
    install_train_metrics_aggregator,
    flush_train_metrics as _flush_train_metrics,
)
from coda.backends.megatron.mixed_precision import KeepFP32Module
from coda.algorithms.kl_policy import build_kl_policies
from coda.backends.megatron.teacher_lm_head import TeacherLMHeads

logger = logging.getLogger(__name__)


class MegatronTrainWorker(TrainWorker):
    """Per-GPU Ray actor that drives Megatron-Core distributed training."""

    @classmethod
    def runtime_env_vars(cls):
        """set custom runtime environment variables."""
        return {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
        }

    @classmethod
    def validate_config(cls, config):
        """validate or fill default megatron config"""

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

        if config.trainer.use_rollout_routing_replay:
            config.megatron.model.moe_enable_routing_replay = True
        if config.trainer.deterministic_mode:
            config.megatron.model.deterministic_mode = True
            config.megatron.model.cross_entropy_loss_fusion = False
            config.trainer.env_vars["NCCL_ALGO"] = "Ring"
            config.trainer.env_vars["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
            config.trainer.env_vars["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            config.trainer.env_vars["FLASH_ATTENTION_DETERMINISTIC"] = "1"
        if config.trainer.use_fp32_lm_head:
            config.megatron.keep_fp32_weights["output_layer"] = True
        if config.megatron.optimizer.optimizer_cpu_offload:
            config.megatron.optimizer.use_precision_aware_optimizer = True
            config.megatron.optimizer.overlap_cpu_optimizer_d2h_h2d = True

        if config.megatron.ddp_config.overlap_param_gather:
            config.megatron.ddp_config.setdefault("align_param_gather", True)
        config.megatron.ddp_config.fp8_param_gather = config.megatron.model.fp8_param

        config.megatron.optimizer.setdefault("min_lr", config.megatron.optimizer.lr * 0.1)
        config.megatron.optimizer.use_distributed_optimizer = config.megatron.ddp_config.use_distributed_optimizer
        config.megatron.optimizer.fp16 = config.megatron.model.fp16
        config.megatron.optimizer.bf16 = config.megatron.model.bf16
        config.megatron.optimizer.fp8_recipe = config.megatron.model.fp8_recipe
        if "params_dtype" in config.megatron.model:
            config.megatron.optimizer.params_dtype = config.megatron.model.params_dtype
        config.megatron.optimizer.overlap_param_gather = config.megatron.ddp_config.overlap_param_gather

        config.megatron.scheduler.setdefault("max_lr", config.megatron.optimizer.lr)
        config.megatron.scheduler.setdefault("min_lr", config.megatron.optimizer.min_lr)
        config.megatron.scheduler.setdefault("init_lr", config.megatron.optimizer.min_lr)
        config.megatron.scheduler.setdefault("start_wd", config.megatron.optimizer.weight_decay)
        config.megatron.scheduler.setdefault("end_wd", config.megatron.optimizer.weight_decay)

        # Parameter validation    TODO: validate data iterator parameters in trainer
        if config.algorithm.is_correction.enable and config.trainer.use_rollout_log_probs:
            raise ValueError(
                "is_correction.enable and trainer.use_rollout_log_probs are mutually exclusive. "
            )

        if config.algorithm.m2po.enable and config.trainer.use_rollout_log_probs:
            raise ValueError(
                "m2po.enable and trainer.use_rollout_log_probs are mutually exclusive. "
                "M2PO requires recomputed old_log_probs."
            )

        if config.algorithm.m2po.enable and config.algorithm.m2po.threshold < 0.0:
            raise ValueError(
                f"m2po.threshold must be non-negative, got {config.algorithm.m2po.threshold}"
            )

        if (
            not config.trainer.use_dynamic_batch_size
            and config.trainer.mini_batch_size % config.trainer.micro_batch_size != 0
        ):
            raise ValueError(
                f"When use_dynamic_batch_size is disabled, mini_batch_size ({config.trainer.mini_batch_size}) "
                f"must be divisible by micro_batch_size ({config.trainer.micro_batch_size})."
            )
        assert config.algorithm.clip_ratio_c > 1.0, "clip_ratio_c must be greater than 1.0"

        if config.algorithm.ref_kl.enable:
            if config.algorithm.ref_kl.kl_type not in {"k1", "k2", "k3"}:
                raise ValueError(
                    "algorithm.ref_kl.kl_type must be one of: k1, k2, k3."
                )
            if not (config.ref_dist_ckpt_path or config.ref_hf_model_path):
                raise ValueError(
                    "algorithm.ref_kl.enable=True requires either "
                    "ref_dist_ckpt_path or ref_hf_model_path to be set."
                )

        return config

    @override
    def init(self, config):
        """Initialize Megatron train worker.

        Steps:
          1. Call super().init() — creates NCCL + Gloo process groups
          2. Initialize parallel communication groups (TP/PP/VPP/CP/EP/ETP)
          3. Initialize random seeds for reproducibility
          4. Detect checkpoint availability
          5. Load model via Megatron-bridge (HF weights or structure only)
          6. Create optimizer
          7. Create learning rate scheduler
          8. Load checkpoint and restore model/optimizer/scheduler/rng state if one exists
          9. Build KL policy instances and set up teacher lm_heads
          10. Install train metrics aggregator for fully_async mode
          11. Set up frozen reference model for ref-KL (weight hot-swap)
        """
        # Step 1: base class init (env vars, NCCL, Gloo)
        super().init(config)
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
        ckpt_dir = find_latest_ckpt_path(config.checkpoint_path)
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
            overrides[k] = v
        # Apply overrides and finalize via the bridge helper so provider paths
        # that need post-override setup (e.g. DeepSeek-V4 hash MoE auto-setting
        # pipeline_model_parallel_layout when PP > 1) run before finalize().
        provider.apply_overrides_and_finalize(overrides=overrides)
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

        # Step 9: build KL policy instances and set up teacher lm_heads if an
        # active policy needs teacher logits (full_kl/full_jsd). The
        # TeacherLMHeads singleton owns the lm_head lifecycle (last PP stage only).
        self.kl_policies = self._init_kl_policies()

        if config.fully_async.enable or config.algorithm.ref_kl.enable:
            self._weights_backuper = TensorBackuper(
                source_getter=lambda: (
                    (f"chunk{i}.{name}", param)
                    for i, m in enumerate(self.model)
                    for name, param in m.named_parameters()
                ),
            )

        self._active_model_tag = "actor"

        if config.fully_async.enable:
            # Step 10: install train metrics aggregator for fully_async mode.
            install_train_metrics_aggregator()
            self._current_step: int = -1
        # Step 11: set up the frozen reference model for ref-KL (weight hot-swap).
        self._setup_ref_model()

        logger.info(f"[Rank {self.rank}] MegatronTrainWorker initialization complete")

    @torch.no_grad()
    def _setup_ref_model(self):
        """Load the frozen reference model and snapshot both actor/ref weights.

        ``TensorBackuper`` stores named CPU snapshots for actor/ref weights, and
        also serves fully-async old_actor snapshots when that mode is enabled.
        Only one tag is restored into the live GPU model at a time. After setup,
        the GPU holds "actor" and ``_active_model_tag`` is "actor".
        """
        ref_kl = self.config.algorithm.ref_kl
        if not ref_kl.enable:
            return

        self._weights_backuper.backup("actor")  # snapshot current student weights

        ref_ckpt = resolve_dist_ckpt_dir(
            self.config.ref_dist_ckpt_path, "ref_dist_ckpt_path"
        )
        if ref_ckpt:
            load_model_weights(self.model, ref_ckpt)
        elif self.config.ref_hf_model_path:
            self.bridge.load_hf_weights(self.model, self.config.ref_hf_model_path)
        else:
            raise ValueError(
                "algorithm.ref_kl.enable=True requires either "
                "ref_dist_ckpt_path or ref_hf_model_path to be set."
            )

        self._weights_backuper.backup("ref")     # snapshot ref weights
        self._weights_backuper.restore("actor")  # switch back to the student
        self._active_model_tag = "actor"
        logger.info(
            f"[Rank {self.rank}] Reference model set up for ref-KL "
            f"(source={'ckpt:' + ref_ckpt if ref_ckpt else 'hf:' + str(self.config.ref_hf_model_path)})"
        )

    @torch.no_grad()
    def switch(self, tag: str) -> None:
        """Hot-swap the GPU-resident weights to ``tag`` ("actor"/"ref"/"old_actor").

        Only restores the requested tag (no backup-before-restore). The caller keeps
        the active snapshot fresh via an explicit ``backup("actor")`` after each
        optimizer step, so restoring never loses training progress. No-op when
        already active.
        """
        if self._active_model_tag == tag:
            return
        self._weights_backuper.restore(tag)
        self._active_model_tag = tag

    def _register_training_hooks(self):
        "following https://github.com/NVIDIA/Megatron-LM/blob/core_v0.16.0/megatron/training/training.py#L2546-L2565"
        for model_chunk in self.model:
            assert isinstance(model_chunk, DDP)
            config = get_model_config(model_chunk)
            ddp_config = model_chunk.ddp_config
            config.grad_scale_func = self.optimizer.scale_loss
            config.finalize_model_grads_func = finalize_model_grads

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

    @override
    def train(self, step, rollout_data_ref):
        """Full train cycle: fetch data -> compute old log-probs -> advantages -> training loop.

        When OPD is enabled in pure-GKD mode, skips the forward-only pass and
        advantage computation.
        """
        torch.cuda.reset_peak_memory_stats()

        # run finalize_fn for pre checkpoint save
        async_calls.maybe_finalize_async_calls()
        rollout_data = get_rollout_data(rollout_data_ref)

        # Determine OPD mode
        opd_enabled = self.config.opd.enable
        gkd_ratio = self.config.opd.gkd_ratio if opd_enabled else 0.0
        pg_ratio = self.config.opd.pg_ratio if opd_enabled else 0.0
        pure_gkd = (gkd_ratio == 1.0)

        if not pure_gkd:
            for m in self.model:
                m.eval()
            # Optionally (re-)compute log-probs from the current policy
            if not self.config.trainer.use_rollout_log_probs:
                rollout_data.update(self._compute_log_probs(step, rollout_data))

            if self.config.algorithm.ref_kl.enable:
                rollout_data.update(self._compute_ref_log_probs(step, rollout_data))

            if self.config.algorithm.m2po.enable:
                rollout_data.update(self._compute_m2po(step, rollout_data))

            rollout_data.update(self._compute_is_correction(step, rollout_data))

            advantages, adv_metrics = compute_advantages(self.config.algorithm, rollout_data)
            rollout_data["adv_metrics"] = adv_metrics

            if pg_ratio > 0:
                self._apply_opd_kl_to_advantages(rollout_data, advantages)
                self._free_pg_fields(rollout_data, gkd_ratio)

            rollout_data["advantages"] = advantages

        # Train
        for m in self.model:
            m.train()
        self._train_actor(step, rollout_data)

        # Refresh the CPU "actor" snapshot after the optimizer step for
        # fully-async/ref-KL weight switching; optionally move the reference
        # toward the current policy (moving-ref).
        ref_kl = self.config.algorithm.ref_kl
        if ref_kl.enable or self.config.fully_async.enable:
            self._weights_backuper.backup("actor")

            if ref_kl.update_interval > 0 and step % ref_kl.update_interval == 0:
                self._weights_backuper.backup("ref")
                logger.info(f"[Rank {self.rank}] Ref-model update at step {step}")

    def _compute_token_metric(
        self,
        name: str,
        values: list[torch.Tensor],
        masks: list[torch.Tensor],
    ) -> float:
        """Compute a per-token metric reduced and normalized according to loss_agg_mode."""
        loss_agg_mode = self.config.algorithm.loss_agg_mode
        device = values[0].device

        metric_sum = torch.tensor(0.0, device=device)
        num_tokens = torch.tensor(0.0, device=device)
        num_seqs = torch.tensor(0.0, device=device)
        for val_i, mask_i in zip(values, masks):
            num_tokens += mask_i.sum()
            if mask_i.any():
                num_seqs += 1
            masked_val = val_i * mask_i
            if loss_agg_mode == "seq-mean-token-mean":
                metric_sum += masked_val.sum() / torch.clamp_min(mask_i.sum(), 1)
            else:
                metric_sum += masked_val.sum()

        raw = {
            name: metric_sum.item(),
            "num_tokens": num_tokens.item(),
            "num_seqs": num_seqs.item(),
        }
        raw = reduce_dict(raw, parallel_state.get_data_parallel_group())
        num_tokens_r = max(raw.pop("num_tokens"), 1)
        num_seqs_r = max(raw.pop("num_seqs"), 1)
        denom = num_seqs_r if loss_agg_mode == "seq-mean-token-mean" else num_tokens_r

        return raw[name] / float(denom)

    @torch.no_grad()
    def _apply_opd_kl_to_advantages(self, rollout_data: RolloutBatch, advantages: list[torch.Tensor]):
        """PG-Style KL penalty: advantages -= pg_ratio * per_token_kl.

        per_token_kl is pre-computed in forward_only via compute_forward_only_outputs.
        """
        if not parallel_state.is_pipeline_last_stage():
            return

        per_token_kl = rollout_data["per_token_kl"]
        pg_ratio = self.config.opd.pg_ratio
        for i in range(len(advantages)):
            advantages[i] = advantages[i] - pg_ratio * per_token_kl[i]

        kl_val = self._compute_token_metric("pg_opd_kl", per_token_kl, rollout_data["loss_masks"])
        rollout_data.setdefault("kl_metrics", {})["train/opd_pg_kl"] = kl_val

    def _free_pg_fields(self, rollout_data: RolloutBatch, gkd_ratio: float):
        """Free PG-only fields after advantages are computed to save memory."""
        rollout_data.pop("per_token_kl", None)
        # GKD off → all teacher_* fields are dead; sweep by the transport prefix
        # so this can never drift from the producer field names.
        if gkd_ratio == 0:
            for k in [k for k in rollout_data if k.startswith("teacher_")]:
                rollout_data.pop(k, None)

    def _init_kl_policies(self):
        """Build KL policy instances and set up teacher lm_heads if needed.

        Only runs on the last PP stage (where hidden states / output_layer live).
        Builds the :class:`TeacherLMHeads` singleton when an active policy needs
        teacher full logits. Returns the policies dict stored as ``self.kl_policies``.
        """
        if not self.config.opd.enable or not parallel_state.is_pipeline_last_stage():
            return {}
        policies = build_kl_policies(self.config)
        if any(p.need_teacher_logits() for p in policies.values()):
            TeacherLMHeads.setup(self.config)
        return policies
    def flush_train_metrics(self):
        """Drain the per-step train-metrics buffer and emit one aggregated record."""
        _flush_train_metrics()

    @override
    @torch.no_grad()
    def save_model(self, step, checkpoint=True, hf=False):
        """Save distributed checkpoint and/or export HF weights.

        Passing ``step=None`` skips scheduling a new save and instead
        drains the pending async-save queue so the last ``finalize_fn``
        (which writes the tracker recording the last step) runs before
        return.

        Args:
            step: Current training step; ``None`` means flush only.
            checkpoint: If True, save the distributed checkpoint.
            hf: If True, export weights in HuggingFace format.
        """
        if step is None:
            async_calls.maybe_finalize_async_calls(blocking=True)
            return

        if checkpoint:
            ckpt_dir = get_ckpt_dir(self.config.checkpoint_path, step)
            async_save = self.config.trainer.async_save

            def _on_save_complete():
                update_latest(self.config.checkpoint_path, step)
                logger.info(f"[Rank {self.rank}] Checkpoint saved for step={step}")

            save_checkpoint(
                self.model, self.optimizer, self.scheduler,
                ckpt_dir, async_save=async_save,
                finalize_fn=_on_save_complete,
                use_distributed_optimizer=self.config.megatron.ddp_config.use_distributed_optimizer,
                optimizer_sharding_type=self.config.megatron.optimizer_sharding_type,
            )

        if hf:
            hf_dir = get_hf_dir(self.config.checkpoint_path, step)
            self.bridge.save_hf_pretrained(self.model, hf_dir)
            logger.info(f"[Rank {self.rank}] HF model exported for step={step}")

    @override
    @torch.no_grad()
    def update_weights(self, channel_meta: ChannelMeta):
        """Update weights"""
        print_memory("before send huggingface weights")
        send_channel = create_sender_channel(channel_meta)
        for name, tensor, in self.bridge.export_hf_weights(self.model):
            send_channel.send((name, tensor))
        send_channel.send(None, flush=True)
        if self.offloaded:
            self._offload_model(move_params=True, move_grads=False)
        clear_memory()
        print_memory("after send huggingface weights")

    @override
    @torch.no_grad()
    def onload(self):
        """Onload: move from CPU memory to GPU memory (CPU RAM -> VRAM)"""
        print_memory("before onload")
        self._onload_model()
        self._move_optimizer(torch.cuda.current_device())
        lm_heads = TeacherLMHeads.get()
        if lm_heads is not None:
            lm_heads.onload(torch.cuda.current_device())
        clear_memory()
        print_memory("after onload")
        self.offloaded = False

    @override
    @torch.no_grad()
    def offload(self):
        """Offload: move from GPU memory to CPU memory (VRAM -> CPU RAM)"""
        print_memory("before offload")
        # keep params in gpu for latter update weights
        self._offload_model(move_params=False, move_grads=True)
        self._move_optimizer("cpu")
        lm_heads = TeacherLMHeads.get()
        if lm_heads is not None:
            lm_heads.offload()
        clear_memory()
        print_memory("after offload")
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
        """
        def _move(t):
            if t is not None:
                t.data = t.data.to(device, non_blocking=True)

        def _move_moments(state):
            for value in state.values():
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
                _move_moments(optimizer.optimizer.state)

    @time_tracker("compute_log_probs")
    def _compute_log_probs(
            self,
            step,
            rollout_data: RolloutBatch,
    ) -> dict[str, list[torch.Tensor]]:
        """Compute behavior-policy log-probs for the rollout batch.

        Uses the same mini-batch splitting as ``_train_actor`` to ensure consistent
        micro-batch boundaries across forward-only and training passes.
        When a PG KL policy is active, also returns per-token PG KL.

        In fully-async mode, repeated calls within one step restore the
        ``old_actor`` snapshot so log-probs reflect the policy that produced the
        rollout. Otherwise the live actor weights are used.
        """
        # Per-step old_actor snapshot
        if self.config.fully_async.enable:
            if step != self._current_step:
                # First call this step: live weights == rollout policy, snapshot.
                self._current_step = step
                self._weights_backuper.backup("old_actor")
            else:
                # Subsequent call same step: swap live weights with old_actor for forward.
                self.switch("old_actor")

        collected_all = self._forward_only_log_probs(rollout_data, step, pg_policy=self.kl_policies.get("pg"))

        if self.config.fully_async.enable:
            self.switch("actor")

        output = {"old_log_probs": collected_all.get("log_probs")}
        if "per_token_kl" in collected_all:
            output["per_token_kl"] = collected_all["per_token_kl"]
        return output

    def _forward_only_log_probs(
            self,
            rollout_data: RolloutBatch,
            step,
            pg_policy=None,
    ) -> dict[str, list[torch.Tensor]]:
        """Run the forward-only pass and collect per-microbatch outputs.

        Shared by :meth:`_compute_log_probs` (current policy) and
        :meth:`_compute_ref_log_probs` (reference policy); the caller controls
        which model weights are live and which ``pg_policy`` (if any) is applied.
        Recovers the original Segment-row order under dynamic / padded batching.
        """
        data_iterator, num_microbatches_list = get_data_iterator(
            self.config.trainer, self.model, rollout_data
        )

        collected_all: dict[str, list[torch.Tensor]] = {}
        for num_microbatches in num_microbatches_list:
            forward_data_store = forward_only(
                self.config, self.model,
                data_iterator=data_iterator,
                num_microbatches=num_microbatches,
                pg_policy=pg_policy,
            )
            for mb in forward_data_store:
                for key, values in mb.items():
                    collected_all.setdefault(key, []).extend(values)

        # Undo the dynamic-batch permutation / -1 padding so outputs align with
        # the Segment rows of rollout_data.
        if collected_all:
            order = sum(data_iterator[0].micro_batch_indices, [])
            n_rows = len(rollout_data["tokens"])
            for key in collected_all:
                reordered: list = [None] * n_rows
                for value, idx in zip(collected_all[key], order):
                    if idx >= 0:
                        reordered[idx] = value
                collected_all[key] = reordered

        return collected_all

    @torch.no_grad()
    @time_tracker("compute_ref_log_probs")
    def _compute_ref_log_probs(self, step, rollout_data: RolloutBatch) -> dict[str, list[torch.Tensor]]:
        """Compute reference-model log-probs via the forward-only pass.

        Swaps the GPU-resident weights to "ref", runs the forward-only log-prob
        pass (no PG policy), then swaps back to "actor". Eval mode is set by the
        caller.
        """
        self.switch("ref")
        collected_all = self._forward_only_log_probs(rollout_data, step, pg_policy=None)
        self.switch("actor")
        return {"ref_log_probs": collected_all.get("log_probs")}

    @torch.no_grad()
    def _compute_m2po(self, step, rollout_data: RolloutBatch):
        """Pre-compute M2PO-updated loss masks before IS correction."""
        old_log_probs = rollout_data.get("old_log_probs")
        # only last pp stage needs to compute m2po
        if old_log_probs is None:
            return {}
        
        rollout_log_probs = rollout_data.get("rollout_log_probs")
        loss_masks = rollout_data.get("loss_masks")
        raw_loss_masks = rollout_data.get("raw_loss_masks")

        new_loss_masks, m2po_metrics = apply_m2po_masking(
            self.config.algorithm.m2po,
            raw_loss_masks,
            old_log_probs,
            rollout_log_probs,
        )

        m2po_metrics = reduce_dict(
            m2po_metrics, parallel_state.get_data_parallel_group()
        )
        num_tokens = m2po_metrics.pop("valid_tokens")
        if num_tokens <= 0:
            logger.warning(
                "[Rank %s] step=%s M2PO: no valid tokens across all DP ranks after reduce; "
                "metrics may be meaningless",
                self.rank,
                step,
            )
            num_tokens = 1
        m2po_clip_count = m2po_metrics.pop("clip_count", 0.0)
        m2po_metrics = {
            f"train/m2po_{k}": v / float(num_tokens)
            for k, v in m2po_metrics.items()
        }
        m2po_metrics["train/m2po_clip_ratio"] = m2po_clip_count / float(num_tokens)
        return {
            "raw_loss_masks": raw_loss_masks,
            "loss_masks": [
                new_mask * loss_mask
                for new_mask, loss_mask in zip(new_loss_masks, loss_masks, strict=True)
            ],
            "m2po_metrics": m2po_metrics,
        }

    @torch.no_grad()
    def _compute_is_correction(self, step, rollout_data: RolloutBatch):
        """Pre-compute IS correction weights and updated loss masks.  """
        old_log_probs = rollout_data.get("old_log_probs")
        # only last pp stage needs to compute is_correction
        if old_log_probs is None:
            return {}

        rollout_log_probs = rollout_data.get("rollout_log_probs")
        loss_masks = rollout_data.get("loss_masks")
        raw_loss_masks = rollout_data.get("raw_loss_masks")

        # ``raw_loss_masks`` is the pre-IS snapshot taken in ``get_rollout_data``,
        # so OPSM and compute_policy_loss can operate on the original mask.
        is_weights, new_loss_masks, is_metrics = apply_is_correction(
            self.config.algorithm.is_correction,
            raw_loss_masks,
            old_log_probs,
            rollout_log_probs,
            loss_agg_mode=self.config.algorithm.loss_agg_mode,
        )
        is_metrics = reduce_dict(is_metrics, parallel_state.get_data_parallel_group())
        num_tokens = max(is_metrics.pop("num_tokens"), 1)
        num_seqs = max(is_metrics.pop("num_seqs"), 1)
        kl_denom = num_seqs if self.config.algorithm.loss_agg_mode == "seq-mean-token-mean" else num_tokens
        approx_k3_kl = is_metrics.pop("train/is_approx_k3_kl") / float(kl_denom)
        for k in list(is_metrics.keys()):
            is_metrics[k] = is_metrics[k] / float(num_tokens)
        is_metrics["train/is_approx_k3_kl"] = approx_k3_kl

        # Always report metrics; only apply weights/masks to loss when enabled
        result = {"is_metrics": is_metrics}
        if self.config.algorithm.is_correction.enable:
            result["is_weights"] = is_weights
            result["loss_masks"] = [
                new_mask * loss_mask
                for new_mask, loss_mask in zip(new_loss_masks, loss_masks, strict=True)
            ]
        return result

    @time_tracker("train_actor")
    def _train_actor(self, step, rollout_data: RolloutBatch):
        """Run training over a rollout consisting of multiple mini batches.

        ``train_minibatch`` is invoked once per mini-batch.

        Args:
            step: Global training step identifier.
            rollout_data: The complete rollout batch for this DP rank.
        """
        data_iterator, num_microbatches = get_data_iterator(
            self.config.trainer, self.model, rollout_data
        )
        global_padding_count, global_padding_ratio = (
            data_iterator[0].get_global_padding_stats()
        )

        avg_metrics: dict[str, float] = {}
        n_minibatches = len(num_microbatches)
        for index in range(n_minibatches):
            is_last_minibatch = index == n_minibatches - 1
            metrics = train_minibatch(
                self.config,
                step,
                data_iterator,
                self.model,
                self.optimizer,
                self.scheduler,
                num_microbatches=num_microbatches[index],
                gkd_policy=self.kl_policies.get("gkd"),
            )
            if is_main_rank():
                for k, v in metrics.items():
                    avg_metrics[k] = avg_metrics.get(k, 0.0) + v
                logger.info(f"[Rank {self.rank}] step: {step}, minibatch: {index}, metrics: {metrics}")
                if not is_last_minibatch:
                    continue
                # Average metrics across minibatches and track
                for k in avg_metrics:
                    avg_metrics[k] /= n_minibatches            
                avg_metrics.update(rollout_data.get("m2po_metrics", {}))
                avg_metrics.update(rollout_data.get("is_metrics", {}))
                avg_metrics.update(rollout_data.get("kl_metrics", {}))
                avg_metrics.update(rollout_data.get("adv_metrics", {}))
                avg_metrics["train/segment_padding_count"] = global_padding_count
                avg_metrics["train/segment_padding_ratio"] = global_padding_ratio
                avg_metrics["perf/train_memory_allocated_max"] = \
                    round(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024), 2)
                avg_metrics["perf/train_memory_reserved_max"] = \
                    round(torch.cuda.max_memory_reserved() / (1024 * 1024 * 1024), 2)
                logger.info(f"[Rank {self.rank}] step: {step}, avg_metrics: {avg_metrics}")
                track(avg_metrics, step)
