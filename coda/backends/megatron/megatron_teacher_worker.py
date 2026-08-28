"""Megatron-backed OPD teacher worker."""

from __future__ import annotations

import logging
import random
from functools import partial
from typing import Optional, get_type_hints

import numpy as np
import ray
import torch
from megatron.bridge import AutoBridge
from megatron.bridge.models import GPTModelProvider
from megatron.core import parallel_state as mpu
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
from omegaconf import DictConfig, OmegaConf

from coda.backends.megatron.model import forward_teacher
from coda.algorithms.kl_policy import build_kl_policies
from coda.backends.megatron.data import (
    concat_rollout_batch,
    slice_rollout_batch,
    get_data_iterator,
    group_rollout_batch,
)
from coda.backends.memory_pool import build_memory_pool
from coda.backends.teacher_worker import TeacherWorker as BaseTeacherWorker
from coda.utils.memory_utils import clear_memory
from coda.backends.megatron.checkpoint import load_model_weights
from coda.backends.megatron.mixed_precision import KeepFP32Module
from coda.utils.checkpoint_utils import resolve_dist_ckpt_dir
from coda.utils.tracking import configure_tracking
from coda.utils.types import RolloutBatch, to_torch_dtype
from coda.utils.tensor_backuper import TensorBackuper

logger = logging.getLogger(__name__)


class MegatronTeacherWorker(BaseTeacherWorker):
    """Per-GPU Ray actor that manages OPD teacher model inference."""

    def __init__(self, world_size: int, rank: int, teacher_index_list: list[int]):
        super().__init__(world_size, rank, teacher_index_list)
        self.model: list | None = None
        self.bridge: AutoBridge | None = None
        self.active_teacher_idx: int | None = None
        self.kl_policies: dict = {}
        self.memory_pool = None
        self._weights_backuper: TensorBackuper | None = None
        self._cpu_backup: list[tuple[torch.nn.Parameter, torch.Tensor]] = []

    @classmethod
    def runtime_env_vars(cls):
        """Set runtime environment variables for teacher actors."""
        return {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }

    @classmethod
    def validate_config(cls, config: DictConfig) -> DictConfig:
        """Validate and derive OPD Megatron teacher config."""
        if "opd" not in config:
            raise ValueError("Missing opd config for TeacherWorker")
        if "model" not in config.opd:
            raise ValueError("Missing opd.model config for TeacherWorker")

        config.opd.model.variable_seq_lengths = True
        config.opd.model.moe_token_dispatcher_type = "alltoall"
        config.opd.model.calculate_per_token_loss = True
        config.opd.model.moe_router_load_balancing_type = "none"

        if config.opd.model.bf16:
            config.opd.model.params_dtype = "torch.bfloat16"
        elif config.opd.model.fp16:
            config.opd.model.params_dtype = "torch.float16"

        config.opd.model.sequence_parallel = int(config.opd.model.tensor_model_parallel_size) > 1
        if config.opd.model.sequence_parallel:
            config.opd.env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        if (
            config.opd.model.virtual_pipeline_model_parallel_size is not None
            and config.opd.model.virtual_pipeline_model_parallel_size > 1
        ):
            config.opd.model.overlap_p2p_comm = True
        config.opd.model.batch_p2p_comm = not config.opd.model.overlap_p2p_comm
        return config

    def init(self, config: DictConfig):
        """Initialize distributed groups and load all assigned teachers."""
        super().init(config)
        self.memory_pool = build_memory_pool(config)

        configure_tracking(config)

        mpu.initialize_model_parallel(
            tensor_model_parallel_size=config.opd.model.tensor_model_parallel_size,
            pipeline_model_parallel_size=config.opd.model.pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size=config.opd.model.virtual_pipeline_model_parallel_size,
            context_parallel_size=config.opd.model.context_parallel_size,
            expert_model_parallel_size=config.opd.model.expert_model_parallel_size,
            expert_tensor_parallel_size=config.opd.model.expert_tensor_parallel_size,
        )
        logger.info(
            "[Teacher rank %s] parallel groups initialized: TP=%s PP=%s EP=%s CP=%s ETP=%s VP=%s",
            self.rank,
            mpu.get_tensor_model_parallel_world_size(),
            mpu.get_pipeline_model_parallel_world_size(),
            mpu.get_expert_model_parallel_world_size(),
            mpu.get_context_parallel_world_size(),
            mpu.get_expert_tensor_parallel_world_size(),
            mpu.get_virtual_pipeline_model_parallel_world_size(),
        )

        seed = config.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model_parallel_cuda_manual_seed(seed)

        # Build ds_index -> teacher_idx mapping from config
        teacher_name_to_idx: dict[str, int] = {}
        for t_idx, t_cfg in enumerate(config.opd.teachers):
            t_cfg_name = t_cfg.name
            teacher_name_to_idx[t_cfg_name] = t_idx

        self.ds_to_teacher_idx: dict[int, int] = {}
        for ds_idx, ds_cfg in enumerate(config.data_sources):
            t_name = ds_cfg.teacher_name
            if not t_name:
                raise ValueError(
                    f"data_sources[{ds_idx}].teacher_name is empty. "
                    f"When OPD is enabled, every data_source must specify a teacher_name."
                )
            if t_name not in teacher_name_to_idx:
                raise ValueError(
                    f"data_sources[{ds_idx}].teacher_name='{t_name}' not found in opd.teachers. "
                    f"Available: {list(teacher_name_to_idx.keys())}"
                )
            self.ds_to_teacher_idx[ds_idx] = teacher_name_to_idx[t_name]
        logger.info("[Teacher rank %s] ds_to_teacher_idx: %s", self.rank, self.ds_to_teacher_idx)

        # Load first teacher to initialize model structure
        self._load_teacher_model(self.teacher_index_list[0])
        self.kl_policies = build_kl_policies(self.config)

        # Only create backuper when multiple teachers need switching
        if len(self.teacher_index_list) > 1:
            self._weights_backuper = TensorBackuper(
                source_getter=lambda: (
                    (f"chunk{i}.{name}", param)
                    for i, m in enumerate(self.model)
                    for name, param in m.named_parameters()
                ),
            )
            self._weights_backuper.backup(str(self.teacher_index_list[0]))

            # Load and backup remaining teachers (reuse model structure)
            for teacher_idx in self.teacher_index_list[1:]:
                self._load_teacher_weights(teacher_idx)
                self._weights_backuper.backup(str(teacher_idx))
                self.active_teacher_idx = teacher_idx

    def _load_teacher_model(self, teacher_idx: int):
        """Load the first teacher model (full initialization with TP/PP, no DDP).

        ``hf_path`` is always required: the bridge builds the Megatron model from
        its config.json, which a dist checkpoint directory does not contain. When
        ``checkpoint_path`` is also set the weights come from that checkpoint and
        the HF weight read is skipped, mirroring how the student resumes.
        """
        hf_path = self.config.opd.teachers[teacher_idx].hf_path
        if not hf_path:
            raise ValueError(f"Missing hf_path for teacher {teacher_idx}")
        ckpt_dir = resolve_dist_ckpt_dir(
            self.config.opd.teachers[teacher_idx].get("dist_ckpt_path"),
            f"opd.teachers[{teacher_idx}].dist_ckpt_path",
        )

        self.bridge = AutoBridge.from_hf_pretrained(hf_path)
        provider = self.bridge.to_megatron_provider(load_weights=ckpt_dir is None)
        self._apply_model_config(provider)
        provider.finalize()

        self.model = provider.provide_distributed_model(
            wrap_with_ddp=False,
            mixed_precision_wrapper=partial(KeepFP32Module, self.config.megatron.keep_fp32_weights),
        )
        if ckpt_dir:
            # load_weights=False left randomly initialized real tensors; overwrite
            # them wholesale, same as the student's resume path.
            load_model_weights(self.model, ckpt_dir)
        self.active_teacher_idx = teacher_idx
        logger.info(
            "[Teacher rank %s] teacher %s loaded (source=%s)",
            self.rank, teacher_idx, f"ckpt:{ckpt_dir}" if ckpt_dir else f"hf:{hf_path}",
        )

    def _load_teacher_weights(self, teacher_idx: int):
        """Overwrite the live model's weights with teacher *teacher_idx*'s."""
        teacher = self.config.opd.teachers[teacher_idx]
        ckpt_dir = resolve_dist_ckpt_dir(
            teacher.get("dist_ckpt_path"), f"opd.teachers[{teacher_idx}].dist_ckpt_path"
        )
        if ckpt_dir:
            load_model_weights(self.model, ckpt_dir)
        else:
            self.bridge.load_hf_weights(self.model, teacher.hf_path)
        logger.info(
            "[Teacher rank %s] teacher %s weights loaded (source=%s)",
            self.rank, teacher_idx,
            f"ckpt:{ckpt_dir}" if ckpt_dir else f"hf:{teacher.hf_path}",
        )

    def _apply_model_config(self, provider):
        model_config_dict = OmegaConf.to_container(self.config.opd.model, resolve=True)
        hints = get_type_hints(GPTModelProvider)
        for key, value in model_config_dict.items():
            if not hasattr(provider, key):
                raise ValueError(f"Invalid opd.model config key: {key}")
            hint = hints.get(key)
            if hint is torch.dtype or hint is Optional[torch.dtype]:
                value = to_torch_dtype(value)
            setattr(provider, key, value)

    @torch.no_grad()
    def switch(self, teacher_idx):
        """Switch current teacher model parameters to another one."""
        if self.active_teacher_idx == teacher_idx:
            return
        self._weights_backuper.restore(str(teacher_idx))
        self.active_teacher_idx = teacher_idx
        clear_memory()

    @torch.no_grad()
    def onload(self):
        """Reload teacher model params to GPU from CPU pinned memory."""
        if self._weights_backuper:
            self._weights_backuper.restore(str(self.active_teacher_idx))
        else:
            for param, cpu_copy in self._cpu_backup:
                param.data.storage().resize_(cpu_copy.numel())
                param.data.copy_(cpu_copy, non_blocking=True)
            torch.cuda.synchronize()

    @torch.no_grad()
    def offload(self):
        """Backup active teacher weights to CPU and free GPU memory."""
        if self._weights_backuper is None and not self._cpu_backup:
            for model in self.model or []:
                for param in model.parameters():
                    self._cpu_backup.append((param, param.data.cpu().pin_memory()))
        for model in self.model or []:
            for param in model.parameters():
                param.data.storage().resize_(0)
        clear_memory()
    
    def compute_teacher(self, rollout_data_ref):
        """Orchestrate teacher forward for all assigned teachers on this worker.

        All ranks participate in forward (PP/TP communication requires this).
        Only output-rank workers (last-PP + TP0 + CP0) organize results by
        train_dp_rank and ray.put them; non-output-rank workers return ``{}``.

        Returns:
            ``dict[int, ray.ObjectRef]`` — keyed by ``train_dp_rank``.
            Each ref resolves to a ``RolloutBatch`` containing the subset of
            trajectories destined for that train DP rank, with fields:
            ``seq_index``, ``teacher_idx``, ``train_dp_ranks``, plus
            per-token teacher outputs (e.g. log-probs, top-k log-probs).
        """
        rollout_data = self._fetch_rollout_data(rollout_data_ref)
        self._attach_teacher_idx(rollout_data)

        teacher_batches = group_rollout_batch(
            rollout_data, "teacher_idx", allowed_keys=self.teacher_index_list,
        )

        if not teacher_batches:
            logger.info("[Teacher rank %s] compute_teacher: no teacher_batches, returning empty", self.rank)
            return {}

        # Process active teacher first to avoid unnecessary weight switch
        ordered_teachers = [self.active_teacher_idx] if self.active_teacher_idx in teacher_batches else []
        ordered_teachers += [t for t in teacher_batches if t != self.active_teacher_idx]
        is_output = self._is_output_rank()

        # All ranks execute forward (PP/TP communication requires all ranks to participate)
        # Accumulate per-teacher results into combined within the same loop on output ranks
        combined: RolloutBatch | None = None
        for teacher_idx in ordered_teachers:
            batch = teacher_batches[teacher_idx]
            result = self._run_single_teacher(batch, teacher_idx)
            logger.info("[Teacher rank %s] compute_teacher: teacher %s forward done", self.rank, teacher_idx)

            if not is_output:
                continue

            n = len(batch["seq_index"])
            result["seq_index"] = batch["seq_index"]
            result["teacher_idx"] = [teacher_idx] * n
            result["train_dp_ranks"] = batch["train_dp_ranks"]

            if combined is None:
                combined = {key: [] for key in result}
            for key, values in result.items():
                combined[key].extend(values)

        if not is_output:
            return {}

        per_rank = group_rollout_batch(combined, "train_dp_ranks")
        return {rank: ray.put(data) for rank, data in per_rank.items()}

    # ------------------------------------------------------------------
    # compute_teacher helpers
    # ------------------------------------------------------------------

    def _fetch_rollout_data(self, rollout_data_ref) -> RolloutBatch:
        """Fetch the rollout data shard(s) that this DP rank should process.

        The returned batch has trajectory-aligned bookkeeping fields embedded:
            - train_dp_ranks[i]: train DP rank that sequence i originated from
            - seq_index[i]:      position of sequence i within its original train DP shard
        """
        train_dp = len(rollout_data_ref)
        dp_size = mpu.get_data_parallel_world_size()
        dp_rank = mpu.get_data_parallel_rank()

        if train_dp == dp_size:
            data = ray.get(rollout_data_ref[dp_rank].ref)
            n = len(data["tokens"])
            data["train_dp_ranks"] = [dp_rank] * n
            data["seq_index"] = list(range(n))
        elif train_dp > dp_size:
            shards_per_dp = train_dp // dp_size
            start = dp_rank * shards_per_dp
            shards = []
            for i in range(start, start + shards_per_dp):
                shard = ray.get(rollout_data_ref[i].ref)
                n = len(shard["tokens"])
                shard["train_dp_ranks"] = [i] * n
                shard["seq_index"] = list(range(n))
                shards.append(shard)
            data = concat_rollout_batch(shards)
        else:
            ranks_per_shard = dp_size // train_dp
            source_idx = dp_rank // ranks_per_shard
            local_rank = dp_rank % ranks_per_shard
            shard = ray.get(rollout_data_ref[source_idx].ref)
            n = len(shard["tokens"])
            shard["train_dp_ranks"] = [source_idx] * n
            shard["seq_index"] = list(range(n))
            data = slice_rollout_batch(shard, ranks_per_shard, local_rank)

        return data

    def _attach_teacher_idx(self, rollout_data: RolloutBatch):
        """Annotate each trajectory with its teacher_idx based on ds_indices."""
        teacher_idx_list: list[int] = []
        for ds_idx in rollout_data["ds_indices"]:
            t_idx = self.ds_to_teacher_idx.get(int(ds_idx))
            if t_idx is None:
                raise ValueError(
                    f"ds_index={ds_idx} not found in ds_to_teacher_idx mapping: {self.ds_to_teacher_idx}"
                )
            teacher_idx_list.append(t_idx)
        rollout_data["teacher_idx"] = teacher_idx_list


    def _is_output_rank(self) -> bool:
        """True for TP-rank-0, CP-rank-0 on the last pipeline stage (one per DP rank)."""
        return (mpu.is_pipeline_last_stage() and
                mpu.get_tensor_model_parallel_rank() == 0 and
                mpu.get_context_parallel_rank() == 0)

    @torch.no_grad()
    def _run_single_teacher(self, rollout_data, teacher_idx: int) -> dict[str, list[torch.Tensor]]:
        """Run a single teacher forward pass. Auto-switches weights if needed.

        Rows are per-Segment (the source flattens at ``put_dp_shards_to_ray``), so
        each Segment is its own packed sequence — attention never crosses Segment
        boundaries, matching how the tokens were produced at rollout time.

        Returns:
            On the last PP stage: per-Segment teacher outputs (e.g. teacher_entropy,
            teacher_log_probs, teacher_topk_*). On other stages: empty dict.
        """
        if self.active_teacher_idx != teacher_idx:
            self.switch(teacher_idx)

        self._move_rollout_data_to_device(rollout_data)

        for module in self.model:
            module.eval()

        data_iterator, num_microbatches_list = get_data_iterator(
            self.config.trainer, self.model, rollout_data, use_single_mini_batch=True
        )

        return forward_teacher(self.config, self.model, data_iterator, num_microbatches_list[0],
                               policies=list(self.kl_policies.values()))

    def _move_rollout_data_to_device(self, rollout_data: RolloutBatch):
        """Move core tensor fields to the current CUDA device for teacher forward."""
        device = torch.cuda.current_device()
        for field in ("tokens", "loss_masks", "rollout_log_probs", "rollout_routed_experts"):
            if rollout_data.get(field):
                rollout_data[field] = [t.to(device) for t in rollout_data[field]]
        return rollout_data






