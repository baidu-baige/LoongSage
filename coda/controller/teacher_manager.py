"""Controller-side manager for OPD teacher workers."""

from __future__ import annotations

import copy
import logging
from typing import Any

import ray
from omegaconf import DictConfig

from coda.backends.megatron.megatron_teacher_worker import MegatronTeacherWorker

logger = logging.getLogger(__name__)


class TeacherManager:
    """Controller-side orchestrator for OPD teacher GPU pools."""

    def __init__(self, config: DictConfig, resource_scheduler):
        self.config: DictConfig = config
        self._validate_config()

        self.teacher_world_size = self._get_teacher_world_size()
        total_world_size = config.opd.teacher_nodes * config.opd.teacher_gpus_per_node
        num_teachers = len(config.opd.teachers)

        group_num = total_world_size // self.teacher_world_size
        group_teacher_num = num_teachers // group_num

        # Build remote worker class
        if config.trainer.backend == "megatron":
            env_vars = MegatronTeacherWorker.runtime_env_vars()
            env_vars.update(dict(config.opd.get("env_vars", {})))
            remote_worker_cls = ray.remote(runtime_env={"env_vars": env_vars})(MegatronTeacherWorker)
        else:
            raise ValueError(f"Unsupported backend '{config.trainer.backend}'")

        # Create worker handlers per group
        self._worker_handlers: list[Any] = []
        for group_index in range(group_num):
            teacher_index_list = list(range(
                group_index * group_teacher_num,
                (group_index + 1) * group_teacher_num,
            ))
            for rank in range(self.teacher_world_size):
                self._worker_handlers.append(
                    resource_scheduler.schedule(remote_worker_cls)[0].remote(
                        self.teacher_world_size, rank, teacher_index_list
                    )
                )

    def _get_teacher_world_size(self) -> int:
        """World size of a single teacher group = dp_per_teacher * TP * PP * CP."""
        tp = self.config.opd.model.tensor_model_parallel_size
        pp = self.config.opd.model.pipeline_model_parallel_size
        cp = self.config.opd.model.context_parallel_size
        gpu_per_dp = tp * pp * cp

        total_world_size = self.config.opd.teacher_nodes * self.config.opd.teacher_gpus_per_node
        num_dp = total_world_size // gpu_per_dp
        num_teachers = len(self.config.opd.teachers)

        if num_dp >= num_teachers:
            dp_per_teacher = num_dp // num_teachers
        else:
            dp_per_teacher = 1

        return dp_per_teacher * gpu_per_dp

    def _validate_config(self):
        """Validate OPD config: teachers, parallelism, train_dp/teacher_dp divisibility."""
        if self.config.trainer.backend == "megatron":
            self.config = MegatronTeacherWorker.validate_config(self.config)

        if not self.config.opd.get("enable", False):
            raise ValueError("TeacherManager requires opd.enable=true")
        if not self.config.opd.get("teachers"):
            raise ValueError("TeacherManager requires at least one teacher in opd.teachers")
        for idx, teacher in enumerate(self.config.opd.teachers):
            hf_path = teacher.get("hf_path") if hasattr(teacher, "get") else teacher.hf_path
            if not hf_path:
                raise ValueError(f"Missing hf_path for opd.teachers[{idx}]")

        total_world_size = int(self.config.opd.teacher_nodes) * int(self.config.opd.teacher_gpus_per_node)
        if total_world_size <= 0:
            raise ValueError("opd.teacher_nodes * opd.teacher_gpus_per_node must be > 0")

        tp = self.config.opd.model.tensor_model_parallel_size
        pp = self.config.opd.model.pipeline_model_parallel_size
        cp = self.config.opd.model.context_parallel_size
        gpu_per_dp = tp * pp * cp
        if gpu_per_dp <= 0:
            raise ValueError("opd.model TP * PP * CP must be > 0")
        if total_world_size % gpu_per_dp != 0:
            raise ValueError(
                f"Teacher world size {total_world_size} must be divisible by TP*PP*CP ({gpu_per_dp})"
            )

        num_dp = total_world_size // gpu_per_dp
        num_teachers = len(self.config.opd.teachers)
        if num_dp % num_teachers != 0 and num_teachers % num_dp != 0:
            raise ValueError(
                f"num_dp ({num_dp}) and num_teachers ({num_teachers}) must have a divisibility relationship"
            )

        # train_dp and teacher_dp must be mutually divisible
        train_world_size = int(self.config.trainer.num_nodes) * int(self.config.trainer.num_gpus_per_node)
        train_gpu_per_dp = (
            int(self.config.megatron.model.tensor_model_parallel_size)
            * int(self.config.megatron.model.pipeline_model_parallel_size)
            * int(self.config.megatron.model.context_parallel_size)
        )
        train_dp = train_world_size // train_gpu_per_dp
        if train_dp % num_dp != 0 and num_dp % train_dp != 0:
            raise ValueError(
                f"train_dp ({train_dp}) and teacher_dp ({num_dp}) must be mutually divisible"
            )
        
        if self.config.opd.pg_ratio <= 0 and self.config.opd.gkd_ratio <= 0:
            raise ValueError("opd.pg_ratio or opd.gkd_ratio must be > 0")

        if self.config.opd.gkd_ratio > 1:
            raise ValueError("gkd_ratio must be <= 1")

        if self.config.opd.pg_ratio > 0 and self.config.opd.gkd_ratio == 1:
            raise ValueError("opd.pg_ratio > 0 and opd.gkd_ratio == 1 are mutually exclusive")

    def async_init(self):
        """Initialize all teacher workers."""
        refs = []
        group_config = None
        for worker_index, worker in enumerate(self._worker_handlers):
            if worker_index % self.teacher_world_size == 0:
                master_addr, master_port = ray.get(worker.get_ip_port.remote())
                group_config = copy.deepcopy(self.config)
                group_config.opd.master_addr = master_addr
                group_config.opd.master_port = master_port
            refs.append(worker.init.remote(group_config))
        return refs

    def onload(self):
        """Onload each worker's assigned teacher onto GPU."""
        refs = [w.onload.remote() for w in self._worker_handlers]
        return ray.get(refs)

    def offload(self):
        """Offload active teacher from GPU."""
        refs = [w.offload.remote() for w in self._worker_handlers]
        return ray.get(refs)

    def compute_teacher(self, rollout_data_ref):
        """Dispatch teacher forward to all workers and collect results.

        Each output-rank worker returns ``dict[int, ray.ObjectRef]`` mapping
        train_dp_rank to teacher output refs. This method aggregates them
        into ``rollout_data_ref[rank].teacher_worker_ref``.

        Returns:
            The updated ``rollout_data_ref`` with teacher refs attached.
        """
        all_refs = [worker.compute_teacher.remote(rollout_data_ref) for worker in self._worker_handlers]
        all_results = ray.get(all_refs)

        for result in all_results:
            if not result:
                continue
            for rank, ref in result.items():
                rollout_data_ref[rank].teacher_worker_ref.append(ref)

        return rollout_data_ref
