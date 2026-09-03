"""Base teacher worker for different backends."""

import logging
import os
from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Any

import pynvml
import ray
import torch.distributed as dist
from omegaconf import DictConfig

from coda.utils import distributed_utils, http_utils, logging_utils

logger = logging.getLogger(__name__)


class TeacherWorker(ABC):
    """Base class for TeacherWorker, defining interfaces that all backends must implement."""

    def __init__(self, world_size: int, rank: int, teacher_index_list: list[int]):
        self.world_size = world_size
        self.rank = rank
        self.teacher_index_list = teacher_index_list
        # Registries are per-process: this Ray actor resolves KL policy names
        # itself, so user extensions have to be loaded here too and not only in
        # the driver entry point.
        import coda.custom  # noqa: F401,PLC0415
        self._set_numa_affinity()

    def _set_numa_affinity(self):
        try:
            pynvml.nvmlInit()
            local_rank = ray.get_gpu_ids()[0]
            handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
            pynvml.nvmlDeviceSetCpuAffinity(handle)
            pynvml.nvmlShutdown()
        except Exception as e:
            logger.warning(f"Failed to set numa affinity: {e}")

    def get_ip_port(self):
        """Discover node IP and a free port."""
        return ray.util.get_node_ip_address(), http_utils.find_available_port(20000)

    @abstractmethod
    def init(self, config: DictConfig, **kwargs):
        """Initialize teacher worker."""
        self.config = config

        logging_utils.configure_logger(level=config.log_level)

        os.environ["MASTER_ADDR"] = config.opd.master_addr
        os.environ["MASTER_PORT"] = str(config.opd.master_port)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["RANK"] = str(self.rank)
        nccl_timeout = (
            timedelta(minutes=config.trainer.nccl_timeout_minutes)
            if config.trainer.nccl_timeout_minutes else None
        )
        if not dist.is_initialized():
            dist.init_process_group("nccl", timeout=nccl_timeout)

        gloo_timeout = (
            timedelta(minutes=config.trainer.gloo_timeout_minutes)
            if config.trainer.gloo_timeout_minutes else None
        )
        distributed_utils.init_gloo_group(timeout=gloo_timeout)

    @abstractmethod
    def onload(self) -> dict[str, Any]:
        """Load teacher to GPU."""

    @abstractmethod
    def offload(self) -> dict[str, Any]:
        """Offload teacher from GPU."""

    @abstractmethod
    def compute_teacher(self, rollout_data_ref) -> dict[str, Any]:
        """Compute teacher outputs for the assigned teachers on this worker."""


