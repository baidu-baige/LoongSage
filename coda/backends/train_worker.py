"""Base train worker for different backends"""
import os
from abc import ABC, abstractmethod
import ray
import pynvml
import torch.distributed as dist
from coda.utils import http_utils
from coda.utils import distributed_utils
from coda.utils import logging_utils
from omegaconf import DictConfig
from datetime import timedelta
from coda.utils.channel_helper import ChannelMeta
import logging

logger = logging.getLogger(__name__)

class TrainWorker(ABC):
    """Base class for TrainWorker, defining interfaces that all backends must implement"""

    def __init__(self, world_size, rank):
        self.world_size = world_size
        self.rank = rank
        self._set_numa_affinity()

    def _set_numa_affinity(self):
        # pynvml
        try:
            pynvml.nvmlInit()
            local_rank = ray.get_gpu_ids()[0]
            handle = pynvml.nvmlDeviceGetHandleByIndex(local_rank)
            pynvml.nvmlDeviceSetCpuAffinity(handle)
            pynvml.nvmlShutdown()
        except Exception as e:
            logger.warning(f"Failed to set numa affinity: {e}")

    def get_ip_port(self):
        """Discover node ip and a free port"""
        return ray.util.get_node_ip_address(), http_utils.find_available_port(20000)

    @abstractmethod
    def init(self, config: DictConfig):
        """Initialize train worker"""
        self.config = config

        logging_utils.configure_logger(level=config.log_level)

        # init nccl process group
        os.environ["MASTER_ADDR"] = self.config.trainer.master_addr
        os.environ["MASTER_PORT"] = str(self.config.trainer.master_port)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["RANK"] = str(self.rank)
        nccl_timeout = (
            timedelta(minutes=config.trainer.nccl_timeout_minutes)
            if config.trainer.nccl_timeout_minutes else None
        )
        dist.init_process_group("nccl", timeout=nccl_timeout)

        # init gloo process group
        gloo_timeout = (
            timedelta(minutes=config.trainer.gloo_timeout_minutes)
            if config.trainer.gloo_timeout_minutes else None
        )
        distributed_utils.init_gloo_group(timeout=gloo_timeout)

    @abstractmethod
    def train(self, step, rollout_data_ref):
        """Training entry point"""

    @abstractmethod
    def save_model(self, step, checkpoint=True, hf=False):
        """Save checkpoint/hf_models. Pass ``step=None`` to only flush pending async saves."""

    @abstractmethod
    def update_weights(self, channel_meta: ChannelMeta):
        """Update weights"""

    @abstractmethod
    def onload(self):
        """Load from CPU RAM to VRAM"""

    @abstractmethod
    def offload(self):
        """Offload from VRAM to CPU RAM"""
