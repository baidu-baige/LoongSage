"""
TrainManager — lifecycle manager for distributed Megatron training workers.
"""
import ray
from omegaconf import DictConfig
from coda.backends.megatron import MegatronTrainWorker
from coda.utils.channel_helper import ChannelMeta

class TrainManager:
    """Manages a pool of ``TrainWorker`` Ray actors."""
    def __init__(self, config: DictConfig, resource_scheduler):
        self.config: DictConfig = config
        self._validate_config()
        world_size = self.config.trainer.num_nodes * self.config.trainer.num_gpus_per_node

        if self.config.trainer.backend == "megatron":
            env_vars = MegatronTrainWorker.runtime_env_vars()
            env_vars.update(dict(self.config.trainer.env_vars))
            remote_worker_cls = ray.remote(runtime_env={"env_vars": env_vars})(MegatronTrainWorker)
        else:
            raise ValueError(f"Unsupported backend '{self.config.trainer.backend}'")

        # Initialize train workers, each worker occupies one GPU
        # Store the Ray actor handle for each worker
        self._worker_handlers = [
            resource_scheduler.schedule(remote_worker_cls)[0].remote(world_size, rank) for rank in range(world_size)
        ]

        # rank0 serves as master
        master_addr, master_port = ray.get(
            self._worker_handlers[0].get_ip_port.remote()
        )
        self.config.trainer.master_addr = master_addr
        self.config.trainer.master_port = master_port

    def _validate_config(self):
        if self.config.trainer.backend == "megatron":
            self.config = MegatronTrainWorker.validate_config(self.config)

    def async_init(self):
        """Initialize train workers"""
        return [worker.init.remote(self.config) for worker in self._worker_handlers]

    def async_train(self, step, rollout_data_ref):
        """Training entry point, input parameter format TBD"""
        return [worker.train.remote(step, rollout_data_ref) for worker in self._worker_handlers]

    def async_flush_train_metrics(self):
        """Trigger each worker to flush its per-step train-metrics aggregator"""
        return [worker.flush_train_metrics.remote() for worker in self._worker_handlers]

    def async_save_model(self, step, checkpoint=True, hf=False):
        """Save checkpoint/hf_models. Pass ``step=None`` to only flush pending async saves."""
        return [worker.save_model.remote(step, checkpoint, hf) for worker in self._worker_handlers]

    def async_update_weights(self, channel_meta: ChannelMeta):
        """Update weights"""
        return [worker.update_weights.remote(channel_meta) for worker in self._worker_handlers]

    def onload(self):
        """Onload: move from CPU memory to GPU memory"""
        return ray.get([worker.onload.remote() for worker in self._worker_handlers])

    def offload(self):
        """Offload: move from GPU memory to CPU memory"""
        return ray.get([worker.offload.remote() for worker in self._worker_handlers])
