"""
TrainManager — lifecycle manager for distributed Megatron training workers.

Always manages the actor (policy) pool. With ``critic.enable`` it also manages
a PPO critic pool pinned onto the same GPUs, and hides it entirely: the
trainer keeps calling the same methods, and only this class knows whether a
critic exists, when it holds the GPU, and how its advantages reach the actor.
"""
import logging

import ray
from omegaconf import DictConfig

from coda.backends.megatron import MegatronActorWorker, MegatronCriticWorker
from coda.transfer_mesh import ChannelMeta
from coda.utils.checkpoint_utils import should_save

logger = logging.getLogger(__name__)


class TrainManager:
    """Manages the actor ``TrainWorker`` Ray actors and, for PPO, the critic ones.

    GPU sharing with a critic: both pools hold one rank per GPU and run one at
    a time. ``_resident`` names the pool that may still hold GPU memory
    (``None`` when neither does); only that pool is offloaded before the other
    is onloaded. ``offload(move_params=False)`` clears ``_resident`` but keeps
    the actor's parameters on GPU for weight sync, and ``update_weights``
    releases them afterwards. A caller that offloads without a following weight
    sync must pass ``move_params=True``, or the parameters stay on GPU while
    the critic onloads.
    """
    def __init__(self, config: DictConfig, resource_scheduler):
        self.config: DictConfig = config
        world_size = self.config.trainer.num_nodes * self.config.trainer.num_gpus_per_node

        if self.config.trainer.backend != "megatron":
            raise ValueError(f"Unsupported backend '{self.config.trainer.backend}'")
        self._validate_config()
        env_vars = dict(self.config.trainer.env_vars)

        # Initialize train workers, each worker occupies one GPU.
        actor_cls = ray.remote(runtime_env={"env_vars": env_vars})(MegatronActorWorker)
        self._actor_worker_handlers = []
        self.bundle_indices: list[int] = []
        for rank in range(world_size):
            prepared, bundle_index = resource_scheduler.schedule(actor_cls)
            self._actor_worker_handlers.append(prepared.remote(world_size, rank))
            self.bundle_indices.append(bundle_index)

        # The critic rank r shares the GPU of actor rank r.
        self._critic_worker_handlers = []
        if self.config.critic.enable:
            critic_cls = ray.remote(runtime_env={"env_vars": env_vars})(MegatronCriticWorker)
            self._critic_worker_handlers = [
                resource_scheduler.schedule(critic_cls, bundle_index=bundle_index)[0].remote(world_size, rank)
                for rank, bundle_index in enumerate(self.bundle_indices)
            ]
        self._resident = None

    def _validate_config(self):
        self.config = MegatronActorWorker.validate_config(self.config)
        if self.config.critic.enable:
            self.config = MegatronCriticWorker.validate_config(self.config)
        elif self.config.algorithm.advantage_estimator == "gae" :
            raise ValueError("algorithm.advantage_estimator=gae requires critic.enable=true.")

    def _make_critic_resident(self):
        """Give the GPU to the critic: release the actor fully if resident, then onload the critic."""
        if self._resident == "critic":
            return
        if self._resident == "actor":
            ray.get([w.offload.remote(True) for w in self._actor_worker_handlers])
        ray.get([w.onload.remote() for w in self._critic_worker_handlers])
        self._resident = "critic"

    # ── public API (the same with or without a critic) ───────────────────

    def async_init(self):
        """Initialize the train workers; returns the actor's init refs.

        With a critic, the critic pool is initialized and fully offloaded
        inside this call first, so two full models + optimizers never sit on
        the GPU together. The actor's port is picked after the critic's
        rendezvous is bound, so the two worlds cannot collide. The actor is
        then initialized asynchronously and stays resident on GPU.
        """
        if self.config.critic.enable:
            critic_addr, critic_port = ray.get(self._critic_worker_handlers[0].get_ip_port.remote())
            ray.get([w.init.remote(self.config, critic_addr, critic_port) for w in self._critic_worker_handlers])
            ray.get([w.offload.remote(True) for w in self._critic_worker_handlers])

        actor_addr, actor_port = ray.get(self._actor_worker_handlers[0].get_ip_port.remote())
        actor_refs = [w.init.remote(self.config, actor_addr, actor_port) for w in self._actor_worker_handlers]
        self._resident = "actor"
        return actor_refs

    def async_train(self, step, rollout_data_ref):
        """Training entry point; returns the actor's train refs.

        With a critic, the critic step runs (blocking) first: it trains the
        value model and attaches each DP rank's advantages to
        ``rollout_data_ref[dp].critic_ref``. It is saved before being offloaded
        (offloaded parameters cannot be saved). During critic warmup
        (``step <= critic.critic_warmup``) the actor is not trained and ``[]``
        is returned.
        """
        if not self.config.critic.enable:
            return [w.train.remote(step, rollout_data_ref) for w in self._actor_worker_handlers]

        self._make_critic_resident()
        outputs = ray.get([w.train.remote(step, rollout_data_ref) for w in self._critic_worker_handlers])
        for out in outputs:
            for dp_rank, ref in out.items():
                rollout_data_ref[dp_rank].critic_ref = ref

        if should_save(self.config, step) and self.config.trainer.save_checkpoint:
            ray.get([w.save_model.remote(step) for w in self._critic_worker_handlers])
        ray.get([w.offload.remote(True) for w in self._critic_worker_handlers])
        self._resident = None

        if step <= int(self.config.critic.critic_warmup):
            logger.info(f"[TrainManager] step {step}: critic warmup, actor not trained")
            return []

        ray.get([w.onload.remote() for w in self._actor_worker_handlers])
        self._resident = "actor"
        return [w.train.remote(step, rollout_data_ref) for w in self._actor_worker_handlers]

    def async_save_model(self, step, checkpoint=True, hf=False):
        """Save the actor's checkpoint/hf_models. Pass ``step=None`` to only flush pending async saves.

        The critic is saved inside :meth:`async_train`. During critic warmup
        the actor has not changed and is not saved. On flush the critic is
        drained first, so the actor's finalize (which writes the shared tracker)
        never points at a critic checkpoint that is still being written.
        """
        if self.config.critic.enable:
            if step is None:
                ray.get([w.save_model.remote(None) for w in self._critic_worker_handlers])
            elif step <= int(self.config.critic.critic_warmup):
                return []
        return [w.save_model.remote(step, checkpoint, hf) for w in self._actor_worker_handlers]

    def async_update_weights(self, channel_meta: ChannelMeta):
        """Push the actor's weights to the rollout engines."""
        return [w.update_weights.remote(channel_meta) for w in self._actor_worker_handlers]

    def onload(self):
        """Onload for the next training step: move from CPU memory to GPU memory.

        With a critic, the critic trains first, so the critic is onloaded (and
        the actor fully released); the actor follows inside :meth:`async_train`.
        """
        if self.config.critic.enable:
            self._make_critic_resident()
            return None
        self._resident = "actor"
        return ray.get([w.onload.remote() for w in self._actor_worker_handlers])

    def offload(self, move_params: bool = False):
        """Offload the resident pool: move from GPU memory to CPU memory.

        A no-op when neither pool is resident.

        Args:
            move_params: Also release the actor's parameters (see
                ``MegatronTrainWorker.offload``). The critic, when resident, is
                always offloaded completely.
        """
        result = None
        if self._resident == "critic":
            result = ray.get([w.offload.remote(True) for w in self._critic_worker_handlers])
        elif self._resident == "actor":
            result = ray.get([w.offload.remote(move_params) for w in self._actor_worker_handlers])
        self._resident = None
        return result
