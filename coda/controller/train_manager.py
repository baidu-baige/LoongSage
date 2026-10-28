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
    a time. ``_resident`` names the pool currently onloaded for training
    (``None`` when neither is). The actor may still keep its parameters on GPU
    while not resident (``offload(move_params=False)`` keeps them for weight
    sync); they are released before the critic onloads.
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
        estimator = self.config.algorithm.advantage_estimator
        if self.config.critic.enable:
            self.config = MegatronCriticWorker.validate_config(self.config)
            # PPO constraints; checked here so the trainer needs no critic knowledge.
            if estimator != "gae":
                raise ValueError(
                    f"critic.enable requires algorithm.advantage_estimator=gae, got {estimator!r}."
                )
            if int(self.config.critic.critic_warmup) < 0:
                raise ValueError(f"critic.critic_warmup must be >= 0, got {self.config.critic.critic_warmup}")
            if self.config.opd.enable:
                raise ValueError("critic.enable and opd.enable cannot be used together yet.")
            for idx, ds in enumerate(self.config.data_sources):
                if int(ds.num_trajectories_per_prompt) != 1:
                    raise ValueError(
                        f"PPO (critic.enable) currently requires num_trajectories_per_prompt == 1, "
                        f"got {ds.num_trajectories_per_prompt} for data_sources[{idx}]."
                    )
        elif estimator == "gae":
            raise ValueError("algorithm.advantage_estimator=gae requires critic.enable=true.")

    # ── worker-pool helpers ──────────────────────────────────────────────

    @staticmethod
    def _master(workers) -> tuple[str, int]:
        """Rank 0 address + a free port for one pool's torch.distributed rendezvous."""
        return ray.get(workers[0].get_ip_port.remote())

    @staticmethod
    def _onload_pool(workers):
        return ray.get([w.onload.remote() for w in workers])

    @staticmethod
    def _offload_pool(workers, move_params: bool):
        return ray.get([w.offload.remote(move_params) for w in workers])

    def _make_critic_resident(self):
        """Give the GPU to the critic: release the actor fully, then onload the critic."""
        if self._resident == "critic":
            return
        # Also drops actor params kept on GPU by offload(move_params=False).
        self._offload_pool(self._actor_worker_handlers, True)
        self._onload_pool(self._critic_worker_handlers)
        self._resident = "critic"

    # ── public API (the same with or without a critic) ───────────────────

    def async_init(self):
        """Initialize the train workers.

        Without a critic this is asynchronous. With a critic the two
        pools are initialized one after the other, inside this call, so two
        full models + optimizers never sit on the GPU together; each pool is
        offloaded right after its init. The critic's port is picked after the
        actor's rendezvous is bound, so the two worlds cannot collide. Returns
        ``[]`` in that case so ``ray.get(async_init())`` still works.
        """
        actor_addr, actor_port = self._master(self._actor_worker_handlers)
        if not self.config.critic.enable:
            return [w.init.remote(self.config, actor_addr, actor_port) for w in self._actor_worker_handlers]

        ray.get([w.init.remote(self.config, actor_addr, actor_port) for w in self._actor_worker_handlers])
        self._offload_pool(self._actor_worker_handlers, True)
        critic_addr, critic_port = self._master(self._critic_worker_handlers)
        ray.get([w.init.remote(self.config, critic_addr, critic_port) for w in self._critic_worker_handlers])
        self._offload_pool(self._critic_worker_handlers, True)
        self._resident = None
        logger.info("[TrainManager] actor and critic pools initialized, both offloaded")
        return []

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
        missing = [dp for dp, box in enumerate(rollout_data_ref) if box.critic_ref is None]
        # TODO: remove once validated in end-to-end testing.
        if missing:
            raise RuntimeError(f"Critic returned no advantages for DP ranks {missing}")

        if should_save(self.config, step) and self.config.trainer.save_checkpoint:
            ray.get([w.save_model.remote(step) for w in self._critic_worker_handlers])
        self._offload_pool(self._critic_worker_handlers, True)
        self._resident = None

        if step <= int(self.config.critic.critic_warmup):
            logger.info(f"[TrainManager] step {step}: critic warmup, actor not trained")
            return []

        self._onload_pool(self._actor_worker_handlers)
        self._resident = "actor"
        return [w.train.remote(step, rollout_data_ref) for w in self._actor_worker_handlers]

    def async_save_model(self, step, checkpoint=True, hf=False):
        """Save the actor's checkpoint/hf_models. Pass ``step=None`` to only flush pending async saves.

        The critic is saved inside :meth:`async_train`. During critic warmup
        the actor has not changed and is not saved.
        """
        if self.config.critic.enable and step is not None and step <= int(self.config.critic.critic_warmup):
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
        return self._onload_pool(self._actor_worker_handlers)

    def offload(self, move_params: bool = False):
        """Offload: move from GPU memory to CPU memory

        Args:
            move_params: Also release the actor's parameters (see
                ``MegatronTrainWorker.offload``). The critic, when resident, is
                always offloaded completely.
        """
        if self._resident == "critic":
            self._offload_pool(self._critic_worker_handlers, True)
        result = self._offload_pool(self._actor_worker_handlers, move_params)
        self._resident = None
        return result
