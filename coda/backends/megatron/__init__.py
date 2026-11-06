"""
Megatron backend: model loading, checkpoint, distributed init, train workers,
and weight transfer.
"""
from coda.backends.megatron.megatron_train_worker import MegatronTrainWorker
from coda.backends.megatron.megatron_actor_worker import MegatronActorWorker
from coda.backends.megatron.megatron_critic_worker import MegatronCriticWorker

__all__ = ["MegatronTrainWorker", "MegatronActorWorker", "MegatronCriticWorker"]
