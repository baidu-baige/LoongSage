"""
Megatron backend: model loading, checkpoint, distributed init, train worker,
and weight transfer.
"""
from coda.backends.megatron.megatron_train_worker import MegatronTrainWorker

__all__ = ["MegatronTrainWorker"]
