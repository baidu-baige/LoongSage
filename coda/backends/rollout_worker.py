"""Rollout worker implementation for rollout inference.

This module provides a Ray actor wrapper around inference engine HTTP servers,
supporting distributed deployment, weight updates, and fault tolerance.
"""

from abc import ABC, abstractmethod
import logging
import ray

from coda.utils.http_utils import find_available_port
logger = logging.getLogger(__name__)

class RolloutWorker(ABC):
    """Ray actor wrapper for inference engine HTTP server.

    Manages the lifecycle of inference engines, including
    initialization, health monitoring, weight updates, and shutdown.
    """

    @staticmethod
    def runtime_env_vars(cls):
        """runtime env"""
        return {}

    def get_ip_port(self, base_port=15000, consecutive=1):
        """Discover ip and free port."""
        return ray.util.get_node_ip_address(), find_available_port(base_port, consecutive)

    def __init__(
        self,
        config,
        rank: int,
        worker_type: str = "regular",
        base_gpu_id: int | None = None,
        engine_args: dict | None = None,
        num_gpus_per_replica: int | None = None,
    ):
        """Initialize the inference engine actor.

        Args:
            config: Configuration arguments.
            rank: The engine rank.
            worker_type: The worker type ("regular", "prefill", or "decode").
            base_gpu_id: The base GPU ID for this engine.
            engine_args: inference engine server arguments.
            num_gpus_per_replica: Number of GPUs per replica.
        """
        self.config = config
        self.rank = rank
        self.worker_type = worker_type
        self.base_gpu_id = base_gpu_id
        self.engine_args = engine_args or {}
        self.num_gpus_per_replica = num_gpus_per_replica
        # Tracks whether the prefix/KV cache has already been flushed for the current
        # weight version. Set when the offload path flushes; consumed and cleared by the
        # next weight update so each version flushes exactly once. See release_memory_occupation
        # and update_weights_from_channel in the concrete engine.
        self._cache_flushed = False

    @abstractmethod
    def init(
        self,
        dist_init_addr,
        port,
        nccl_port,
        host=None,
        disaggregation_bootstrap_port=None,
        router_ip=None,
        router_port=None,
    ):
        """Initialize the inference engine and launch the HTTP server.

        Args:
            dist_init_addr: Distributed initialization address.
            port: Server port.
            nccl_port: NCCL communication port.
            host: Host address.
            disaggregation_bootstrap_port: Port for prefill-decode disaggregation.
            router_ip: Router IP address.
            router_port: Router port.
        """

    @abstractmethod
    def health_generate(self, timeout: float = 5.0) -> bool:
        """Run /health_generate on the underlying inference engine HTTP server.

        Args:
            timeout: Timeout for the health request in seconds.

        Returns:
            True if the server responds with HTTP 200.

        Raises:
            requests.RequestException: If the request fails for any reason, including timeout.
        """

    @abstractmethod
    def shutdown(self, timeout: float = 30.0):
        """Shutdown the inference engine and clean up resources."""

    @abstractmethod
    def release_memory_occupation(self):
        """Release memory occupation for offloading."""

    @abstractmethod
    def resume_memory_occupation(self, tags: list[str] = None):
        """Resume memory occupation after offloading.

        Args:
            tags: Available tags for multi-stage resume: weights, kv_cache.
        """

    @abstractmethod
    def update_weights_from_channel(self, meta_dict: dict):
        """Pull new weights over the already-established transfer-mesh channel.

        Args:
            meta_dict: The ChannelMeta dict
        """