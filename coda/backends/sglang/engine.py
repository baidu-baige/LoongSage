"""Rollout worker implementation for rollout inference.

This module provides a Ray actor wrapper around SGLang HTTP servers,
supporting distributed deployment, weight updates, and fault tolerance.
"""

import logging
import multiprocessing
import os
import time
from typing import override

import requests
from omegaconf import OmegaConf
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree
from coda.backends.rollout_worker import RolloutWorker
from coda.utils.logging_utils import redact_secrets
logger = logging.getLogger(__name__)

class SglangEngine(RolloutWorker):
    """Ray actor wrapper for SGLang HTTP server.

    Manages the lifecycle of SGLang inference engines, including
    initialization, health monitoring, weight updates, and shutdown.
    """
    @classmethod
    def runtime_env_vars(cls) -> dict:
        """Return runtime environment variables for the SGLang engine."""
        return  {
            key: os.environ.get(key, default_val)
            for key, default_val in {
                "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "false",
                "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
                "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",
                "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "true",
                "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
                "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "false",
                # Extend detokenizer watchdog timeout to survive weight-sync (NCCL broadcast
                # for large models can block the GPU for >20s, triggering the default 20s timeout).
                "SGLANG_HEALTH_CHECK_TIMEOUT": "300",

                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            }.items()
        }

    @classmethod
    def validate_config(cls, config):
        sglang_args = config.rollout.sglang_args
        for _, replica_cfg in config.rollout.sglang_replicas.items():
            replica_cfg.sglang_args = OmegaConf.merge(sglang_args, replica_cfg.get("sglang_args", {}))

        # regular and (prefill or decode) are exclusive
        if config.rollout.sglang_replicas.regular.num_nodes > 0:
            assert config.rollout.sglang_replicas.prefill.num_nodes == 0
            assert config.rollout.sglang_replicas.decode.num_nodes == 0
        if config.rollout.sglang_replicas.regular.num_nodes == 0:
            assert config.rollout.sglang_replicas.prefill.num_nodes > 0
            assert config.rollout.sglang_replicas.decode.num_nodes > 0

        return config
    

    @override
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
        """Initialize the SGLang engine and launch the HTTP server.

        Args:
            dist_init_addr: Distributed initialization address.
            port: Server port.
            nccl_port: NCCL communication port.
            host: Host address.
            disaggregation_bootstrap_port: Port for prefill-decode disaggregation.
            router_ip: Router IP address.
            router_port: Router port.
        """
        self.router_ip = router_ip
        self.router_port = router_port

        server_args_dict = _compute_server_args(
            self.config,
            self.rank,
            dist_init_addr,
            nccl_port,
            host,
            port,
            self.worker_type,
            disaggregation_bootstrap_port,
            base_gpu_id=self.base_gpu_id,
            engine_args=self.engine_args,
            num_gpus_per_replica=self.num_gpus_per_replica,
        )

        self.node_rank = server_args_dict["node_rank"]
        self.server_host = server_args_dict["host"]  # with [] if ipv6
        self.server_port = server_args_dict["port"]

        self._init_server(server_args_dict)

    def _init_server(self, server_args_dict, timeout: float = 30.0):
        """Initialize the SGLang server.

        Args:
            server_args_dict: Dictionary containing server configuration arguments.
        """
        logger.info(f"Launch HttpServerEngineAdapter at: {self.server_host}:{self.server_port}")
        self.process = launch_server_process(ServerArgs(**server_args_dict))

        if self.node_rank == 0 and self.router_ip and self.router_port:
            payload = {
                "worker_url": f"http://{self.server_host}:{self.server_port}",
            }
            if self.worker_type == "prefill":
                payload["bootstrap_port"] = server_args_dict["disaggregation_bootstrap_port"]
            response = requests.post(
                f"http://{self.router_ip}:{self.router_port}/add_worker",
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()

    def _make_request(self, endpoint: str, payload: dict | None = None, timeout: float | None = None):
        """Make a POST request to the specified endpoint with the given payload.

        Args:
            endpoint: The API endpoint to call
            payload: The JSON payload to send (default: empty dict)
            timeout: Timeout for the request in seconds (falls back to 30.0)

        Returns:
            The JSON response from the server, or None on non-zero node_rank.
        """
        if self.node_rank != 0:
            return

        url = f"http://{self.server_host}:{self.server_port}/{endpoint}"
        response = requests.post(url, json=payload or {}, timeout=timeout or 30.0)
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            e.add_note(f"{response.text=}")
            raise
        return response.json()

    @override
    def health_generate(self, timeout: float = 5.0) -> bool:
        """Run /health_generate on the underlying SGLang HTTP server.

        Args:
            timeout: Timeout for the health request in seconds.

        Returns:
            True if the server responds with HTTP 200.

        Raises:
            requests.RequestException: If the request fails for any reason, including timeout.
        """
        if self.node_rank != 0:
            return True

        response = requests.get(
            f"http://{self.server_host}:{self.server_port}/health_generate",
            timeout=timeout,
        )
        response.raise_for_status()
        return True

    def _flush_cache(self, timeout: float = 30.0):
        """Flush the cache of the server."""
        if self.node_rank != 0:
            return
        # flush cache will not return status_code 200 when there are pending requests
        for _ in range(20):
            try:
                response = requests.get(
                    f"http://{self.server_host}:{self.server_port}/flush_cache",
                    timeout=timeout,
                )
                if response.status_code == 200:
                    break
                response.raise_for_status()
            except Exception as e:
                logger.info(f"Error flushing cache: {e}")
                time.sleep(1)
                continue
        else:
            raise TimeoutError("Timeout while flushing cache.")

    @override
    def shutdown(self, timeout: float = 30.0):
        """Shutdown the SGLang engine and clean up resources."""
        logger.info(f"Shutdown engine {self.server_host}:{self.server_port}...")
        if self.node_rank == 0:
            worker_url = f"http://{self.server_host}:{self.server_port}"
            try:
                response = requests.get(
                    f"http://{self.router_ip}:{self.router_port}/list_workers",
                    timeout=timeout,
                )
                response.raise_for_status()
                all_workers = response.json()["active_workers"]
                for worker in all_workers:
                    if worker == worker_url:
                        response = requests.put(
                            f"http://{self.router_ip}:{self.router_port}/exclude_worker",
                            json={"worker_url": worker},
                            timeout=timeout,
                        )
                        if response.status_code == 404:
                            logger.info(f"Worker {worker} already removed from router (404).")
                        else:
                            response.raise_for_status()
                        break
                else:
                    logger.warning(f"Worker {worker_url} not found in router during shutdown.")
            except Exception as e:
                logger.warning(f"Failed to fetch workers list or remove worker: {e}")

        if self.process is not None:
            kill_process_tree(self.process.pid)


    @override
    def update_weights_from_channel(self, meta_dict: dict):
        """Pull new weights over the already-established transfer-mesh channel.

        Args:
            meta_dict: The ChannelMeta dict
        """
        # Ensure the prefix/KV cache computed under the old weights is dropped before
        # the new weights take effect. The colocate offload path (release_memory_occupation)
        # already flushed and set the marker; in that case we skip the redundant flush here.
        # The fully-async path does not offload, so it flushes on demand. Either way the
        # marker is cleared afterwards so the next version flushes exactly once.
        if not self._cache_flushed:
            self._flush_cache()
        result = self._make_request(
            "update_weights_from_channel",
            meta_dict,
            timeout=600.0,
        )
        self._cache_flushed = False
        return result

    @override
    def release_memory_occupation(self):
        """Release memory occupation for offloading."""
        self._flush_cache()
        self._cache_flushed = True
        return self._make_request("release_memory_occupation", timeout=180.0)

    @override
    def resume_memory_occupation(self, tags: list[str] = None):
        """Resume memory occupation after offloading.

        Args:
            tags: Available tags for multi-stage resume: weights, kv_cache.
        """
        return self._make_request(
            "resume_memory_occupation",
            {"tags": tags},
            timeout=180.0,
        )

def _compute_server_args(
    config,
    rank,
    dist_init_addr,
    nccl_port,
    host,
    port,
    worker_type: str = "regular",
    disaggregation_bootstrap_port: int | None = None,
    base_gpu_id: int | None = None,
    engine_args: dict | None = None,
    num_gpus_per_replica: int | None = None,
):
    """Compute SGLang server arguments from configuration.

    Args:
        config: Configuration arguments.
        rank: The engine rank.
        dist_init_addr: Distributed initialization address.
        nccl_port: NCCL communication port.
        host: Server host address.
        port: Server port.
        worker_type: Worker type ("regular", "prefill", or "decode").
        disaggregation_bootstrap_port: Port for prefill-decode disaggregation.
        base_gpu_id: Base GPU ID for this engine.
        engine_args: SGLang server arguments.
        num_gpus_per_replica: Number of GPUs per replica.

    Returns:
        dict: Dictionary of SGLang ServerArgs.
    """
    nnodes = max(1, num_gpus_per_replica // config.rollout.num_gpus_per_node)
    node_rank = rank % nnodes
    kwargs = {
        "trust_remote_code": True,
        "random_seed": 42 + rank,
        # memory
        "enable_memory_saver": config.colocate,
        "model_path": config.hf_model_path,
        # distributed
        "host": host,
        "port": port,
        "nccl_port": nccl_port,
        "nnodes": nnodes,
        "node_rank": node_rank,
        "dist_init_addr": dist_init_addr,
        "gpu_id_step": 1,
        "base_gpu_id": base_gpu_id,
        # parallel
        "tp_size": num_gpus_per_replica // engine_args.get("pp_size", 1),
        # always skip warmup to prevent warmup timeout.
        "skip_server_warmup": True,
        # always enable draft weights cpu backup so that we run training without mtp weights.
        "enable_draft_weights_cpu_backup": True,
        "enable_return_routed_experts": config.trainer.use_rollout_routing_replay,
        "enable_fp32_lm_head": config.trainer.use_fp32_lm_head,
        "dtype": "auto",
    }
    if config.colocate:
        # Breakable CUDA graph is not compatible with memory saver mode
        kwargs["disable_prefill_cuda_graph"] = True

    if worker_type == "prefill":
        kwargs["disaggregation_mode"] = "prefill"
        kwargs["load_balance_method"] = "round_robin"
        assert (
            disaggregation_bootstrap_port is not None
        ), "disaggregation_bootstrap_port must be set for prefill worker"
        kwargs["disaggregation_bootstrap_port"] = disaggregation_bootstrap_port
    elif worker_type == "decode":
        kwargs["disaggregation_mode"] = "decode"
        kwargs["prefill_round_robin_balance"] = True

    for key, value in engine_args.items():
        kwargs[key] = value

    # sglang_args is a verbatim ServerArgs passthrough and may carry api_key.
    logger.info("sglang kwargs: %s", redact_secrets(kwargs))

    return kwargs

def launch_server_process(server_args: ServerArgs) -> multiprocessing.Process | None:
    """Launch a SGLang HTTP server in a separate process.

    Args:
        server_args: ServerArgs containing the server configuration.

    Returns:
        multiprocessing.Process: The server process, or None for non-master nodes.
    """
    from sglang.srt.entrypoints.http_server import launch_server

    multiprocessing.set_start_method("spawn", force=True)
    server_args.host = server_args.host.strip("[]")
    p = multiprocessing.Process(target=launch_server, args=(server_args,))
    p.start()

    if server_args.node_rank != 0:
        return None

    _wait_server_healthy(
        base_url=server_args.url(),
        api_key=server_args.api_key,
        is_process_alive=lambda: p.is_alive(),
    )

    return p

def _wait_server_healthy(base_url, api_key, is_process_alive):
    """Wait for the SGLang server to become healthy and ready.

    Args:
        base_url: The base URL of the server.
        api_key: The API key for authentication.
        is_process_alive: Callable to check if the server process is alive.

    Raises:
        Exception: If the server process terminates unexpectedly.
    """
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {api_key}",
    }

    with requests.Session() as session:
        while True:
            try:
                response = session.get(f"{base_url}/health_generate", headers=headers)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                logger.info("Health generate request failed, retrying...")
                pass

            if not is_process_alive():
                raise Exception("Server process terminated unexpectedly.")

            time.sleep(2)

        # use flush_cache to make sure the working queue is empty, so that we can do offload
        while True:
            try:
                response = session.get(f"{base_url}/flush_cache", headers=headers)
                if response.status_code == 200:
                    break

            except requests.RequestException:
                logger.info("Flush cache request failed, retrying...")
                pass

            if not is_process_alive():
                raise Exception("Server process terminated unexpectedly.")

            time.sleep(2)
