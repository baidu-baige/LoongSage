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
from coda.utils import replica_utils
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
        num_gpus_per_node = config.rollout.num_gpus_per_node
        engines = {}
        for worker_type, replica_cfg in config.rollout.sglang_replicas.items():
            replica_cfg.sglang_args = OmegaConf.merge(sglang_args, replica_cfg.get("sglang_args", {}))
            engines[worker_type] = replica_utils.num_engines(replica_cfg, num_gpus_per_node)
            if engines[worker_type] == 0:
                continue
            # The weight channel's rank_base stride is num_gpus_per_replica, so it
            # must match the engine's own world size (tp_size * pp_size).
            pp_size = replica_cfg.sglang_args.get("pp_size", 1)
            num_gpus = replica_cfg.num_gpus_per_replica
            tp_size = replica_cfg.sglang_args.get("tp_size", num_gpus // pp_size)
            assert tp_size * pp_size == num_gpus, (
                f"{worker_type}: tp_size({tp_size}) * pp_size({pp_size}) must equal "
                f"num_gpus_per_replica({num_gpus})"
            )

        if engines["regular"] > 0:
            assert engines["prefill"] == 0 and engines["decode"] == 0, \
                "regular and prefill/decode replicas are mutually exclusive"
        else:
            assert engines["prefill"] > 0 and engines["decode"] > 0, \
                "PD disaggregation needs both prefill and decode replicas"

        # The Rust server's PD hook demands bootstrap_port == port, while the
        # framework allocates port + 2.
        if engines["prefill"] > 0:
            rust_server = str(config.rollout.env_vars.get("SGLANG_RUST_SERVER", "")).lower()
            assert rust_server not in ("1", "true"), \
                "PD disaggregation is incompatible with SGLANG_RUST_SERVER"

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
        self.server_host = server_args_dict["host"]
        self.server_port = server_args_dict["port"]

        # Must precede launch_server_process: the server and its scheduler children
        # inherit this environment, and that is where mooncake reads the GID index.
        _export_gid_index()

        self._init_server(server_args_dict)

    def _init_server(self, server_args_dict, timeout: float = 30.0):
        """Initialize the SGLang server.

        Args:
            server_args_dict: Dictionary containing server configuration arguments.
        """
        logger.info(f"Launch HttpServerEngineAdapter at: {self.server_host}:{self.server_port}")
        server_args = ServerArgs(**server_args_dict)
        self.process = launch_server_process(server_args)

        if self.node_rank == 0 and self.router_ip and self.router_port:
            payload = {
                "worker_url": f"http://{self.server_host}:{self.server_port}",
                # Which pool the router registers this engine in.
                "worker_type": self.worker_type,
            }
            if self.worker_type == "prefill":
                payload["bootstrap_port"] = server_args_dict["disaggregation_bootstrap_port"]
            # DP attention: report the attention-DP group count so the router can pin
            # each request to one via routed_dp_rank, instead of letting SGLang's
            # internal DataParallelController distribute. 1 = no DP attention.
            if server_args.attn_dp_size > 1:
                payload["dp_size"] = server_args.attn_dp_size
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
    def release_memory_occupation(self, tags: list[str] = None):
        """Release memory occupation for offloading.

        Args:
            tags: Memory tags to release (e.g. kv_cache, cuda_graph). None releases all.
        """
        self._flush_cache()
        self._cache_flushed = True
        return self._make_request(
            "release_memory_occupation",
            {"tags": tags},
            timeout=180.0,
        )

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

_NCCL_CONF_PATH = "/etc/nccl.conf"


def _parse_nccl_ib_hca(raw: str) -> str | None:
    """Turn an ``NCCL_IB_HCA`` value into mooncake's bare device list.

    Drops NCCL's leading ``=`` and ``:port`` suffixes. A ``^`` exclusion list has
    no device list to derive, so it yields None.
    """
    value = raw.strip()
    if value.startswith("^"):
        logger.warning("NCCL_IB_HCA %r is an exclusion list; cannot derive a device list from it", raw)
        return None
    devices = [d.strip().split(":")[0] for d in value.lstrip("=").split(",")]
    return ",".join(dict.fromkeys(d for d in devices if d)) or None


def _nccl_conf_value(path: str, key: str) -> str | None:
    """Value of ``key`` in an nccl.conf-style ``KEY=value`` file, or None."""
    try:
        with open(path) as handle:
            for line in handle:
                entry, sep, value = line.strip().partition("=")
                if sep and entry.strip() == key:
                    return value
    except OSError as e:
        logger.warning("Cannot read %s: %s", path, e)
    return None


def _resolve_ib_devices(configured: str | None) -> str | None:
    """RDMA devices for mooncake's KV transfer, resolved on this engine's machine.

    Precedence: explicit config > /etc/nccl.conf > $NCCL_IB_HCA > None (mooncake
    auto-discovery).
    """
    if configured:
        return configured
    for source, raw in (
        (_NCCL_CONF_PATH, _nccl_conf_value(_NCCL_CONF_PATH, "NCCL_IB_HCA")),
        ("$NCCL_IB_HCA", os.environ.get("NCCL_IB_HCA")),
    ):
        if raw and (devices := _parse_nccl_ib_hca(raw)):
            logger.info("disaggregation_ib_device resolved from %s: %s", source, devices)
            return devices
    logger.warning(
        "No usable NCCL_IB_HCA found; mooncake auto-discovery may pick NICs with no "
        "route into the RDMA fabric"
    )
    return None


def _export_gid_index() -> None:
    """Export nccl.conf's NCCL_IB_GID_INDEX for mooncake, which reads it from env only.

    Skipped when NCCL_IB_GID_INDEX or MC_GID_INDEX is already set.
    """
    if os.environ.get("NCCL_IB_GID_INDEX") or os.environ.get("MC_GID_INDEX"):
        return
    raw = _nccl_conf_value(_NCCL_CONF_PATH, "NCCL_IB_GID_INDEX")
    if raw is None:
        return
    gid_index = raw.strip()
    if not gid_index.isdecimal():
        logger.warning("Ignoring invalid NCCL_IB_GID_INDEX %r in %s", raw, _NCCL_CONF_PATH)
        return
    os.environ["NCCL_IB_GID_INDEX"] = gid_index
    logger.info("Exported NCCL_IB_GID_INDEX=%s from %s", gid_index, _NCCL_CONF_PATH)


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
        "routed_experts_dtype": config.trainer.routed_experts_dtype,
        "enable_fp32_lm_head": config.trainer.use_fp32_lm_head,
        "dtype": "auto",
    }
    if config.colocate:
        # Breakable CUDA graph is not compatible with memory saver mode
        kwargs["disable_prefill_cuda_graph"] = True

    if worker_type == "prefill":
        kwargs["disaggregation_mode"] = "prefill"
        # Inert while the router pins routed_dp_rank; explicit so prefill does not
        # inherit SGLang's PD default (follow_bootstrap_room).
        kwargs["load_balance_method"] = "round_robin"
        assert (
            disaggregation_bootstrap_port is not None
        ), "disaggregation_bootstrap_port must be set for prefill worker"
        kwargs["disaggregation_bootstrap_port"] = disaggregation_bootstrap_port
    elif worker_type == "decode":
        kwargs["disaggregation_mode"] = "decode"

    for key, value in engine_args.items():
        kwargs[key] = value

    if worker_type in ("prefill", "decode"):
        kwargs["disaggregation_ib_device"] = _resolve_ib_devices(
            kwargs.get("disaggregation_ib_device")
        )

    # sglang_args is a verbatim ServerArgs passthrough and may carry api_key.
    logger.info("sglang kwargs: %s", redact_secrets(kwargs))

    return kwargs

def _run_server(server_args: ServerArgs):
    """Entrypoint of the spawned SGLang server process.

    Args:
        server_args: ServerArgs containing the server configuration.
    """
    from sglang.srt.entrypoints import http_server

    # SGLang posts /freeze_gc from its warmup thread, which runs during FastAPI lifespan
    # startup, i.e. before uvicorn binds the port. With skip_server_warmup there is no
    # retry loop ahead of it, so it always fails with ECONNREFUSED. _wait_server_healthy
    # does it instead, once the server is known to serve.
    if hasattr(http_server, "_freeze_gc_after_server_warmup"):
        http_server._freeze_gc_after_server_warmup = lambda *_, **__: None

    http_server.launch_server(server_args)

def launch_server_process(server_args: ServerArgs) -> multiprocessing.Process | None:
    """Launch a SGLang HTTP server in a separate process.

    Args:
        server_args: ServerArgs containing the server configuration.

    Returns:
        multiprocessing.Process: The server process, or None for non-master nodes.
    """
    multiprocessing.set_start_method("spawn", force=True)
    p = multiprocessing.Process(target=_run_server, args=(server_args,))
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

        # SGLang's own post-warmup freeze_gc is disabled in _run_server; do it here,
        # where the server is known to be serving.
        try:
            response = session.post(f"{base_url}/freeze_gc", headers=headers, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.warning("freeze_gc failed, GC stays unfrozen: %s", e)
