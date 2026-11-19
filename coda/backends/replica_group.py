"""Replica group."""

import logging
 
import ray
from typing import Any
from omegaconf import DictConfig
 
from coda.backends.sglang.engine import SglangEngine
 
logger = logging.getLogger(__name__)

# Port scan start per worker type. A probe only checks that a port is free *now*,
# so prefill and decode starting on the same node need disjoint ranges. All bases
# stay below the kernel ephemeral range: SGLang later binds port+5.. too, and an
# outbound connection could be handed one of those first.
_BASE_PORT_BY_WORKER_TYPE = {
    "regular": 2000,
    "prefill": 4000,
    "decode": 6000,
}


class ReplicaGroup:
    """A replica group of homogeneous rollout engines with the same configuration.

    All engines in a replica group share the same num_gpus_per_replica / scheduler.
    A RolloutManager may contain multiple ReplicaGroups (e.g. prefill vs decode
    in PD disaggregation).
    """
 
    def __init__(self, config: DictConfig, replica_config: DictConfig, scheduler: Any, num_engines: int,
                  worker_type: str = "regular", rank_offset: int = 0):
        self.config = config
        self.replica_config = replica_config
        self.scheduler = scheduler
        self.worker_type = worker_type
        self.rank_offset = rank_offset

        self.num_new_engines = 0
        self.all_engines = num_engines * [None]
        # Bundle index (into scheduler.reorder_bundle_list) each engine slot was placed on.
        # Recorded on first allocation so a recovered engine lands back on the same GPUs.
        self.engine_bundle_indices: list[int | None] = num_engines * [None]
 
    @property
    def _nodes_per_engine(self):
        """Calculate the number of nodes per engine.
 
        Returns:
            int: Number of nodes required per engine.
        """
        return max(1, self.replica_config.num_gpus_per_replica // self.config.rollout.num_gpus_per_node)
 
    @property
    def engines(self):
        """Node-0 engines only (for multi-node serving)."""
        return self.all_engines[:: self._nodes_per_engine]
 
    def start_engines(
        self,
        recover: bool = False,
    ) -> list:
        """Create Ray actors, allocate ports, and fire ``engine.init()`` without waiting.

        Returns a list of Ray ObjectRefs. The caller should ``ray.get()`` on them
        to block until the engines are healthy.
        """

        num_gpu_per_engine = min(self.replica_config.num_gpus_per_replica, self.config.rollout.num_gpus_per_node)

        assert self.config.rollout.backend == "sglang"

        env_vars = SglangEngine.runtime_env_vars()
        env_vars.update(dict(self.config.rollout.env_vars))
        rollout_worker_cls = ray.remote(runtime_env={"env_vars": env_vars})(SglangEngine)
 
        rollout_engines = []
        for i in range(len(self.all_engines)):
            if self.all_engines[i] is not None:
                continue
 
            global_rank = self.rank_offset + i

            # When recovering, reuse the bundle this slot was originally placed on so the
            # engine comes back on the same GPUs; otherwise take the next bundles from the
            # scheduler cursor (which already accounts for train/teacher allocations).
            pinned_bundle_index = -1
            if recover and self.engine_bundle_indices[i] is not None:
                pinned_bundle_index = self.engine_bundle_indices[i]

            rollout_actor_cls, bundle_index = self.scheduler.schedule(
                rollout_worker_cls, num_gpu_per_engine, bundle_index=pinned_bundle_index
            )
            self.engine_bundle_indices[i] = bundle_index

            # sglang needs the physical GPU id of the bundle the engine actually landed on.
            base_gpu_id = self.scheduler.reorder_bundle_list[bundle_index]["gpu_id"]

            rollout_engine = rollout_actor_cls.remote(
                self.config,
                rank=global_rank,
                worker_type=self.worker_type,
                base_gpu_id=base_gpu_id,
                engine_args=self.replica_config.sglang_args,
                num_gpus_per_replica=self.replica_config.num_gpus_per_replica,
            )
 
            rollout_engines.append((global_rank, rollout_engine))
            self.all_engines[i] = rollout_engine
 
        self.num_new_engines = len(rollout_engines)
 
        if self.num_new_engines == 0:
            return []
 
        addr_and_ports = _allocate_rollout_engine_addr_and_ports(
            rollout_engines=rollout_engines,
            num_gpus_per_node=self.config.rollout.num_gpus_per_node,
            num_gpus_per_replica=self.replica_config.num_gpus_per_replica,
            worker_type=self.worker_type,
            rank_offset=self.rank_offset,
            base_port=_BASE_PORT_BY_WORKER_TYPE[self.worker_type],
        )
 
        init_handles = [
            engine.init.remote(
                **(addr_and_ports[rank]),
                router_ip=self.config.agentflow.router.ip,
                router_port=self.config.agentflow.router.port,
            )
            for rank, engine in rollout_engines
        ]
        return init_handles

    def kill_engine(self, rollout_engine_id: int):
        """Kill an engine and all its replicas in the replica group.

        Args:
            rollout_engine_id: The ID of the engine to kill.
        """
        logger.info(f"Killing replica group {rollout_engine_id}...")
        for i in range(
            rollout_engine_id * self._nodes_per_engine,
            (rollout_engine_id + 1) * self._nodes_per_engine,
        ):
            engine = self.all_engines[i]
            if engine:
                logger.info(f"Shutting down and killing engine at index {i}")
                try:
                    ray.get(engine.shutdown.remote())
                    ray.kill(engine)
                    logger.info(f"Successfully killed engine at index {i}")
                except Exception as e:
                    logger.warning(f"Fail to kill engine at index {i} (e: {e})")
            else:
                logger.info(f"Engine at index {i} is already None")
            self.all_engines[i] = None

    def offload(self, tags: list[str] | None = None):
        """Fire release_memory_occupation on all engines (non-blocking).

        Returns a list of Ray ObjectRefs.
        """
        return [
            engine.release_memory_occupation.remote(tags=tags)
            for engine in self.engines
            if engine is not None
        ]

    def onload(self, tags: list[str] | None = None):
        """Fire resume_memory_occupation on all engines (non-blocking).
 
        Returns a list of Ray ObjectRefs.
        """
        return [engine.resume_memory_occupation.remote(tags=tags) for engine in self.engines if engine is not None]

 
def _allocate_rollout_engine_addr_and_ports(
    rollout_engines,
    num_gpus_per_node,
    num_gpus_per_replica,
    worker_type="regular",
    rank_offset=0,
    base_port=2000,
):
    """Allocate network addresses and ports for rollout engines.
 
    Args:
        rollout_engines: List of rollout engine tuples (rank, engine).
        num_gpus_per_node: Number of GPUs per node.
        num_gpus_per_replica: Number of GPUs per replica.
        worker_type: Worker type ("regular", "prefill", or "decode").
        rank_offset: Rank offset for this sglang engine.
        base_port: Base port number for allocation.
 
    Returns:
        dict: maps rank to a dict of address/ports
    """
    # Ports per engine, offset from ``port``: +0 server, +1 nccl, +2 prefill bootstrap, +3 dist_init.
    num_node_per_replica = max(1, num_gpus_per_replica // num_gpus_per_node)
    addr_and_ports: dict[int, dict] = {}
    # Probe ports on each engine's own node, keeping one scan cursor per host so
    # engines sharing a node get disjoint port blocks before any of them binds.
    next_port_by_host: dict[str, int] = {}
    for rank, engine in rollout_engines:
        host = ray.get(engine.get_node_ip.remote())
        _, port = ray.get(engine.get_ip_port.remote(next_port_by_host.get(host, base_port), 4))
        next_port_by_host[host] = port + 100
        # A multi-node replica's first rank hosts dist_init; the rest join it. The
        # head is always started too, since kill_engine tears down whole replicas.
        head_rank = rank - (rank - rank_offset) % num_node_per_replica
        entry = {
            "host": host,
            "port": port,
            "nccl_port": port + 1,
            "dist_init_addr": (
                f"{host}:{port + 3}" if head_rank == rank
                else addr_and_ports[head_rank]["dist_init_addr"]
            ),
        }
        if worker_type == "prefill":
            entry["disaggregation_bootstrap_port"] = port + 2
        addr_and_ports[rank] = entry
        logger.info("Ports for engine %d: %s", rank, entry)
    return addr_and_ports