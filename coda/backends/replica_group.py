"""Replica group."""

import logging
import threading
 
import ray
from typing import Any
from omegaconf import DictConfig
 
from coda.backends.sglang.engine import SglangEngine
 
logger = logging.getLogger(__name__)

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
        self.ready = num_engines * [False]
        # Shared by the health monitor thread and recovery.
        self.lock = threading.RLock()
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

    def _slots(self, rollout_engine_id: int) -> range:
        """all_engines indices of one engine (one per node it spans)."""
        return range(rollout_engine_id * self._nodes_per_engine, (rollout_engine_id + 1) * self._nodes_per_engine)

    def is_ready(self, rollout_engine_id: int) -> bool:
        """Whether every node of the engine has finished init()."""
        return all(self.ready[i] for i in self._slots(rollout_engine_id))

    def mark_ready(self) -> None:
        """Record that the init() handles from start_engines() have all returned."""
        self.ready = [engine is not None for engine in self.all_engines]

    def kill_engine_with_lock(self, rollout_engine_id: int, engine) -> bool:
        """Kill the engine unless its slot now holds a different actor. Returns True if killed."""
        with self.lock:
            if self.all_engines[self._slots(rollout_engine_id)[0]] is not engine:
                logger.info(f"Rollout engine {rollout_engine_id} was already restarted, not killing it")
                return False
            self.kill_engine(rollout_engine_id=rollout_engine_id)
            return True
 
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
        with self.lock:
            logger.info(f"Killing replica group {rollout_engine_id}...")
            for i in self._slots(rollout_engine_id):
                self.ready[i] = False
                engine = self.all_engines[i]
                if engine:
                    logger.info(f"Shutting down and killing engine at index {i}")
                    try:
                        ray.get(engine.shutdown.remote(), timeout=180)
                    except Exception as e:
                        logger.warning(f"Shutdown of engine at index {i} failed, killing it anyway (e: {e})")
                    try:
                        ray.kill(engine)
                        logger.info(f"Successfully killed engine at index {i}")
                    except Exception as e:
                        logger.warning(f"Fail to kill engine at index {i} (e: {e})")
                else:
                    logger.info(f"Engine at index {i} is already None")
                self.all_engines[i] = None

    def kill_unresponsive_engines(self, timeout: float = 30.0) -> None:
        """Probe every live engine once; kill the ones that do not answer."""
        probes = []
        for rollout_engine_id, engine in enumerate(self.engines):
            if engine is None:
                continue
            ref = engine.health_generate.remote(timeout=timeout)
            probes.append((rollout_engine_id, engine, ref))

        for rollout_engine_id, engine, ref in probes:
            try:
                ray.get(ref, timeout=timeout + 60)
            except Exception as e:
                logger.error(
                    f"Rollout engine {rollout_engine_id} did not answer a health probe, restarting it (e: {e})"
                )
                self.kill_engine_with_lock(rollout_engine_id, engine)

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
    # get ports
    # there are 4 ports we need to allocate
    # 1. server port
    # 2. nccl port
    # 3. disaggregation_bootstrap_port (prefill node)
    # 4. dist_init_addr port
    num_replicas_per_node = max(1, num_gpus_per_node // num_gpus_per_replica)
    addr_and_ports: dict[int, dict] = {}
 
    visited_nodes = set()
    for rank, engine in rollout_engines:
        local_rank = rank - rank_offset
        node_index = local_rank // num_replicas_per_node
        if node_index in visited_nodes:
            continue
        visited_nodes.add(node_index)
 
        # we will set port for all replicas on this node starting with this rank.
        # e.g. for 8 gpus, if we are restarting replica on gpu 3, we will set port for engine 3,4,5,6,7 on this node.
        num_replicas_on_this_node = num_replicas_per_node - (local_rank % num_replicas_per_node)

        start_port = base_port
        for i in range(num_replicas_on_this_node):
            host_addr, port = ray.get(engine.get_ip_port.remote(start_port, 4))
            start_port = port + 100
            current_rank = rank + i
            addr_and_ports.setdefault(current_rank, {})
            addr_and_ports[current_rank]["host"] = host_addr
            addr_and_ports[current_rank]["port"] = port
            addr_and_ports[current_rank]["nccl_port"] = port + 1
            if worker_type == "prefill":
                addr_and_ports[current_rank]["disaggregation_bootstrap_port"] = port + 2
            # multi node
            if num_gpus_per_replica > num_gpus_per_node:
                num_node_per_replica = num_gpus_per_replica // num_gpus_per_node
                if local_rank % num_node_per_replica == 0:
                    # this is the first node in the replica, we need to allocate the dist_init_addr port
                    dist_init_addr = f"{host_addr}:{port+3}"
                    for j in range(num_node_per_replica):
                        addr_and_ports.setdefault(rank + j, {})
                        addr_and_ports[rank + j]["dist_init_addr"] = dist_init_addr
            # single node
            else:
                addr_and_ports[current_rank]["dist_init_addr"] = f"{host_addr}:{port+3}"
 
    for i, _ in rollout_engines:
        for key in ["port", "nccl_port", "dist_init_addr"]:
            assert key in addr_and_ports[i], f"Engine {i} {key} is not set."
        logger.info(f"Ports for engine {i}: {addr_and_ports[i]}")
 
    return addr_and_ports