"""Rollout manager for managing rollout workers.
 
This module provides classes for managing inference engines,
including replicas and health monitoring.
"""
 
import contextlib
import dataclasses
import logging
 
import ray
import copy
from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS 
from coda.utils.health_monitor import RolloutHealthMonitor
from coda.transfer_mesh import ChannelMeta
from coda.backends.sglang.engine import SglangEngine
from coda.backends.replica_group import ReplicaGroup
 
logger = logging.getLogger(__name__)

class RolloutManager:
    """The class to run rollout manager, A model served behind a shared router, with one or more inference replicas.

    A RolloutManager may contain multiple ReplicaGroups with different
    ``num_gpus_per_replica`` (e.g. prefill TP=2, decode TP=4).
    """

    def __init__(self, config, scheduler):
        """Initialize the rollout manager.
 
        Args:
            config: Configuration arguments.
            scheduler: Ray scheduler for resource allocation.
        """
        self.config = config
        self._validate_config()
        self.replica_groups = self._create_replica_groups(scheduler)

        self._health_monitors = []
        if self.config.rollout.use_fault_tolerance:
            for replica_group in self.replica_groups:
                monitor = RolloutHealthMonitor(replica_group)
                monitor.start()
                self._health_monitors.append(monitor)

    def _validate_config(self):
        """Validates the configuration for the rollout manager."""

        if self.config.rollout.backend == "sglang":
            self.config = SglangEngine.validate_config(self.config)
            
        # todo validate


    def _create_replica_groups(self, scheduler):
        """
        Creates groups of inference replicas.
        Engine replicas may have different ``num_gpus_per_replica``
        (e.g. for PD disaggregation where prefill and decode use different TP sizes).
        """
        rank_offset = 0
        replica_groups: list[ReplicaGroup] = []
        new_engines_per_group: list = []
        all_init_handles: list = []

        num_gpus_per_node = self.config.rollout.num_gpus_per_node
        for worker_type, replica_cfg in self.config.rollout.sglang_replicas.items():
            num_gpus_per_replica = replica_cfg.num_gpus_per_replica
            num_nodes = replica_cfg.num_nodes
            num_gpu_per_engine = min(num_gpus_per_replica, num_gpus_per_node)
            num_engines = num_nodes * num_gpus_per_node // num_gpu_per_engine
            if num_engines == 0:
                continue

            replica_group = ReplicaGroup(
                config=self.config,
                replica_config=replica_cfg,
                scheduler=scheduler,
                num_engines=num_engines,
                worker_type=worker_type,
                rank_offset=rank_offset,
            )
            new_engines, handles = replica_group.start_engines()
            new_engines_per_group.append(new_engines)
            all_init_handles.extend(handles)
            replica_groups.append(replica_group)

            rank_offset += num_engines

        init_results = ray.get(all_init_handles) if all_init_handles else []
        _publish_engines(replica_groups, new_engines_per_group, init_results)
 
        return replica_groups

    def _health_monitoring_pause(self) -> None:
        """Pause health monitoring for all rollout engines."""
        for monitor in self._health_monitors:
            monitor.pause()
 
    def health_monitoring_resume(self) -> None:
        """Resume health monitoring for all rollout engines."""
        for monitor in self._health_monitors:
            monitor.resume()
 
    @property
    def num_new_engines(self):
        """Get the total number of newly created engines across all replicas.

        ``Trainer._init_channel_meta`` checks num_new_engines to determine whether the
        gloo master address must be refreshed for newly connected inference engines,
        and clears the counter in the same pass, before the weight update runs.

        Returns:
            int: Total count of newly created engines.
        """
        return sum(r.num_new_engines for r in self.replica_groups)

    def clear_num_new_engines(self):
        """Clear the counter of newly created engines."""
        for r in self.replica_groups:
            r.num_new_engines = 0
 
    def offload(self, tags: list[str] | None = None):
        """Offload all rollout engines to free memory.

        Args:
            tags: Memory tags to release (e.g. kv_cache, cuda_graph). None releases all.
        """
        self._health_monitoring_pause()
        handles = []
        for g in self.replica_groups:
            handles.extend(g.offload(tags))
        return ray.get(handles) if handles else []

    def onload(self, tags: list[str] | None = None):
        """Onload all rollout engines with specified tags.
 
        Args:
            tags: Memory tags to onload (e.g., weights, kv_cache).
        """
        handles = []
        for g in self.replica_groups:
            handles.extend(g.onload(tags))
        result = ray.get(handles) if handles else []
        return result
 
    def offload_kv(self):
        """Release only the KV cache (and the cuda graphs sharing that budget)."""
        self.offload(tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH])

    def onload_weights(self):
        """Onload only the weights for all rollout engines."""
        self.onload(tags=[GPU_MEMORY_TYPE_WEIGHTS])
 
    def onload_kv(self):
        """Onload only the KV cache for all rollout engines."""
        self.onload(tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH])
 
    def recover_faulty_engines(self):
        """Restart any dead rollout engines and update num_new_engines for update_weights detection.

        Holds every replica group's lock, so a kill the health monitor has in progress finishes
        before the dead slots are read, and no kill starts until the new engines are published.
        New engines are published only after their init() returned, so the monitor never probes
        an engine that is still starting.
        """
        with contextlib.ExitStack() as stack:
            for r in self.replica_groups:
                stack.enter_context(r.lock)

            # Start all replicas concurrently.
            new_engines_per_replica = []
            all_handles = []
            try:
                for r in self.replica_groups:
                    new_engines, handles = r.start_engines(True)
                    new_engines_per_replica.append(new_engines)
                    all_handles.extend(handles)
                init_results = ray.get(all_handles) if all_handles else []
                self._restore_colocated_memory(new_engines_per_replica)
            except BaseException:
                # Unpublished actors are invisible to the health monitor, so nobody else would kill
                # them; they would keep the pinned GPUs and the next recovery would fail on them too.
                for new_engines in new_engines_per_replica:
                    for _, engine in new_engines:
                        try:
                            ray.kill(engine)
                        except Exception:
                            logger.exception("Failed to kill a rollout engine whose recovery failed")
                raise

            _publish_engines(self.replica_groups, new_engines_per_replica, init_results)

    def _restore_colocated_memory(self, new_engines_per_replica):
        """Post-recovery: offload then onload weights for newly created engines in colocate mode."""
        release_handles = []
        new_engines_all = []
        for r, new_engines in zip(self.replica_groups, new_engines_per_replica, strict=True):
            logger.info(f"Recovered {len(new_engines)} dead rollout engines (worker_type={r.worker_type})")
            if r.config.colocate and new_engines:
                engines = [engine for _, engine in new_engines]
                release_handles.extend(engine.release_memory_occupation.remote() for engine in engines)
                new_engines_all.extend(engines)

        if release_handles:
            ray.get(release_handles)
            ray.get(
                [engine.resume_memory_occupation.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS]) for engine in new_engines_all]
            )
 
    def async_update_weights(self, meta: ChannelMeta, weight_version: int):
        """Async update weights for all rollout engines.

        Args:
            meta: The ChannelMeta.
            weight_version: Rollout model version label to expose in SGLang responses.
        """
        engines = [engine for replica in self.replica_groups for engine in replica.engines]
        handles = []
        for i in range(len(engines)):
            engine_meta = copy.deepcopy(meta)
            engine_meta.engine_id = i
            meta_dict = dataclasses.asdict(engine_meta)
            meta_dict["weight_version"] = str(weight_version)
            handles.append(engines[i].update_weights_from_channel.remote(meta_dict))
        return handles


def _publish_engines(replica_groups, new_engines_per_group, init_results) -> None:
    """Publish each group's new engines with their init() results (the registered worker URLs).

    ``init_results`` is the flattened ``ray.get`` of every group's init handles, in group order;
    each group has one handle per new engine.
    """
    offset = 0
    for group, new_engines in zip(replica_groups, new_engines_per_group, strict=True):
        group.publish_engines(new_engines, init_results[offset:offset + len(new_engines)])
        offset += len(new_engines)
