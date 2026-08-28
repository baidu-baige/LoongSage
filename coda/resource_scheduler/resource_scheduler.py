"""
ResourceScheduler is used to schedule actors onto a Ray placement group.

In colocate mode, each role has its own cursor from index 0, so train and rollout actors of the
same index share a GPU; in non-colocate mode, all roles share one cursor in creation order
(train -> teacher -> rollout), so GPUs never overlap.
"""

import logging
import random
import socket

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.actor import ActorClass
from coda.utils import logging_utils
from omegaconf import DictConfig 

logger = logging.getLogger(__name__)


def _ip_sort_key(ip: str):
    """Sort key that orders IPv4 addresses numerically instead of lexicographically.

    Plain string sorting puts "10.0.0.9" after "10.0.0.84", which scrambles node
    ordering and therefore anything derived from it (e.g. which nodes get paired into a
    multi-node inference replica). Falls back to the raw string for non-IPv4 addresses
    (IPv6, hostnames) so ordering stays deterministic rather than raising.
    """
    octets = ip.split(".")
    if len(octets) == 4:
        try:
            return (0, tuple(int(o) for o in octets), "")
        except ValueError:
            pass
    return (1, (), ip)


@ray.remote(num_gpus=1)
class Probe:
    """Actor to get IP and GPU ID of a bundle in a Ray placement group."""

    def get_ip_and_gpu_id(self):
        """Get IP address and GPU ID of current ray node and bundle"""
        ip = ray.util.get_node_ip_address()
        gpu_ids = ray.get_gpu_ids()
        return ip, gpu_ids

    def get_free_port(self, port_num: int = 1, min_port: int = 15000, max_port: int = 50000, max_tries: int = 100):
        """Find consecutive free ports in specified range on the current node.

        Args:
            port_num: Number of consecutive free ports to return (default 1)
            min_port: Minimum port number (inclusive), default 15000
            max_port: Maximum port number (inclusive), default 50000
            max_tries: Maximum number of starting ports to try before giving up

        Returns:
            A list of consecutive free port numbers (e.g., [25000, 25001, 25002])

        Raises:
            RuntimeError: If unable to find consecutive free ports after max_tries attempts
        """
        attempts = 0

        while attempts < max_tries:
            attempts += 1
            # Randomly pick a starting port that leaves room for port_num ports
            start_port = random.randint(min_port, max_port - port_num + 1)
            free_ports = []

            # Try to bind consecutive ports
            for offset in range(port_num):
                port = start_port + offset
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    try:
                        s.bind(("", port))
                        free_ports.append(port)
                    except OSError:
                        # Port in use, break and try a different starting port
                        break

            # If we found all consecutive ports, return them
            if len(free_ports) == port_num:
                return free_ports

        raise RuntimeError(
            f"Unable to find {port_num} consecutive free port(s) in range [{min_port}-{max_port}] "
            f"after {attempts} attempts"
        )


class ResourceScheduler:
    """
    ResourceScheduler is used to schedule actors onto a Ray placement group.
    """
    def __init__(self, config):
        logging_utils.configure_logger(level=config.log_level)
        self.reorder_bundle_list = []
        self.config = config
        self.role_cursors = {}  # role -> cursor (colocate mode)
                                # _global -> cursor (non-colocate mode)
        assert isinstance(config, (dict, DictConfig)), "config must be a dict"
        assert "trainer" in config and "rollout" in config, "trainer and rollout are required in config"

        self.colocate = config.colocate

        trainer_config = config.trainer
        rollout_config = config.rollout

        if self.colocate:
            num_gpus = trainer_config.num_gpus_per_node * trainer_config.num_nodes
        else:
            assert rollout_config.backend == "sglang"
            rollout_num_gpus = sum(
                replica.num_nodes * self.config.rollout.num_gpus_per_node
                for _, replica in rollout_config.sglang_replicas.items()
            )
            teacher_num_gpus = 0
            if "opd" in config and config.opd.get("enable", False):
                teacher_num_gpus = config.opd.teacher_nodes * config.opd.teacher_gpus_per_node
            num_gpus = trainer_config.num_gpus_per_node * trainer_config.num_nodes + \
                rollout_num_gpus + teacher_num_gpus

        self.create_placement_group(num_gpus)


    MAX_PROBE_RETRIES = 3

    def _get_bundle_ip_and_gpu(self, pg, bundle_idx):
        """
        Get the IP address and GPU ID of a bundle in a Ray placement group.
        Retries up to MAX_PROBE_RETRIES times on failure, then raises.
        """
        last_exception = None
        for attempt in range(1, self.MAX_PROBE_RETRIES + 1):
            probe = Probe.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg, placement_group_bundle_index=bundle_idx
                )
            ).remote()
            try:
                ip, gpu_ids = ray.get(probe.get_ip_and_gpu_id.remote())
                return ip, gpu_ids[0] if gpu_ids else -1
            except Exception as e:
                last_exception = e
                logging.warning(
                    f"Probe failed for bundle {bundle_idx} "
                    f"(attempt {attempt}/{self.MAX_PROBE_RETRIES}): {e}"
                )
            finally:
                ray.kill(probe)
        raise RuntimeError(
            f"Failed to probe bundle {bundle_idx} after {self.MAX_PROBE_RETRIES} attempts"
        ) from last_exception

    def get_gloo_master_address(
        self, 
        bundle_idx: int = 0, 
        min_port: int = 15000, 
        max_port: int = 50000
    ) -> tuple[str, int]:
        """Get the master IP and free port for gloo ring initialization.

        Args:
            bundle_idx: The bundle index to get IP from (default 0)
            min_port: Minimum port number (inclusive), default 15000
            max_port: Maximum port number (inclusive), default 50000

        Returns:
            A tuple of (ip_address, free_port) for the gloo master

        Raises:
            RuntimeError: If unable to probe the bundle or find a free port
        """
        if not self.reorder_bundle_list:
            raise RuntimeError("Placement group not initialized, no bundles available")

        if bundle_idx >= len(self.reorder_bundle_list):
            raise RuntimeError(
                f"Bundle index {bundle_idx} out of range, only {len(self.reorder_bundle_list)} bundles available"
            )

        bundle_info = self.reorder_bundle_list[bundle_idx]
        ip = bundle_info["ip"]
        pg = bundle_info["pg"]
        p_idx = bundle_info["p_idx"]

        last_exception = None
        for attempt in range(1, self.MAX_PROBE_RETRIES + 1):
            probe = None
            try:
                probe = Probe.options(
                    num_cpus=0.1,
                    num_gpus=0.1,
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=p_idx
                    )
                ).remote()
                logger.info(
                    f"get_gloo_master_address: Probe actor created (attempt {attempt}), "
                    f"waiting for get_free_port result..."
                )
                # transfer_mesh gloo init requires 3 consecutive ports
                ports_ref = probe.get_free_port.remote(3, min_port, max_port)
                ready, _ = ray.wait([ports_ref], timeout=60)
                if not ready:
                    raise RuntimeError(
                        f"get_free_port timed out after 60s on bundle {bundle_idx} "
                        f"(pg_bundle_index={p_idx}). Probe actor may be pending due to "
                        f"insufficient resources (Probe requests num_gpus=0.1, "
                        f"but the bundle GPU may already be fully occupied by scheduled workers)."
                    )
                ports = ray.get(ready[0])
                if not ports:
                    raise RuntimeError(f"get_free_port returned empty list for bundle {bundle_idx}")
                logger.info(
                    f"get_gloo_master_address: success, ip={ip}, port={ports[0]}, "
                    f"all_ports={ports}"
                )
                return ip, ports[0]
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"get_gloo_master_address: failed on bundle {bundle_idx} "
                    f"(attempt {attempt}/{self.MAX_PROBE_RETRIES}): {type(e).__name__}: {e}"
                )
            finally:
                if probe is not None:
                    try:
                        ray.kill(probe)
                        logger.info(f"get_gloo_master_address: Probe actor killed (attempt {attempt})")
                    except Exception as kill_err:
                        logger.warning(f"get_gloo_master_address: failed to kill Probe: {kill_err}")
        raise RuntimeError(
            f"Failed to get gloo master address for bundle {bundle_idx} after {self.MAX_PROBE_RETRIES} attempts"
        ) from last_exception

    def create_placement_group(self, num_bundles):
        """
        Create a Ray placement group with the specified number of bundles.
        """
        new_pg = placement_group([{"CPU": 1, "GPU": 1} for _ in range(num_bundles)], strategy="PACK")
        # waiting for all bundles are ready
        ray.get(new_pg.ready())
        # 1. get bundle ip and gpu id
        temp_info = []
        for i in range(num_bundles):
            ip, gpu_id = self._get_bundle_ip_and_gpu(new_pg, i)
            temp_info.append({"pg": new_pg, "p_idx": i, "ip": ip, "gpu_id": gpu_id})
        # 2. sort by ip and gpu_id, to guarantee the bundles on the same machine are in order of gpu id.
        # Sort the IP numerically per octet, not lexicographically: "10.0.0.9" must come before
        # "10.0.0.84", otherwise node ordering (and any pairing derived from it) silently scrambles.
        self.reorder_bundle_list = sorted(temp_info, key=lambda x: (_ip_sort_key(x["ip"]), x["gpu_id"]))


    def schedule(
        self, ray_actor: ActorClass, num_bundles: int = 1, recover_bundle_index: int = -1
    ) -> tuple[ActorClass, int]:
        """
        Schedule an actor onto a Ray placement group.

        Args:
            ray_actor: The Ray ActorClass.
            num_bundles: The number of bundles to allocate.
            recover_bundle_index: The bundle index of the dead actor.

        Returns:
            (prepared_actor, bundle_index): An ActorClass with resources and placement-group
                scheduling bound, together with the ``reorder_bundle_list`` index of the
                primary bundle it was placed on.
        """
        role = getattr(ray_actor, '__name__', None) or ray_actor.__ray_metadata__.class_name

        if recover_bundle_index == -1:
            # Get the cursor: per-role in colocate mode, shared in non-colocate mode
            cursor_key = role if self.colocate else "_global"
            if cursor_key not in self.role_cursors:
                self.role_cursors[cursor_key] = 0
            cursor = self.role_cursors[cursor_key]
            bundle_index = cursor

            allocated_bundles = []
            for _ in range(num_bundles):
                if cursor >= len(self.reorder_bundle_list):
                    raise RuntimeError(f"No available bundles to allocate for role '{role}'")
                bundle_info = self.reorder_bundle_list[cursor]
                allocated_bundles.append(bundle_info)
                cursor += 1

            # Update the cursor
            self.role_cursors[cursor_key] = cursor

            primary_bundle = allocated_bundles[0]
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=primary_bundle["pg"],
                placement_group_bundle_index=primary_bundle["p_idx"],
            )
        else:
            bundle_index = recover_bundle_index
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=self.reorder_bundle_list[recover_bundle_index]["pg"],
                placement_group_bundle_index=self.reorder_bundle_list[recover_bundle_index]["p_idx"],
            )

        prepared_actor = ray_actor.options(
            # we don't need user to apply any resources anymore, scheduling resources are useless
            # but ray require it to schedule, so we apply a small value instead.
            scheduling_strategy=scheduling_strategy,
            num_cpus=0.1,
            num_gpus=0.1,
        )

        return prepared_actor, bundle_index
