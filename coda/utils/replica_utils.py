"""Topology helpers for ``rollout.sglang_replicas`` groups.

Engine counts and GPU footprints are derived here so the rollout manager, the
trainer's gloo ring and the resource scheduler cannot drift apart.

A replica group is sized either by its node allotment (``num_nodes``, the GPU
budget being ``num_nodes * rollout.num_gpus_per_node``) or, when
``num_replicas`` is set, by an explicit replica count. The explicit form is
needed whenever a group should get fewer GPUs than one whole node, e.g. prefill
and decode sharing a single 8-GPU node with ``num_gpus_per_replica: 4``.
"""


def nodes_per_engine(num_gpus_per_replica: int, num_gpus_per_node: int) -> int:
    """Nodes spanned by one replica. >1 only for multi-node serving."""
    return max(1, int(num_gpus_per_replica) // int(num_gpus_per_node))


def gpus_per_engine(num_gpus_per_replica: int, num_gpus_per_node: int) -> int:
    """GPUs driven by a single engine process (one per node of a replica)."""
    return min(int(num_gpus_per_replica), int(num_gpus_per_node))


def num_engines(replica_cfg, num_gpus_per_node: int) -> int:
    """Engine processes to launch for one replica group.

    Counts engine *processes*, not logical replicas: a replica spanning N nodes
    contributes N engines.
    """
    num_gpus_per_replica = int(replica_cfg.num_gpus_per_replica)
    num_replicas = replica_cfg.get("num_replicas")
    if num_replicas:
        return int(num_replicas) * nodes_per_engine(num_gpus_per_replica, num_gpus_per_node)
    return (
        int(replica_cfg.num_nodes)
        * int(num_gpus_per_node)
        // gpus_per_engine(num_gpus_per_replica, num_gpus_per_node)
    )


def group_num_gpus(replica_cfg, num_gpus_per_node: int) -> int:
    """GPUs occupied by one replica group."""
    return num_engines(replica_cfg, num_gpus_per_node) * gpus_per_engine(
        int(replica_cfg.num_gpus_per_replica), num_gpus_per_node
    )


def rollout_num_gpus(rollout_cfg) -> int:
    """GPUs occupied by all enabled replica groups, i.e. the rollout world size."""
    num_gpus_per_node = rollout_cfg.num_gpus_per_node
    return sum(
        group_num_gpus(replica_cfg, num_gpus_per_node)
        for _, replica_cfg in rollout_cfg.sglang_replicas.items()
    )
