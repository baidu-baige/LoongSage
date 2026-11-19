"""Unit tests for ``coda.utils.replica_utils`` topology math.

Three call sites (rollout manager, trainer gloo ring, resource scheduler) must
agree on how many engines a replica group gets and how many GPUs that costs;
these tests pin the shared derivation.
"""

from omegaconf import OmegaConf

from coda.utils import replica_utils


def group(**fields):
    fields.setdefault("num_nodes", 0)
    fields.setdefault("num_replicas", None)
    return OmegaConf.create(fields)


def rollout(num_gpus_per_node: int, **groups):
    return OmegaConf.create(
        {
            "num_gpus_per_node": num_gpus_per_node,
            "sglang_replicas": {name: dict(cfg) for name, cfg in groups.items()},
        }
    )


class TestNumEngines:
    def test_whole_node_groups_derive_from_num_nodes(self):
        cfg = group(num_nodes=1, num_gpus_per_replica=8)
        assert replica_utils.num_engines(cfg, 8) == 1

    def test_several_replicas_fit_in_one_node(self):
        cfg = group(num_nodes=1, num_gpus_per_replica=2)
        assert replica_utils.num_engines(cfg, 8) == 4

    def test_multi_node_replica_counts_one_engine_per_node(self):
        cfg = group(num_nodes=2, num_gpus_per_replica=16)
        assert replica_utils.num_engines(cfg, 8) == 2

    def test_disabled_group_has_no_engines(self):
        cfg = group(num_nodes=0, num_gpus_per_replica=8)
        assert replica_utils.num_engines(cfg, 8) == 0

    def test_num_replicas_overrides_num_nodes(self):
        cfg = group(num_nodes=4, num_replicas=1, num_gpus_per_replica=4)
        assert replica_utils.num_engines(cfg, 8) == 1

    def test_num_replicas_enables_a_sub_node_group(self):
        """Half a node: not expressible through num_nodes at all."""
        cfg = group(num_nodes=0, num_replicas=1, num_gpus_per_replica=4)
        assert replica_utils.num_engines(cfg, 8) == 1

    def test_num_replicas_expands_to_one_engine_per_node(self):
        cfg = group(num_replicas=2, num_gpus_per_replica=16)
        assert replica_utils.num_engines(cfg, 8) == 4

    def test_missing_num_replicas_key_falls_back(self):
        cfg = OmegaConf.create({"num_nodes": 1, "num_gpus_per_replica": 8})
        assert replica_utils.num_engines(cfg, 8) == 1


class TestRolloutNumGpus:
    def test_colocate_single_group(self):
        cfg = rollout(
            8,
            regular={"num_nodes": 1, "num_gpus_per_replica": 8},
            prefill={"num_nodes": 0, "num_gpus_per_replica": 8},
            decode={"num_nodes": 0, "num_gpus_per_replica": 8},
        )
        assert replica_utils.rollout_num_gpus(cfg) == 8

    def test_pd_sharing_one_node_does_not_overcount(self):
        """The bug this field fixes: 2x4 GPUs on an 8-GPU node, not 16."""
        cfg = rollout(
            8,
            regular={"num_nodes": 0, "num_gpus_per_replica": 8},
            prefill={"num_nodes": 0, "num_replicas": 1, "num_gpus_per_replica": 4},
            decode={"num_nodes": 0, "num_replicas": 1, "num_gpus_per_replica": 4},
        )
        assert replica_utils.rollout_num_gpus(cfg) == 8

    def test_pd_across_nodes(self):
        cfg = rollout(
            8,
            regular={"num_nodes": 0, "num_gpus_per_replica": 8},
            prefill={"num_nodes": 2, "num_gpus_per_replica": 8},
            decode={"num_nodes": 4, "num_gpus_per_replica": 2},
        )
        assert replica_utils.rollout_num_gpus(cfg) == 48

    def test_multi_node_replica_is_counted_once(self):
        cfg = rollout(
            8,
            regular={"num_nodes": 2, "num_gpus_per_replica": 16},
        )
        assert replica_utils.rollout_num_gpus(cfg) == 16
