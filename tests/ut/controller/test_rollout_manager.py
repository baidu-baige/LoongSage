"""Unit tests for engine recovery in coda/controller/rollout_manager.py."""
import threading
from unittest.mock import Mock, patch

import pytest
from omegaconf import OmegaConf

pytest.importorskip("sglang", reason="SGLang is not installed")

from coda.backends.replica_group import ReplicaGroup  # noqa: E402
from coda.controller.rollout_manager import RolloutManager  # noqa: E402


def _manager(group):
    manager = object.__new__(RolloutManager)
    manager.replica_groups = [group]
    manager._health_monitors = []
    return manager


def _two_node_group():
    """One 16-GPU engine on two 8-GPU nodes: two actor slots."""
    config = OmegaConf.create({"colocate": False, "rollout": {"num_gpus_per_node": 8, "backend": "sglang"}})
    group = ReplicaGroup(config, OmegaConf.create({"num_gpus_per_replica": 16}), scheduler=None, num_engines=2)
    group.all_engines = [Mock(name="node0"), Mock(name="node1")]
    return group


def test_new_engines_are_published_after_init():
    group = _two_node_group()
    group.all_engines = [None, None]
    new0, new1 = Mock(name="new0"), Mock(name="new1")
    group.start_engines = Mock(return_value=([(0, new0), (1, new1)], ["init0", "init1"]))
    slots_during_init = []

    def fake_get(refs):
        slots_during_init.append(list(group.all_engines))
        return ["http://10.0.0.1:2000", None]

    with patch("coda.controller.rollout_manager.ray.get", side_effect=fake_get):
        _manager(group).recover_faulty_engines()

    assert slots_during_init == [[None, None]]
    assert group.all_engines == [new0, new1]
    assert group.worker_urls == ["http://10.0.0.1:2000", None]
    assert group.num_new_engines == 2


def test_failed_recovery_kills_new_engines_and_leaves_slots_empty():
    group = _two_node_group()
    group.all_engines = [None, None]
    new0, new1 = Mock(name="new0"), Mock(name="new1")
    group.start_engines = Mock(return_value=([(0, new0), (1, new1)], ["init0", "init1"]))

    with patch("coda.controller.rollout_manager.ray.get", side_effect=RuntimeError("init failed")), \
            patch("coda.controller.rollout_manager.ray.kill") as mock_kill:
        with pytest.raises(RuntimeError, match="init failed"):
            _manager(group).recover_faulty_engines()

    assert [c.args[0] for c in mock_kill.call_args_list] == [new0, new1]
    assert group.all_engines == [None, None]
    assert group.num_new_engines == 0
    assert not group.lock.locked()


def test_recovery_waits_for_a_kill_in_progress():
    """Recovery does not read the slots while the monitor is halfway through killing a two-node engine."""
    group = _two_node_group()
    group.all_engines[1].shutdown.remote.return_value = "shutdown_node1"
    in_node1_shutdown, release_shutdown = threading.Event(), threading.Event()

    def blocking_get(ref, timeout=None):
        if ref == "shutdown_node1":
            in_node1_shutdown.set()
            release_shutdown.wait(timeout=10)

    slots_at_recovery = []

    def fake_start_engines(recover=False):
        slots_at_recovery.append(list(group.all_engines))
        return [], []

    group.start_engines = fake_start_engines

    # rollout_manager and replica_group share one ray module, so this patch covers both;
    # fake_start_engines returns no handles, so recovery itself never calls ray.get.
    with patch("coda.backends.replica_group.ray.get", side_effect=blocking_get), \
            patch("coda.backends.replica_group.ray.kill"):
        killer = threading.Thread(target=group.kill_engine, args=(0,))
        killer.start()
        assert in_node1_shutdown.wait(timeout=10)
        assert group.all_engines[0] is None  # node-0 is already gone, node-1 is not

        recovery = threading.Thread(target=_manager(group).recover_faulty_engines)
        recovery.start()
        recovery.join(timeout=0.5)
        assert recovery.is_alive()
        assert slots_at_recovery == []

        release_shutdown.set()
        killer.join(timeout=10)
        recovery.join(timeout=10)

    assert slots_at_recovery == [[None, None]]
