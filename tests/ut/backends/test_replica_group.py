"""Unit tests for engine bookkeeping in coda/backends/replica_group.py."""
from unittest.mock import Mock, patch

import pytest
from omegaconf import OmegaConf

pytest.importorskip("sglang", reason="SGLang is not installed")

from coda.backends.replica_group import ReplicaGroup  # noqa: E402


def _config():
    return OmegaConf.create({
        "colocate": False,
        "rollout": {"num_gpus_per_node": 8, "backend": "sglang", "env_vars": {}},
        "agentflow": {"router": {"ip": "127.0.0.1", "port": 30000}},
    })


def _two_node_group():
    """One 16-GPU engine on two 8-GPU nodes: two actor slots."""
    group = ReplicaGroup(_config(), OmegaConf.create({"num_gpus_per_replica": 16}), scheduler=None, num_engines=2)
    group.all_engines = [Mock(name="node0"), Mock(name="node1")]
    return group


def _ray_get(ref, timeout=None):
    if isinstance(ref, Exception):
        raise ref
    return ref


def test_kill_engine_kills_every_node_even_if_shutdown_fails():
    group = _two_node_group()
    node0, node1 = group.all_engines
    node0.shutdown.remote.return_value = TimeoutError("shutdown timed out")
    node1.shutdown.remote.return_value = None

    with patch("coda.backends.replica_group.ray.get", side_effect=_ray_get), \
            patch("coda.backends.replica_group.ray.kill") as mock_kill:
        group.kill_engine(rollout_engine_id=0)

    assert [c.args[0] for c in mock_kill.call_args_list] == [node0, node1]
    assert group.all_engines == [None, None]


def test_kill_engine_removes_worker_from_router_before_shutdown():
    """The router stops routing to the engine even when the actor never answers shutdown()."""
    group = _two_node_group()
    node0, node1 = group.all_engines
    group.worker_urls = ["http://10.0.0.1:2000", None]
    order = []
    node0.shutdown.remote.side_effect = lambda: order.append("shutdown") or TimeoutError("hung")
    node1.shutdown.remote.return_value = None

    def fake_post(url, json, timeout):
        order.append(("post", url, json["worker_url"]))
        return Mock(status_code=200)

    with patch("coda.backends.replica_group.requests.post", side_effect=fake_post), \
            patch("coda.backends.replica_group.ray.get", side_effect=_ray_get), \
            patch("coda.backends.replica_group.ray.kill"):
        group.kill_engine(rollout_engine_id=0)

    assert order[0] == ("post", "http://127.0.0.1:30000/remove_worker", "http://10.0.0.1:2000")
    assert order[1:] == ["shutdown"]
    assert group.worker_urls == [None, None]
    assert group.all_engines == [None, None]


def test_kill_engine_still_kills_when_router_unreachable():
    group = _two_node_group()
    node0, node1 = group.all_engines
    group.worker_urls = ["http://10.0.0.1:2000", None]

    with patch("coda.backends.replica_group.requests.post", side_effect=ConnectionError("refused")), \
            patch("coda.backends.replica_group.ray.get", side_effect=_ray_get), \
            patch("coda.backends.replica_group.ray.kill") as mock_kill:
        group.kill_engine(rollout_engine_id=0)

    assert [c.args[0] for c in mock_kill.call_args_list] == [node0, node1]
    assert group.worker_urls == [None, None]


def test_start_engines_leaves_slots_empty_until_published():
    """New actors stay out of all_engines (and out of the health monitor's view) until published."""
    scheduler = Mock()
    scheduler.reorder_bundle_list = [{"gpu_id": 0}, {"gpu_id": 4}]
    actor_cls = Mock()
    actor_cls.remote.side_effect = [Mock(name="new1")]
    scheduler.schedule.return_value = (actor_cls, 1)
    group = ReplicaGroup(
        _config(), OmegaConf.create({"num_gpus_per_replica": 4, "sglang_args": {}}),
        scheduler=scheduler, num_engines=2, rank_offset=3,
    )
    alive = Mock(name="alive0")
    group.all_engines = [alive, None]
    group.engine_bundle_indices = [0, 1]

    with patch("coda.backends.replica_group.ray.remote", return_value=lambda cls: cls), \
            patch("coda.backends.replica_group._allocate_rollout_engine_addr_and_ports",
                  return_value={4: {"port": 2000}}):
        new_engines, handles = group.start_engines(recover=True)

    assert scheduler.schedule.call_args.kwargs["bundle_index"] == 1
    assert [slot for slot, _ in new_engines] == [1]
    assert len(handles) == 1
    assert group.all_engines == [alive, None]

    group.publish_engines(new_engines, ["http://10.0.0.1:2000"])

    assert group.all_engines == [alive, new_engines[0][1]]
    assert group.worker_urls == [None, "http://10.0.0.1:2000"]
    assert group.num_new_engines == 1
