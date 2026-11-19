"""Unit tests for ``_allocate_rollout_engine_addr_and_ports``.

ray and sglang are stubbed so the allocator can be driven with fake engines whose
node placement is chosen by the test.
"""

import sys
import types
from unittest.mock import MagicMock

import pytest

_ORIGINAL_SYS_MODULES: dict[str, object] = {}
allocate = None


def _stub(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def setup_module(module):
    global allocate
    names = ["ray", "coda.backends.sglang.engine", "coda.backends.replica_group"]
    for name in names:
        _ORIGINAL_SYS_MODULES[name] = sys.modules.get(name)
    sys.modules.pop("coda.backends.replica_group", None)
    # Fake engines return plain values from .remote(), so ray.get is the identity.
    _stub("ray", get=lambda x: x)
    _stub("coda.backends.sglang.engine", SglangEngine=MagicMock())
    from coda.backends import replica_group

    allocate = replica_group._allocate_rollout_engine_addr_and_ports


def teardown_module(module):
    for name, original in _ORIGINAL_SYS_MODULES.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


class _Method:
    def __init__(self, fn):
        self.remote = fn


class FakeEngine:
    """Engine pinned to ``host``; port probes are served by the host's ledger."""

    def __init__(self, host: str, taken: dict[str, set[int]]):
        self.host = host
        self.taken = taken
        self.probes: list[int] = []
        self.get_node_ip = _Method(lambda: self.host)
        self.get_ip_port = _Method(self._probe)

    def _probe(self, base_port, consecutive):
        # Deterministic stand-in for find_available_port: skip ports bound on this host.
        self.probes.append(base_port)
        used = self.taken.setdefault(self.host, set())
        port = base_port
        while any(p in used for p in range(port, port + consecutive)):
            port += 1
        return self.host, port


def _engines(hosts: list[str], rank_offset: int, taken=None):
    taken = {} if taken is None else taken
    return [(rank_offset + i, FakeEngine(h, taken)) for i, h in enumerate(hosts)]


def test_group_starting_mid_node_gets_each_engines_own_host():
    # 1P:3D at TP4 on 2x8 GPUs: prefill on node0 GPU0-3, decode on node0 GPU4-7 + node1.
    engines = _engines(["n0", "n1", "n1"], rank_offset=1)
    out = allocate(engines, num_gpus_per_node=8, num_gpus_per_replica=4,
                   worker_type="decode", rank_offset=1, base_port=22000)

    assert sorted(out) == [1, 2, 3]
    assert [out[r]["host"] for r in (1, 2, 3)] == ["n0", "n1", "n1"]
    # Every engine is probed by itself, never on behalf of a neighbour.
    assert all(len(e.probes) == 1 for _, e in engines)
    # Same-host engines get disjoint 4-port blocks.
    assert out[3]["port"] >= out[2]["port"] + 4
    # First engine on each host starts from the group's base.
    assert out[1]["port"] == 22000 and out[2]["port"] == 22000
    for r in (1, 2, 3):
        assert out[r]["nccl_port"] == out[r]["port"] + 1
        assert out[r]["dist_init_addr"] == f"{out[r]['host']}:{out[r]['port'] + 3}"
        assert "disaggregation_bootstrap_port" not in out[r]


def test_prefill_gets_bootstrap_port():
    engines = _engines(["n0", "n0"], rank_offset=0)
    out = allocate(engines, num_gpus_per_node=8, num_gpus_per_replica=4,
                   worker_type="prefill", rank_offset=0, base_port=12000)
    for r in (0, 1):
        assert out[r]["disaggregation_bootstrap_port"] == out[r]["port"] + 2
    assert out[1]["port"] >= out[0]["port"] + 4


def test_multi_node_replica_shares_head_dist_init_addr():
    # Two TP16 replicas over four nodes: ranks (0,1) and (2,3).
    engines = _engines(["n0", "n1", "n2", "n3"], rank_offset=0)
    out = allocate(engines, num_gpus_per_node=8, num_gpus_per_replica=16,
                   worker_type="regular", rank_offset=0, base_port=2000)
    assert out[0]["dist_init_addr"] == out[1]["dist_init_addr"] == f"n0:{out[0]['port'] + 3}"
    assert out[2]["dist_init_addr"] == out[3]["dist_init_addr"] == f"n2:{out[2]['port'] + 3}"


def test_multi_node_head_uses_local_rank_with_offset():
    # A TP16 decode group after one TP16 prefill replica (rank_offset=2).
    engines = _engines(["n2", "n3"], rank_offset=2)
    out = allocate(engines, num_gpus_per_node=8, num_gpus_per_replica=16,
                   worker_type="decode", rank_offset=2, base_port=22000)
    assert out[3]["dist_init_addr"] == out[2]["dist_init_addr"] == f"n2:{out[2]['port'] + 3}"


def test_recover_single_engine_skips_ports_held_by_survivors():
    taken = {"n0": {22000, 22001, 22002, 22003}}
    engines = [(2, FakeEngine("n0", taken))]
    out = allocate(engines, num_gpus_per_node=8, num_gpus_per_replica=4,
                   worker_type="decode", rank_offset=1, base_port=22000)
    assert list(out) == [2]
    assert out[2]["port"] not in taken["n0"]


@pytest.mark.parametrize("hosts", [["n0"] * 8, ["n0"] * 4 + ["n1"] * 4])
def test_tp1_blocks_disjoint_per_host(hosts):
    engines = _engines(hosts, rank_offset=0)
    out = allocate(engines, num_gpus_per_node=8, num_gpus_per_replica=1,
                   worker_type="regular", rank_offset=0, base_port=2000)
    by_host: dict[str, list[int]] = {}
    for r, info in out.items():
        by_host.setdefault(info["host"], []).append(info["port"])
    for ports in by_host.values():
        ports.sort()
        assert all(b - a >= 4 for a, b in zip(ports, ports[1:]))
