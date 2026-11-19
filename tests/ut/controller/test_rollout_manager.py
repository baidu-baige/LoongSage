"""Unit tests for RolloutManager's weight-channel rank plan and staged offload.

``coda.controller.rollout_manager`` pulls in ray and sglang, neither of which is
needed for the logic under test, so both are stubbed in ``setup_module`` and torn
down afterwards (the same approach ``test_trainer.py`` uses for Megatron).
"""

import sys
import types
from unittest.mock import MagicMock

import pytest

_ORIGINAL_SYS_MODULES: dict[str, object] = {}

RolloutManager = None
PD_WORKER_TYPES = None
ChannelMeta = None


def _stub(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def setup_module(module):
    """Install ray/sglang stubs and import the module under test."""
    global RolloutManager, PD_WORKER_TYPES, ChannelMeta
    for name in [
        "ray",
        "requests",
        "coda.utils.tracking",
        "sglang",
        "sglang.srt",
        "sglang.srt.constants",
        "sglang.srt.server_args",
        "sglang.srt.utils",
        "coda.controller.rollout_manager",
        "coda.backends.sglang.engine",
        "coda.backends.replica_group",
        "coda.utils.health_monitor",
    ]:
        _ORIGINAL_SYS_MODULES[name] = sys.modules.get(name)

    ray_stub = _stub("ray")
    # ray.get is only used as a barrier here; return the refs unchanged so the
    # staged offload order can be read straight off the recorded calls.
    ray_stub.get = lambda refs: list(refs)
    ray_stub.remote = lambda *args, **kwargs: MagicMock()
    ray_stub.kill = MagicMock()
    ray_stub.util = MagicMock()
    _stub("requests", post=MagicMock(), get=MagicMock())
    # coda.controller.__init__ imports rollout_sampler, which pulls in
    # coda.utils.tracking (mlflow + pandas). None of it is involved here.
    _stub(
        "coda.utils.tracking",
        track=lambda *args, **kwargs: (lambda fn: fn),
        configure_tracking=MagicMock(),
        time_marker=MagicMock(),
        TimeMarkerAcc=MagicMock(),
    )
    _stub("sglang")
    _stub("sglang.srt")
    _stub(
        "sglang.srt.constants",
        GPU_MEMORY_TYPE_CUDA_GRAPH="cuda_graph",
        GPU_MEMORY_TYPE_KV_CACHE="kv_cache",
        GPU_MEMORY_TYPE_WEIGHTS="weights",
    )
    _stub("sglang.srt.server_args", ServerArgs=MagicMock())
    _stub("sglang.srt.utils", kill_process_tree=MagicMock())
    # Pulls in mlflow through coda.utils.tracking; nothing here starts a monitor.
    _stub("coda.utils.health_monitor", RolloutHealthMonitor=MagicMock())

    from coda.controller import rollout_manager as rm
    from coda.transfer_mesh import ChannelMeta as _ChannelMeta

    RolloutManager = rm.RolloutManager
    PD_WORKER_TYPES = rm.PD_WORKER_TYPES
    ChannelMeta = _ChannelMeta


def teardown_module(module):
    """Restore the real modules so later test files are unaffected."""
    for name, original in _ORIGINAL_SYS_MODULES.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


class FakeEngine:
    """Records the ChannelMeta dict it was handed and the memory ops it received."""

    def __init__(self, label: str, journal: list):
        self.label = label
        self.journal = journal
        self.meta_dicts: list[dict] = []
        self.update_weights_from_channel = MagicMock()
        self.update_weights_from_channel.remote = self._update_remote

    def _update_remote(self, meta_dict):
        self.meta_dicts.append(meta_dict)
        return {"success": True}


class FakeReplicaGroup:
    """Minimal stand-in: a worker_type, a GPU count, and a list of node-0 engines."""

    def __init__(self, worker_type: str, num_gpus_per_replica: int, engines, journal):
        self.worker_type = worker_type
        self.replica_config = types.SimpleNamespace(
            num_gpus_per_replica=num_gpus_per_replica
        )
        self.engines = engines
        self.journal = journal

    def offload(self, tags=None):
        self.journal.append(("offload", self.worker_type, tags))
        return [f"offload:{self.worker_type}"]

    def onload(self, tags=None):
        self.journal.append(("onload", self.worker_type, tags))
        return [f"onload:{self.worker_type}"]


def make_manager(spec: list[tuple[str, int, int]], journal: list | None = None):
    """Build a RolloutManager over fake replica groups.

    ``spec`` is a list of ``(worker_type, num_gpus_per_replica, num_engines)``.
    ``__init__`` is bypassed on purpose: it would build a Ray placement group.
    """
    journal = journal if journal is not None else []
    manager = object.__new__(RolloutManager)
    manager.config = None
    manager._health_monitors = []
    groups = []
    for worker_type, num_gpus, num_engines in spec:
        engines = [
            FakeEngine(f"{worker_type}-{i}", journal) for i in range(num_engines)
        ]
        groups.append(FakeReplicaGroup(worker_type, num_gpus, engines, journal))
    manager.replica_groups = groups
    return manager, journal


def channel_meta(rollout_world_size: int, train_world_size: int = 16) -> "ChannelMeta":
    return ChannelMeta(
        master_addr="127.0.0.1",
        world_size=train_world_size + rollout_world_size,
        train_world_size=train_world_size,
    )


# ---------------------------------------------------------------------------
# rank_base plan
# ---------------------------------------------------------------------------


class TestRankBasePlan:
    """Each engine gets an explicit base that tiles the rollout rank range."""

    def test_regular_topology_keeps_the_legacy_numbering(self):
        """Homogeneous engines: the explicit base equals the old derived one."""
        manager, _ = make_manager([("regular", 8, 4)])
        manager.async_update_weights(channel_meta(32), weight_version=3)

        metas = [m for g in manager.replica_groups for e in g.engines for m in e.meta_dicts]
        assert [m["rank_base"] for m in metas] == [0, 8, 16, 24]
        assert [m["engine_id"] for m in metas] == [0, 1, 2, 3]
        assert {m["weight_version"] for m in metas} == {"3"}

    def test_heterogeneous_tp_tiles_without_gaps_or_overlap(self):
        """prefill TP=8 x2 then decode TP=16 x2: the derived formula would not fit."""
        manager, _ = make_manager([("prefill", 8, 2), ("decode", 16, 2)])
        manager.async_update_weights(channel_meta(48), weight_version=1)

        metas = [m for g in manager.replica_groups for e in g.engines for m in e.meta_dicts]
        bases = [m["rank_base"] for m in metas]
        assert bases == [0, 8, 16, 32]
        # Every rollout rank is claimed exactly once.
        covered = sorted(
            base + local
            for base, world in zip(bases, [8, 8, 16, 16])
            for local in range(world)
        )
        assert covered == list(range(48))

    def test_coverage_shortfall_is_caught(self):
        """A plan that does not reach rollout_world_size fails loudly, not silently."""
        manager, _ = make_manager([("prefill", 8, 2), ("decode", 16, 2)])
        with pytest.raises(AssertionError, match="rank_base coverage"):
            manager.async_update_weights(channel_meta(64), weight_version=1)


# ---------------------------------------------------------------------------
# Staged offload / onload
# ---------------------------------------------------------------------------


class TestStagedMemoryOps:
    """Release has a direction; resume does not. Regular keeps the single fan-out."""

    @staticmethod
    def _record_barriers(monkeypatch, journal):
        """Log every ``ray.get`` into the journal.

        Staged and single fan-out visit the replica groups in the same order, so the
        order alone cannot tell them apart. What the staged path adds is a ``ray.get``
        *between* the groups, so the barrier has to be observable.
        """
        import ray

        original = ray.get

        def get(refs):
            journal.append(("barrier", None, None))
            return original(refs)

        monkeypatch.setattr(ray, "get", get)

    def test_regular_offload_stays_single_stage(self):
        manager, journal = make_manager([("regular", 8, 2)])
        manager.offload(tags=["kv_cache"])
        assert journal == [("offload", "regular", ["kv_cache"])]

    def test_pd_offload_waits_for_prefill_before_decode(self, monkeypatch):
        """Prefill may still be RDMA-writing into regions decode is about to drop."""
        manager, journal = make_manager([("prefill", 8, 1), ("decode", 2, 2)])
        self._record_barriers(monkeypatch, journal)
        manager.offload(tags=["kv_cache"])
        # A barrier between the stages is the whole point of staging.
        assert [entry[0] for entry in journal] == [
            "offload", "barrier", "offload", "barrier",
        ]
        assert [e[1] for e in journal if e[0] == "offload"] == ["prefill", "decode"]

    def test_pd_onload_kv_is_a_single_fan_out(self, monkeypatch):
        """Resume has no cross-engine dependency, so both halves go concurrently.

        Each half only registers its own pool with its own NIC, decode's new address
        reaches prefill through the next step's per-request bootstrap rather than here,
        and no request can arrive before health monitoring resumes.
        """
        manager, journal = make_manager([("prefill", 8, 1), ("decode", 2, 2)])
        self._record_barriers(monkeypatch, journal)
        manager.onload_kv()
        assert [entry[0] for entry in journal] == ["onload", "onload", "barrier"]
        assert {entry[2][0] for entry in journal if entry[0] == "onload"} == {"kv_cache"}

    def test_onload_weights_is_not_staged(self, monkeypatch):
        """The WEIGHTS tag never touches the RDMA memory regions."""
        manager, journal = make_manager([("prefill", 8, 1), ("decode", 2, 1)])
        self._record_barriers(monkeypatch, journal)
        manager.onload_weights()
        assert [entry[0] for entry in journal] == ["onload", "onload", "barrier"]
        assert {entry[2][0] for entry in journal if entry[0] == "onload"} == {"weights"}

    def test_is_pd_disaggregated_reflects_the_groups(self):
        pd, _ = make_manager([("prefill", 8, 1), ("decode", 8, 1)])
        regular, _ = make_manager([("regular", 8, 1)])
        assert pd.is_pd_disaggregated is True
        assert regular.is_pd_disaggregated is False
        assert PD_WORKER_TYPES == ("prefill", "decode")
