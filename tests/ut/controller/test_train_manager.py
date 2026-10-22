"""Unit tests for TrainManager: config validation and actor/critic orchestration."""
import pytest
from omegaconf import OmegaConf
from unittest.mock import patch

pytest.importorskip("megatron", reason="Megatron-Core is not installed")

import coda.controller.train_manager as tm_mod
from coda.controller.train_manager import TrainManager
from coda.backends.megatron import MegatronActorWorker, MegatronCriticWorker
from coda.data_factory.data_processor import Box


def make_minimal_config(**overrides):
    """Build a minimal config that satisfies _validate_config without Ray/GPU.

    ``trainer.backend`` has no in-code default — it comes from conf/default.yaml
    (``backend: megatron``) — so _validate_config raises without it.
    """
    base = {
        "trainer": {
            "backend": "megatron",
            "num_nodes": 1,
            "num_gpus_per_node": 1,
            "hf_model_path": "/tmp/fake_model",
            "checkpoint_path": "/tmp/fake_ckpt",
        },
        "megatron": {
            "model": {
                "tensor_model_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
            },
            "optimizer": {"lr": 1e-6},
            "scheduler": {},
        },
        "algorithm": {"advantage_estimator": "grpo"},
        "critic": {"enable": False},
    }
    cfg = OmegaConf.create(base)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return cfg


def make_ppo_config(**overrides):
    cfg = make_minimal_config(
        trainer={"num_gpus_per_node": 2, "env_vars": {}, "save_freq": 2, "save_checkpoint": True},
        total_steps=10,
        algorithm={"advantage_estimator": "gae"},
        critic={"enable": True, "critic_warmup": 1},
        opd={"enable": False},
        data_sources=[{"num_trajectories_per_prompt": 1}],
    )
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return cfg


def call_validate_config(config):
    """Call TrainManager._validate_config without constructing the full object."""
    obj = object.__new__(TrainManager)
    obj.config = config
    obj._validate_config()
    return obj.config


class TestTrainManagerValidateConfig:
    def test_megatron_backend_delegates_to_worker_and_adopts_result(self):
        """The worker's returned config replaces self.config, not just mutates it."""
        config = make_minimal_config()
        validated = make_minimal_config(trainer={"seed": 7})

        with patch.object(
            MegatronActorWorker, "validate_config", return_value=validated
        ) as mock_vc:
            result = call_validate_config(config)

        mock_vc.assert_called_once_with(config)
        assert result is validated

    def test_non_megatron_backend_raises_before_validation(self):
        """The backend is checked in __init__ before any worker validation runs."""
        config = make_minimal_config(trainer={"backend": "other"})

        with patch.object(MegatronActorWorker, "validate_config") as mock_vc:
            with pytest.raises(ValueError, match="Unsupported backend 'other'"):
                TrainManager(config, resource_scheduler=None)

        mock_vc.assert_not_called()

    def test_missing_backend_key_raises(self):
        """A config without trainer.backend is a hard error, not a silent default."""
        config = make_minimal_config()
        del config.trainer.backend

        with pytest.raises(Exception, match="backend"):
            TrainManager(config, resource_scheduler=None)

    def test_critic_validation_runs_only_with_critic(self):
        identity = lambda cfg: cfg  # noqa: E731
        with patch.object(MegatronActorWorker, "validate_config", side_effect=identity), \
                patch.object(MegatronCriticWorker, "validate_config", side_effect=identity) as critic_vc:
            call_validate_config(make_minimal_config())
            critic_vc.assert_not_called()
            call_validate_config(make_ppo_config())
            critic_vc.assert_called_once()


def validate_ppo(config):
    identity = lambda cfg: cfg  # noqa: E731
    with patch.object(MegatronActorWorker, "validate_config", side_effect=identity), \
            patch.object(MegatronCriticWorker, "validate_config", side_effect=identity):
        return call_validate_config(config)


class TestPPOValidation:
    def test_valid_ppo_config_passes(self):
        validate_ppo(make_ppo_config())

    @pytest.mark.parametrize(
        "overrides, message",
        [
            ({"critic": {"enable": False}}, "gae requires critic.enable"),
            ({"algorithm": {"advantage_estimator": "grpo"}}, "critic.enable requires algorithm.advantage_estimator=gae"),
            ({"critic": {"critic_warmup": -1}}, "critic_warmup must be >= 0"),
            ({"opd": {"enable": True}}, "opd.enable"),
            ({"data_sources": [{"num_trajectories_per_prompt": 4}]}, "num_trajectories_per_prompt == 1"),
        ],
    )
    def test_invalid_ppo_configs_raise(self, overrides, message):
        with pytest.raises(ValueError, match=message):
            validate_ppo(make_ppo_config(**overrides))

    def test_grpo_config_passes(self):
        validate_ppo(make_minimal_config())


# ---------------------------------------------------------------------------
# Orchestration with a fake Ray: every remote call is recorded in order.
# ---------------------------------------------------------------------------

class _FakeMethod:
    def __init__(self, worker, name):
        self.worker, self.name = worker, name

    def remote(self, *args):
        w = self.worker
        w.log.append((w.role, w.rank, self.name, args))
        if self.name == "get_ip_port":
            return (f"{w.role}-host", w.port)
        if self.name == "train" and w.role == "critic":
            return {w.rank: f"adv-ref-{w.rank}"}  # one output rank per DP group
        return f"{w.role}{w.rank}.{self.name}"


class _FakeWorker:
    def __init__(self, role, rank, log, port):
        self.role, self.rank, self.log, self.port = role, rank, log, port

    def __getattr__(self, name):
        return _FakeMethod(self, name)


class _FakeActorClass:
    def __init__(self, cls, log):
        self.role = "critic" if cls is MegatronCriticWorker else "actor"
        self.log = log

    def remote(self, world_size, rank):
        return _FakeWorker(self.role, rank, self.log, 20000 if self.role == "actor" else 20001)


class _FakeRay:
    def __init__(self, log):
        self.log = log

    def remote(self, **_):
        return lambda cls: _FakeActorClass(cls, self.log)

    @staticmethod
    def get(x):
        return x


class _FakeScheduler:
    def __init__(self):
        self.calls = []

    def schedule(self, cls, num_bundles=1, bundle_index=-1):
        self.calls.append((cls.role, bundle_index))
        if bundle_index == -1:
            bundle_index = 10 + sum(1 for c in self.calls if c[1] == -1) - 1
        return cls, bundle_index


@pytest.fixture
def build(monkeypatch):
    def _build(config):
        log = []
        monkeypatch.setattr(tm_mod, "ray", _FakeRay(log))
        monkeypatch.setattr(MegatronActorWorker, "validate_config", classmethod(lambda cls, cfg: cfg))
        monkeypatch.setattr(MegatronCriticWorker, "validate_config", classmethod(lambda cls, cfg: cfg))
        scheduler = _FakeScheduler()
        manager = TrainManager(config, scheduler)
        return manager, log, scheduler
    return _build


def calls(log, start=0):
    return [(role, name, args) for role, _rank, name, args in log[start:] if name != "get_ip_port"]


def phases(log, start=0):
    """Collapse per-rank calls into (role, method, args) phases in order."""
    out = []
    for c in calls(log, start):
        if not out or out[-1] != c:
            out.append(c)
    return out


class TestCriticOrchestration:
    def test_critic_ranks_are_pinned_to_actor_bundles(self, build):
        manager, _log, scheduler = build(make_ppo_config())
        assert manager.bundle_indices == [10, 11]
        assert scheduler.calls == [("actor", -1), ("actor", -1), ("critic", 10), ("critic", 11)]

    def test_init_is_sequential_with_separate_masters(self, build):
        config = make_ppo_config()
        manager, log, _ = build(config)
        assert manager.async_init() == []
        assert phases(log) == [
            ("actor", "init", (config, "actor-host", 20000)),
            ("actor", "offload", (True,)),
            ("critic", "init", (config, "critic-host", 20001)),
            ("critic", "offload", (True,)),
        ]
        # The masters are passed explicitly, never written into the shared config.
        assert "master_addr" not in config.trainer

    def test_step_after_warmup_runs_critic_then_actor(self, build):
        manager, log, _ = build(make_ppo_config())
        manager.async_init()
        boxes = [Box("shard0"), Box("shard1")]
        start = len(log)

        refs = manager.async_train(3, boxes)

        assert phases(log, start) == [
            ("actor", "offload", (True,)),
            ("critic", "onload", ()),
            ("critic", "train", (3, boxes)),
            ("critic", "offload", (True,)),
            ("actor", "onload", ()),
            ("actor", "train", (3, boxes)),
        ]
        assert [b.critic_ref for b in boxes] == ["adv-ref-0", "adv-ref-1"]
        assert refs == ["actor0.train", "actor1.train"]

    def test_save_step_saves_critic_before_offloading_it(self, build):
        manager, log, _ = build(make_ppo_config())
        start = len(log)
        manager.async_train(4, [Box("s0"), Box("s1")])
        names = [(r, n) for r, n, _ in phases(log, start)]
        assert names.index(("critic", "save_model")) < names.index(("critic", "offload"))

    def test_warmup_step_skips_actor(self, build):
        manager, log, _ = build(make_ppo_config())
        start = len(log)
        assert manager.async_train(1, [Box("s0"), Box("s1")]) == []
        assert not any(role == "actor" and name in ("onload", "train") for role, name, _ in calls(log, start))
        assert manager.async_save_model(1) == []
        assert manager.async_save_model(None) == ["actor0.save_model", "actor1.save_model"]
        assert manager.async_save_model(2) == ["actor0.save_model", "actor1.save_model"]

    def test_colocate_onload_gives_gpu_to_critic_once(self, build):
        manager, log, _ = build(make_ppo_config())
        manager.onload()
        start = len(log)
        manager.async_train(3, [Box("s0"), Box("s1")])
        assert ("critic", "onload", ()) not in calls(log, start)

    def test_offload_releases_resident_critic_and_actor(self, build):
        manager, log, _ = build(make_ppo_config())
        manager.onload()
        start = len(log)
        manager.offload()
        assert phases(log, start) == [("critic", "offload", (True,)), ("actor", "offload", (False,))]

    def test_missing_dp_advantages_raise(self, build):
        manager, _log, _ = build(make_ppo_config(trainer={"num_gpus_per_node": 1}))
        with pytest.raises(RuntimeError, match="DP ranks \\[1\\]"):
            manager.async_train(3, [Box("s0"), Box("s1")])

    def test_without_critic_everything_targets_actor(self, build):
        config = make_minimal_config(trainer={"num_gpus_per_node": 2, "env_vars": {}})
        manager, log, scheduler = build(config)
        assert scheduler.calls == [("actor", -1), ("actor", -1)]
        assert manager.async_init() == ["actor0.init", "actor1.init"]
        manager.onload()
        assert manager.async_train(1, ["s0"]) == ["actor0.train", "actor1.train"]
        manager.offload(True)
        assert {role for role, *_ in log} == {"actor"}
