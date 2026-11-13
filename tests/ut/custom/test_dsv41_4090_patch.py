"""Unit tests for coda/custom/dsv41_4090/patch.py (stock controller + streaming recipe)."""
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

pytest.importorskip("megatron", reason="Megatron-Core is not installed")

import coda.controller.rollout_manager as rm_mod
import coda.controller.train_manager as tm_mod
from coda.backends.megatron import MegatronActorWorker
from coda.custom.dsv41_4090 import patch as dsv_patch
from coda.custom.dsv41_4090.stream_worker import MegatronStreamActorWorker

dsv_patch.apply()


def make_config(stream):
    cfg = {
        "hf_model_path": "/ckpt/serve",
        "trainer": {"backend": "megatron", "num_nodes": 1, "num_gpus_per_node": 1, "env_vars": {}},
        "algorithm": {"advantage_estimator": "grpo"},
        "critic": {"enable": False},
    }
    if stream:
        cfg["dsv41_4090"] = {"stream": True}
    return OmegaConf.create(cfg)


def build_train_manager(config):
    scheduler = MagicMock()
    scheduler.schedule.return_value = (MagicMock(), 0)
    with patch.object(tm_mod.ray, "remote", side_effect=lambda **_: (lambda cls: cls)):
        manager = tm_mod.TrainManager(config, scheduler)
    return manager, scheduler.schedule.call_args[0][0]


def make_rollout_manager(config, n_engines=2):
    manager = object.__new__(rm_mod.RolloutManager)
    manager.config = config
    engines = [MagicMock() for _ in range(n_engines)]
    manager.replica_groups = [MagicMock(engines=engines)]
    return manager, engines


class TestTrainManager:
    def test_stream_recipe_builds_the_stream_worker(self):
        manager, actor_cls = build_train_manager(make_config(stream=True))
        assert actor_cls is MegatronStreamActorWorker
        assert tm_mod.MegatronActorWorker is MegatronActorWorker
        assert manager.async_update_weights(channel_meta=None) == []

    def test_other_recipes_keep_the_megatron_worker(self):
        config = make_config(stream=False)
        with patch.object(MegatronActorWorker, "validate_config", return_value=config):
            manager, actor_cls = build_train_manager(config)
        assert actor_cls is MegatronActorWorker
        manager._actor_worker_handlers = [MagicMock()]
        assert len(manager.async_update_weights(channel_meta="meta")) == 1

    def test_stream_worker_rejects_a_critic(self):
        config = make_config(stream=True)
        config.critic.enable = True
        with pytest.raises(ValueError, match="critic"):
            build_train_manager(config)


class TestRolloutManager:
    def test_initial_sync_only_sets_the_version(self):
        manager, engines = make_rollout_manager(make_config(stream=True))
        manager.async_update_weights(meta=None, weight_version=1)
        for e in engines:
            e._make_request.remote.assert_called_once_with(
                "update_weight_version", {"new_version": "1", "abort_all_requests": False}, 60.0)
            e._flush_cache.remote.assert_not_called()

    def test_later_syncs_flush_then_reload_from_disk(self):
        manager, engines = make_rollout_manager(make_config(stream=True))
        manager.async_update_weights(meta=None, weight_version=1)
        for e in engines:
            e.reset_mock()
        refs = manager.async_update_weights(meta=None, weight_version=2)
        assert len(refs) == 2 * len(engines)
        for e in engines:
            e._flush_cache.remote.assert_called_once_with()
            e._make_request.remote.assert_called_once_with(
                "update_weights_from_disk",
                {"model_path": "/ckpt/serve", "load_format": "auto", "weight_version": "2"}, 1800.0)
            e.update_weights_from_channel.remote.assert_not_called()

    def test_other_recipes_use_the_channel(self):
        manager, engines = make_rollout_manager(make_config(stream=False))
        with patch.object(rm_mod.copy, "deepcopy", return_value=MagicMock()), \
                patch.object(rm_mod.dataclasses, "asdict", return_value={}):
            manager.async_update_weights(meta=MagicMock(), weight_version=3)
        for e in engines:
            e.update_weights_from_channel.remote.assert_called_once()
            e._make_request.remote.assert_not_called()
