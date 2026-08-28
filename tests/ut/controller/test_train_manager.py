"""Unit tests for TrainManager._validate_config."""
import pytest
from omegaconf import OmegaConf
from unittest.mock import patch

pytest.importorskip("megatron", reason="Megatron-Core is not installed")

from coda.controller.train_manager import TrainManager
from coda.backends.megatron import MegatronTrainWorker


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
        "algorithm": {},
    }
    cfg = OmegaConf.create(base)
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
            MegatronTrainWorker, "validate_config", return_value=validated
        ) as mock_vc:
            result = call_validate_config(config)

        mock_vc.assert_called_once_with(config)
        assert result is validated

    def test_non_megatron_backend_is_left_untouched(self):
        """_validate_config only handles megatron; other backends fail later in __init__."""
        config = make_minimal_config(trainer={"backend": "other"})

        with patch.object(MegatronTrainWorker, "validate_config") as mock_vc:
            result = call_validate_config(config)

        mock_vc.assert_not_called()
        assert result.trainer.backend == "other"

    def test_missing_backend_key_raises(self):
        """A config without trainer.backend is a hard error, not a silent default."""
        config = make_minimal_config()
        del config.trainer.backend

        with pytest.raises(Exception, match="backend"):
            call_validate_config(config)
