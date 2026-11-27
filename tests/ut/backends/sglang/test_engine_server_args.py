"""Unit tests for how coda.backends.sglang.engine._compute_server_args sets `enable_multimodal`."""

import unittest

import pytest
from omegaconf import OmegaConf

pytest.importorskip("sglang", reason="SGLang is not installed")

from coda.backends.sglang.engine import _compute_server_args  # noqa: E402


def _config(*, multimodal_enable=False):
    return OmegaConf.create(
        {
            "colocate": False,
            "hf_model_path": "/models/m",
            "multimodal": {"enable": multimodal_enable},
            "rollout": {"num_gpus_per_node": 8},
            "trainer": {"use_rollout_routing_replay": False, "use_fp32_lm_head": False},
        }
    )


def _server_args(config, engine_args=None):
    return _compute_server_args(
        config,
        rank=0,
        dist_init_addr="127.0.0.1:1",
        nccl_port=2,
        host="127.0.0.1",
        port=3,
        base_gpu_id=0,
        engine_args=engine_args or {},
        num_gpus_per_replica=8,
    )


class TestEnableMultimodal(unittest.TestCase):
    def test_enabled_defaults_to_true(self):
        self.assertIs(_server_args(_config(multimodal_enable=True))["enable_multimodal"], True)

    def test_disabled_defaults_to_false(self):
        self.assertIs(_server_args(_config(multimodal_enable=False))["enable_multimodal"], False)

    def test_sglang_args_override_the_default(self):
        # engine_args come from rollout.sglang_args and are applied after the default derived from multimodal.enable.
        args = _server_args(_config(multimodal_enable=True), {"enable_multimodal": False})
        self.assertIs(args["enable_multimodal"], False)
        args = _server_args(_config(multimodal_enable=False), {"enable_multimodal": True})
        self.assertIs(args["enable_multimodal"], True)


if __name__ == "__main__":
    unittest.main()
