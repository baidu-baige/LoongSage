"""Unit tests for SglangEngine.validate_config."""

import pytest
from omegaconf import OmegaConf

pytest.importorskip("sglang", reason="SGLang is not installed")

from coda.backends.sglang.engine import SglangEngine  # noqa: E402


def _config(*, score_centering=False, sglang_args=None, completion_params=None):
    return OmegaConf.create(
        {
            "trainer": {"temperature": 1.0},
            "algorithm": {"score_centering": {"enable": score_centering, "topk": 128}},
            "data_sources": [{"completion_params": completion_params or {}}],
            "rollout": {
                "env_vars": {},
                "sglang_args": sglang_args or {},
                "sglang_replicas": {
                    "regular": {"num_nodes": 1},
                    "prefill": {"num_nodes": 0},
                    "decode": {"num_nodes": 0},
                },
            },
        }
    )


def test_score_centering_disabled_skips_sampling_checks():
    SglangEngine.validate_config(_config(completion_params={"top_p": 0.9}))


def test_score_centering_rejects_truncated_sampling():
    with pytest.raises(ValueError, match="top_p=0.9"):
        SglangEngine.validate_config(_config(score_centering=True, completion_params={"top_p": 0.9}))


def test_score_centering_checks_merged_replica_args():
    # Shared rollout.sglang_args are merged into every replica before the check.
    with pytest.raises(ValueError, match="speculative"):
        SglangEngine.validate_config(_config(score_centering=True, sglang_args={"speculative_algorithm": "EAGLE"}))
