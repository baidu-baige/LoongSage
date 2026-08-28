"""Unit tests for TeacherManager._validate_config."""

from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

pytest.importorskip("megatron", reason="Megatron-Core is not installed")

from coda.backends.megatron.megatron_teacher_worker import MegatronTeacherWorker
from coda.controller.teacher_manager import TeacherManager


def make_config(pg_ratio=0.0, gkd_ratio=1.0):
    return OmegaConf.create(
        {
            "trainer": {
                "backend": "megatron",
                "num_nodes": 1,
                "num_gpus_per_node": 1,
            },
            "megatron": {
                "model": {
                    "tensor_model_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                    "context_parallel_size": 1,
                }
            },
            "opd": {
                "enable": True,
                "pg_ratio": pg_ratio,
                "gkd_ratio": gkd_ratio,
                "teacher_nodes": 1,
                "teacher_gpus_per_node": 1,
                "teachers": [{"name": "teacher", "hf_path": "/tmp/teacher"}],
                "model": {
                    "tensor_model_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                    "context_parallel_size": 1,
                },
            },
        }
    )


def validate(config):
    manager = object.__new__(TeacherManager)
    manager.config = config
    with patch.object(MegatronTeacherWorker, "validate_config", side_effect=lambda cfg: cfg):
        manager._validate_config()


@pytest.mark.parametrize("gkd_ratio", [0.0, 0.5, 1.0])
def test_gkd_ratio_accepts_closed_unit_interval(gkd_ratio):
    pg_ratio = 1.0 if gkd_ratio == 0.0 else 0.0
    validate(make_config(pg_ratio=pg_ratio, gkd_ratio=gkd_ratio))


@pytest.mark.parametrize(
    ("pg_ratio", "gkd_ratio", "message"),
    [
        # Neither objective is active.
        (0.0, 0.0, r"opd\.pg_ratio or opd\.gkd_ratio must be > 0"),
        # gkd_ratio is a fraction of the loss, so it cannot exceed 1.
        (0.0, 1.1, r"gkd_ratio must be <= 1"),
        # gkd_ratio == 1 is pure distillation, which leaves no room for PG.
        (0.5, 1.0, r"mutually exclusive"),
    ],
)
def test_opd_ratios_reject_invalid_combinations(pg_ratio, gkd_ratio, message):
    with pytest.raises(ValueError, match=message):
        validate(make_config(pg_ratio=pg_ratio, gkd_ratio=gkd_ratio))
