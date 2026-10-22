"""Unit tests for the Megatron train worker base and the actor worker."""
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf

# The worker module imports Megatron-Core / Transformer-Engine at module scope;
# validate_config itself is pure config manipulation.
pytest.importorskip("megatron", reason="Megatron-Core is not installed")

from coda.backends.megatron import MegatronTrainWorker, MegatronActorWorker


def make_minimal_config(**overrides):
    """Build a minimal config that satisfies validate_config without Ray/GPU.

    Every key read by ``validate_config`` must be present: OmegaConf raises on a
    missing attribute, so a key added to the production validator without being
    added here fails every test in this file.
    """
    base = {
        "fully_async": {
            "enable": False,
        },
        "trainer": {
            "num_nodes": 1,
            "num_gpus_per_node": 1,
            "hf_model_path": "/tmp/fake_model",
            "checkpoint_path": "/tmp/fake_ckpt",
            "env_vars": {},
            "seed": 42,
            "async_save": True,
            "deterministic_mode": False,
            "use_rollout_log_probs": False,
            "use_rollout_routing_replay": False,
            "use_fp32_lm_head": False,
            "use_dynamic_batch_size": True,
            "mini_batch_size": 1,
            "micro_batch_size": 1,
        },
        "megatron": {
            "model": {
                "bf16": True,
                "fp16": False,
                "fp8_param": False,
                "fp8_recipe": None,
                "tensor_model_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
                "virtual_pipeline_model_parallel_size": None,
                "context_parallel_size": 1,
                "expert_model_parallel_size": 1,
                "expert_tensor_parallel_size": None,
                "overlap_p2p_comm": False,
            },
            "ddp_config": {
                "use_distributed_optimizer": True,
                "overlap_param_gather": False,
            },
            "keep_fp32_weights": {},
            "optimizer": {
                "lr": 1e-6,
                "weight_decay": 0.01,
                "optimizer_cpu_offload": False,
                "chunked_optimizer_state_offload": False,
            },
            "scheduler": {},
        },
        "algorithm": {
            "loss_agg_mode": "token-mean",
            "clip_ratio_c": 3.0,
            "is_correction": {
                "enable": False,
            },
            "m2po": {
                "enable": False,
                "threshold": 0.0,
            },
            "ref_kl": {
                "enable": False,
                "kl_type": "k1",
            },
        },
        "ref_dist_ckpt_path": None,
        "ref_hf_model_path": None,
    }
    cfg = OmegaConf.create(base)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return cfg


# ---------------------------------------------------------------------------
# MegatronActorWorker.validate_config — forced parameters
# ---------------------------------------------------------------------------

class TestMegatronValidateConfigForced:
    def _run(self, **overrides):
        config = make_minimal_config(**overrides)
        return MegatronActorWorker.validate_config(config)

    def test_variable_seq_lengths_always_true(self):
        result = self._run(megatron={"model": {"variable_seq_lengths": False}})
        assert result.megatron.model.variable_seq_lengths is True

    def test_moe_token_dispatcher_type_always_alltoall(self):
        result = self._run(megatron={"model": {"moe_token_dispatcher_type": "other"}})
        assert result.megatron.model.moe_token_dispatcher_type == "alltoall"


# ---------------------------------------------------------------------------
# MegatronActorWorker.validate_config — derived parameters
# ---------------------------------------------------------------------------

class TestMegatronValidateConfigDerived:
    def _run(self, **overrides):
        config = make_minimal_config(**overrides)
        return MegatronActorWorker.validate_config(config)

    def test_param_dtype_set_to_bfloat16_when_bf16_true(self):
        result = self._run(megatron={"model": {"bf16": True}})
        assert result.megatron.model.params_dtype == "torch.bfloat16"

    def test_param_dtype_not_set_when_bf16_false(self):
        result = self._run(megatron={"model": {"bf16": False}})
        assert result.megatron.model.get("params_dtype") is None

    def test_sequence_parallel_true_when_tp_greater_than_1(self):
        result = self._run(megatron={"model": {"tensor_model_parallel_size": 2}})
        assert result.megatron.model.sequence_parallel is True

    def test_sequence_parallel_false_when_tp_equals_1(self):
        result = self._run(megatron={"model": {"tensor_model_parallel_size": 1}})
        assert result.megatron.model.sequence_parallel is False


class TestMegatronValidateConfigM2PO:
    def _run(self, **overrides):
        config = make_minimal_config(**overrides)
        return MegatronActorWorker.validate_config(config)

    def test_m2po_defaults_to_disabled(self):
        result = self._run()
        assert result.algorithm.m2po.enable is False
        assert result.algorithm.m2po.threshold == 0.0

    def test_m2po_rejects_rollout_log_probs_as_old_policy(self):
        with pytest.raises(ValueError, match="m2po.enable and trainer.use_rollout_log_probs"):
            self._run(
                fully_async={"enable": True},
                trainer={"use_rollout_log_probs": True},
                algorithm={"m2po": {"enable": True, "threshold": 0.05}},
            )


def test_compute_m2po_updates_loss_masks_before_is_correction(monkeypatch):
    import coda.backends.megatron.megatron_actor_worker as worker_mod

    monkeypatch.setattr(worker_mod.parallel_state, "get_data_parallel_group", lambda: None)
    monkeypatch.setattr(worker_mod, "reduce_dict", lambda metrics, group: metrics)

    worker = MegatronActorWorker.__new__(MegatronActorWorker)
    worker.rank = 0
    worker.config = OmegaConf.create(
        {
            "algorithm": {
                "loss_agg_mode": "token-mean",
                "is_correction": {
                    "enable": False,
                    "action": "clip",
                    "level": "token",
                    "lower_bound": 0.5,
                    "upper_bound": 2.0,
                },
                "m2po": {"enable": True, "threshold": 0.05},
            }
        }
    )
    # get_rollout_data seeds raw_loss_masks as a clone of loss_masks, so M2PO
    # always sees a pre-filtering snapshot; mirror that here.
    loss_masks = [torch.ones(3, dtype=torch.float32)]
    raw_loss_masks = [m.clone() for m in loss_masks]
    rollout_data = {
        "loss_masks": loss_masks,
        "raw_loss_masks": raw_loss_masks,
        "old_log_probs": [torch.tensor([3.0, 0.2, 0.1])],
        "rollout_log_probs": [torch.zeros(3)],
    }

    result = worker._compute_m2po(step=1, rollout_data=rollout_data)

    # raw_loss_masks is passed through untouched, so downstream IS correction and
    # loss aggregation still see the pre-M2PO mask.
    assert torch.equal(result["raw_loss_masks"][0], torch.ones(3))
    assert result["raw_loss_masks"][0] is raw_loss_masks[0]
    assert result["raw_loss_masks"][0] is not result["loss_masks"][0]
    # The largest squared log-ratio token (index 0) is masked out.
    assert torch.equal(result["loss_masks"][0], torch.tensor([0.0, 1.0, 1.0]))
    assert result["m2po_metrics"]["train/m2po_clip_ratio"] == pytest.approx(1 / 3)
    assert result["m2po_metrics"]["train/m2po_m2"] == pytest.approx(9.05 / 3, abs=1e-6)


def test_compute_m2po_returns_empty_without_old_log_probs():
    """Non-last pipeline stages carry no old_log_probs, so M2PO is a no-op there."""
    worker = MegatronActorWorker.__new__(MegatronActorWorker)
    worker.rank = 0

    result = worker._compute_m2po(step=1, rollout_data={"loss_masks": [torch.ones(3)]})

    assert result == {}


def test_compute_is_correction_uses_existing_loss_masks(monkeypatch):
    import coda.backends.megatron.megatron_actor_worker as worker_mod

    monkeypatch.setattr(worker_mod.parallel_state, "get_data_parallel_group", lambda: None)
    monkeypatch.setattr(worker_mod, "reduce_dict", lambda metrics, group: metrics)

    worker = MegatronActorWorker.__new__(MegatronActorWorker)
    worker.config = OmegaConf.create(
        {
            "algorithm": {
                "loss_agg_mode": "token-mean",
                "is_correction": {
                    "enable": True,
                    "action": "mask",
                    "level": "token",
                    "lower_bound": 0.5,
                    "upper_bound": 2.0,
                },
                "m2po": {"enable": False, "threshold": 0.0},
            }
        }
    )
    loss_masks = [torch.tensor([0.0, 1.0, 1.0])]
    rollout_data = {
        "loss_masks": loss_masks,
        "raw_loss_masks": [m.clone() for m in loss_masks],
        "old_log_probs": [torch.log(torch.tensor([0.25, 0.5, 1.0]))],
        "rollout_log_probs": [torch.zeros(3)],
    }

    result = worker._compute_is_correction(step=1, rollout_data=rollout_data)

    assert torch.equal(result["loss_masks"][0], torch.tensor([0.0, 1.0, 1.0]))
    assert result["is_metrics"]["train/is_clip_ratio"] == pytest.approx(0.0)


def test_compute_is_correction_preserves_m2po_raw_loss_masks(monkeypatch):
    """IS correction must not overwrite raw_loss_masks set by M2PO."""
    import coda.backends.megatron.megatron_actor_worker as worker_mod

    monkeypatch.setattr(worker_mod.parallel_state, "get_data_parallel_group", lambda: None)
    monkeypatch.setattr(worker_mod, "reduce_dict", lambda metrics, group: metrics)

    worker = MegatronActorWorker.__new__(MegatronActorWorker)
    worker.config = OmegaConf.create(
        {
            "algorithm": {
                "loss_agg_mode": "token-mean",
                "is_correction": {
                    "enable": True,
                    "action": "clip",
                    "level": "token",
                    "lower_bound": 0.5,
                    "upper_bound": 2.0,
                },
                "m2po": {"enable": True, "threshold": 0.05},
            }
        }
    )
    m2po_raw = [torch.ones(3, dtype=torch.float32)]
    rollout_data = {
        "raw_loss_masks": m2po_raw,
        "loss_masks": [torch.tensor([0.0, 1.0, 1.0])],
        "old_log_probs": [torch.tensor([0.0, 0.0, 0.0])],
        "rollout_log_probs": [torch.zeros(3)],
    }

    result = worker._compute_is_correction(step=1, rollout_data=rollout_data)

    assert "raw_loss_masks" not in result
    assert rollout_data["raw_loss_masks"] is m2po_raw


# ---------------------------------------------------------------------------
# Base / actor split
# ---------------------------------------------------------------------------

def test_base_worker_is_abstract():
    """The base only provides shared machinery; ``train`` must come from a subclass."""
    with pytest.raises(TypeError, match="abstract"):
        MegatronTrainWorker(1, 0)


def test_base_validate_config_is_idempotent():
    """Actor and critic both run the shared validation on the same driver config."""
    once = MegatronTrainWorker.validate_config(make_minimal_config())
    twice = MegatronTrainWorker.validate_config(OmegaConf.create(OmegaConf.to_container(once)))
    assert OmegaConf.to_container(once) == OmegaConf.to_container(twice)


def test_finalize_optim_sched_derives_from_its_own_lr():
    config = make_minimal_config()
    opt = OmegaConf.create({
        "lr": 2e-5, "weight_decay": 0.1,
        "optimizer_cpu_offload": False, "chunked_optimizer_state_offload": False,
    })
    sched = OmegaConf.create({})
    MegatronTrainWorker._finalize_optim_sched(opt, sched, config)
    assert opt.min_lr == pytest.approx(2e-6)
    assert sched.max_lr == pytest.approx(2e-5)
    assert sched.min_lr == pytest.approx(2e-6)
    assert sched.start_wd == sched.end_wd == pytest.approx(0.1)
    assert opt.bf16 is True and opt.use_distributed_optimizer is True


def test_base_validate_config_leaves_policy_fields_to_actor():
    cfg = make_minimal_config(trainer={"use_fp32_lm_head": True, "use_rollout_routing_replay": True})
    base = MegatronTrainWorker.validate_config(cfg.copy())
    assert "output_layer" not in base.megatron.keep_fp32_weights
    assert base.megatron.model.get("moe_enable_routing_replay") is None

    actor = MegatronActorWorker.validate_config(cfg.copy())
    assert actor.megatron.keep_fp32_weights["output_layer"] is True
    assert actor.megatron.model.moe_enable_routing_replay is True


def test_ckpt_dir_name_defaults_to_dist_ckpt():
    assert MegatronTrainWorker.CKPT_DIR_NAME == "dist_ckpt"
    assert MegatronActorWorker.CKPT_DIR_NAME == "dist_ckpt"


@pytest.mark.parametrize(
    "calls, expected",
    [
        ([False], False),
        ([True], True),
        ([False, True], True),   # release params kept on GPU for weight sync
        ([True, False], True),   # never claims offloaded params are back on GPU
    ],
)
def test_offload_params_flag_tracks_where_params_live(monkeypatch, calls, expected):
    import coda.backends.megatron.megatron_train_worker as base_mod

    monkeypatch.setattr(base_mod, "print_memory", lambda *a, **k: None)
    monkeypatch.setattr(base_mod, "clear_memory", lambda: None)
    worker = MegatronActorWorker.__new__(MegatronActorWorker)
    worker.offloaded = False
    worker.params_offloaded = False
    worker._offload_model = MagicMock()
    worker._move_optimizer = MagicMock()

    for move_params in calls:
        MegatronTrainWorker.offload(worker, move_params)

    assert worker.offloaded is True
    assert worker.params_offloaded is expected


# ---------------------------------------------------------------------------
# Actor advantages: own estimator vs. the critic's (PPO)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("critic_enable", [False, True])
def test_actor_advantages_source(monkeypatch, critic_enable):
    import coda.backends.megatron.megatron_actor_worker as actor_mod

    critic_advs = [torch.tensor([0.3, -0.3])]
    own_advs = [torch.tensor([1.0, 1.0])]
    rollout_data = {"critic_advantages": critic_advs} if critic_enable else {}
    calls = []

    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(actor_mod.async_calls, "maybe_finalize_async_calls", lambda: None)
    monkeypatch.setattr(actor_mod, "get_rollout_data", lambda ref: rollout_data)

    def fake_compute(config, data):
        calls.append("estimator")
        return own_advs, own_advs, {"train/adv_zero_ratio": 0.0}

    monkeypatch.setattr(actor_mod, "compute_advantages_and_returns", fake_compute)

    worker = MegatronActorWorker.__new__(MegatronActorWorker)
    worker.config = OmegaConf.create({
        "opd": {"enable": False},
        "critic": {"enable": critic_enable},
        "trainer": {"use_rollout_log_probs": True},
        "algorithm": {"ref_kl": {"enable": False}, "m2po": {"enable": False}},
    })
    worker.model = [MagicMock()]
    worker._compute_is_correction = lambda step, data: {}
    worker._train_actor = lambda step, data: calls.append(("train", data["advantages"]))

    worker.train(1, "refs")

    if critic_enable:
        assert calls == [("train", critic_advs)]
        assert "critic_advantages" not in rollout_data
        assert rollout_data["adv_metrics"] == {}
    else:
        assert calls == ["estimator", ("train", own_advs)]
        assert rollout_data["adv_metrics"] == {"train/adv_zero_ratio": 0.0}
