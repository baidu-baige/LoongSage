"""Unit tests for the PPO critic worker (framework parts; no GPU / Ray)."""
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf

pytest.importorskip("megatron", reason="Megatron-Core is not installed")

import coda.backends.megatron.megatron_critic_worker as critic_mod
from coda.backends.megatron import MegatronCriticWorker, MegatronTrainWorker
from tests.ut.backends.megatron.test_megatron_train_worker import make_minimal_config


def make_critic_config(**overrides):
    cfg = make_minimal_config(
        megatron={
            "optimizer": {"lr": 1e-6, "weight_decay": 0.01, "adam_beta2": 0.98},
            "scheduler": {"lr_decay_style": "cosine"},
        },
        critic={
            "enable": True,
            "critic_warmup": 2,
            "optimizer": {
                "lr": 5e-6,
                "weight_decay": 0.0,
                "optimizer_cpu_offload": False,
                "chunked_optimizer_state_offload": False,
            },
            "scheduler": {"lr_warmup_steps": 3},
        },
    )
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return cfg


def make_worker(config=None):
    worker = MegatronCriticWorker.__new__(MegatronCriticWorker)
    worker.rank = 0
    worker.config = config if config is not None else make_critic_config()
    return worker


class TestValidateConfig:
    def test_critic_optimizer_is_independent_of_the_actor(self):
        cfg = MegatronCriticWorker.validate_config(make_critic_config())

        opt, sched = cfg.critic.optimizer, cfg.critic.scheduler
        assert opt.lr == pytest.approx(5e-6)
        assert "adam_beta2" not in opt                        # not taken from the actor
        assert opt.min_lr == pytest.approx(5e-7)              # from the critic's lr
        assert sched.max_lr == pytest.approx(5e-6)
        assert sched.init_lr == pytest.approx(5e-7)
        assert sched.start_wd == sched.end_wd == 0.0          # from the critic's weight_decay
        assert "lr_decay_style" not in sched                  # not taken from the actor
        assert sched.lr_warmup_steps == 3
        assert opt.bf16 is True and opt.use_distributed_optimizer is True

        # The actor's config is left as the shared validation made it.
        assert cfg.megatron.optimizer.lr == pytest.approx(1e-6)
        assert cfg.megatron.scheduler.max_lr == pytest.approx(1e-6)

    def test_validate_config_is_idempotent(self):
        once = MegatronCriticWorker.validate_config(make_critic_config())
        twice = MegatronCriticWorker.validate_config(OmegaConf.create(OmegaConf.to_container(once)))
        assert OmegaConf.to_container(once) == OmegaConf.to_container(twice)

    def test_critic_checkpoint_dir_name(self):
        assert MegatronCriticWorker.CKPT_DIR_NAME == "critic_dist_ckpt"
        assert MegatronCriticWorker.CKPT_DIR_NAME != MegatronTrainWorker.CKPT_DIR_NAME


def test_init_rewrites_only_the_process_local_config(monkeypatch):
    cfg = MegatronCriticWorker.validate_config(
        make_critic_config(trainer={"use_rollout_routing_replay": True})
    )
    cfg.megatron.model.moe_enable_routing_replay = True
    cfg.megatron.model.mtp_num_layers = 1
    cfg.megatron.keep_fp32_weights = {"output_layer": True, "router": False}  # use_fp32_lm_head
    seen = {}

    def fake_base_init(self, config, master_addr, master_port):
        seen.update(config=OmegaConf.to_container(config), addr=master_addr, port=master_port)

    monkeypatch.setattr(MegatronTrainWorker, "init", fake_base_init)
    worker = make_worker()
    MegatronCriticWorker.init(worker, cfg, "host", 1234)

    assert seen["addr"] == "host" and seen["port"] == 1234
    assert seen["config"]["megatron"]["optimizer"] == seen["config"]["critic"]["optimizer"]
    assert seen["config"]["megatron"]["scheduler"] == seen["config"]["critic"]["scheduler"]
    assert seen["config"]["trainer"]["use_rollout_routing_replay"] is False
    assert seen["config"]["megatron"]["model"]["moe_enable_routing_replay"] is False
    assert seen["config"]["megatron"]["model"]["mtp_num_layers"] is None
    # The value head follows the model precision; other fp32 layers are kept.
    assert seen["config"]["megatron"]["keep_fp32_weights"] == {"router": False}


@pytest.mark.parametrize("step, update_tracker", [(1, True), (2, True), (3, False)])
def test_save_model_writes_tracker_only_during_warmup(step, update_tracker):
    worker = make_worker()  # critic_warmup = 2
    worker._save_dist_checkpoint = MagicMock()

    worker.save_model(step)

    worker._save_dist_checkpoint.assert_called_once_with(step, async_save=False, update_tracker=update_tracker)


def test_save_model_flush_and_disabled_are_noops():
    worker = make_worker()
    worker._save_dist_checkpoint = MagicMock()
    worker.save_model(None)
    worker.save_model(4, checkpoint=False)
    worker._save_dist_checkpoint.assert_not_called()


def _train_harness(monkeypatch, last_stage):
    order = []
    rollout_data = {"tokens": [torch.zeros(3)]}
    monkeypatch.setattr(critic_mod, "get_rollout_data", lambda ref: rollout_data)
    ps = MagicMock()
    ps.is_pipeline_last_stage.return_value = last_stage
    monkeypatch.setattr(critic_mod, "parallel_state", ps)

    advs, rets = [torch.tensor([0.5])], [torch.tensor([1.0])]

    def fake_compute(config, data):
        order.append("advantages")
        assert data["old_values"] == ["v"]
        return advs, rets, {"train/adv_zero_ratio": 0.0}

    monkeypatch.setattr(critic_mod, "compute_advantages_and_returns", fake_compute)

    worker = make_worker()
    model_chunk = MagicMock()
    model_chunk.eval.side_effect = lambda: order.append("eval")
    model_chunk.train.side_effect = lambda: order.append("train_mode")
    worker.model = [model_chunk]
    worker._compute_old_values = lambda step, data: order.append("old_values") or (["v"] if last_stage else None)
    worker._train_critic = lambda step, data: order.append(("update", dict(data.get("adv_metrics", {}))))
    worker._export_advantages = lambda a: order.append("export") or {"exported": a}
    return worker, order, rollout_data, advs, rets


def test_train_exports_advantages_from_pre_update_values(monkeypatch):
    worker, order, rollout_data, advs, rets = _train_harness(monkeypatch, last_stage=True)

    out = worker.train(5, "refs")

    assert order == [
        "eval", "old_values", "advantages", "train_mode",
        ("update", {"train/adv_zero_ratio": 0.0}), "export",
    ]
    assert out == {"exported": advs}
    assert rollout_data["returns"] is rets


def test_train_skips_advantages_off_the_last_pp_stage(monkeypatch):
    worker, order, rollout_data, _, _ = _train_harness(monkeypatch, last_stage=False)

    out = worker.train(5, "refs")

    assert "advantages" not in order
    assert ("update", {}) in order
    assert "returns" not in rollout_data
    assert out == {"exported": None}


class _FakeChunk:
    def __init__(self, post_process):
        self.post_process = post_process
        self.output_layer = "lm_head"


class _FakeProvider:
    hidden_size = 16
    sequence_parallel = False
    share_embeddings_and_output_weights = True

    def __init__(self):
        self.hooks = []

    def register_pre_wrap_hook(self, hook, prepend=False):
        self.hooks.append(hook)


def test_configure_provider_unties_and_appends_value_head_hook():
    provider = _FakeProvider()
    provider.hooks.append(lambda model: model)  # stands for the HF weight-loading hook
    worker = make_worker(make_critic_config(seed=11, critic={"value_head_init_std": 0.05}))

    worker._configure_provider(provider)

    assert provider.share_embeddings_and_output_weights is False
    assert len(provider.hooks) == 2                     # appended after HF loading
    chunks = [_FakeChunk(False), _FakeChunk(True)]
    out = provider.hooks[-1](chunks)
    assert out is chunks                                # hooks are chained
    assert chunks[0].output_layer == "lm_head"          # only the post_process chunk
    head = chunks[1].output_layer
    assert isinstance(head, critic_mod.LinearValueHead)
    assert head.in_features == 16 and head.out_features == 1
    assert head.bias.item() == 0.0


def test_compute_forward_only_values_collects_per_row_values(monkeypatch):
    monkeypatch.setattr(critic_mod, "response_values", lambda config, batch, out: ["row0", "row1"])
    loss, collected = critic_mod.compute_forward_only_values(
        None, {}, None, make_critic_config(), None, torch.zeros(1)
    )
    assert loss.item() == 0.0
    assert collected == {"values": ["row0", "row1"]}


@pytest.mark.parametrize(
    "last_stage, tp_rank, cp_rank, exported",
    [(True, 0, 0, True), (False, 0, 0, False), (True, 1, 0, False), (True, 0, 1, False)],
)
def test_export_advantages_only_on_output_rank(monkeypatch, last_stage, tp_rank, cp_rank, exported):
    ps = MagicMock()
    ps.is_pipeline_last_stage.return_value = last_stage
    ps.get_tensor_model_parallel_rank.return_value = tp_rank
    ps.get_context_parallel_rank.return_value = cp_rank
    ps.get_data_parallel_rank.return_value = 3
    monkeypatch.setattr(critic_mod, "parallel_state", ps)
    monkeypatch.setattr(critic_mod.ray, "put", lambda payload: ("ref", payload))

    out = make_worker()._export_advantages([torch.tensor([1.0, 2.0])])

    if exported:
        (dp_rank, (tag, payload)), = out.items()
        assert dp_rank == 3 and tag == "ref"
        assert torch.equal(payload["critic_advantages"][0], torch.tensor([1.0, 2.0]))
    else:
        assert out == {}
