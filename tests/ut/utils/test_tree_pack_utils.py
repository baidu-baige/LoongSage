"""Unit tests for coda/utils/tree_pack_utils.py."""

from omegaconf import OmegaConf

from coda.utils.tree_pack_utils import tree_pack_unsupported_reason


def _cfg(**model):
    m = {"context_parallel_size": 1, "attention_backend": "flash"} | model
    return OmegaConf.create({"megatron": {"model": m}, "opd": {"enable": False},
                             "trainer": {"use_dynamic_batch_size": True}})


def test_supported_config():
    assert tree_pack_unsupported_reason(_cfg()) is None


def test_unsupported_configs():
    assert "context_parallel" in tree_pack_unsupported_reason(_cfg(context_parallel_size=2))
    assert "attention_backend" in tree_pack_unsupported_reason(_cfg(attention_backend="auto"))
    assert "fusion" in tree_pack_unsupported_reason(_cfg(gdn_pre_gated_delta_rule_fusion=True))
    cfg = _cfg()
    cfg.opd.enable = True
    assert "OPD" in tree_pack_unsupported_reason(cfg)
    cfg = _cfg()
    cfg.trainer.use_dynamic_batch_size = False
    assert "dynamic" in tree_pack_unsupported_reason(cfg)
