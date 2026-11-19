"""Unit tests for ``SglangEngine.validate_config``'s startup-time checks.

SGLang defers P/D consistency checks to the first request that needs pairing, and
reports them by crashing decode's scheduler loop, so "both sides started" proves
nothing. These checks cover the complementary set: what SGLang cannot see and the
framework therefore has to reject up front.

sglang itself is stubbed — only ``validate_config`` is exercised, no server.
"""

import sys
import types
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

_ORIGINAL_SYS_MODULES: dict[str, object] = {}

SglangEngine = None


def _stub(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def setup_module(module):
    """Install ray/sglang stubs and import the engine class."""
    global SglangEngine
    for name in [
        "ray",
        "requests",
        "sglang",
        "sglang.srt",
        "sglang.srt.server_args",
        "sglang.srt.utils",
        "coda.backends.sglang.engine",
    ]:
        _ORIGINAL_SYS_MODULES[name] = sys.modules.get(name)
    ray_stub = _stub("ray")
    ray_stub.remote = lambda *args, **kwargs: (lambda cls: cls)
    ray_stub.get = lambda refs: refs
    _stub("requests", post=MagicMock(), get=MagicMock())
    _stub("sglang")
    _stub("sglang.srt")
    _stub("sglang.srt.server_args", ServerArgs=MagicMock())
    _stub("sglang.srt.utils", kill_process_tree=MagicMock())

    from coda.backends.sglang.engine import SglangEngine as _SglangEngine

    SglangEngine = _SglangEngine


def teardown_module(module):
    """Restore the real modules so later test files are unaffected."""
    for name, original in _ORIGINAL_SYS_MODULES.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


def make_config(
    *,
    replicas: dict,
    env_vars: dict | None = None,
    sglang_args: dict | None = None,
    num_gpus_per_node: int = 8,
):
    """A config carrying just the keys validate_config reads."""
    return OmegaConf.create(
        {
            "rollout": {
                "num_gpus_per_node": num_gpus_per_node,
                "sglang_args": sglang_args or {},
                "sglang_replicas": replicas,
                "env_vars": env_vars or {},
            }
        }
    )


def regular_replicas(**overrides) -> dict:
    replicas = {
        "regular": {"num_nodes": 1, "num_gpus_per_replica": 8},
        "prefill": {"num_nodes": 0, "num_gpus_per_replica": 8},
        "decode": {"num_nodes": 0, "num_gpus_per_replica": 8},
    }
    for key, value in overrides.items():
        replicas[key].update(value)
    return replicas


def pd_replicas(**overrides) -> dict:
    replicas = {
        "regular": {"num_nodes": 0, "num_gpus_per_replica": 8},
        "prefill": {"num_nodes": 2, "num_gpus_per_replica": 8},
        "decode": {"num_nodes": 4, "num_gpus_per_replica": 2},
    }
    for key, value in overrides.items():
        replicas[key].update(value)
    return replicas


class TestParallelismSelfConsistency:
    """The rank_base stride must match the engine's own process group."""

    def test_regular_and_pd_defaults_are_accepted(self):
        assert SglangEngine.validate_config(make_config(replicas=regular_replicas()))
        assert SglangEngine.validate_config(make_config(replicas=pd_replicas()))

    def test_explicit_tp_times_pp_must_equal_num_gpus_per_replica(self):
        config = make_config(replicas=pd_replicas(decode={"sglang_args": {"tp_size": 4}}))
        with pytest.raises(AssertionError, match="must equal num_gpus_per_replica"):
            SglangEngine.validate_config(config)

    def test_consistent_tp_pp_override_is_accepted(self):
        config = make_config(
            replicas=pd_replicas(prefill={"sglang_args": {"tp_size": 4, "pp_size": 2}})
        )
        SglangEngine.validate_config(config)

    def test_disabled_pools_are_not_checked(self):
        """An unused group may carry anything; only enabled ones are validated."""
        config = make_config(
            replicas=regular_replicas(decode={"sglang_args": {"tp_size": 4}})
        )
        SglangEngine.validate_config(config)

    def test_global_sglang_args_are_merged_into_every_replica(self):
        config = make_config(replicas=pd_replicas(), sglang_args={"page_size": 64})
        SglangEngine.validate_config(config)
        assert config.rollout.sglang_replicas.prefill.sglang_args.page_size == 64
        assert config.rollout.sglang_replicas.decode.sglang_args.page_size == 64


class TestGroupEnablement:
    """A group counts as enabled by its engine count, not by ``num_nodes`` alone."""

    def test_num_replicas_enables_a_sub_node_group(self):
        """prefill/decode sharing one node: num_nodes=0 but num_replicas=1 each."""
        replicas = {
            "regular": {"num_nodes": 0, "num_gpus_per_replica": 8},
            "prefill": {"num_nodes": 0, "num_replicas": 1, "num_gpus_per_replica": 4},
            "decode": {"num_nodes": 0, "num_replicas": 1, "num_gpus_per_replica": 4},
        }
        assert SglangEngine.validate_config(make_config(replicas=replicas))

    def test_num_replicas_group_is_validated(self):
        """Enabled via num_replicas, so its tp/pp mismatch must still be caught."""
        replicas = {
            "regular": {"num_nodes": 0, "num_gpus_per_replica": 8},
            "prefill": {"num_nodes": 0, "num_replicas": 1, "num_gpus_per_replica": 4},
            "decode": {
                "num_nodes": 0, "num_replicas": 1, "num_gpus_per_replica": 4,
                "sglang_args": {"tp_size": 8},
            },
        }
        with pytest.raises(AssertionError, match="must equal num_gpus_per_replica"):
            SglangEngine.validate_config(make_config(replicas=replicas))

    def test_num_replicas_regular_excludes_pd(self):
        replicas = {
            "regular": {"num_nodes": 0, "num_replicas": 1, "num_gpus_per_replica": 4},
            "prefill": {"num_nodes": 0, "num_replicas": 1, "num_gpus_per_replica": 4},
            "decode": {"num_nodes": 0, "num_gpus_per_replica": 4},
        }
        with pytest.raises(AssertionError, match="mutually exclusive"):
            SglangEngine.validate_config(make_config(replicas=replicas))


class TestRustServerExclusion:

    """The PD hook demands bootstrap_port == port, which the framework cannot give."""

    def test_pd_with_rust_server_is_rejected(self):
        config = make_config(replicas=pd_replicas(), env_vars={"SGLANG_RUST_SERVER": "1"})
        with pytest.raises(AssertionError, match="SGLANG_RUST_SERVER"):
            SglangEngine.validate_config(config)

    def test_regular_with_rust_server_is_untouched(self):
        config = make_config(
            replicas=regular_replicas(), env_vars={"SGLANG_RUST_SERVER": "true"}
        )
        SglangEngine.validate_config(config)
