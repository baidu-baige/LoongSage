"""Unit tests for per-machine ``disaggregation_ib_device`` resolution.

RDMA device names differ between machines, so the list cannot be pinned in the
config of a heterogeneous cluster. ``_resolve_ib_devices`` runs inside the engine
actor and falls back through ``/etc/nccl.conf`` and ``$NCCL_IB_HCA`` so each
machine answers for its own NICs.

sglang itself is stubbed — no server is launched.
"""

import os
import sys
import types
from unittest.mock import MagicMock

import pytest

_ORIGINAL_SYS_MODULES: dict[str, object] = {}

engine = None


def _stub(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def setup_module(module):
    """Install ray/sglang stubs and import the engine module."""
    global engine
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

    import coda.backends.sglang.engine as _engine

    engine = _engine


def teardown_module(module):
    """Restore the real modules so later test files are unaffected."""
    for name, original in _ORIGINAL_SYS_MODULES.items():
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


@pytest.fixture
def no_env(monkeypatch):
    monkeypatch.delenv("NCCL_IB_HCA", raising=False)


class TestParseNcclIbHca:
    """NCCL's own syntax has to be stripped down to bare device names."""

    def test_strips_exact_match_prefix(self):
        # The real /etc/nccl.conf reads `NCCL_IB_HCA==mlx5_8,...`; the leading `=`
        # is NCCL's "match whole device name" marker, not part of a name.
        assert engine._parse_nccl_ib_hca("=mlx5_8,mlx5_6") == "mlx5_8,mlx5_6"

    def test_strips_port_suffix_and_dedups_preserving_order(self):
        assert engine._parse_nccl_ib_hca("mlx5_8:1,mlx5_6:1,mlx5_8:1") == "mlx5_8,mlx5_6"

    def test_tolerates_whitespace(self):
        assert engine._parse_nccl_ib_hca(" mlx5_1 , mlx5_2 ") == "mlx5_1,mlx5_2"

    @pytest.mark.parametrize("raw", ["^mlx5_0", "^mlx5_0,mlx5_1", " ^mlx5_0"])
    def test_exclusion_list_is_rejected(self, raw):
        # A blocklist cannot be inverted without enumerating local NICs; treating
        # it as a device list would hand mooncake names like "^mlx5_0".
        assert engine._parse_nccl_ib_hca(raw) is None

    @pytest.mark.parametrize("raw", ["", "   ", ",", "="])
    def test_empty_values_are_rejected(self, raw):
        assert engine._parse_nccl_ib_hca(raw) is None


class TestNcclConfFile:
    """Reading one entry out of an nccl.conf-style KEY=value file."""

    def test_reads_the_hca_line(self, tmp_path):
        conf = tmp_path / "nccl.conf"
        conf.write_text(
            "IB_GID_INDEX=3\n"
            "NVSHMEM_HCA_LIST=mlx5_8:1,mlx5_6:1\n"
            "NCCL_IB_HCA==mlx5_8,mlx5_6,mlx5_15\n"
            "NCCL_DEBUG=INFO\n"
        )
        assert engine._nccl_conf_value(str(conf), "NCCL_IB_HCA") == "=mlx5_8,mlx5_6,mlx5_15"

    def test_missing_file_returns_none(self, tmp_path):
        assert engine._nccl_conf_value(str(tmp_path / "absent.conf"), "NCCL_IB_HCA") is None

    def test_commented_out_entry_is_not_read(self, tmp_path):
        conf = tmp_path / "nccl.conf"
        conf.write_text("NCCL_DEBUG=INFO\n# NCCL_IB_HCA=mlx5_9\n")
        assert engine._nccl_conf_value(str(conf), "NCCL_IB_HCA") is None

    def test_valueless_line_does_not_crash(self, tmp_path):
        conf = tmp_path / "nccl.conf"
        conf.write_text("NCCL_IB_HCA\nNCCL_IB_HCA=mlx5_4\n")
        assert engine._nccl_conf_value(str(conf), "NCCL_IB_HCA") == "mlx5_4"


class TestResolvePrecedence:
    """config > /etc/nccl.conf > $NCCL_IB_HCA > None (mooncake auto-discovery)."""

    @pytest.fixture
    def conf_file(self, tmp_path, monkeypatch):
        def write(contents: str | None):
            path = tmp_path / "nccl.conf"
            if contents is not None:
                path.write_text(contents)
            monkeypatch.setattr(engine, "_NCCL_CONF_PATH", str(path))

        return write

    def test_explicit_config_wins_over_the_file(self, conf_file, no_env):
        conf_file("NCCL_IB_HCA==mlx5_8,mlx5_6\n")
        assert engine._resolve_ib_devices("mlx5_1,mlx5_2") == "mlx5_1,mlx5_2"

    def test_unset_config_falls_back_to_the_file(self, conf_file, no_env):
        conf_file("NCCL_IB_HCA==mlx5_8,mlx5_6\n")
        assert engine._resolve_ib_devices(None) == "mlx5_8,mlx5_6"

    def test_file_wins_over_the_environment(self, conf_file, monkeypatch):
        conf_file("NCCL_IB_HCA==mlx5_8\n")
        monkeypatch.setenv("NCCL_IB_HCA", "mlx5_99")
        assert engine._resolve_ib_devices(None) == "mlx5_8"

    def test_missing_file_falls_back_to_the_environment(self, conf_file, monkeypatch):
        conf_file(None)
        monkeypatch.setenv("NCCL_IB_HCA", "=mlx5_3:1,mlx5_5")
        assert engine._resolve_ib_devices(None) == "mlx5_3,mlx5_5"

    def test_unusable_file_entry_falls_back_to_the_environment(self, conf_file, monkeypatch):
        conf_file("NCCL_IB_HCA=^mlx5_0\n")
        monkeypatch.setenv("NCCL_IB_HCA", "mlx5_7")
        assert engine._resolve_ib_devices(None) == "mlx5_7"

    def test_no_source_hands_over_to_auto_discovery(self, conf_file, no_env):
        conf_file(None)
        assert engine._resolve_ib_devices(None) is None

    def test_unusable_environment_hands_over_to_auto_discovery(self, conf_file, monkeypatch):
        conf_file(None)
        monkeypatch.setenv("NCCL_IB_HCA", "^mlx5_0")
        assert engine._resolve_ib_devices(None) is None

    def test_empty_config_string_is_treated_as_unset(self, conf_file, no_env):
        conf_file("NCCL_IB_HCA=mlx5_8\n")
        assert engine._resolve_ib_devices("") == "mlx5_8"


class TestComputeServerArgsWiring:
    """Resolution happens in the engine actor, and only for the PD worker types."""

    @pytest.fixture
    def config(self):
        from omegaconf import OmegaConf

        return OmegaConf.create(
            {
                "colocate": False,
                "hf_model_path": "/models/stub",
                "rollout": {"num_gpus_per_node": 8},
                "trainer": {
                    "use_rollout_routing_replay": False,
                    "use_fp32_lm_head": False,
                    "routed_experts_dtype": "auto",
                },
            }
        )

    def _args(self, config, worker_type, engine_args):
        return engine._compute_server_args(
            config, 0, "127.0.0.1:5000", 20000, "127.0.0.1", 30000, worker_type,
            disaggregation_bootstrap_port=30002, base_gpu_id=0,
            engine_args=engine_args, num_gpus_per_replica=4,
        )

    @pytest.mark.parametrize("worker_type", ["prefill", "decode"])
    def test_pd_workers_resolve_from_the_local_file(self, worker_type, config, monkeypatch, tmp_path, no_env):
        path = tmp_path / "nccl.conf"
        path.write_text("NCCL_IB_HCA==mlx5_8,mlx5_6\n")
        monkeypatch.setattr(engine, "_NCCL_CONF_PATH", str(path))
        args = self._args(config, worker_type, {})
        assert args["disaggregation_ib_device"] == "mlx5_8,mlx5_6"

    def test_explicit_sglang_arg_is_passed_through(self, config, monkeypatch, tmp_path, no_env):
        path = tmp_path / "nccl.conf"
        path.write_text("NCCL_IB_HCA==mlx5_8,mlx5_6\n")
        monkeypatch.setattr(engine, "_NCCL_CONF_PATH", str(path))
        args = self._args(config, "prefill", {"disaggregation_ib_device": "mlx5_1"})
        assert args["disaggregation_ib_device"] == "mlx5_1"

    def test_colocate_worker_is_left_alone(self, config, monkeypatch, tmp_path, no_env):
        # No PD link, so no device list is invented and no warning is emitted.
        path = tmp_path / "nccl.conf"
        path.write_text("NCCL_IB_HCA==mlx5_8\n")
        monkeypatch.setattr(engine, "_NCCL_CONF_PATH", str(path))
        args = self._args(config, "regular", {})
        assert "disaggregation_ib_device" not in args


class TestExportGidIndex:
    """NCCL_IB_GID_INDEX is the one NCCL var mooncake reads, and it reads it from env."""

    @pytest.fixture
    def conf_file(self, tmp_path, monkeypatch):
        # Swap os.environ for a throwaway copy: _export_gid_index writes to it directly,
        # and monkeypatch only restores vars it set itself, so a real write would leak
        # NCCL_IB_GID_INDEX into every later test.
        sandbox = dict(os.environ)
        sandbox.pop("NCCL_IB_GID_INDEX", None)
        sandbox.pop("MC_GID_INDEX", None)
        monkeypatch.setattr(os, "environ", sandbox)

        def write(contents: str | None):
            path = tmp_path / "nccl.conf"
            if contents is not None:
                path.write_text(contents)
            monkeypatch.setattr(engine, "_NCCL_CONF_PATH", str(path))

        return write

    def test_exports_the_value_from_the_file(self, conf_file):
        conf_file("IB_GID_INDEX=3\nNCCL_IB_GID_INDEX=3\nNCCL_DEBUG=INFO\n")
        engine._export_gid_index()
        assert os.environ["NCCL_IB_GID_INDEX"] == "3"

    def test_index_zero_is_exported(self, conf_file):
        # 0 is a real GID table slot; only a parse failure or a negative value is junk.
        conf_file("NCCL_IB_GID_INDEX=0\n")
        engine._export_gid_index()
        assert os.environ["NCCL_IB_GID_INDEX"] == "0"

    @pytest.mark.parametrize("contents", ["NCCL_IB_GID_INDEX=three\n", "NCCL_IB_GID_INDEX=-1\n", "NCCL_IB_GID_INDEX=\n"])
    def test_unusable_values_are_not_exported(self, conf_file, contents):
        conf_file(contents)
        engine._export_gid_index()
        assert "NCCL_IB_GID_INDEX" not in os.environ

    def test_missing_file_exports_nothing(self, conf_file):
        conf_file(None)
        engine._export_gid_index()
        assert "NCCL_IB_GID_INDEX" not in os.environ

    def test_existing_env_value_is_not_overridden(self, conf_file):
        conf_file("NCCL_IB_GID_INDEX=3\n")
        os.environ["NCCL_IB_GID_INDEX"] = "1"
        engine._export_gid_index()
        assert os.environ["NCCL_IB_GID_INDEX"] == "1"

    def test_mc_gid_index_suppresses_the_export(self, conf_file):
        # MC_GID_INDEX wins inside mooncake, so injecting a conflicting value would
        # only make the environment lie about which index is in use.
        conf_file("NCCL_IB_GID_INDEX=3\n")
        os.environ["MC_GID_INDEX"] = "1"
        engine._export_gid_index()
        assert "NCCL_IB_GID_INDEX" not in os.environ
