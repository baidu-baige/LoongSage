"""Unit tests for coda/utils/packing.py."""

import sys
from importlib.metadata import EntryPoint, EntryPoints

import pytest
from omegaconf import OmegaConf

import coda.utils.packing as packing
from coda.utils.packing import MicroBatchPacker, build_packer, get_packer_class

PLUGIN = '''
from coda.utils.packing import MicroBatchPacker, register_packer

@register_packer("{name}")
class Plugin(MicroBatchPacker):
    reason = None
    setup_reason = None

    @classmethod
    def unsupported_reason(cls, config):
        return cls.reason

    def setup(self, model):
        self.model = model
        return self.setup_reason
'''


def _config(packer="rows"):
    return OmegaConf.create({"trainer": {"packer": packer, "packer_options": {}}})


@pytest.fixture
def plugin(tmp_path, monkeypatch, request):
    """Install a fake plugin module exposed through the packer entry point group."""
    name = f"fake_{request.node.name}"
    mod = f"fake_packer_{request.node.name}"
    (tmp_path / f"{mod}.py").write_text(PLUGIN.format(name=name))
    monkeypatch.syspath_prepend(str(tmp_path))
    eps = EntryPoints([EntryPoint(name, f"{mod}:Plugin", packing.ENTRY_POINT_GROUP)])
    monkeypatch.setattr(packing, "entry_points", lambda group: eps if group == packing.ENTRY_POINT_GROUP else EntryPoints([]))
    yield name
    packing.PACKER_REGISTRY._registry.pop(name, None)
    sys.modules.pop(mod, None)


def test_default_is_rows():
    packer = build_packer(_config())
    assert type(packer) is MicroBatchPacker
    assert packer.dp_loads is None
    assert packer.plan({}, [], 1) is None
    assert packer.pack({}, None, 0, training=True, routed_rows=list) is None


def test_unknown_packer_lists_choices():
    with pytest.raises(ValueError, match="'nope'.*rows"):
        get_packer_class("nope")


def test_plugin_is_loaded_from_its_entry_point(plugin):
    assert plugin not in packing.PACKER_REGISTRY
    cls = get_packer_class(plugin)
    assert cls.__name__ == "Plugin"
    assert plugin in packing.PACKER_REGISTRY


def test_build_runs_setup_on_the_model(plugin):
    model = object()
    packer = build_packer(_config(plugin), model)
    assert type(packer).__name__ == "Plugin"
    assert packer.model is model


def test_unsupported_config_falls_back_to_rows(plugin):
    get_packer_class(plugin).reason = "CP > 1"
    assert type(build_packer(_config(plugin), object())) is MicroBatchPacker


def test_rejected_model_falls_back_to_rows(plugin):
    get_packer_class(plugin).setup_reason = "no adapter"
    assert type(build_packer(_config(plugin))).__name__ == "Plugin"   # no model: setup not run
    assert type(build_packer(_config(plugin), object())) is MicroBatchPacker
