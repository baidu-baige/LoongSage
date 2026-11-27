"""Unit tests for coda/utils/packing.py."""

import pytest
from omegaconf import OmegaConf

import coda.utils.packing as packing
from coda.utils.packing import MicroBatchPacker, build_packer, get_packer_class


def _config(packer="rows"):
    return OmegaConf.create({"trainer": {"packer": packer, "packer_options": {}}})


@pytest.fixture
def plugin(request):
    """Register a packer the way a ``coda/custom/`` module does (decorator on import)."""
    name = f"fake_{request.node.name}"

    @packing.register_packer(name)
    class Plugin(MicroBatchPacker):
        reason = None
        setup_reason = None

        @classmethod
        def unsupported_reason(cls, config):
            return cls.reason

        def setup(self, model):
            self.model = model
            return self.setup_reason

    yield name
    packing.PACKER_REGISTRY._registry.pop(name, None)


def test_default_is_rows():
    packer = build_packer(_config())
    assert type(packer) is MicroBatchPacker
    assert packer.dp_loads is None
    assert packer.plan({}, [], 1) is None
    assert packer.pack({}, None, 0, training=True, routed_rows=list) is None


def test_unknown_packer_lists_choices():
    with pytest.raises(ValueError, match="'nope'.*rows.*Skipped custom module"):
        get_packer_class("nope")


def test_custom_packer_is_found(plugin):
    assert get_packer_class(plugin).__name__ == "Plugin"


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
