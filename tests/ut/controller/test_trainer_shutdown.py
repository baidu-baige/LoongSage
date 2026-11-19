"""SIGTERM/SIGINT handling in the trainer entry point."""

import signal
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

pytest.importorskip("megatron", reason="Megatron-Core is not installed")

from coda.controller import trainer as trainer_mod  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_signal_handlers():
    saved = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    yield
    for s, h in saved.items():
        signal.signal(s, h)


class _FakeTrainer:
    def __init__(self):
        self.cleanups = 0

    def delete_all_sandboxes(self):
        self.cleanups += 1


def test_first_signal_cleans_up_then_exits():
    fake = _FakeTrainer()
    trainer_mod.install_shutdown_handler(fake)
    handler = signal.getsignal(signal.SIGTERM)

    with pytest.raises(SystemExit) as exc:
        handler(signal.SIGTERM, None)

    assert exc.value.code == 128 + signal.SIGTERM
    assert fake.cleanups == 1
    assert signal.getsignal(signal.SIGINT) is handler


def test_second_signal_exits_immediately(monkeypatch):
    fake = _FakeTrainer()
    trainer_mod.install_shutdown_handler(fake)
    handler = signal.getsignal(signal.SIGTERM)
    with pytest.raises(SystemExit):
        handler(signal.SIGTERM, None)

    class _Exited(Exception):
        pass

    def fake_exit(code):
        raise _Exited(code)

    monkeypatch.setattr(trainer_mod.os, "_exit", fake_exit)
    with pytest.raises(_Exited) as exc:
        handler(signal.SIGINT, None)

    assert exc.value.args == (128 + signal.SIGINT,)
    assert fake.cleanups == 1  # no second cleanup


def _trainer_with(agentflow):
    t = trainer_mod.Trainer.__new__(trainer_mod.Trainer)
    t.config = OmegaConf.create({"agentflow": {"sandbox_cleanup_timeout_seconds": 42}})
    t.rollout_sampler = SimpleNamespace(agentflow=agentflow)
    return t


def test_trainer_delete_all_sandboxes_passes_configured_timeout():
    calls = []
    t = _trainer_with(SimpleNamespace(delete_all_sandboxes=calls.append))

    t.delete_all_sandboxes()

    assert calls == [42.0]


def test_trainer_delete_all_sandboxes_skips_without_agentflow():
    _trainer_with(None).delete_all_sandboxes()
    bare = trainer_mod.Trainer.__new__(trainer_mod.Trainer)  # failed before rollout_sampler
    bare.delete_all_sandboxes()


def test_trainer_delete_all_sandboxes_swallows_errors():
    def boom(timeout):
        raise RuntimeError("engine down")

    _trainer_with(SimpleNamespace(delete_all_sandboxes=boom)).delete_all_sandboxes()
