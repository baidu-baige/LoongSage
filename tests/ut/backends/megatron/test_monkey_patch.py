"""Unit tests for the FA4 causal-window patch in coda/backends/megatron/monkey_patch.py.

A fake ``flash_attn.cute.interface`` carries a copy of the window resolver from
flash-attn-4 4.0.0b30+, so no GPU or FA4 install is required. Megatron-Core must
still be installed, because ``monkey_patch`` imports it at module scope.
"""

import sys
import types

import pytest

pytest.importorskip("megatron", reason="Megatron-Core is not installed")

from coda.backends.megatron import monkey_patch  # noqa: E402


def _resolve_b30(causal, window_size_left, window_size_right, mask_mod=None):
    """``_resolve_causal_local_window`` as shipped in flash-attn-4 4.0.0b30 - b39."""
    if mask_mod is not None:
        return False, False, window_size_left, window_size_right
    if (window_size_left is not None and window_size_right is not None
            and window_size_left < 0 and window_size_right < 0):
        window_size_left = None
        window_size_right = None
    if causal:
        window_size_right = 0
    if window_size_left is not None or window_size_right is not None:
        if window_size_left is None and window_size_right == 0:
            causal, local = True, False
            window_size_right = None
        else:
            causal, local = False, True
    else:
        local = False
    return causal, local, window_size_left, window_size_right


@pytest.fixture
def fake_fa4(monkeypatch):
    module = types.ModuleType("flash_attn.cute.interface")
    module._resolve_causal_local_window = _resolve_b30
    monkeypatch.setitem(sys.modules, "flash_attn.cute.interface", module)
    return module


def test_unpatched_resolver_turns_te_causal_into_local(fake_fa4):
    assert fake_fa4._resolve_causal_local_window(True, -1, 0) == (False, True, -1, 0)


@pytest.mark.parametrize(
    "args, expected",
    [
        ((True, -1, 0), (True, False, None, None)),  # TE causal -> plain causal
        ((True, 128, 0), (False, True, 128, 0)),  # sliding window stays local
        ((False, 7, -8), (False, True, 7, -8)),  # non-causal window untouched
        ((False, -1, -1), (False, False, None, None)),  # unbounded stays unbounded
    ],
)
def test_patched_resolver(fake_fa4, args, expected):
    monkey_patch._patch_fa4_causal_unbounded_window()
    assert fake_fa4._resolve_causal_local_window(*args) == expected


def test_patch_is_applied_once(fake_fa4):
    monkey_patch._patch_fa4_causal_unbounded_window()
    patched = fake_fa4._resolve_causal_local_window
    monkey_patch._patch_fa4_causal_unbounded_window()
    assert fake_fa4._resolve_causal_local_window is patched


def test_missing_fa4_is_ignored(monkeypatch):
    monkeypatch.setitem(sys.modules, "flash_attn.cute.interface", None)
    monkey_patch._patch_fa4_causal_unbounded_window()
