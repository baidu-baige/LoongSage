"""Repo-wide pytest configuration and shared skip markers.

Importable from any test module as::

    from tests.conftest import requires_cuda, requires_megatron

Nothing heavy is imported at module scope: ``torch`` is only imported when it is
actually installed, so the pure-Python part of the suite still collects in an
environment without it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types

import pytest


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _cuda_device_count() -> int:
    if not _has_module("torch"):
        return 0
    import torch

    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


_HAS_TORCH = _has_module("torch")
_HAS_MEGATRON = _has_module("megatron")
_CUDA_COUNT = _cuda_device_count()

requires_torch = pytest.mark.skipif(not _HAS_TORCH, reason="torch is not installed")

requires_megatron = pytest.mark.skipif(
    not _HAS_MEGATRON, reason="Megatron-Core is not installed"
)

requires_cuda = pytest.mark.skipif(_CUDA_COUNT < 1, reason="CUDA not available")

requires_multi_gpu = pytest.mark.skipif(
    _CUDA_COUNT < 2, reason="Requires >= 2 CUDA GPUs"
)

_SGLANG_SERVING_CHAT = "sglang.srt.entrypoints.openai.serving_chat"


def _normalize_assistant_tool_call_arguments(message, *, strict=True):
    """Same as sglang's: parse assistant tool_call ``arguments`` JSON strings into dicts in place."""
    if message.get("role") != "assistant" or not isinstance(message.get("tool_calls"), list):
        return
    for item in message["tool_calls"]:
        function = item.get("function") if isinstance(item, dict) else None
        if not isinstance(function, dict) or not isinstance(function.get("arguments"), str):
            continue
        try:
            parsed = json.loads(function["arguments"])
        except json.JSONDecodeError as exc:
            if strict:
                raise ValueError("Assistant tool call function.arguments must be valid JSON.") from exc
            continue
        if not isinstance(parsed, dict):
            if strict:
                raise ValueError("Assistant tool call function.arguments must be a JSON object.")
            continue
        function["arguments"] = parsed


@pytest.fixture
def stub_sglang_serving_chat(monkeypatch):
    """Let ``TokenizerManager.tokenize`` run where sglang is not installed.

    ``tokenize`` imports only ``normalize_assistant_tool_call_arguments`` from sglang; without
    sglang a copy of it is installed in ``sys.modules``. With sglang installed the real one is used.
    """
    try:
        importlib.import_module(_SGLANG_SERVING_CHAT)
    except ImportError:
        stub = types.ModuleType(_SGLANG_SERVING_CHAT)
        stub.normalize_assistant_tool_call_arguments = _normalize_assistant_tool_call_arguments
        monkeypatch.setitem(sys.modules, _SGLANG_SERVING_CHAT, stub)
