"""Repo-wide pytest configuration and shared skip markers.

Importable from any test module as::

    from tests.conftest import requires_cuda, requires_megatron

Nothing heavy is imported at module scope: ``torch`` is only imported when it is
actually installed, so the pure-Python part of the suite still collects in an
environment without it.
"""

from __future__ import annotations

import importlib.util

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
