"""Unit tests for transfer_mesh.channel (no multi-process, no real GPU required).

Tests buffer allocation via send() using mocks to avoid actual distributed
communication. CUDA-dependent tests are skipped when no GPU is present.
"""

import pytest
from unittest.mock import MagicMock

import torch

from coda.transfer_mesh.channel import (
    Role,
    TransferMeshChannel,
)


# ── Buffer allocation via send() ─────────────────────────────────────────────


class TestAllocateBuffer:
    """The staging buffer is dtype-agnostic: uint8, sized in bytes."""

    def _make_sender(self, buffer_size_bytes):
        return TransferMeshChannel(
            master_addr="127.0.0.1", addr="127.0.0.1",
            gpu_id=0, world_size=2, rank=0,
            role=Role.SENDER, src_rank=0,
            buffer_size_bytes=buffer_size_bytes,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_buffer_is_byte_sized_regardless_of_dtype(self, dtype):
        ch = self._make_sender(buffer_size_bytes=1024)
        ch._flush_bucket = MagicMock()

        ch.send(("w", torch.randn(1, dtype=dtype, device="cuda:0")))

        assert ch._buffer.dtype == torch.uint8
        assert ch._buffer.numel() == 1024

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA")
    def test_buffer_grows_to_fit_an_oversized_tensor(self):
        ch = self._make_sender(buffer_size_bytes=64)
        ch._flush_bucket = MagicMock()

        # 100 float32 elements = 400 bytes > the 64-byte nominal buffer.
        ch.send(("big", torch.randn(100, dtype=torch.float32, device="cuda:0")))

        assert ch._buffer.numel() == 400
