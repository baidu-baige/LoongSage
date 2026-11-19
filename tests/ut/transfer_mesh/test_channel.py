"""Unit tests for transfer_mesh.channel (no multi-process, no real GPU required).

Tests buffer allocation via send() using mocks to avoid actual distributed
communication. CUDA-dependent tests are skipped when no GPU is present.
"""

import pytest
from unittest.mock import MagicMock, patch

import torch

from coda.transfer_mesh.channel import (
    ChannelMeta,
    Role,
    TransferMeshChannel,
    _compute_rank,
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


# ── Rank assignment ──────────────────────────────────────────────────────────


class TestComputeRank:
    """The rank base may be derived or handed down; heterogeneous TP needs the latter."""

    @staticmethod
    def _patched_dist(rank: int, world_size: int):
        """Pretend torch.distributed is up with the given rank/world_size."""
        return patch.multiple(
            "coda.transfer_mesh.channel.dist",
            is_initialized=MagicMock(return_value=True),
            get_rank=MagicMock(return_value=rank),
            get_world_size=MagicMock(return_value=world_size),
        )

    def test_derived_base_is_used_when_rank_base_is_absent(self):
        """Pre-existing behaviour: engine_id * world_size, unchanged."""
        with self._patched_dist(rank=3, world_size=8):
            sender = _compute_rank(2, train_world_size=16, role=Role.SENDER)
            receiver = _compute_rank(2, train_world_size=16, role=Role.RECEIVER)
        assert sender == 2 * 8 + 3
        assert receiver == 2 * 8 + 3 + 16

    def test_explicit_base_overrides_the_derived_one(self):
        with self._patched_dist(rank=1, world_size=16):
            rank = _compute_rank(
                engine_id=3, train_world_size=16, role=Role.RECEIVER, rank_base=24
            )
        assert rank == 24 + 1 + 16

    def test_derived_base_overlaps_under_heterogeneous_tp(self):
        """Why the explicit base exists: the derived formula does not tile.

        prefill TP=8 on engines 0-1, decode TP=16 on engines 2-3. The derived base
        skips ranks 16..31 and runs past the end of the rollout range, so the
        rendezvous can never be completed.
        """
        derived = []
        for engine_id, world_size in [(0, 8), (1, 8), (2, 16), (3, 16)]:
            with self._patched_dist(rank=0, world_size=world_size):
                derived.append(_compute_rank(engine_id, 0, Role.SENDER))
        assert derived == [0, 8, 32, 48]  # expected 0, 8, 16, 32

    def test_explicit_bases_tile_the_rollout_range(self):
        """What the sender computes instead: contiguous, no gaps, no overlap."""
        plan = [(0, 8), (8, 8), (16, 16), (32, 16)]
        ranks = []
        for base, world_size in plan:
            for local_rank in range(world_size):
                with self._patched_dist(rank=local_rank, world_size=world_size):
                    ranks.append(
                        _compute_rank(
                            engine_id=0,
                            train_world_size=0,
                            role=Role.SENDER,
                            rank_base=base,
                        )
                    )
        assert sorted(ranks) == list(range(48))

    def test_uninitialized_distributed_raises(self):
        with patch.multiple(
            "coda.transfer_mesh.channel.dist",
            is_initialized=MagicMock(return_value=False),
        ):
            with pytest.raises(RuntimeError, match="torch.distributed is not initialized"):
                _compute_rank(0, 0, Role.SENDER, rank_base=0)


class TestChannelMetaDefaults:
    """``rank_base`` is optional so an unpatched receiver keeps the old behaviour."""

    def test_rank_base_defaults_to_none(self):
        meta = ChannelMeta(master_addr="127.0.0.1", world_size=4, train_world_size=2)
        assert meta.rank_base is None
