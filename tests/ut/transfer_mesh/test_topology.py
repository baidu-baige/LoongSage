"""Unit tests for transfer_mesh.topology."""

import pytest

from coda.transfer_mesh.topology import (
    RankInfo,
    Role,
    partition_receivers,
)


# ── partition_receivers ──────────────────────────────────────────────────────


class TestPartitionReceivers:

    @staticmethod
    def _make_rank_infos(infos: list[tuple[int, Role, int, str]]) -> list[RankInfo]:
        """Build RankInfo list from (rank, role, gpu_id, ip) tuples."""
        return [
            RankInfo(rank=r, gpu_id=g, ip=ip, role=role)
            for r, role, g, ip in infos
        ]

    def test_all_colocated(self, sample_rank_info_colocated):
        """Each receiver shares GPU with a sender -> all IPC."""
        rank_infos, _ = sample_rank_info_colocated

        ipc, nccl = partition_receivers(rank_infos)

        assert nccl == []
        for s_rank in range(8):
            assert len(ipc[s_rank]) == 1
            assert ipc[s_rank][0] == s_rank + 8

    def test_all_separated(self):
        """No sender-receiver pairs share a GPU -> all NCCL."""
        rank_infos = self._make_rank_infos([
            (0, Role.SENDER,   0, "10.0.0.1"),
            (1, Role.SENDER,   1, "10.0.0.1"),
            (2, Role.RECEIVER, 2, "10.0.0.1"),
            (3, Role.RECEIVER, 3, "10.0.0.1"),
        ])
        ipc, nccl = partition_receivers(rank_infos)

        assert set(nccl) == {2, 3}
        assert ipc[0] == []
        assert ipc[1] == []

    def test_mixed(self):
        """Rank 2 on same GPU as sender 0 (IPC), rank 3 on separate GPU (NCCL)."""
        rank_infos = self._make_rank_infos([
            (0, Role.SENDER,   0, "10.0.0.1"),
            (1, Role.SENDER,   1, "10.0.0.1"),
            (2, Role.RECEIVER, 0, "10.0.0.1"),
            (3, Role.RECEIVER, 3, "10.0.0.1"),
        ])
        ipc, nccl = partition_receivers(rank_infos)

        assert ipc[0] == [2]
        assert ipc[1] == []
        assert nccl == [3]

    def test_empty_receivers(self):
        rank_infos = self._make_rank_infos([(0, Role.SENDER, 0, "10.0.0.1")])
        ipc, nccl = partition_receivers(rank_infos)
        assert nccl == []
        assert ipc[0] == []

    def test_empty_senders(self):
        """No senders -> all receivers go to NCCL."""
        rank_infos = self._make_rank_infos([
            (0, Role.RECEIVER, 0, "10.0.0.1"),
            (1, Role.RECEIVER, 0, "10.0.0.1"),
        ])
        ipc, nccl = partition_receivers(rank_infos)
        assert ipc == {}
        assert set(nccl) == {0, 1}

    def test_multiple_receivers_on_same_gpu_as_sender(self):
        """Two receivers on the same GPU as one sender."""
        rank_infos = self._make_rank_infos([
            (0, Role.SENDER,   0, "10.0.0.1"),
            (1, Role.RECEIVER, 0, "10.0.0.1"),
            (2, Role.RECEIVER, 0, "10.0.0.1"),
        ])
        ipc, nccl = partition_receivers(rank_infos)

        assert set(ipc[0]) == {1, 2}
        assert nccl == []

    def test_duplicate_sender_location_raises(self):
        """Two senders on the same (ip, gpu_id) must raise ValueError."""
        rank_infos = self._make_rank_infos([
            (0, Role.SENDER,   0, "10.0.0.1"),
            (1, Role.SENDER,   0, "10.0.0.1"),  # duplicate location
            (2, Role.RECEIVER, 1, "10.0.0.1"),
        ])
        with pytest.raises(ValueError, match="Duplicate sender location"):
            partition_receivers(rank_infos)
