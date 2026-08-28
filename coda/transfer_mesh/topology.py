"""Topology detection for TransferMesh.

Detects optimal communication method between rank pairs based on GPU placement.
"""

from dataclasses import dataclass
from enum import Enum


class Role(str, Enum):
    """Role of a participant in the transfer."""
    SENDER = "sender"
    RECEIVER = "receiver"


@dataclass
class RankInfo:
    """Information about a single rank's placement.

    Attributes:
        rank: Global rank ID
        gpu_id: Physical GPU index
        ip: IP address of the node
        role: Role enum value (SENDER or RECEIVER); None if not yet assigned
    """
    rank: int
    gpu_id: int
    ip: str
    role: Role | None = None


def partition_receivers(
    all_rank_info: list[RankInfo],
) -> tuple[dict[int, list[int]], list[int]]:
    """Partition receivers into IPC / NCCL groups based on GPU co-location.

    - Same GPU as a sender → IPC (zero-copy)
    - No co-located sender  → NCCL (broadcast from src_rank)

    Args:
        all_rank_info: List of RankInfo with role set.

    Returns:
        Tuple of:
        - ipc_assignments: ``{sender_rank: [co-located receiver ranks]}``
        - nccl_receivers: receiver ranks without a local sender

    Raises:
        ValueError: If two senders share the same (ip, gpu_id) location.
    """
    sender_ranks: list[int] = []
    receiver_ranks: list[int] = []

    for info in all_rank_info:
        if info.role == Role.SENDER:
            sender_ranks.append(info.rank)
        else:
            receiver_ranks.append(info.rank)

    # Build (ip, gpu_id) → sender_rank lookup, detecting duplicates
    rank_to_info = {info.rank: info for info in all_rank_info}
    sender_by_location: dict[tuple[str, int], int] = {}
    for sr in sender_ranks:
        info = rank_to_info[sr]
        key = (info.ip, info.gpu_id)
        if key in sender_by_location:
            raise ValueError(
                f"Duplicate sender location: ip={info.ip}, gpu_id={info.gpu_id} "
                f"(ranks {sender_by_location[key]} and {sr})"
            )
        sender_by_location[key] = sr

    # Assign each receiver to a co-located sender or to NCCL
    ipc_assignments: dict[int, list[int]] = {sr: [] for sr in sender_ranks}
    nccl_receivers: list[int] = []

    for rr in receiver_ranks:
        info = rank_to_info[rr]
        key = (info.ip, info.gpu_id)
        if key in sender_by_location:
            ipc_assignments[sender_by_location[key]].append(rr)
        else:
            nccl_receivers.append(rr)

    return ipc_assignments, nccl_receivers
