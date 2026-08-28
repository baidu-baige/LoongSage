"""TransferMesh - Efficient Weight Synchronization Library.

TransferMesh enables efficient model weight transfer between senders and receivers
using topology-aware communication strategies.

- Same-card transfers use IPC handles for zero-copy
- Cross-card transfers use NCCL
- Gloo for metadata and IPC handle transmission
"""

from .channel import (
    TransferMeshChannel,
    Role,
    create_channel,
)
from .protocol import (
    TensorSpec,
    MetaFrame,
    str_to_dtype,
)
from .topology import (
    RankInfo,
    partition_receivers,
)
__version__ = "0.1.0"

__all__ = [
    # Channel
    "TransferMeshChannel",
    "Role",
    "create_channel",
    # Protocol
    "TensorSpec",
    "MetaFrame",
    "str_to_dtype",
    # Topology
    "RankInfo",
    "partition_receivers",
]