"""Helper for managing a process-local TransferMeshChannel instance.

Each process that imports this module maintains its own module-level channel.
Use ``create_sender_channel(meta)`` or ``create_receiver_channel(meta)`` as
the entry points: they create, recreate, or reuse the channel based on the
``recreate`` flag inside *meta*.

Pass a ``ChannelMeta`` instance to either helper.  See ``ChannelMeta``
for the full list of fields and their defaults.

Dynamically resolved (not needed in ChannelMeta):
    addr: automatically obtained via HOST_IP env var or Ray
    gpu_id: automatically obtained from CUDA_VISIBLE_DEVICES (physical index)
    rank: engine_id * dist.get_world_size() + dist.get_rank() [+ train_world_size for receivers]
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import ray
import torch
import torch.distributed as dist

from coda.transfer_mesh.channel import TransferMeshChannel, create_channel, Role


@dataclass
class ChannelMeta:
    """Configuration for creating / recreating a TransferMeshChannel.

    Required fields
    ---------------
    master_addr : str
        IP address of rank-0 used as the Gloo rendezvous point.
    world_size : int
        Total number of participants (senders + receivers).
    train_world_size : int
        Training world size, added to ``dist.get_rank()`` when computing
        the global rank so that inference ranks do not collide with training
        ranks.

    Optional fields
    ---------------
    recreate : bool
        If ``True``, close the existing process-local channel and create a
        fresh one.  Defaults to ``False``.
    buffer_size_bytes : int
        Bucket size threshold in bytes.  Defaults to 1 GiB.
    timeout : float
        Operation timeout in seconds.  Defaults to 300.0.
    gloo_port : int
        TCP port for the Gloo TCPStore rendezvous.  Defaults to 29500.
    local_ip : str or None
        If specified, use this IP directly instead of auto-detecting.
        Otherwise, detection order: HOST_IP env var → Ray.
    engine_id : int
        Engine identifier.  Use ``0`` for single-engine setups; set to a
        positive integer for multi-engine setups so that the rank formula
        ``engine_id * dist.get_world_size() + dist.get_rank() + train_world_size``
        is used.
    """

    # --- required ---
    master_addr: str
    world_size: int
    train_world_size: int

    # --- optional ---
    engine_id: int = 0
    recreate: bool = False
    buffer_size_bytes: int = 1024 * 1024 * 1024  # 1 GiB
    timeout: float = 300.0
    gloo_port: int = 29500
    local_ip: str | None = None

logger = logging.getLogger(__name__)

_channel: TransferMeshChannel | None = None

def _get_physical_gpu_id() -> int:
    """Return the physical GPU device index visible to the current process.

    Ray actors set CUDA_VISIBLE_DEVICES to a single device, so
    ``torch.cuda.current_device()`` always returns 0 (the only visible device).
    We parse CUDA_VISIBLE_DEVICES to get the actual physical GPU id instead.
    Falls back to torch.cuda.current_device() when the variable is unset, cannot be
    parsed, or does not cover the local device index.
    """
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        try:
            # CUDA_VISIBLE_DEVICES may be "0,1,2" or just "3"
            local_idx = torch.cuda.current_device()
            ids = [int(x.strip()) for x in cuda_visible.split(",") if x.strip()]
            if ids and local_idx < len(ids):
                return ids[local_idx]
        except (ValueError, IndexError):
            pass
    return torch.cuda.current_device()



def _get_ip_from_ray() -> str | None:
    """Try to get local IP from Ray runtime."""
    try:
        return ray.util.get_node_ip_address()
    except Exception:
        return None


def _get_local_ip(local_ip: str | None) -> str:
    """Resolve local IP: use specified value, then env var HOST_IP, then Ray.

    Raises RuntimeError if all methods fail.
    """
    if local_ip:
        return local_ip

    ip = os.environ.get("HOST_IP")
    if ip:
        logger.info("Local IP resolved via HOST_IP env var: %s", ip)
        return ip

    ip = _get_ip_from_ray()
    if ip:
        logger.info("Local IP resolved via Ray: %s", ip)
        return ip

    raise RuntimeError(
        "Failed to resolve local IP address. "
        "Set ChannelMeta.local_ip explicitly, set HOST_IP env var, or ensure Ray is available."
    )


def _compute_rank(engine_id: int, train_world_size: int, role: Role) -> int:
    """Compute the rank for the channel.

    Base rank: engine_id * dist.get_world_size() + dist.get_rank()
    For receivers, train_world_size is added so that receiver ranks
    start after sender ranks and do not collide.

    - sender:   engine_id * dist.get_world_size() + dist.get_rank()
    - receiver: engine_id * dist.get_world_size() + dist.get_rank() + train_world_size
    """
    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed is not initialized. "
            "This helper requires the caller to have already called dist.init_process_group()."
        )

    rank = engine_id * dist.get_world_size() + dist.get_rank()
    if role == Role.RECEIVER:
        rank += train_world_size
    return rank


def _get_or_recreate_channel(meta: ChannelMeta, role: Role) -> TransferMeshChannel:
    """Return a process-local TransferMeshChannel, creating or recreating as needed.

    Args:
        meta: A ``ChannelMeta`` instance describing the channel configuration.
            See ``ChannelMeta`` for field descriptions.
        role: ``Role.SENDER`` or ``Role.RECEIVER``.

        The following are resolved automatically:
            - addr: Local IP address.
            - gpu_id: Physical CUDA device index.
            - rank: Computed based on engine_id and dist.get_rank().

    Returns:
        An initialised TransferMeshChannel instance.
    """
    global _channel

    if meta.recreate and _channel is not None:
        logger.info("Recreating channel (closing existing instance).")
        _channel.close()
        _channel = None

    if _channel is not None and _channel.role != role:
        raise RuntimeError(
            f"Cached channel has role {_channel.role}, but {role} was requested. "
            f"Set recreate=True to close the existing channel first."
        )

    if _channel is None:
        # Resolve dynamic fields
        addr = _get_local_ip(meta.local_ip)
        rank = _compute_rank(meta.engine_id, meta.train_world_size, role)
        gpu_id = _get_physical_gpu_id()

        _channel = create_channel(
            master_addr=meta.master_addr,
            addr=addr,
            gpu_id=gpu_id,
            world_size=meta.world_size,
            rank=rank,
            role=role,
            src_rank=0,  # TODO: make configurable if needed
            buffer_size_bytes=meta.buffer_size_bytes,
            timeout=meta.timeout,
            gloo_port=meta.gloo_port,
        )
        logger.info(
            "Channel created (rank=%s, role=%s, engine_id=%s, gpu_id=%s).",
            rank, role, meta.engine_id, gpu_id
        )

    return _channel


def create_sender_channel(meta: ChannelMeta) -> TransferMeshChannel:
    """Create (or reuse) a process-local channel with ``Role.SENDER``."""
    return _get_or_recreate_channel(meta, Role.SENDER)


def create_receiver_channel(meta: ChannelMeta) -> TransferMeshChannel:
    """Create (or reuse) a process-local channel with ``Role.RECEIVER``."""
    return _get_or_recreate_channel(meta, Role.RECEIVER)
