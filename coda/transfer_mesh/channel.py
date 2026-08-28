"""TransferMesh Channel implementation.

Topology-Aware Transfer
- Same-card transfers use IPC handles
- Cross-card transfers use NCCL, broadcast from the configured ``src_rank``
"""

import datetime
import logging
import os
import pickle
from collections.abc import Iterator
from contextlib import contextmanager
from typing import cast

import torch
import torch.distributed as dist

from ._c10d_compat import (
    _get_distributed_c10d_symbols,
    _register_process_group_ranks,
)
from .protocol import MetaFrame, TensorSpec, str_to_dtype
from .topology import RankInfo, Role, partition_receivers

logger = logging.getLogger(__name__)


def _expandable_segments_enabled() -> bool:
    """Whether this process started with expandable CUDA segments enabled.

    There is no torch API to read the current allocator settings back, so the
    launch environment is the only source of truth.  Parsed once at import.
    """
    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    for entry in conf.split(","):
        key, _, value = entry.partition(":")
        if key.strip() == "expandable_segments":
            return value.strip().lower() == "true"
    return False


_EXPANDABLE_SEGMENTS = _expandable_segments_enabled()


def _set_expandable_segments(enabled: bool) -> None:
    setter = getattr(torch._C, "_accelerator_setAllocatorSettings", None)
    if setter is None:  # torch < 2.9 keeps it under torch.cuda.memory
        setter = torch.cuda.memory._set_allocator_settings
    setter(f"expandable_segments:{enabled}")


@contextmanager
def _classic_ipc_segments():
    """Allocate inside this block from a non-expandable CUDA segment.

    When ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` is in effect,
    ``_share_cuda_()`` stops returning a plain ``cudaIpcMemHandle`` and instead
    hands back a descriptor the receiver must import via ``pidfd_open`` +
    ``pidfd_getfd`` on the sender's process.  That import fails with ``EBADF`` on
    some kernel/container configurations, so every IPC weight transfer breaks —
    while the training process genuinely wants expandable segments to keep the
    repeatedly resized param/grad buffers from fragmenting the heap.

    A segment records the mode it was created under, so allocating just the
    shared bucket buffer with the flag off yields a classic ``cudaIpcMemHandle``
    and leaves the rest of the process expandable.  The buffer is discarded and
    reallocated on every flush, so it gains nothing from expandable segments
    anyway.

    No-op when the process never enabled expandable segments.  The switch is
    process-global, so do not allocate from other threads inside this block or
    they will get classic segments too.
    """
    if not _EXPANDABLE_SEGMENTS:
        yield
        return
    _set_expandable_segments(False)
    try:
        yield
    finally:
        _set_expandable_segments(True)


class TransferMeshChannel:
    """TransferMesh channel for weight synchronization.

    Key features:
    - Gloo for metadata and IPC handle transmission (point-to-point)
    - NCCL for bulk tensor data (cross-card broadcast)
    - IPC handles for same-card zero-copy
    - Bucket-based aggregation for reduced communication rounds
    """

    def __init__(
        self,
        master_addr: str,
        addr: str,
        gpu_id: int,
        world_size: int,
        rank: int,
        role: Role,
        src_rank: int,
        buffer_size_bytes: int = 1024 * 1024 * 1024,  # 1GB default
        timeout: float = 300.0,
        gloo_port: int = 29500,
    ):
        """Initialize TransferMesh channel.

        Args:
            master_addr: Rank 0 address for coordination
            addr: Local IP address
            gpu_id: Physical GPU index, used for topology/IPC routing
            world_size: Total participant count (senders + receivers)
            rank: This node's rank in the global group
            role: Whether this rank is a sender or receiver
            src_rank: The primary sender rank
            buffer_size_bytes: Bucket size threshold
            timeout: Operation timeout in seconds
            gloo_port: TCP port for the Gloo TCPStore rendezvous.
        """
        self.master_addr: str = master_addr
        self.addr: str = addr
        self.gpu_id: int = gpu_id
        self.world_size: int = world_size
        self.rank: int = rank
        self.role: Role = role
        self.src_rank: int = src_rank
        self.buffer_size_bytes: int = buffer_size_bytes
        self.timeout: float = timeout
        self.gloo_port: int = gloo_port
        # Local CUDA device index (always 0 in Ray actors with CUDA_VISIBLE_DEVICES
        # restricted to a single device; gpu_id may be the physical index used for
        # topology/IPC routing which can differ from the local index).
        self._local_device: int = torch.cuda.current_device()

        # Communication groups (initialized in init_groups)
        self._gloo_group: dist.ProcessGroup | None = None  # main coordination group
        self._nccl_data_group: dist.ProcessGroup | None = None
        self._nccl_meta_group: dist.ProcessGroup | None = None

        # Topology info
        self._ipc_assignments: dict[int, list[int]] = {}
        self._nccl_receivers: list[int] = []

        # Buffer for bucket accumulation
        self._buffer: torch.Tensor | None = None
        self._buffer_offset: int = 0
        self._pending_specs: list[TensorSpec] = []

    def init_groups(self) -> None:
        """Initialize communication groups and auto-discover topology.

        Steps:
        1. Create an independent Gloo process group via _new_process_group_helper.
           This works even when torch.distributed is already initialized with another
           backend (e.g. NCCL for TP workers) because it bypasses init_process_group
           and directly creates a new group object.
        2. Single all_gather to collect role + RankInfo from all ranks
        3. Compute IPC/NCCL topology partition
        4. Create NCCL data group and NCCL metadata group
        """
        if self._gloo_group is not None:
            logger.warning("rank %d init_groups() called but groups already initialized, skipping", self.rank)
            return
        timeout_td = datetime.timedelta(seconds=self.timeout)

        all_ranks = list(range(self.world_size))
        self._gloo_group = self._create_pg(
            all_ranks, self.rank, dist.Backend("gloo"),
            port_offset=0, group_name="transfer_mesh_gloo", timeout=timeout_td,
        )

        # Single gather: collect RankInfo from all ranks at once.
        my_info = RankInfo(
            rank=self.rank,
            gpu_id=self.gpu_id,
            ip=self.addr,
            role=self.role,
        )
        gathered: list[RankInfo] = cast(
            list[RankInfo],
            [None] * self.world_size,
        )
        dist.all_gather_object(gathered, my_info, group=self._gloo_group)

        # Partition receivers into IPC (same-GPU) and NCCL (cross-GPU) groups
        self._ipc_assignments, self._nccl_receivers = partition_receivers(gathered)

        # Create NCCL groups for cross-GPU receivers (data + metadata)
        if self._nccl_receivers:
            participants = [self.src_rank] + self._nccl_receivers
            if self.rank in participants:
                local_rank = participants.index(self.rank)

                self._nccl_data_group = self._create_pg(
                    participants, local_rank, dist.Backend("nccl"),
                    port_offset=1, group_name="transfer_mesh_nccl_data", timeout=timeout_td,
                )
                self._nccl_meta_group = self._create_pg(
                    participants, local_rank, dist.Backend("gloo"),
                    port_offset=2, group_name="transfer_mesh_nccl_meta", timeout=timeout_td,
                )

        logger.info(
            "rank %d init_groups done: role=%s, ipc_assignments=%s, nccl_receivers=%s",
            self.rank, self.role, self._ipc_assignments, self._nccl_receivers,
        )

    def _create_pg(
        self,
        participants: list[int],
        local_rank: int,
        backend: object,
        *,
        port_offset: int,
        group_name: str,
        timeout: datetime.timedelta,
    ) -> dist.ProcessGroup:
        """Create a standalone process group with its own TCPStore.

        All groups created here are independent (not derived from a default
        process group), so [] is passed as ranks to _new_process_group_helper
        to avoid an internal _get_default_group() call.
        """
        PrefixStore, _new_pg_helper, _world = _get_distributed_c10d_symbols()

        store = dist.TCPStore(
            self.master_addr,
            port=self.gloo_port + port_offset,
            world_size=len(participants),
            is_master=(local_rank == 0),
            timeout=timeout,
        )
        pg = cast(
            dist.ProcessGroup,
            _new_pg_helper(
                len(participants),
                local_rank,
                [],
                backend,
                PrefixStore(group_name, store),
                group_name=group_name,
                timeout=timeout,
            )[0],
        )
        _register_process_group_ranks(
            _world, pg, {i: participants[i] for i in range(len(participants))}
        )
        return pg

    def _flush_bucket(self, is_end: bool = False) -> None:
        """Send accumulated bucket to receivers. Sender only."""
        if self.role != Role.SENDER:
            raise RuntimeError("_flush_bucket must only be called by sender")

        if not self._pending_specs and not is_end:
            return

        meta = MetaFrame(
            is_end=is_end,
            tensor_specs=self._pending_specs.copy(),
            payload_numel=self._buffer_offset,
        )

        # Broadcast metadata to NCCL receivers (only src_rank sends)
        if self._nccl_receivers and self.rank == self.src_rank:
            self._broadcast_nccl_metadata(meta)

        payload = self._buffer[:self._buffer_offset] if self._pending_specs and self._buffer is not None else None

        my_ipc_receivers = self._ipc_assignments.get(self.rank, [])
        if my_ipc_receivers:
            self._send_ipc(payload, my_ipc_receivers, meta)

        if payload is not None and self._nccl_receivers and self.rank == self.src_rank:
            self._send_nccl(payload)

        # Discard buffer; next send() will allocate a fresh one.
        self._buffer = None
        self._buffer_offset = 0
        self._pending_specs.clear()

        # Promptly reclaim shared memory no longer in use by receivers
        torch.cuda.ipc_collect()

    def _broadcast_nccl_metadata(self, meta: MetaFrame) -> None:
        """Broadcast metadata to NCCL receivers via _nccl_meta_group."""
        if self._nccl_meta_group is None:
            return

        meta_list: list[bytes | None] = [meta.serialize()] if self.rank == self.src_rank else [None]
        dist.broadcast_object_list(meta_list, src=self.src_rank, group=self._nccl_meta_group)

    def _send_ipc(self, tensor: torch.Tensor | None, receiver_ranks: list[int], meta: MetaFrame) -> None:
        """Send tensor (or end marker) to same-card receivers via IPC handle.

        Args:
            tensor: Payload tensor, or None for end marker.
            receiver_ranks: List of receiver ranks on the same card.
            meta: MetaFrame describing the bucket.
        """
        if tensor is not None:
            ipc_handle: object = tensor.untyped_storage()._share_cuda_()
            handle_data: dict[str, object] = {
                "ipc_handle": ipc_handle,
                "size": tensor.numel(),
                "dtype": str(tensor.dtype),
                "shape": tuple(tensor.shape),
                "meta": meta,
            }
        else:
            handle_data = {"meta": meta}

        handle_bytes = pickle.dumps(handle_data)
        payload = torch.tensor(list(handle_bytes), dtype=torch.uint8)
        header = torch.tensor([len(handle_bytes)], dtype=torch.int64)

        for recv_rank in receiver_ranks:
            _ = dist.send(header, dst=recv_rank, group=self._gloo_group)
            _ = dist.send(payload, dst=recv_rank, group=self._gloo_group)

    def _send_nccl(self, tensor: torch.Tensor) -> None:
        """Broadcast tensor to cross-card receivers via NCCL data group."""
        if self._nccl_data_group is not None:
            _ = dist.broadcast(tensor, src=self.src_rank, group=self._nccl_data_group)

    def _recv_ipc(self, sender_rank: int) -> tuple[torch.Tensor, MetaFrame]:
        """Receive tensor and metadata from same-card sender via IPC handle."""
        # Phase 1: receive 8-byte header containing payload size
        header = torch.empty(1, dtype=torch.int64)
        _ = dist.recv(header, src=sender_rank, group=self._gloo_group)
        size = int(header.item())
        if size <= 0:
            raise RuntimeError(
                f"_recv_ipc: received non-positive payload size {size} from "
                f"rank {sender_rank}, likely data corruption or protocol mismatch"
            )

        # Phase 2: receive payload of exact size
        payload = torch.empty(size, dtype=torch.uint8)
        _ = dist.recv(payload, src=sender_rank, group=self._gloo_group)

        handle_bytes = payload.numpy().tobytes()
        handle_data: dict[str, object] = cast(dict[str, object], pickle.loads(handle_bytes))
        meta = cast(MetaFrame, handle_data["meta"])

        if meta.is_end and not meta.tensor_specs:
            return torch.empty(0, device=self._local_device), meta

        # The IPC handle tuple from _share_cuda_() has the sender's local device
        # ordinal (always 0 for Ray actors with CUDA_VISIBLE_DEVICES=N) as the first
        # element.  Replace it with self._local_device so _new_shared_cuda opens
        # the shared memory directly on the receiver's target device — no copy needed.
        from torch.storage import UntypedStorage
        ipc_handle = cast(tuple, handle_data["ipc_handle"])
        if ipc_handle[0] != self._local_device:
            ipc_handle = (self._local_device,) + ipc_handle[1:]
        storage = UntypedStorage._new_shared_cuda(*ipc_handle)
        dtype = str_to_dtype(cast(str, handle_data["dtype"]))
        size = cast(int, handle_data["size"])
        tensor = torch.empty(size, dtype=dtype, device=self._local_device)
        tensor.set_(storage)

        return tensor, meta

    def _recv_nccl(self, numel: int, dtype: torch.dtype) -> torch.Tensor:
        """Receive tensor via NCCL broadcast from data group."""
        tensor = torch.empty(numel, dtype=dtype, device=self._local_device)
        if self._nccl_data_group is not None:
            _ = dist.broadcast(tensor, src=self.src_rank, group=self._nccl_data_group)
        return tensor

    def send(self, named_tensor: tuple[str, torch.Tensor] | None, flush: bool = False) -> None:
        """Send tensor through channel.

        Args:
            named_tensor: (name, tensor) tuple, or None for end marker
            flush: Force flush even if under buffer threshold; sets is_end=True
        """
        if self.role != Role.SENDER:
            raise RuntimeError("Only senders can call send()")

        if named_tensor is None or flush:
            self._flush_bucket(is_end=True)
            return

        name, tensor = named_tensor

        # Convert tensor to bytes for dtype-agnostic buffering
        tensor_bytes = tensor.numel() * tensor.element_size()

        if self._buffer is None or tensor_bytes > self._buffer.numel() - self._buffer_offset:
            self._flush_bucket()
            alloc_bytes = max(self.buffer_size_bytes, tensor_bytes)
            # Shared with co-located receivers via _share_cuda_(); must come from
            # a classic segment so the handle stays IPC-importable.
            with _classic_ipc_segments():
                self._buffer = torch.empty(
                    alloc_bytes, dtype=torch.uint8, device=self._local_device
                )

        # Copy tensor as raw bytes into uint8 buffer. contiguous() is a no-op if
        # already contiguous; otherwise it makes a copy so view(uint8) doesn't
        # raise on strided/non-contiguous input.
        flat_tensor = tensor.flatten().contiguous().view(torch.uint8)
        self._buffer[self._buffer_offset:self._buffer_offset + tensor_bytes] = flat_tensor
        self._pending_specs.append(TensorSpec(
            name=name,
            shape=tuple(tensor.shape),
            offset=self._buffer_offset,
            dtype=str(tensor.dtype),
        ))
        self._buffer_offset += tensor_bytes

        if self._buffer_offset >= self.buffer_size_bytes:
            self._flush_bucket()

    def get_iterator(
        self,
        yield_buckets: bool = False,
    ) -> Iterator[tuple[str, torch.Tensor] | tuple[torch.Tensor, MetaFrame]]:
        """Get iterator for receiving tensors.

        Args:
            yield_buckets: If False (default), yields (name, tensor) for each tensor
                in the bucket. If True, yields (payload, meta) for each bucket,
                where payload is the raw flat GPU tensor and meta contains tensor_specs
                describing the layout — allowing the caller to unpack tensors themselves.

        Yields:
            yield_buckets=False: (name, tensor) per tensor
            yield_buckets=True:  (payload: torch.Tensor, meta: MetaFrame) per bucket
        """
        if self.role != Role.RECEIVER:
            raise RuntimeError("Only receivers can call get_iterator()")

        my_sender = None
        for sender_rank, receivers in self._ipc_assignments.items():
            if self.rank in receivers:
                my_sender = sender_rank
                break
        use_ipc = my_sender is not None

        if not use_ipc and not (self._nccl_receivers and self._nccl_meta_group is not None):
            raise RuntimeError(
                f"Receiver rank {self.rank} has no IPC sender and no NCCL "
                f"metadata group; check topology configuration"
            )

        payload = torch.empty(0)
        while True:
            if use_ipc:
                try:
                    payload, meta = self._recv_ipc(my_sender)
                except RuntimeError as e:
                    if "Closed" in str(e) or "EOF" in str(e):
                        break
                    raise
            else:
                meta_list: list[bytes | None] = [None]
                dist.broadcast_object_list(meta_list, src=self.src_rank, group=self._nccl_meta_group)
                raw_meta = meta_list[0]
                if raw_meta is None:
                    raise RuntimeError(
                        "broadcast_object_list returned None; expected serialised MetaFrame"
                    )
                meta = MetaFrame.deserialize(raw_meta)
                if meta.tensor_specs:
                    payload = self._recv_nccl(meta.payload_numel, torch.uint8)

            if meta.tensor_specs:
                if yield_buckets:
                    yield (payload, meta)
                else:
                    for spec in meta.tensor_specs:
                        tensor_dtype = str_to_dtype(spec.dtype)
                        num_bytes = spec.numel() * tensor_dtype.itemsize
                        raw_bytes = payload[spec.offset:spec.offset + num_bytes]
                        tensor = raw_bytes.view(tensor_dtype).view(spec.shape).clone()
                        yield (spec.name, tensor)

            if meta.is_end:
                break

    def close(self) -> None:
        """Clean up resources."""
        logger.info("rank %d closing channel", self.rank)
        if self._nccl_data_group is not None:
            dist.destroy_process_group(self._nccl_data_group)
            self._nccl_data_group = None
        if self._nccl_meta_group is not None:
            dist.destroy_process_group(self._nccl_meta_group)
            self._nccl_meta_group = None
        if self._gloo_group is not None:
            dist.destroy_process_group(self._gloo_group)
            self._gloo_group = None
        if self.role == Role.SENDER:
            # Promptly reclaim shared memory no longer in use by receivers
            torch.cuda.ipc_collect()

        # Clear internal state to prevent misuse after close
        self._buffer = None
        self._buffer_offset = 0
        self._pending_specs.clear()
        self._ipc_assignments.clear()
        self._nccl_receivers.clear()


def create_channel(
    master_addr: str,
    addr: str,
    gpu_id: int,
    world_size: int,
    rank: int,
    role: Role,
    src_rank: int,
    buffer_size_bytes: int = 1024 * 1024 * 1024,
    timeout: float = 300.0,
    gloo_port: int = 29500,
) -> TransferMeshChannel:
    """Create and initialize a TransferMeshChannel.

    Args:
        master_addr: Rank 0 address for coordination
        addr: Local IP address
        gpu_id: Physical GPU index, used for topology/IPC routing
        world_size: Total participant count
        rank: This node's rank
        role: Role enum value (Role.SENDER or Role.RECEIVER)
        src_rank: Broadcast source rank
        buffer_size_bytes: Bucket size threshold
        timeout: Operation timeout in seconds
        gloo_port: TCP port for the Gloo TCPStore rendezvous

    Returns:
        Initialized TransferMeshChannel instance
    """
    channel = TransferMeshChannel(
        master_addr=master_addr,
        addr=addr,
        gpu_id=gpu_id,
        world_size=world_size,
        rank=rank,
        role=role,
        src_rank=src_rank,
        buffer_size_bytes=buffer_size_bytes,
        timeout=timeout,
        gloo_port=gloo_port,
    )

    channel.init_groups()

    return channel
