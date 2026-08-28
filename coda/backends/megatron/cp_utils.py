"""CP utilities. Two modes via ``cp_partition_mode``: ``zigzag`` (TE default, per-traj slice)
and ``contiguous`` (DSv4-required)."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from typing import Callable, Literal

from megatron.core import parallel_state as mpu
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_config import TransformerConfig


CPPartitionMode = Literal["zigzag", "contiguous"]


def _contiguous_align_size(cp_size: int, tp_size: int) -> int:
    """Return per-traj alignment size for contiguous CP.

    ``tp_size * 2 * cp_size`` when CP is enabled, else just ``tp_size``.
    """
    if cp_size <= 1:
        return max(tp_size, 1)
    return max(tp_size, 1) * 2 * cp_size


def _contiguous_padded_length(raw_len: int, cp_size: int, tp_size: int) -> int:
    """Round a trajectory's raw length up to the contiguous CP alignment."""
    align = _contiguous_align_size(cp_size, tp_size)
    return ((raw_len + align - 1) // align) * align


def _pad_seq_dim(tensor: torch.Tensor, pad_len: int, pad_value: int) -> torch.Tensor:
    """Tail-pad ``tensor`` along dim 0 (the sequence dim), whatever its rank.

    ``F.pad``'s spec is ordered from the last dim backwards, so the bare
    ``(0, pad_len)`` that is correct for the 1-D token/mask/log-prob tensors pads
    the feature dim of a 2-D ``[seq, hidden]`` tensor instead -- which silently
    turns teacher hidden states into ``[seq, hidden + pad_len]`` and leaves the
    sequence dim short of the ``2 * cp_size * chunk_size`` that zigzag slicing
    assumes.
    """
    spec = (0, 0) * (tensor.dim() - 1) + (0, pad_len)
    return F.pad(tensor, spec, value=pad_value)


def slice_cp_with_zigzag(
    tensor: torch.Tensor,
    pad_value: int | Callable,
) -> torch.Tensor:
    """Zigzag CP slice: pad to 2*cp chunks, extract symmetric front+back for this rank."""
    cp_rank = mpu.get_context_parallel_rank()
    cp_size = mpu.get_context_parallel_world_size()

    if cp_size == 1:
        return tensor

    token_len = tensor.size(0)
    chunk_size = (token_len + 2 * cp_size - 1) // (2 * cp_size)

    # Pad to exact multiple of 2 * cp_size * chunk_size
    pad_len = 2 * cp_size * chunk_size - token_len
    if pad_len > 0:
        if callable(pad_value):
            tensor = pad_value(tensor, pad_len)
        else:
            tensor = _pad_seq_dim(tensor, pad_len, pad_value)

    # Symmetric two-chunk extraction for THD CP (zigzag)
    start_1 = chunk_size * cp_rank
    end_1 = chunk_size * (cp_rank + 1)
    start_2 = chunk_size * (2 * cp_size - cp_rank - 1)
    end_2 = chunk_size * (2 * cp_size - cp_rank)
    return torch.cat([tensor[start_1:end_1], tensor[start_2:end_2]], dim=0)


def slice_cp_packed(
    tensors: list[torch.Tensor],
    cp_partition_mode: CPPartitionMode,
    pad_value: int | Callable,
    pad_multiplier: int = 0,
) -> tuple[torch.Tensor, list[int]]:
    """CP-slice multiple per-traj tensors into a packed buffer. Returns (packed, padded_lens)."""
    cp_size = mpu.get_context_parallel_world_size()
    tp_size = mpu.get_tensor_model_parallel_world_size()

    if cp_partition_mode == "zigzag":
        sliced = [slice_cp_with_zigzag(t, pad_value) for t in tensors]
        padded_lens = [s.size(0) for s in sliced]
        packed = torch.cat(sliced, dim=0)
        # Tail-pad for SP scatter when pad_multiplier is specified.
        if pad_multiplier > 0:
            pad_size = tp_size * pad_multiplier
            tail = (pad_size - packed.size(0) % pad_size) % pad_size
            if tail != 0:
                if callable(pad_value):
                    packed = pad_value(packed, tail)
                else:
                    packed = _pad_seq_dim(packed, tail, pad_value)

    elif cp_partition_mode == "contiguous":
        cp_rank = mpu.get_context_parallel_rank()
        raw_lens = [int(t.size(0)) for t in tensors]
        padded_lens = [_contiguous_padded_length(L, cp_size, tp_size) for L in raw_lens]
        total_padded = sum(padded_lens)

        dtype = tensors[0].dtype
        device = tensors[0].device
        if callable(pad_value):
            padded_list: list[torch.Tensor] = []
            for t, L, pl in zip(tensors, raw_lens, padded_lens):
                if pl > L:
                    t = pad_value(t, pl - L)
                padded_list.append(t)
            full_packed = torch.cat(padded_list, dim=0)
        else:
            # Trailing dims come from the inputs: the 2-D [seq, hidden] teacher
            # hidden states go through here too, and a flat [total_padded] buffer
            # would reject the per-traj assignment below.
            full_packed = torch.full(
                (total_padded, *tensors[0].shape[1:]),
                pad_value, dtype=dtype, device=device,
            )
            offset = 0
            for tok, L, PL in zip(tensors, raw_lens, padded_lens):
                if L > 0:
                    full_packed[offset : offset + L] = tok
                offset += PL

        if cp_size > 1:
            local_len = total_padded // cp_size
            packed = full_packed.narrow(0, cp_rank * local_len, local_len).contiguous()
        else:
            packed = full_packed

    else:
        raise ValueError(f"Unsupported cp_partition_mode: {cp_partition_mode!r}")

    return packed, padded_lens


def prepare_packed_seq_params(
    tokens_list: list[torch.Tensor],
    pad_token_id: int = 0,
    pad_multiplier: int = 128,
    cp_partition_mode: CPPartitionMode = "zigzag",
) -> tuple[torch.Tensor, PackedSeqParams]:
    """Pack token tensors for Megatron THD attention with CP slicing and build PackedSeqParams."""
    if not tokens_list:
        raise ValueError("tokens_list must be non-empty")

    cp_size = mpu.get_context_parallel_world_size()

    packed, padded_lens = slice_cp_packed(
        tokens_list, cp_partition_mode, pad_token_id, pad_multiplier
    )

    if cp_partition_mode == "zigzag":
        # cu_seqlens from per-traj local lengths; THD needs pre-CP lengths (* cp_size).
        cu_seqlens = [0]
        for pl in padded_lens:
            cu_seqlens.append(cu_seqlens[-1] + pl)
        # Tail-pad segment (if any) is already included in packed but not in padded_lens.
        if packed.size(0) > cu_seqlens[-1]:
            cu_seqlens.append(packed.size(0))
        cu_seqlens_t = torch.tensor(
            cu_seqlens, dtype=torch.int32, device=packed.device
        ) * cp_size
        max_seqlen = (cu_seqlens_t[1:] - cu_seqlens_t[:-1]).max().item()

        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=cu_seqlens_t,
            cu_seqlens_kv=cu_seqlens_t,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            qkv_format="thd",
            cp_partition_mode="zigzag",
        )
        return packed, packed_seq_params

    if cp_partition_mode == "contiguous":
        # cu_seqlens{,_padded} both describe the *global* padded THD layout.
        cu_padded = [0]
        for PL in padded_lens:
            cu_padded.append(cu_padded[-1] + PL)
        cu_padded_t = torch.tensor(cu_padded, dtype=torch.int32, device=packed.device)
        max_seqlen = max(padded_lens) if padded_lens else 0
        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=cu_padded_t,
            cu_seqlens_kv=cu_padded_t,
            cu_seqlens_q_padded=cu_padded_t,
            cu_seqlens_kv_padded=cu_padded_t,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
            qkv_format="thd",
            cp_partition_mode="contiguous",
        )
        return packed, packed_seq_params

    raise ValueError(f"Unsupported cp_partition_mode: {cp_partition_mode!r}")


def prepare_routing_replay_data(
    routed_experts_list: list[torch.Tensor],
    model_config: TransformerConfig,
    pad_multiplier: int = 128,
) -> torch.Tensor | None:
    """CP/TP-slice routed experts for the whole batch, matching the model's packed inputs."""
    tp_size = mpu.get_tensor_model_parallel_world_size()
    tp_rank = mpu.get_tensor_model_parallel_rank()

    assert model_config.num_moe_experts > 0, "model_config must have 'num_moe_experts' > 0"
    num_moe_experts = model_config.num_moe_experts
    cp_partition_mode: CPPartitionMode = model_config.cp_partition_mode

    def pad_func(tensor: torch.Tensor, pad_len: int) -> torch.Tensor:
        pad_tensor = (
            torch.arange(
                pad_len * tensor.size(1) * tensor.size(2),
                device=tensor.device,
                dtype=tensor.dtype,
            ).reshape((pad_len, tensor.size(1), tensor.size(2)))
            % num_moe_experts
        )
        return torch.cat([tensor, pad_tensor], dim=0)

    packed, _ = slice_cp_packed(
        routed_experts_list, cp_partition_mode, pad_func, pad_multiplier
    )

    # TP slice (sequence parallel)
    if model_config.sequence_parallel:
        seqlen = packed.size(0)
        assert seqlen % tp_size == 0
        start = seqlen // tp_size * tp_rank
        end = seqlen // tp_size * (tp_rank + 1)
        packed = packed[start:end]

    return packed


def gather_and_reconstruct_cp(
    packed_tensor: torch.Tensor,
    total_lengths: list[int],
    cp_partition_mode: CPPartitionMode = "zigzag",
) -> list[torch.Tensor]:
    """All-gather CP-local tensor and reconstruct per-traj full sequences."""
    cp_size = mpu.get_context_parallel_world_size()
    cp_rank = mpu.get_context_parallel_rank()
    cp_group = mpu.get_context_parallel_group()

    # all_gather the detached copies, then drop our own local rank back in
    # (un-detached) so autograd flows only through this rank's contribution —
    # avoids the ×cp_size gradient amplification a plain all_gather would cause.
    gathered = [torch.empty_like(packed_tensor) for _ in range(cp_size)]
    dist.all_gather(gathered, packed_tensor.detach(), group=cp_group)
    gathered[cp_rank] = packed_tensor

    if cp_partition_mode == "zigzag":
        result: list[torch.Tensor] = []
        local_offset = 0
        for total_len in total_lengths:
            chunk_size = (total_len + 2 * cp_size - 1) // (2 * cp_size)
            local_len = 2 * chunk_size

            # Inverse zigzag reconstruction:
            #   rank j owns [j*C, (j+1)*C) as front chunk
            #              [(2N-1-j)*C, (2N-j)*C) as back chunk
            # Concatenation order: rank0_front, rank1_front, ..., rankN-1_back, ..., rank0_back
            front = [
                gathered[j][local_offset : local_offset + chunk_size]
                for j in range(cp_size)
            ]
            back = [
                gathered[j][local_offset + chunk_size : local_offset + local_len]
                for j in reversed(range(cp_size))
            ]
            result.append(torch.cat(front + back))
            local_offset += local_len
        return result

    if cp_partition_mode == "contiguous":
        # rank-order concat rebuilds the full [total_padded] buffer that was
        # ``narrow``-sliced in prepare_packed_seq_params.
        full_packed = torch.cat(gathered, dim=0)
        tp_size = mpu.get_tensor_model_parallel_world_size()

        result_c: list[torch.Tensor] = []
        offset = 0
        for total_len in total_lengths:
            padded_len = _contiguous_padded_length(int(total_len), cp_size, tp_size)
            result_c.append(full_packed[offset : offset + padded_len])
            offset += padded_len
        return result_c

    raise ValueError(f"Unsupported cp_partition_mode: {cp_partition_mode!r}")


def gather_and_slice_response(
    per_token_values: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    cp_partition_mode: CPPartitionMode = "zigzag",
) -> list[torch.Tensor]:
    """CP all-gather per-token values and slice out response-only portions.

    Args:
        per_token_values: Per-microbatch CP-local per-token tensors.
        total_lengths: Pre-CP total (prompt + response) length per trajectory.
        response_lengths: Response-only length per trajectory.
        cp_partition_mode: Must match how the values were originally CP-split.
    """
    cp_size = mpu.get_context_parallel_world_size()
    flat = torch.cat(per_token_values)
    offset = -1
    if cp_size > 1:
        reconstructed = gather_and_reconstruct_cp(flat, total_lengths, cp_partition_mode)
        result = []
        for val, total_len, resp_len in zip(reconstructed, total_lengths, response_lengths):
            prompt_len = total_len - resp_len
            result.append(val[offset + prompt_len : offset + total_len])
        return result
    else:
        result = []
        for total_len, resp_len in zip(total_lengths, response_lengths):
            prompt_len = total_len - resp_len
            result.append(flat[offset + prompt_len : offset + total_len])
            offset += total_len
        return result