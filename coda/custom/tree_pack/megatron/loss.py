"""Per-row log-probs / entropy of tree micro-batches on TP-sharded node logits.

Outputs match LoongSage's ``compute_log_probs`` / ``compute_entropy`` (per-row lists).
"""
from __future__ import annotations

import torch
import torch.distributed as dist

from coda.custom.tree_pack.core.loss_index import TreeLossIndex


# from https://github.com/volcengine/verl/blob/
#   0bdf7f469854815177e73dcfe9e420836c952e6e/verl/utils/megatron/tensor_parallel.py#L99
# (as in LoongSage's logits_utils)
class _VocabParallelEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, vocab_parallel_logits: torch.Tensor, process_group: dist.ProcessGroup) -> torch.Tensor:
        """Per-token entropy over the TP-sharded vocabulary (numerically stable log-sum-exp)."""
        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=process_group)
        normalized_vocab_parallel_logits = vocab_parallel_logits - logits_max
        normalized_exp_logits = normalized_vocab_parallel_logits.exp_()
        normalized_sum_exp_logits = normalized_exp_logits.sum(dim=-1, keepdim=True)
        dist.all_reduce(normalized_sum_exp_logits, group=process_group)
        softmax_logits = normalized_exp_logits.div_(normalized_sum_exp_logits)
        sum_softmax_times_logits = (softmax_logits * vocab_parallel_logits).sum(dim=-1, keepdim=True)
        dist.all_reduce(sum_softmax_times_logits, group=process_group)
        entropy = logits_max + normalized_sum_exp_logits.log() - sum_softmax_times_logits
        ctx.save_for_backward(vocab_parallel_logits, softmax_logits, sum_softmax_times_logits)
        return entropy.squeeze(dim=-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        """dH/dx_v = -p_v * (x_v - E[x]) * grad_output."""
        vocab_parallel_logits, softmax_logits, sum_softmax_times_logits = ctx.saved_tensors
        # reuse softmax_logits as grad
        vocab_parallel_logits.sub_(sum_softmax_times_logits)
        softmax_logits.mul_(vocab_parallel_logits)
        softmax_logits.mul_(grad_output.unsqueeze(dim=-1))
        # recover vocab_parallel_logits
        vocab_parallel_logits.add_(sum_softmax_times_logits)
        softmax_logits.mul_(-1)
        return softmax_logits, None


# Upper bound on the per-chunk temporary of the metric-only entropy.  At TP=8
# with fp32 logits this is ~17k tokens per chunk and replaces the two full
# [tokens, vocab_shard] temporaries of ``_VocabParallelEntropy.forward``
# (~60 GiB at a 262k-token micro-batch).
_ENTROPY_CHUNK_BYTES = 2 * 1024 ** 3


def _entropy_chunk_rows(logits: torch.Tensor) -> int:
    """Rows (tokens) per chunk so one chunk's fp32 temporaries fit the budget."""
    # One fp32 chunk for exp(x - max); a non-fp32 input adds its fp32 copy.
    copies = 1 if logits.dtype == torch.float32 else 2
    bytes_per_row = max(1, logits.size(-1)) * 4 * copies
    return max(1, _ENTROPY_CHUNK_BYTES // bytes_per_row)


@torch.no_grad()
def _vocab_parallel_entropy_chunked(
    logits: torch.Tensor,
    process_group: dist.ProcessGroup,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Metric-only TP-sharded entropy, chunked along the token dimension.

    Uses ``H = M + log(S1) - S2 / S1`` with ``M = max x``,
    ``S1 = sum exp(x - M)`` and ``S2 = sum exp(x - M) * x`` over the full
    (TP-gathered) vocabulary.  Never mutates ``logits``: the fused CE that
    follows consumes them in place.  Both collectives sit outside the chunk
    loops, so every TP rank issues exactly two regardless of its chunk count.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    num_tokens = logits.size(0)
    chunk = _entropy_chunk_rows(logits)
    inv_t = 1.0 / temperature

    logits_max = torch.empty(num_tokens, 1, dtype=torch.float32, device=logits.device)
    for start in range(0, num_tokens, chunk):
        logits_max[start:start + chunk] = logits[start:start + chunk].amax(dim=-1, keepdim=True).float()
    if temperature != 1.0:
        logits_max.mul_(inv_t)
    dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=process_group)

    sums = torch.empty(num_tokens, 2, dtype=torch.float32, device=logits.device)
    for start in range(0, num_tokens, chunk):
        x = logits[start:start + chunk].float()
        if temperature != 1.0:
            x = x * inv_t
        e = torch.sub(x, logits_max[start:start + chunk])
        e.exp_()
        sums[start:start + chunk, 0] = e.sum(dim=-1)
        e.mul_(x)
        sums[start:start + chunk, 1] = e.sum(dim=-1)
        del e, x
    dist.all_reduce(sums, group=process_group)

    sum_exp, sum_exp_x = sums[:, 0:1], sums[:, 1:2]
    return (logits_max + sum_exp.log() - sum_exp_x / sum_exp).squeeze(-1)


def _logits_2d(output_tensor: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """``[T, V/tp]`` logits divided by ``temperature`` (a view when it is 1)."""
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    logits = output_tensor.squeeze(0) if output_tensor.dim() == 3 else output_tensor
    return logits / temperature if temperature != 1.0 else logits


def tree_compute_log_probs(idx: TreeLossIndex, output_tensor: torch.Tensor,
                           temperature: float = 1.0) -> list[torch.Tensor]:
    """Per-row response log-probs of a tree micro-batch.

    Like ``compute_log_probs`` it modifies the logits in place (fused CE).

    Args:
        idx: Loss index on the logits' device.
        output_tensor: LM-head output, ``[1, T, V/tp]`` or ``[T, V/tp]``.
        temperature: Sampling temperature.

    Returns:
        One ``[resp_len]`` tensor per row.
    """
    from megatron.core import parallel_state as mpu
    from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy

    tp_group = mpu.get_tensor_model_parallel_group()
    logits = _logits_2d(output_tensor, temperature).contiguous()
    # copy the branch-point rows before the in-place CE
    extra_logits = logits.index_select(0, idx.extra_nodes)
    node_logp = -fused_vocab_parallel_cross_entropy(
        logits.unsqueeze(1), idx.node_targets.unsqueeze(1), tp_group).squeeze(-1)
    parts = [node_logp]
    if idx.extra_nodes.numel():
        parts.append(-fused_vocab_parallel_cross_entropy(
            extra_logits.unsqueeze(1), idx.extra_targets.unsqueeze(1),
            tp_group).squeeze(-1))
    flat = torch.cat(parts).index_select(0, idx.logp_src)
    return list(flat.split(idx.resp_lens))


def tree_compute_entropy(idx: TreeLossIndex, output_tensor: torch.Tensor,
                         temperature: float = 1.0, metric_only: bool = False) -> list[torch.Tensor]:
    """Per-row response entropy of a tree micro-batch; ``output_tensor`` is left untouched.

    Args:
        idx: Loss index on the logits' device.
        output_tensor: LM-head output, ``[1, T, V/tp]`` or ``[T, V/tp]``.
        temperature: Sampling temperature.
        metric_only: No gradient; computed in token chunks without full-size temporaries.

    Returns:
        One ``[resp_len]`` tensor per row.
    """
    from megatron.core import parallel_state as mpu

    tp_group = mpu.get_tensor_model_parallel_group()
    if metric_only:
        ent = _vocab_parallel_entropy_chunked(_logits_2d(output_tensor), tp_group, temperature)
        return list(ent.index_select(0, idx.logit_nodes).split(idx.resp_lens))
    # response rows only; the copy also shields the logits from the in-place backward
    logits = _logits_2d(output_tensor, temperature).index_select(0, idx.logit_nodes)
    ent = _VocabParallelEntropy.apply(logits, tp_group)
    return list(ent.split(idx.resp_lens))

