"""Vocab-parallel logits primitives (entropy / log-probs / top-k).

Leaf module holding the TP/CP-aware functions that turn raw model logits into
per-token entropy, log-probs, and global top-k.  Kept separate from
:mod:`coda.backends.megatron.loss` so that :mod:`coda.backends.megatron.kl_ctx`
can consume them with a plain top-level import: previously ``loss`` imported
``KLCtx`` from ``kl_ctx`` while ``kl_ctx`` imported ``compute_entropy`` from
``loss``, a cycle broken only by a lazy in-method import.  With the primitives
here (a leaf depending only on Megatron parallel state +
:mod:`coda.backends.megatron.cp_utils`), both ``loss`` and ``kl_ctx`` import
from this module and the cycle disappears.

Abstraction note: :func:`compute_entropy` / :func:`compute_log_probs` are
full-pipeline (they CP all-gather + response-slice internally and return
per-trajectory lists), whereas :func:`compute_topk` is a lower-level pure TP
primitive on raw logits — CP aggregation stays with the caller (``TeacherCtx.topk``).
:func:`compute_topk_overlap` builds on :func:`compute_topk` for the student's
global top-k and leaves CP aggregation + loss-mask reduction to the caller
(it is returned as a per-token metric from ``compute_kl`` and reduced uniformly
in ``loss.loss_function``).
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from megatron.core import parallel_state as mpu
from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy

from coda.backends.megatron.cp_utils import CPPartitionMode, gather_and_slice_response


def compute_topk(
    logits: torch.Tensor, k: int, need_log_prob: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute global top-k across TP ranks, returning ``(values, global_idx)``.

    With ``need_log_prob=True`` (default) the returned values are global
    log_softmax probabilities (TP=1: direct; TP>1: distributed log_softmax with
    2 all-reduce ops).  With ``need_log_prob=False`` the log_softmax is skipped
    entirely: the top-k runs on the raw logits, so no vocab-wide temporaries
    (``shifted`` / ``exp`` / ``log_softmax_local``) are allocated and the 2
    all-reduce ops are saved.  The top-k *indices* are identical either way
    because log_softmax is order-preserving — callers that only need indices
    (e.g. top-k overlap) should pass ``False`` to avoid the memory + comm cost
    (the returned values are then raw logits, not probabilities).

    Does NOT modify logits in-place and does NOT aggregate across CP ranks.
    """
    tp_size = mpu.get_tensor_model_parallel_world_size()

    if tp_size == 1:
        scores = torch.log_softmax(logits, dim=-1) if need_log_prob else logits
        return scores.topk(k, dim=-1)

    tp_group = mpu.get_tensor_model_parallel_group()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    local_vocab_size = logits.size(-1)

    if need_log_prob:
        # Distributed log_softmax: compute correct denominator across all TP ranks
        local_max = logits.max(dim=-1, keepdim=True).values
        global_max = local_max.clone()
        dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=tp_group)

        shifted = logits - global_max  # temporary tensor, released after topk
        local_exp_sum = shifted.exp().sum(dim=-1, keepdim=True)
        global_exp_sum = local_exp_sum.clone()
        dist.all_reduce(global_exp_sum, op=dist.ReduceOp.SUM, group=tp_group)

        local_scores = shifted - global_exp_sum.log()
    else:
        # Indices-only: raw logits give the same top-k ordering, with no
        # vocab-wide temporaries and no all-reduce.
        local_scores = logits

    # Local top-k
    local_topk_vals, local_topk_idx = local_scores.topk(k, dim=-1)
    # Convert to global vocab indices
    local_topk_idx = local_topk_idx + tp_rank * local_vocab_size

    # All-gather across TP ranks
    all_topk_vals = [torch.empty_like(local_topk_vals) for _ in range(tp_size)]
    all_topk_idx = [torch.empty_like(local_topk_idx) for _ in range(tp_size)]
    dist.all_gather(all_topk_vals, local_topk_vals, group=tp_group)
    dist.all_gather(all_topk_idx, local_topk_idx, group=tp_group)

    # Concatenate and select global top-k
    all_vals = torch.cat(all_topk_vals, dim=-1)  # [seq_len, k * tp_size]
    all_idx = torch.cat(all_topk_idx, dim=-1)    # [seq_len, k * tp_size]

    global_topk_vals, select = all_vals.topk(k, dim=-1)
    global_topk_idx = all_idx.gather(-1, select)

    return global_topk_vals, global_topk_idx


@torch.no_grad()
def compute_topk_overlap(
    student_logits: list[torch.Tensor],
    teacher_topk_indices: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Per-token student/teacher top-k overlap ratio (CP-local, per-traj).

    Returns one ``[seq]`` overlap tensor per trajectory (values in ``[0, 1]``);
    CP aggregation + loss-mask reduction stay with the caller.
    """
    k = teacher_topk_indices[0].size(-1)

    per_token_overlap_list = []
    for s_logits, t_idx in zip(student_logits, teacher_topk_indices):
        # vals unused; need_log_prob=False -> indices only, no vocab-wide temps.
        _, student_topk_global = compute_topk(s_logits, k, need_log_prob=False)

        # Overlap via sort + adjacent comparison (memory efficient).
        combined = torch.cat([student_topk_global, t_idx], dim=-1)  # [seq, 2k]
        sorted_combined = combined.sort(dim=-1).values
        overlap = (sorted_combined[:, 1:] == sorted_combined[:, :-1]).sum(dim=-1).float() / k
        per_token_overlap_list.append(overlap)

    return per_token_overlap_list


@torch.compile(dynamic=True)
def _mul_reduce(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(dim=-1, keepdim=True)


# from https://github.com/volcengine/verl/blob/
#   0bdf7f469854815177e73dcfe9e420836c952e6e/verl/utils/megatron/tensor_parallel.py#L99
class _VocabParallelEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, vocab_parallel_logits: torch.Tensor, process_group: dist.ProcessGroup) -> torch.Tensor:
        """Compute per-token entropy over the TP-sharded vocabulary using numerically stable log-sum-exp."""
        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=process_group)
        normalized_vocab_parallel_logits = vocab_parallel_logits - logits_max
        normalized_exp_logits = normalized_vocab_parallel_logits.exp_()
        normalized_sum_exp_logits = normalized_exp_logits.sum(dim=-1, keepdim=True)
        dist.all_reduce(normalized_sum_exp_logits, group=process_group)
        softmax_logits = normalized_exp_logits.div_(normalized_sum_exp_logits)
        sum_softmax_times_logits = _mul_reduce(softmax_logits, vocab_parallel_logits)
        dist.all_reduce(sum_softmax_times_logits, group=process_group)
        entropy = logits_max + normalized_sum_exp_logits.log() - sum_softmax_times_logits
        ctx.save_for_backward(vocab_parallel_logits, softmax_logits, sum_softmax_times_logits)
        return entropy.squeeze(dim=-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Backpropagate entropy gradient: dH/dx_v = -p_v * (x_v - E[x]) * grad_output."""
        vocab_parallel_logits, softmax_logits, sum_softmax_times_logits = ctx.saved_tensors
        # reuse softmax_logits as grad
        vocab_parallel_logits.sub_(sum_softmax_times_logits)
        softmax_logits.mul_(vocab_parallel_logits)
        softmax_logits.mul_(grad_output.unsqueeze(dim=-1))
        # recover vocab_parallel_logits
        vocab_parallel_logits.add_(sum_softmax_times_logits)
        softmax_logits.mul_(-1)
        return softmax_logits, None


def compute_entropy(
    total_lengths: list[int],
    response_lengths: list[int],
    output_tensor: torch.Tensor,
    temperature: float = 1.0,
    cp_partition_mode: CPPartitionMode = "zigzag",
) -> list[torch.Tensor]:
    """Compute per-sample response-only entropy from model output.

    Handles CP all-gather + response slicing internally.
    WARNING: modifies logits in-place via _VocabParallelEntropy.
    If caller needs original logits afterward, pass output_tensor.clone().

    Args:
        cp_partition_mode: Must match the model's ``TransformerConfig.cp_partition_mode``.
    """
    logits = output_tensor.squeeze(0) if output_tensor.dim() == 3 else output_tensor
    if temperature != 1.0:
        logits = logits / temperature

    tp_group = mpu.get_tensor_model_parallel_group()
    logits = logits.contiguous()
    entropy = _VocabParallelEntropy.apply(logits, tp_group)

    return gather_and_slice_response([entropy], total_lengths, response_lengths, cp_partition_mode)


def compute_log_probs(
    packed_targets: torch.Tensor,
    total_lengths: list[int],
    response_lengths: list[int],
    output_tensor: torch.Tensor,
    temperature: float = 1.0,
    cp_partition_mode: CPPartitionMode = "zigzag",
) -> list[torch.Tensor]:
    """Compute per-trajectory response-only log-probs from model output.

    Performs CP all-gather + response slicing internally, returning
    response-only tensors.

    WARNING: modifies logits in-place via fused_vocab_parallel_cross_entropy.

    Args:
        cp_partition_mode: Must match the model's ``TransformerConfig.cp_partition_mode``.
    """
    logits = output_tensor.squeeze(0) if output_tensor.dim() == 3 else output_tensor
    if temperature != 1.0:
        logits = logits / temperature

    tp_group = mpu.get_tensor_model_parallel_group()
    logits = logits.contiguous()
    log_probs = -fused_vocab_parallel_cross_entropy(
        logits.unsqueeze(1), packed_targets.unsqueeze(1), tp_group
    ).squeeze(-1)

    return gather_and_slice_response([log_probs], total_lengths, response_lengths, cp_partition_mode)
