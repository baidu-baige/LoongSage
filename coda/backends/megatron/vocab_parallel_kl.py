"""Vocab-parallel KL divergence implementations using custom autograd.Function.

Provides TP-aware KL computation for OPD (Online Policy Distillation):
- VocabParallelTopkKL: Reverse KL with teacher top-k input
- VocabParallelFullKL: Reverse KL with teacher full logits
- VocabParallelTopkJSD: Jensen-Shannon divergence with teacher top-k input
- VocabParallelFullJSD: Jensen-Shannon divergence with teacher full logits

All functions expect student logits as TP-local partitions [N, V/tp].

The vocab-parallel autograd design (distributed softmax over TP partitions via
``calculate_logits_max`` + TP all-reduce, teacher top-k gather into the local
vocab range, and the renormalized-top-k RKL/JSD math) is adapted from
verl-recipe's GKD ``megatron_distill_losses.py`` (Apache-2.0):
https://github.com/verl-project/verl-recipe/blob/e7f889574b8301cc0f0fc1d57c6d67f31ffeb689/gkd/megatron/megatron_distill_losses.py
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
from megatron.core.fusions.fused_cross_entropy import calculate_logits_max
from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.core.tensor_parallel.utils import VocabUtility


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════

def _get_local_vocab_range(partition_vocab_size: int):
    rank = get_tensor_model_parallel_rank()
    world_size = get_tensor_model_parallel_world_size()
    return VocabUtility.vocab_range_from_per_partition_vocab_size(
        partition_vocab_size, rank, world_size
    )


# ══════════════════════════════════════════════════════════════
# 1) Top-K Reverse KL: KL(Q_hat || P_hat) on renormalized top-k
# ══════════════════════════════════════════════════════════════

class VocabParallelTopkKL(torch.autograd.Function):
    """Reverse KL(Q_hat || P_hat) on renormalized top-k support."""

    @staticmethod
    def forward(ctx, vocab_parallel_logits, target_topk_logps, target_topk_indices):
        """Compute per-token reverse KL on renormalized top-k support.

        1. Distributed softmax on student logits (TP all-reduce for max & sum).
        2. Map teacher top-k indices to local vocab partition, gather Q at those positions.
        3. Renormalize P_hat and Q_hat over the top-k support.
        4. Compute RKL = sum(Q_hat * log(Q_hat / P_hat)) with TP all-reduce.
        """
        eps = 1e-10
        tp_group = get_tensor_model_parallel_group()

        # Student softmax
        vocab_parallel_logits, logits_max = calculate_logits_max(vocab_parallel_logits)
        partition_vocab_size = vocab_parallel_logits.size(-1)
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=tp_group)
        vocab_parallel_logits -= logits_max.unsqueeze(-1)
        vocab_parallel_logits.exp_()
        sum_exp = vocab_parallel_logits.sum(dim=-1)
        dist.all_reduce(sum_exp, op=dist.ReduceOp.SUM, group=tp_group)
        Q_full = vocab_parallel_logits
        Q_full.div_(sum_exp.unsqueeze(-1))

        # Map topk indices to local partition
        vocab_start, vocab_end = _get_local_vocab_range(partition_vocab_size)
        topk_in_vocab = (target_topk_indices >= vocab_start) & (target_topk_indices < vocab_end)
        topk_idx_local = (target_topk_indices - vocab_start).clone()
        topk_idx_local[~topk_in_vocab] = 0

        # Teacher P (local pieces only)
        P_topk_part = torch.exp(target_topk_logps).clone()
        P_topk_part[~topk_in_vocab] = 0.0

        # Gather Q at topk positions
        topk = target_topk_indices.size(-1)
        Q_full_2d = Q_full.view(-1, partition_vocab_size)
        row = torch.arange(Q_full_2d.size(0), device=Q_full_2d.device)
        Q_topk = Q_full_2d[row.unsqueeze(-1), topk_idx_local.view(-1, topk)]
        Q_topk = Q_topk.view_as(target_topk_indices).clone()
        Q_topk[~topk_in_vocab] = 0.0

        # Renormalization sums
        P_sum = P_topk_part.sum(dim=-1).clone()
        dist.all_reduce(P_sum, op=dist.ReduceOp.SUM, group=tp_group)
        Q_sum = Q_topk.sum(dim=-1).clone()
        dist.all_reduce(Q_sum, op=dist.ReduceOp.SUM, group=tp_group)

        # Normalized distributions
        Q_hat = Q_topk / (Q_sum.unsqueeze(-1) + eps)
        P_hat = P_topk_part / (P_sum.unsqueeze(-1) + eps)

        # RKL = sum(Q_hat * (log Q_hat - log P_hat))
        per_token_rkl = torch.sum(Q_hat * (torch.log(Q_hat + eps) - torch.log(P_hat + eps)), dim=-1)
        per_token_rkl_out = per_token_rkl.clone()
        dist.all_reduce(per_token_rkl_out, op=dist.ReduceOp.SUM, group=tp_group)

        ctx.save_for_backward(Q_full, P_topk_part, topk_idx_local, Q_sum, P_sum)
        return per_token_rkl_out

    @staticmethod
    def backward(ctx, grad_output):
        """Gradient of RKL w.r.t. student logits (softmax Jacobian factored out).

        Uses identity: d/dz_j = Q_j/Z * (a_j - mean_a) where a_j = log(Q_hat_j) + 1 - log(P_hat_j).
        Scatters grad from top-k positions back to the local vocab partition.
        """
        eps = 1e-10
        Q_full, P_topk_part, topk_idx_local, Q_sum, P_sum = ctx.saved_tensors
        tp_group = get_tensor_model_parallel_group()

        partition_vocab_size = Q_full.size(-1)
        topk = topk_idx_local.size(-1)

        # Re-gather Q_topk
        Q_full_2d = Q_full.view(-1, partition_vocab_size)
        row = torch.arange(Q_full_2d.size(0), device=Q_full_2d.device)
        Q_topk = Q_full_2d[row.unsqueeze(-1), topk_idx_local.view(-1, topk)].view_as(P_topk_part)
        topk_mask = P_topk_part > 0
        Q_topk = torch.where(topk_mask, Q_topk, torch.zeros_like(Q_topk))

        Z = Q_sum.unsqueeze(-1) + eps
        T = P_sum.unsqueeze(-1) + eps
        Q_hat = Q_topk / Z
        P_hat = P_topk_part / T

        a = torch.log(Q_hat + eps) + 1.0 - torch.log(P_hat + eps)
        mean_a = torch.sum(Q_hat * a, dim=-1).clone()
        dist.all_reduce(mean_a, op=dist.ReduceOp.SUM, group=tp_group)

        # Gradient on topk positions, scatter to vocab partition
        grad_topk = (Q_topk / Z) * (a - mean_a.unsqueeze(-1))
        grad_input = torch.zeros_like(Q_full)
        grad_2d = grad_input.view(-1, partition_vocab_size)
        grad_topk_2d = torch.where(
            topk_mask.view(-1, topk),
            grad_topk.view(-1, topk),
            torch.zeros_like(grad_topk.view(-1, topk)),
        )
        idx_2d = topk_idx_local.view(-1, topk)
        row2 = torch.arange(grad_2d.size(0), device=grad_2d.device).unsqueeze(-1)
        grad_2d[row2, idx_2d] += grad_topk_2d

        grad_input.mul_(grad_output.unsqueeze(-1))
        return grad_input, None, None


# ══════════════════════════════════════════════════════════════
# 2) Full Reverse KL: KL(Q || P) on full local vocabulary partition
# ══════════════════════════════════════════════════════════════

class VocabParallelFullKL(torch.autograd.Function):
    """Reverse KL(Q || P) where both Q and P are TP-local full logits."""

    @staticmethod
    def forward(ctx, student_logits, teacher_logits):
        """Compute per-token reverse KL(Q||P) on full vocabulary.

        Memory-optimized: P is converted to log_P inplace to avoid holding
        both P and log_P simultaneously.
        """
        eps = 1e-20
        tp_group = get_tensor_model_parallel_group()

        # Student softmax (inplace)
        student_logits, s_max = calculate_logits_max(student_logits)
        dist.all_reduce(s_max, op=dist.ReduceOp.MAX, group=tp_group)
        student_logits -= s_max.unsqueeze(-1)
        student_logits.exp_()
        s_sum = student_logits.sum(dim=-1)
        dist.all_reduce(s_sum, op=dist.ReduceOp.SUM, group=tp_group)
        Q = student_logits
        Q.div_(s_sum.unsqueeze(-1))

        # Teacher softmax (inplace), then convert to log_P inplace
        teacher_logits, t_max = calculate_logits_max(teacher_logits)
        dist.all_reduce(t_max, op=dist.ReduceOp.MAX, group=tp_group)
        teacher_logits -= t_max.unsqueeze(-1)
        teacher_logits.exp_()
        t_sum = teacher_logits.sum(dim=-1)
        dist.all_reduce(t_sum, op=dist.ReduceOp.SUM, group=tp_group)
        teacher_logits.div_(t_sum.unsqueeze(-1))
        log_P = teacher_logits.clamp_min_(eps).log_()  # inplace: P -> log_P

        # RKL = sum(Q * log_Q) - sum(Q * log_P)
        # Split into two reductions; each only needs one transient [N, V/TP].
        # Step 1: cross-entropy term sum(Q * log_P) — transient Q*log_P freed after sum
        cross_ent = torch.sum(Q * log_P, dim=-1)  # [N]
        # Step 2: neg-entropy term sum(Q * log_Q).
        # torch.xlogy(Q, Q) = Q*log(Q), handles Q=0 correctly, single transient
        neg_entropy = torch.sum(torch.xlogy(Q, Q), dim=-1)  # [N]

        per_token_rkl = neg_entropy - cross_ent
        del neg_entropy, cross_ent
        per_token_rkl_out = per_token_rkl.clone()
        dist.all_reduce(per_token_rkl_out, op=dist.ReduceOp.SUM, group=tp_group)

        ctx.save_for_backward(Q, log_P)
        return per_token_rkl_out

    @staticmethod
    def backward(ctx, grad_output):
        """Gradient of RKL(Q||P) w.r.t. student logits.

        d/dz_j = Q_j * (a_j - mean_a), where a_j = log(Q_j) - log(P_j) + 1.
        """
        eps = 1e-20
        Q, log_P = ctx.saved_tensors
        tp_group = get_tensor_model_parallel_group()

        a = Q.clamp(min=eps).log() - log_P + 1.0
        mean_a = torch.sum(Q * a, dim=-1, keepdim=True)
        dist.all_reduce(mean_a, op=dist.ReduceOp.SUM, group=tp_group)

        grad_input = Q * (a - mean_a)
        grad_input.mul_(grad_output.unsqueeze(-1))
        return grad_input, None


# ══════════════════════════════════════════════════════════════
# 3) Top-K JSD: JSD_beta(P_topk, Q_full) with analytic rest term
# ══════════════════════════════════════════════════════════════

class VocabParallelTopkJSD(torch.autograd.Function):
    """Jensen-Shannon divergence with teacher top-k input."""

    @staticmethod
    def forward(ctx, vocab_parallel_logits, target_topk_logps, target_topk_indices, beta):
        """Compute per-token JSD_beta(P, Q) using teacher top-k logprobs.

        1. Distributed softmax on student logits; map teacher top-k to local partition.
        2. Compute mixture M = beta*P + (1-beta)*Q on top-k positions.
        3. Compute KL(P||M) and KL(Q||M) on top-k; add analytic rest term
           (non-topk tokens where M_j = (1-beta)*Q_j -> KL contribution = -log(1-beta)).
        4. JSD = beta*KL(P||M) + (1-beta)*KL(Q||M).
        """
        beta = min(max(float(beta), 1e-6), 1.0 - 1e-6)
        one_minus_beta = 1.0 - beta
        eps = 1e-10
        tp_group = get_tensor_model_parallel_group()

        # Student softmax
        vocab_parallel_logits, logits_max = calculate_logits_max(vocab_parallel_logits)
        partition_vocab_size = vocab_parallel_logits.size(-1)
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=tp_group)
        vocab_parallel_logits -= logits_max.unsqueeze(-1)
        vocab_parallel_logits.exp_()
        sum_exp = vocab_parallel_logits.sum(dim=-1)
        dist.all_reduce(sum_exp, op=dist.ReduceOp.SUM, group=tp_group)
        Q_full = vocab_parallel_logits
        Q_full.div_(sum_exp.unsqueeze(-1))

        # Map topk to local partition
        vocab_start, vocab_end = _get_local_vocab_range(partition_vocab_size)
        topk_in_vocab = (target_topk_indices >= vocab_start) & (target_topk_indices < vocab_end)
        topk_idx_local = (target_topk_indices - vocab_start).clone()
        topk_idx_local[~topk_in_vocab] = 0

        # Teacher P_topk
        P_topk = torch.exp(target_topk_logps).clone()
        P_topk[~topk_in_vocab] = 0.0
        logP_topk = target_topk_logps.clone()
        logP_topk[~topk_in_vocab] = 0.0

        # Gather Q at topk positions
        topk = target_topk_indices.size(-1)
        Q_full_2d = Q_full.view(-1, partition_vocab_size)
        row = torch.arange(Q_full_2d.size(0), device=Q_full_2d.device)
        Q_topk = Q_full_2d[row.unsqueeze(-1), topk_idx_local.view(-1, topk)]
        Q_topk = Q_topk.view_as(target_topk_indices).clone()
        Q_topk[~topk_in_vocab] = 0.0
        logQ_topk = torch.log(Q_topk + eps)

        # M = beta*P + (1-beta)*Q on topk
        M_topk = beta * P_topk + one_minus_beta * Q_topk
        logM_topk = torch.log(M_topk + eps)

        # KL(P||M) and KL(Q||M) on topk (local)
        kl_P_M_local = torch.sum(P_topk * (logP_topk - logM_topk), dim=-1)
        kl_Q_M_topk_local = torch.sum(Q_topk * (logQ_topk - logM_topk), dim=-1)

        # Analytic rest term: for non-topk, M_j = (1-beta)*Q_j
        Q_topk_sum = Q_topk.sum(dim=-1).clone()
        dist.all_reduce(Q_topk_sum, op=dist.ReduceOp.SUM, group=tp_group)
        log_one_minus_beta = math.log(one_minus_beta)
        kl_Q_M_rest = (1.0 - Q_topk_sum) * (-log_one_minus_beta)

        # JSD
        tmp = beta * kl_P_M_local + one_minus_beta * kl_Q_M_topk_local
        dist.all_reduce(tmp, op=dist.ReduceOp.SUM, group=tp_group)
        per_token_jsd = tmp + one_minus_beta * kl_Q_M_rest

        ctx.save_for_backward(Q_full, P_topk, topk_idx_local)
        ctx.beta = beta
        return per_token_jsd

    @staticmethod
    def backward(ctx, grad_output):
        """Gradient of JSD w.r.t. student logits.

        d/dz_j JSD = (1-beta) * Q_j * (A_j - KL(Q||M)), where A_j = log(Q_j/M_j).
        For non-topk positions, A_j = -log(1-beta) (since M_j = (1-beta)*Q_j).
        """
        Q_full, P_topk, topk_idx_local = ctx.saved_tensors
        beta = ctx.beta
        one_minus_beta = 1.0 - beta
        eps = 1e-10
        tp_group = get_tensor_model_parallel_group()

        partition_vocab_size = Q_full.size(-1)
        topk = topk_idx_local.size(-1)

        # Re-gather Q_topk
        Q_full_2d = Q_full.view(-1, partition_vocab_size)
        row = torch.arange(Q_full_2d.size(0), device=Q_full_2d.device)
        Q_topk = Q_full_2d[row.unsqueeze(-1), topk_idx_local.view(-1, topk)].view_as(P_topk)
        topk_mask = P_topk > 0
        Q_topk = torch.where(topk_mask, Q_topk, torch.zeros_like(Q_topk))

        M_topk = beta * P_topk + one_minus_beta * Q_topk
        logQ_topk = torch.log(Q_topk + eps)
        logM_topk = torch.log(M_topk + eps)

        # KL(Q||M) for baseline
        KL_Q_M_topk_local = torch.sum(Q_topk * (logQ_topk - logM_topk), dim=-1)
        Q_topk_sum = Q_topk.sum(dim=-1).clone()
        dist.all_reduce(Q_topk_sum, op=dist.ReduceOp.SUM, group=tp_group)
        log_one_minus_beta = math.log(one_minus_beta)
        KL_Q_M_rest = (1.0 - Q_topk_sum) * (-log_one_minus_beta)
        KL_Q_M_topk_global = KL_Q_M_topk_local.clone()
        dist.all_reduce(KL_Q_M_topk_global, op=dist.ReduceOp.SUM, group=tp_group)
        KL_Q_M = KL_Q_M_topk_global + KL_Q_M_rest

        # A_j = log(Q_j / M_j); for non-topk: -log(1-beta)
        A = torch.full_like(Q_full, -log_one_minus_beta)
        A_topk = logQ_topk - logM_topk
        A_2d = A.view(-1, partition_vocab_size)
        idx_2d = topk_idx_local.view(-1, topk)
        row2 = torch.arange(A_2d.size(0), device=A_2d.device).unsqueeze(-1)
        A_2d[row2, idx_2d] = A_topk.view(-1, topk)

        # d/dz JSD = (1-beta) * Q * (A - KL(Q||M))
        grad_input = one_minus_beta * Q_full * (A - KL_Q_M.unsqueeze(-1))
        grad_input.mul_(grad_output.unsqueeze(-1))
        return grad_input, None, None, None


# ══════════════════════════════════════════════════════════════
# 4) Full JSD: JSD_beta(P, Q) on full local vocabulary partition
# ══════════════════════════════════════════════════════════════

class VocabParallelFullJSD(torch.autograd.Function):
    """Jensen-Shannon divergence where both P and Q are TP-local full logits."""

    @staticmethod
    def forward(ctx, student_logits, teacher_logits, beta):
        """Compute per-token JSD_beta(P, Q) on full vocabulary.

        Memory-optimized: avoids holding log_Q, log_P, log_M simultaneously.
        """
        beta = min(max(float(beta), 1e-6), 1.0 - 1e-6)
        one_minus_beta = 1.0 - beta
        eps = 1e-20
        tp_group = get_tensor_model_parallel_group()

        # Student softmax (inplace)
        student_logits, s_max = calculate_logits_max(student_logits)
        dist.all_reduce(s_max, op=dist.ReduceOp.MAX, group=tp_group)
        student_logits -= s_max.unsqueeze(-1)
        student_logits.exp_()
        s_sum = student_logits.sum(dim=-1)
        dist.all_reduce(s_sum, op=dist.ReduceOp.SUM, group=tp_group)
        Q = student_logits
        Q.div_(s_sum.unsqueeze(-1))

        # Teacher softmax (inplace)
        teacher_logits, t_max = calculate_logits_max(teacher_logits)
        dist.all_reduce(t_max, op=dist.ReduceOp.MAX, group=tp_group)
        teacher_logits -= t_max.unsqueeze(-1)
        teacher_logits.exp_()
        t_sum = teacher_logits.sum(dim=-1)
        dist.all_reduce(t_sum, op=dist.ReduceOp.SUM, group=tp_group)
        P = teacher_logits
        P.div_(t_sum.unsqueeze(-1))

        # M = beta*P + (1-beta)*Q, compute log_M inplace on M
        M = beta * P + one_minus_beta * Q
        log_M = M.clamp_min_(eps).log_()  # inplace: M -> log_M

        # KL(P||M) = sum(P*log_P) - sum(P*log_M), split to reduce peak memory
        kl_P_M = torch.sum(torch.xlogy(P, P), dim=-1) - torch.sum(P * log_M, dim=-1)

        # KL(Q||M) = sum(Q*log_Q) - sum(Q*log_M)
        kl_Q_M = torch.sum(torch.xlogy(Q, Q), dim=-1) - torch.sum(Q * log_M, dim=-1)
        del log_M

        # JSD = beta*KL(P||M) + (1-beta)*KL(Q||M)
        per_token_jsd = beta * kl_P_M + one_minus_beta * kl_Q_M
        per_token_jsd_out = per_token_jsd.clone()
        dist.all_reduce(per_token_jsd_out, op=dist.ReduceOp.SUM, group=tp_group)

        ctx.save_for_backward(Q, P)
        ctx.beta = beta
        return per_token_jsd_out

    @staticmethod
    def backward(ctx, grad_output):
        """Gradient of JSD w.r.t. student logits.

        d/dz_j JSD = (1-beta) * Q_j * (A_j - KL(Q||M)), where A_j = log(Q_j/M_j).
        """
        Q, P = ctx.saved_tensors
        beta = ctx.beta
        one_minus_beta = 1.0 - beta
        eps = 1e-20
        tp_group = get_tensor_model_parallel_group()

        M = beta * P + one_minus_beta * Q
        log_Q = Q.clamp(min=eps).log()
        log_M = M.clamp_min_(eps).log_()  # inplace

        # KL(Q||M) for baseline
        kl_Q_M_local = torch.sum(Q * (log_Q - log_M), dim=-1)
        kl_Q_M = kl_Q_M_local.clone()
        dist.all_reduce(kl_Q_M, op=dist.ReduceOp.SUM, group=tp_group)

        # A_j = log(Q_j / M_j)
        A = log_Q - log_M
        del log_Q, log_M

        # d/dz JSD = (1-beta) * Q * (A - KL(Q||M))
        grad_input = one_minus_beta * Q * (A - kl_Q_M.unsqueeze(-1))
        grad_input.mul_(grad_output.unsqueeze(-1))
        return grad_input, None, None
