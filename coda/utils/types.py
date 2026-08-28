"""Shared type aliases for the CODA framework."""

from __future__ import annotations

import torch

# A dict-based batch produced along the rollout -> training path.
# In Megatron backend, several fields are converted to torch.Tensor lists on GPU
# before being consumed by data iterators
# (see coda.backends.megatron.data.get_rollout_data).
#
# Required fields (one entry per Segment row):
#   tokens          : list[torch.Tensor]   – token ids per Segment (variable length)
#   loss_masks      : list[torch.Tensor]   – final per-token train mask
#   response_lengths: list[int]            – number of response tokens per Segment
#   total_lengths   : list[int]            – total sequence length per Segment
#   prompt_id       : list[str]            – prompt group key for advantage norm,
#                                             replicated across a trajectory's Segments
#   trajectory_id   : list[int]            – trajectory id (local per shard,
#                                             contiguous), used to re-aggregate Segments
#                                             into their trajectory for advantage and
#                                             mini-batch cutting
#   rewards         : list[float]          – scalar trajectory reward, replicated per Segment
#
# Filled after advantage computation:
#   advantages      : list[torch.Tensor]   – per-token advantage
#   old_log_probs   : list[torch.Tensor]   – log-probs from the behaviour policy
#
# Filled after M2PO computation:
#   raw_loss_masks : list[torch.Tensor] – train mask before M2PO filtering
#   loss_masks     : list[torch.Tensor] – M2PO-updated final train mask
#   m2po_metrics   : dict[str, float]   – M2PO masking metrics
#
# Filled after is weights computation:
#   is_weights      : list[torch.Tensor]   – per-token importance-sampling
#                                             correction weight, multiplied into
#                                             the per-token loss
#   loss_masks      : list[torch.Tensor]   – per-token loss mask, possibly
#                                             zeroed at OOB tokens/seqs by
#                                             ``is_correction.action="mask"``
#   raw_loss_masks  : list[torch.Tensor]   – pre-IS loss mask snapshot, used
#                                             by OPSM and compute_policy_loss
#                                             so they see the original batch
#                                             (same object as loss_masks when
#                                             ``action="clip"`` or IS disabled)
#   is_metrics      : dict[str, float]     – IS correction metrics
#
# Optional:
#   rollout_log_probs       : list[torch.Tensor] | None
RolloutBatch = dict[
    str, list[torch.Tensor] | list[int] | list[float] | list[str] | dict[str, float] | None
]

# Data returned by a teacher worker to a train worker.
# Each teacher worker ``ray.put``s one TeacherData per train_dp_rank.
#
# Fields:
#   teacher_log_probs      : list[torch.Tensor] | None
#   teacher_topk_logprobs  : list[torch.Tensor] | None
#   teacher_topk_indices   : list[torch.Tensor] | None
#   teacher_hidden_states  : list[torch.Tensor] | None
#   seq_index              : list[int]  – position of each seq within the original
#                                        rollout_data_ref[train_dp_rank].ref
TeacherData = dict[str, list[torch.Tensor] | list[int] | None]

def to_torch_dtype(v: torch.dtype | str) -> torch.dtype:
    """Convert string dtype like "torch.bfloat16" to actual torch.dtype"""
    if isinstance(v, str):
        dtype_name = v.split(".")[-1]
        if not hasattr(torch, dtype_name):
            raise ValueError(f"Unsupported dtype string: {v}")
        v = getattr(torch, dtype_name)
    return v
