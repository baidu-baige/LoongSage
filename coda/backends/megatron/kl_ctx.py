"""Megatron distributed context objects for OPD KL policies.

:class:`TeacherCtx` and :class:`KLCtx` wrap the TP/CP/SP plumbing (vocab-parallel
log-prob/top-k, CP slicing, all-gather, reconstruction) so the math-only policy
layer in :mod:`coda.algorithms.kl_policy` never touches Megatron internals.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from megatron.core import parallel_state as mpu
from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy

from coda.backends.megatron.cp_utils import (
    CPPartitionMode,
    slice_cp_with_zigzag,
    slice_cp_packed,
    gather_and_reconstruct_cp,
    _contiguous_padded_length,
)
from coda.backends.megatron.logits_utils import (
    compute_entropy,
    compute_topk,
)

# NOTE: ``coda.backends.megatron.teacher_lm_head`` (TeacherLMHeads) is imported
# lazily inside :meth:`KLCtx.teacher_logits` to avoid importing the teacher
# lm_head stack until a full-vocab policy actually reconstructs logits.


# ════════════════════════════════════════════════════════════════════════
# Teacher-forward distributed primitives
# ════════════════════════════════════════════════════════════════════════

def _split_packed(values: torch.Tensor, total_lengths: list[int]) -> list[torch.Tensor]:
    """Split a packed tensor into per-seq list by total_lengths (full-seq)."""
    offset = 0
    result = []
    for total_length in total_lengths:
        result.append(values[offset:offset + total_length])
        offset += total_length
    return result


# ════════════════════════════════════════════════════════════════════════
# TeacherCtx — per-microbatch memoized teacher primitives
# ════════════════════════════════════════════════════════════════════════

class TeacherCtx:
    """Per-microbatch teacher-forward primitives with memoization.

    Exposes both a generic keyed cache (:meth:`get`) and five convenience
    primitives (:meth:`logits`, :meth:`topk`, :meth:`log_probs`,
    :meth:`hidden_states`, :meth:`entropy`).

    Immutability hazards:
      * :meth:`logits` is the immutable base tensor; when ``temperature == 1.0``
        it aliases the model output, so callers must never mutate it.
      * :meth:`log_probs` CLONES before the in-place
        ``fused_vocab_parallel_cross_entropy`` so call order does not matter.
      * :meth:`entropy` CLONES before the in-place ``_VocabParallelEntropy``.
      * :meth:`topk` only reads logits; the caller supplies ``k``.
    """

    def __init__(
        self,
        output_tensor: torch.Tensor,
        total_lengths: list[int],
        response_lengths: list[int],
        packed_targets: torch.Tensor | None = None,
        hidden: torch.Tensor | None = None,
        temperature: float = 1.0,
        cp_partition_mode: CPPartitionMode = "zigzag",
    ):
        self.output_tensor = output_tensor
        self.total_lengths = total_lengths
        self.response_lengths = response_lengths
        self.packed_targets = packed_targets
        self._hidden = hidden  # raw popped hook tensor [seq, batch, hidden] or None
        self.temperature = temperature
        # Partition mode is fixed for the microbatch: reused by every CP
        # all-gather / reconstruction below to match the model's THD layout.
        self.cp_partition_mode: CPPartitionMode = cp_partition_mode
        self._cache: dict = {}

    def get(self, key: str, compute_fn):
        """Memoize *compute_fn* under *key*."""
        if key not in self._cache:
            self._cache[key] = compute_fn()
        return self._cache[key]

    def logits(self) -> torch.Tensor:
        """Base (immutable) temperature-scaled logits."""
        def _compute():
            logits = self.output_tensor.squeeze(0) if self.output_tensor.dim() == 3 else self.output_tensor
            if self.temperature != 1.0:
                logits = logits / self.temperature
            return logits
        return self.get("logits", _compute)

    def topk(self, k: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Global top-k teacher log-probs/indices (CP-truncated to total_length).

        *k* is supplied by the caller (the policy's configured top-k); memoized
        per *k* so distinct policies asking for the same *k* share one compute.
        """
        def _compute():
            logits = self.logits()
            tp_size = mpu.get_tensor_model_parallel_world_size()
            vocabulary_size = logits.size(-1) * tp_size  # vocab_size * TP
            kk = min(k, vocabulary_size)
            topk_logprobs, topk_indices = compute_topk(logits, kk)
            cp_size = mpu.get_context_parallel_world_size()
            if cp_size > 1:
                # Gather across CP and truncate to total_length (remove CP padding)
                # so the student can re-slice with its own CP size.
                reconstructed_lp = gather_and_reconstruct_cp(
                    topk_logprobs, self.total_lengths, self.cp_partition_mode
                )
                reconstructed_idx = gather_and_reconstruct_cp(
                    topk_indices, self.total_lengths, self.cp_partition_mode
                )
                lp = [t[:tl] for t, tl in zip(reconstructed_lp, self.total_lengths)]
                idx = [t[:tl] for t, tl in zip(reconstructed_idx, self.total_lengths)]
            else:
                lp = _split_packed(topk_logprobs, self.total_lengths)
                idx = _split_packed(topk_indices, self.total_lengths)
            return lp, idx
        return self.get(f"topk_{k}", _compute)

    def log_probs(self) -> list[torch.Tensor]:
        """Per-trajectory CP-merged full-sequence teacher log-probs.

        Mirrors :meth:`topk` / :meth:`hidden_states`: gathered across CP and
        truncated to ``total_length`` (CP padding removed) so the student can
        re-slice with its own CP size.  CLONES logits first because
        ``fused_vocab_parallel_cross_entropy`` mutates them in-place.
        """
        def _compute():
            logits = self.logits().clone().contiguous()
            tp_group = mpu.get_tensor_model_parallel_group()
            log_probs = -fused_vocab_parallel_cross_entropy(
                logits.unsqueeze(1), self.packed_targets.unsqueeze(1), tp_group
            ).squeeze(-1)
            cp_size = mpu.get_context_parallel_world_size()
            if cp_size > 1:
                reconstructed = gather_and_reconstruct_cp(
                    log_probs, self.total_lengths, self.cp_partition_mode
                )
                return [t[:tl] for t, tl in zip(reconstructed, self.total_lengths)]
            return _split_packed(log_probs, self.total_lengths)
        return self.get("log_probs", _compute)

    def hidden_states(self) -> list[torch.Tensor]:
        """Per-trajectory teacher hidden states (SP all-gather + CP truncate)."""
        def _compute():
            hidden = self._hidden.squeeze(1)

            # With sequence parallelism, hidden is sharded across TP in the seq dim.
            # All-gather to reconstruct the full sequence before splitting by trajectory.
            tp_size = mpu.get_tensor_model_parallel_world_size()
            if tp_size > 1 and hidden.size(0) < sum(self.total_lengths):
                tp_group = mpu.get_tensor_model_parallel_group()
                gathered = [torch.empty_like(hidden) for _ in range(tp_size)]
                dist.all_gather(gathered, hidden.contiguous(), group=tp_group)
                hidden = torch.cat(gathered, dim=0)

            cp_size = mpu.get_context_parallel_world_size()
            if cp_size > 1:
                # Truncate to total_length (remove CP padding) so the student can
                # re-slice with its own CP size.
                trajectories = [
                    t[:tl] for t, tl in
                    zip(
                        gather_and_reconstruct_cp(
                            hidden, self.total_lengths, self.cp_partition_mode
                        ),
                        self.total_lengths,
                    )
                ]
            else:
                offset = 0
                trajectories = []
                for total_len in self.total_lengths:
                    trajectories.append(hidden[offset:offset + total_len])
                    offset += total_len
            return trajectories
        return self.get("hidden_states", _compute)

    def entropy(self) -> list[torch.Tensor]:
        """Per-trajectory teacher response entropy (CLONES logits first)."""
        def _compute():
            return compute_entropy(
                self.total_lengths, self.response_lengths, self.logits().clone(),
                temperature=1.0, cp_partition_mode=self.cp_partition_mode,
            )
        return self.get("entropy", _compute)


# ════════════════════════════════════════════════════════════════════════
# KLCtx — student/teacher logit access during compute_kl
# ════════════════════════════════════════════════════════════════════════

class KLCtx:
    """Student/teacher logit access shared by ``compute_kl`` in PG and GKD.

    Holds the per-microbatch student output plus the ``batch`` dict carrying
    teacher fields (``teacher_log_probs`` / ``teacher_topk_*`` /
    ``teacher_hidden_states``).  Reconstructed teacher logits live only in
    ``self._cache`` (never written back into ``batch``).
    """

    def __init__(
        self,
        batch: dict,
        output_tensor: torch.Tensor,
        packed_seq_params,
        temperature: float = 1.0,
        cp_partition_mode: CPPartitionMode = "zigzag",
    ):
        self.batch = batch
        self.output_tensor = output_tensor
        self.packed_seq_params = packed_seq_params
        self.total_lengths = batch["total_lengths"]
        self.response_lengths = batch["response_lengths"]
        self.temperature = temperature
        self.cp_partition_mode: CPPartitionMode = cp_partition_mode
        self._cache: dict = {}

    def student_logits(self) -> list[torch.Tensor]:
        """CP-local per-trajectory student logits (temperature-scaled)."""
        if "student_logits" not in self._cache:
            logits = self.output_tensor.squeeze(0) if self.output_tensor.dim() == 3 else self.output_tensor
            if self.temperature != 1.0:
                logits = logits / self.temperature
            cp_size = mpu.get_context_parallel_world_size()
            if self.cp_partition_mode == "zigzag":
                # Zigzag: cu_seqlens_q is pre-CP (scaled by cp_size), divide back.
                actual_cu_seqlens = self.packed_seq_params.cu_seqlens_q // cp_size
                self._cache["student_logits"] = [
                    logits[actual_cu_seqlens[i]:actual_cu_seqlens[i + 1]]
                    for i in range(len(self.total_lengths))
                ]
            elif self.cp_partition_mode == "contiguous":
                # Contiguous: the local slab is a flat narrow of the global packed buffer.
                # Compute intersection of each traj's global range with this rank's slab.
                cp_rank = mpu.get_context_parallel_rank()
                tp_size = mpu.get_tensor_model_parallel_world_size()
                padded_lens = [
                    _contiguous_padded_length(tl, cp_size, tp_size)
                    for tl in self.total_lengths
                ]
                total_padded = sum(padded_lens)
                local_len = total_padded // cp_size
                slab_start = cp_rank * local_len
                slab_end = slab_start + local_len

                local_lens = []
                cu = 0
                for pl in padded_lens:
                    traj_start = cu
                    traj_end = cu + pl
                    # Intersection of [traj_start, traj_end) and [slab_start, slab_end)
                    overlap = max(0, min(traj_end, slab_end) - max(traj_start, slab_start))
                    local_lens.append(overlap)
                    cu += pl

                self._cache["student_logits"] = list(
                    logits[:sum(local_lens)].split(local_lens)
                )
        return self._cache["student_logits"]

    def teacher_logits(self) -> list[torch.Tensor]:
        """CP-local per-trajectory teacher logits (reconstructed once).

        Reconstructs hidden→logits via the process-wide :class:`TeacherLMHeads`
        singleton on first access and caches the result, so the policy never
        sees lm_head details.  The consumed ``teacher_hidden_states`` are dropped
        from ``batch`` here (reconstruct is purely functional), since they are a
        framework-level transport detail of "needs teacher logits", not a
        policy-owned field.
        """
        if "teacher_logits" not in self._cache:
            from coda.backends.megatron.teacher_lm_head import TeacherLMHeads
            teacher_hidden_states = self.batch["teacher_hidden_states"]
            teacher_idx = self.batch["teacher_idx"]
            lm_heads = TeacherLMHeads.get()

            if self.cp_partition_mode == "zigzag":
                # Per-traj independent slice + lm_head (each rank gets same local_len per traj)
                logits_list = []
                for hidden, tidx in zip(teacher_hidden_states, teacher_idx):
                    hidden_cp = slice_cp_with_zigzag(hidden, pad_value=0)
                    head = lm_heads._by_idx[tidx]
                    # The head's dtype is config-stated (fp32 under
                    # trainer.use_fp32_lm_head) while the hidden states carry the
                    # teacher forward's dtype; F.linear will not promote for us.
                    logits_list.append(head(hidden_cp.to(head.weight.dtype)))
            else:
                # Contiguous: pack all teacher hiddens the same way as student,
                # apply lm_head on the local slab, then split by overlap lengths.
                packed_hidden, _ = slice_cp_packed(
                    teacher_hidden_states, "contiguous", pad_value=0,
                    pad_multiplier=0,
                )
                # Determine which teacher head to use per token in the local slab.
                # All trajectories share the same local slab layout as student_logits.
                # Apply lm_head per-segment based on overlap boundaries.
                student_lens = [t.size(0) for t in self.student_logits()]
                logits_list = []
                offset = 0
                for seg_len, tidx in zip(student_lens, teacher_idx):
                    head = lm_heads._by_idx[tidx]
                    if seg_len > 0:
                        seg = packed_hidden[offset:offset + seg_len]
                        logits_list.append(head(seg.to(head.weight.dtype)))
                    else:
                        # This traj has no tokens on this rank
                        logits_list.append(
                            packed_hidden.new_empty(0, head.out_features)
                        )
                    offset += seg_len

            self._cache["teacher_logits"] = [l / self.temperature for l in logits_list]
            del self.batch["teacher_hidden_states"]
        return self._cache["teacher_logits"]

    def student_log_probs(self) -> list[torch.Tensor]:
        """CP-local student log-probs (lazy from student_logits).

        Targets are the shifted tokens, CP-sliced to the local chunk so they
        align with the CP-local student logits.  ``torch.cat`` already produces a
        fresh tensor, so the in-place mutation by
        ``fused_vocab_parallel_cross_entropy`` does not touch the cached logits.
        """
        if "student_log_probs" not in self._cache:
            logits_list = self.student_logits()
            s_cat = torch.cat(logits_list)
            shifted = [
                torch.cat([toks[1:], toks.new_full((1,), 0)])
                for toks in self.batch["tokens"]
            ]
            targets, _ = slice_cp_packed(
                shifted, self.cp_partition_mode, pad_value=0, pad_multiplier=0
            )
            # Truncate tail-pad from targets to match logits length.
            targets = targets[:s_cat.size(0)]
            tp_group = mpu.get_tensor_model_parallel_group()
            lp_cat = -fused_vocab_parallel_cross_entropy(
                s_cat.unsqueeze(1), targets.unsqueeze(1), tp_group
            ).squeeze(-1)
            self._cache["student_log_probs"] = list(
                lp_cat.split([t.size(0) for t in logits_list])
            )
        return self._cache["student_log_probs"]

    def teacher_field(self, name: str) -> list[torch.Tensor]:
        """CP-local teacher primitive for *name*; CP-slices once, cached.

        The teacher forward emits CP-merged full sequences; re-slice to the
        local CP chunk so they align with the student.  Works for any plain
        per-token field (``teacher_log_probs`` / ``teacher_topk_logprobs`` /
        ``teacher_topk_indices``).  The *name* is supplied by the owning policy,
        so :class:`KLCtx` holds no teacher field-name literals.
        """
        if name not in self._cache:
            packed, _ = slice_cp_packed(
                self.batch[name], self.cp_partition_mode, pad_value=0, pad_multiplier=0
            )
            # Split back into per-traj list using student logits lengths as reference.
            self._cache[name] = list(
                packed.split([t.size(0) for t in self.student_logits()])
            )
        return self._cache[name]

    def free_teacher_logits(self) -> None:
        """Free reconstructed teacher logits from the cache."""
        self._cache.pop("teacher_logits", None)