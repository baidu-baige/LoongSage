"""Loss computation for the CODA training pipeline.

Main components
---------------
* ``_aggregate_loss``  – aggregates packed per-token values (loss, entropy,
  etc.) into a scalar, either as a per-trajectory mean or a per-token sum.
* ``loss_function``    – framework-level loss wrapper consumed by the
  Megatron forward-backward scheduler.
"""

from __future__ import annotations

import torch

from megatron.core import parallel_state as mpu
from omegaconf import DictConfig

from coda.algorithms.policy_loss import compute_policy_loss
from coda.algorithms.kl_policy import compute_approx_kl
from coda.backends.megatron.kl_ctx import KLCtx
from coda.backends.megatron.logits_utils import compute_entropy, compute_log_probs
from coda.algorithms.off_policy_seq_masking import compute_opsm_mask
from coda.utils.types import RolloutBatch
from coda.backends.megatron.cp_utils import (
    prepare_packed_seq_params,
    gather_and_slice_response,
)
from megatron.core.packed_seq_params import PackedSeqParams

# ════════════════════════════════════════════════════════════════════════
# Loss aggregation
# ════════════════════════════════════════════════════════════════════════

def _aggregate_loss(
    values: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    loss_agg_mode: str = "token-mean",
) -> torch.Tensor:
    """Aggregate per-trajectory loss tensors into a scalar.

    Each element of ``values`` is a 1-D per-token loss tensor for one trajectory;
    ``loss_masks`` are the corresponding token masks.

    * ``"seq-mean-token-mean"``: per-trajectory token mean, summed across trajectories.
    * ``"token-mean"``: sum of all masked tokens (Megatron divides by num_tokens externally).
    """
    if loss_agg_mode == "seq-mean-token-mean":
        return sum(
            (x_i * m).sum() / torch.clamp_min(m.sum(), 1)
            for x_i, m in zip(
                values, loss_masks, strict=False
            )
        )
    elif loss_agg_mode == "token-mean":
        loss_mask_tensor = torch.cat(loss_masks)
        values_tensor = torch.cat(values)
        return (loss_mask_tensor * values_tensor).sum()
    else:
        raise ValueError(f"Unknown loss_agg_mode: {loss_agg_mode!r}")


# ════════════════════════════════════════════════════════════════════════
# Framework loss function (Megatron callback)
# ════════════════════════════════════════════════════════════════════════


def loss_function(
    config: DictConfig,
    batch: RolloutBatch,
    packed_seq_params: PackedSeqParams,
    gkd_policy,
    output_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, list[str] | torch.Tensor]]:
    """Framework-level loss wrapper called inside the Megatron scheduler.

    Supports three modes based on OPD config:
    1. Pure GKD (gkd_ratio==1, pg_ratio==0): only KL loss, no RL.
    2. Mixed (0 < gkd_ratio < 1): weighted combination of RL + KL.
    3. Pure RL (gkd_ratio==0 or OPD disabled): standard policy gradient.

    Returns the Megatron-expected 3-tuple ``(loss, denominator, metrics_dict)``,
    where ``denominator`` is num_tokens (token-mean) or num_sequences (seq-mean).
    """
    cp_size = mpu.get_context_parallel_world_size()
    cp_partition_mode = config.megatron.model.cp_partition_mode
    loss_masks = batch["loss_masks"]
    loss_agg_mode = config.algorithm.loss_agg_mode

    num_tokens = torch.cat(loss_masks).sum().item()
    num_sequences = sum(1 for m in loss_masks if m.any())

    reported_metrics = {
        "num_tokens": num_tokens,
        "num_sequences": num_sequences,
    }

    collected = {}
    temperature = config.trainer.temperature

    # Always compute entropy: reported as a metric in every mode and reused as
    # the entropy-loss term when entropy_coef != 0.
    entropy_list = compute_entropy(
        batch["total_lengths"], batch["response_lengths"],
        output_tensor.clone(), temperature,
        cp_partition_mode=cp_partition_mode,
    )
    entropy_sum = (torch.cat(entropy_list) * torch.cat(loss_masks)).sum().item()
    reported_metrics["train/entropy"] = entropy_sum
    collected["entropy"] = entropy_list

    entropy_coef = config.algorithm.entropy_coef
    rl_loss = torch.tensor(0.0, device=output_tensor.device)
    gkd_loss = torch.tensor(0.0, device=output_tensor.device)

    gkd_ratio = config.opd.gkd_ratio if config.opd.enable else 0.0
    pure_gkd = gkd_ratio == 1.0
    if not pure_gkd:
        target_list = [
            torch.cat([t[1:], t.new_full((1,), 0)])
            for t in batch["tokens"]
        ]
        packed_targets, _ = prepare_packed_seq_params(
            target_list, cp_partition_mode=cp_partition_mode,
        )

        log_probs_list = compute_log_probs(
            packed_targets,
            batch["total_lengths"],
            batch["response_lengths"],
            # compute_log_probs mutates logits in-place; clone only when the KL
            # path below (gkd_ratio > 0) still needs the original output_tensor.
            output_tensor.clone() if gkd_ratio > 0 else output_tensor,
            temperature=temperature,
            cp_partition_mode=cp_partition_mode,
        )
        collected["log_probs"] = log_probs_list
        old_log_probs = (
            batch.get("rollout_log_probs")
            if config.trainer.use_rollout_log_probs
            else batch.get("old_log_probs")
        )
        is_weights = batch.get("is_weights")

        # compute_policy_loss uses the IS/M2PO-modified loss_masks for per-token
        # loss computation, with raw_loss_masks available via kwargs for loss
        # functions that need the original mask (e.g. for metric normalization).
        # When IS correction is disabled or uses ``action="clip"``, and M2PO is
        # disabled, raw and current masks hold the same values so this is a no-op.
        raw_loss_masks = batch.get("raw_loss_masks") or loss_masks

        # OPSM: compute per-sample mask but leave ``loss_masks`` untouched so
        # ``compute_policy_loss`` and all downstream aggregation see the full
        # pre-OPSM batch. The mask is applied to the per-token policy loss
        # below.
        opsm_metrics: dict[str, float] = {}
        opsm_masks: list[torch.Tensor] | None = None
        if config.algorithm.opsm.enable:
            opsm_masks, opsm_metrics = compute_opsm_mask(
                config,
                log_probs=collected["log_probs"],
                old_log_probs=old_log_probs,
                advantages=batch.get("advantages"),
                loss_masks=raw_loss_masks,
            )

        per_token_loss, rl_metrics = compute_policy_loss(
            config=config.algorithm,
            old_log_prob=old_log_probs,
            log_prob=collected["log_probs"],
            advantages=batch.get("advantages"),
            loss_masks=loss_masks,
            raw_loss_masks=raw_loss_masks,
        )

        # Apply OPSM mask AFTER compute_policy_loss: zero the per-token loss for
        # filtered sequences while keeping the original ``loss_masks`` as the
        # aggregation denominator.
        if opsm_masks is not None:
            per_token_loss = [l * m for l, m in zip(per_token_loss, opsm_masks)]

        if is_weights is not None:
            per_token_loss = [l * w for l, w in zip(per_token_loss, is_weights)]

        pg_loss = _aggregate_loss(per_token_loss, loss_masks, loss_agg_mode)
        rl_loss = pg_loss
        if entropy_coef != 0.0:
            entropy_loss = _aggregate_loss(
                collected["entropy"], loss_masks, loss_agg_mode,
            )
            rl_loss = pg_loss - entropy_coef * entropy_loss

        # Reference-model KL penalty. Cat all three log-prob lists, one
        # compute_approx_kl call, aggregate via _aggregate_loss, then add to rl_loss.
        ref_kl = config.algorithm.ref_kl
        if ref_kl.enable:
            lp_cat = torch.cat(collected["log_probs"])
            old_cat = torch.cat(old_log_probs)
            ref_cat = torch.cat(batch["ref_log_probs"])
            imp = torch.exp(lp_cat - old_cat) if ref_kl.use_unbiased_kl else None
            kl_cat = compute_approx_kl(lp_cat, ref_cat, ref_kl.kl_type, imp)
            kl_list = list(kl_cat.split([t.size(0) for t in collected["log_probs"]]))
            kl_loss = _aggregate_loss(kl_list, loss_masks, loss_agg_mode)
            rl_loss = rl_loss + ref_kl.coef * kl_loss
            reported_metrics["train/ref_loss"] = kl_loss.item()

        reported_metrics.update(rl_metrics)
        reported_metrics.update(opsm_metrics)
        reported_metrics["train/pg_loss"] = pg_loss.item()

    if gkd_ratio > 0:
        ctx = KLCtx(
            batch,
            output_tensor,
            packed_seq_params,
            temperature=temperature,
            cp_partition_mode=cp_partition_mode,
        )

        # Compute KL — returns CP-local full-seq per_token_kl plus a dict of
        # extra per-token metrics in the SAME CP-local full-seq format.
        per_token_kl, kl_metrics = gkd_policy.compute_kl(config, ctx)

        # All policies return CP-local full-seq; gather + response-slice uniformly.
        per_token_kl = gather_and_slice_response(
            per_token_kl, batch["total_lengths"], batch["response_lengths"],
            cp_partition_mode=cp_partition_mode,
        )

        gkd_loss = _aggregate_loss(per_token_kl, loss_masks, loss_agg_mode)
        reported_metrics["train/opd_loss"] = gkd_loss.item()

        # Loss-mask-weighted sums; _collect_loss_metrics divides each by
        # num_tokens to yield the per-token mean.  "kl" is the mean per-token KL
        # (distinct from gkd_loss, which follows loss_agg_mode).  Each kl_metric
        # shares per_token_kl's CP-local full-seq shape, so it goes through the
        # same gather + response-slice + masked-sum pipeline.
        loss_mask_cat = torch.cat(loss_masks)
        reported_metrics["train/opd_kl"] = (torch.cat(per_token_kl) * loss_mask_cat).sum().item()
        for name, per_token_values in kl_metrics.items():
            sliced = gather_and_slice_response(
                per_token_values, batch["total_lengths"], batch["response_lengths"],
                cp_partition_mode=cp_partition_mode,
            )
            reported_metrics[f"train/opd_{name}"] = (torch.cat(sliced) * loss_mask_cat).sum().item()

        # Entropy gap: |Σ H(teacher) - Σ H(student)|
        teacher_entropy_list = batch["teacher_entropy"]
        teacher_entropy_sum = (torch.cat(teacher_entropy_list) * torch.cat(loss_masks)).sum().item()
        reported_metrics["train/opd_teacher_entropy"] = teacher_entropy_sum
        reported_metrics["train/opd_entropy_gap"] = abs(teacher_entropy_sum - reported_metrics["train/entropy"])

    loss = (1 - gkd_ratio) * rl_loss + gkd_ratio * gkd_loss
    reported_metrics["train/loss"] = loss.item()

    # Megatron averages the loss over CP ranks; pre-multiply so the summed
    # per-token loss is preserved after that division.
    loss = loss * cp_size

    return (
        loss,
        torch.tensor(
            num_tokens if loss_agg_mode == "token-mean" else num_sequences,
            dtype=torch.int,
            device=output_tensor.device,
        ),
        reported_metrics,
    )
