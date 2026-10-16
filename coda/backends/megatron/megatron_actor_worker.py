"""Megatron actor worker — the policy model trained by RL.

Extends :class:`~coda.backends.megatron.megatron_train_worker.MegatronTrainWorker`
with everything specific to the policy: old / ref log-probs, IS correction and
M2PO, advantages, the policy loss, OPD teacher lm_heads, HF export and weight
sync to the rollout engines.
"""

import logging
from typing import override

import torch

from megatron.core import parallel_state
from coda.transfer_mesh import ChannelMeta, create_sender_channel

from coda.backends.megatron.megatron_train_worker import MegatronTrainWorker
from coda.backends.megatron.checkpoint import async_calls, load_model_weights
from coda.utils.tensor_backuper import TensorBackuper
from coda.utils.memory_utils import print_memory, clear_memory
from coda.utils.checkpoint_utils import resolve_dist_ckpt_dir, get_hf_dir
from coda.utils.types import RolloutBatch
from coda.algorithms.advantage import compute_advantages
from coda.algorithms.is_correction import apply_is_correction
from coda.algorithms.second_moment_trust_policy_optimization import apply_m2po_masking
from coda.backends.megatron.data import (
    get_rollout_data,
    get_data_iterator,
)
from coda.backends.megatron.model import forward_only, train_minibatch, is_main_rank, reduce_dict
from coda.utils.tracking import time_tracker, track
from coda.algorithms.kl_policy import build_kl_policies
from coda.backends.megatron.teacher_lm_head import TeacherLMHeads

logger = logging.getLogger(__name__)


class MegatronActorWorker(MegatronTrainWorker):
    """Per-GPU Ray actor that trains the policy with Megatron-Core."""

    @classmethod
    def validate_config(cls, config):
        """Validate or fill the policy-specific config on top of the shared one."""
        config = super().validate_config(config)

        if config.trainer.use_rollout_routing_replay:
            config.megatron.model.moe_enable_routing_replay = True
        if config.trainer.use_fp32_lm_head:
            config.megatron.keep_fp32_weights["output_layer"] = True

        if config.algorithm.is_correction.enable and config.trainer.use_rollout_log_probs:
            raise ValueError(
                "is_correction.enable and trainer.use_rollout_log_probs are mutually exclusive. "
            )

        if config.algorithm.m2po.enable and config.trainer.use_rollout_log_probs:
            raise ValueError(
                "m2po.enable and trainer.use_rollout_log_probs are mutually exclusive. "
                "M2PO requires recomputed old_log_probs."
            )

        if config.algorithm.m2po.enable and config.algorithm.m2po.threshold < 0.0:
            raise ValueError(
                f"m2po.threshold must be non-negative, got {config.algorithm.m2po.threshold}"
            )

        assert config.algorithm.clip_ratio_c > 1.0, "clip_ratio_c must be greater than 1.0"

        if config.algorithm.ref_kl.enable:
            if config.algorithm.ref_kl.kl_type not in {"k1", "k2", "k3"}:
                raise ValueError(
                    "algorithm.ref_kl.kl_type must be one of: k1, k2, k3."
                )
            if not (config.ref_dist_ckpt_path or config.ref_hf_model_path):
                raise ValueError(
                    "algorithm.ref_kl.enable=True requires either "
                    "ref_dist_ckpt_path or ref_hf_model_path to be set."
                )

        return config

    @override
    def init(self, config, master_addr, master_port):
        """Initialize the shared Megatron state, then the policy-specific parts.

        After the base steps:
          9. Build KL policy instances and set up teacher lm_heads
          10. Set up frozen reference model for ref-KL (weight hot-swap)
        """
        super().init(config, master_addr, master_port)

        # Step 9: build KL policy instances and set up teacher lm_heads if an
        # active policy needs teacher logits (full_kl/full_jsd). The
        # TeacherLMHeads singleton owns the lm_head lifecycle (last PP stage only).
        self.kl_policies = self._init_kl_policies()

        if config.algorithm.ref_kl.enable:
            self._weights_backuper = TensorBackuper(
                source_getter=lambda: (
                    (f"chunk{i}.{name}", param)
                    for i, m in enumerate(self.model)
                    for name, param in m.named_parameters()
                ),
            )

        self._active_model_tag = "actor"

        # Step 10: set up the frozen reference model for ref-KL (weight hot-swap).
        self._setup_ref_model()

        logger.info(f"[Rank {self.rank}] MegatronActorWorker initialization complete")

    @override
    @torch.no_grad()
    def save_model(self, step, checkpoint=True, hf=False):
        """Save the distributed checkpoint and/or export HF weights.

        See :meth:`MegatronTrainWorker.save_model` for ``step=None``.
        """
        super().save_model(step, checkpoint=checkpoint, hf=hf)
        if step is None:
            return

        if hf:
            hf_dir = get_hf_dir(self.config.checkpoint_path, step)
            self.bridge.save_hf_pretrained(self.model, hf_dir)
            logger.info(f"[Rank {self.rank}] HF model exported for step={step}")

    @override
    @torch.no_grad()
    def onload(self):
        """Onload model / optimizer state, plus the OPD teacher lm_heads if any."""
        super().onload()
        lm_heads = TeacherLMHeads.get()
        if lm_heads is not None:
            lm_heads.onload(torch.cuda.current_device())
            clear_memory()

    @override
    @torch.no_grad()
    def offload(self, move_params: bool = False):
        """Offload model / optimizer state, plus the OPD teacher lm_heads if any."""
        lm_heads = TeacherLMHeads.get()
        if lm_heads is not None:
            lm_heads.offload()
        super().offload(move_params)

    @torch.no_grad()
    def _setup_ref_model(self):
        """Load the frozen reference model and snapshot both actor/ref weights.

        ``TensorBackuper`` stores named CPU snapshots for actor/ref weights.
        Only one tag is restored into the live GPU model at a time. After setup,
        the GPU holds "actor" and ``_active_model_tag`` is "actor".
        """
        ref_kl = self.config.algorithm.ref_kl
        if not ref_kl.enable:
            return

        self._weights_backuper.backup("actor")  # snapshot current student weights

        ref_ckpt = resolve_dist_ckpt_dir(
            self.config.ref_dist_ckpt_path, "ref_dist_ckpt_path"
        )
        if ref_ckpt:
            load_model_weights(self.model, ref_ckpt)
        elif self.config.ref_hf_model_path:
            self.bridge.load_hf_weights(self.model, self.config.ref_hf_model_path)
        else:
            raise ValueError(
                "algorithm.ref_kl.enable=True requires either "
                "ref_dist_ckpt_path or ref_hf_model_path to be set."
            )

        self._weights_backuper.backup("ref")     # snapshot ref weights
        self._weights_backuper.restore("actor")  # switch back to the student
        self._active_model_tag = "actor"
        logger.info(
            f"[Rank {self.rank}] Reference model set up for ref-KL "
            f"(source={'ckpt:' + ref_ckpt if ref_ckpt else 'hf:' + str(self.config.ref_hf_model_path)})"
        )


    @torch.no_grad()
    def switch(self, tag: str) -> None:
        """Hot-swap the GPU-resident weights to ``tag`` ("actor"/"ref").

        Only restores the requested tag (no backup-before-restore). The caller keeps
        the active snapshot fresh via an explicit ``backup("actor")`` after each
        optimizer step, so restoring never loses training progress. No-op when
        already active.
        """
        if self._active_model_tag == tag:
            return
        self._weights_backuper.restore(tag)
        self._active_model_tag = tag


    @override
    def train(self, step, rollout_data_ref):
        """Full train cycle: fetch data -> compute old log-probs -> advantages -> training loop.

        When OPD is enabled in pure-GKD mode, skips the forward-only pass and
        advantage computation.
        """
        torch.cuda.reset_peak_memory_stats()

        # run finalize_fn for pre checkpoint save
        async_calls.maybe_finalize_async_calls()
        rollout_data = get_rollout_data(rollout_data_ref)

        # Determine OPD mode
        opd_enabled = self.config.opd.enable
        gkd_ratio = self.config.opd.gkd_ratio if opd_enabled else 0.0
        pg_ratio = self.config.opd.pg_ratio if opd_enabled else 0.0
        pure_gkd = (gkd_ratio == 1.0)

        if not pure_gkd:
            for m in self.model:
                m.eval()
            # Optionally (re-)compute log-probs from the current policy
            if not self.config.trainer.use_rollout_log_probs:
                rollout_data.update(self._compute_log_probs(step, rollout_data))

            if self.config.algorithm.ref_kl.enable:
                rollout_data.update(self._compute_ref_log_probs(step, rollout_data))

            if self.config.algorithm.m2po.enable:
                rollout_data.update(self._compute_m2po(step, rollout_data))

            rollout_data.update(self._compute_is_correction(step, rollout_data))

            advantages, adv_metrics = compute_advantages(self.config.algorithm, rollout_data)
            rollout_data["adv_metrics"] = adv_metrics

            if pg_ratio > 0:
                self._apply_opd_kl_to_advantages(rollout_data, advantages)
                self._free_pg_fields(rollout_data, gkd_ratio)

            rollout_data["advantages"] = advantages

        # Train
        for m in self.model:
            m.train()
        self._train_actor(step, rollout_data)

        # Refresh the CPU "actor" snapshot after the optimizer step for ref-KL
        # weight switching; optionally move the reference toward the current
        # policy (moving-ref).
        ref_kl = self.config.algorithm.ref_kl
        if ref_kl.enable:
            self._weights_backuper.backup("actor")

            if ref_kl.update_interval > 0 and step % ref_kl.update_interval == 0:
                self._weights_backuper.backup("ref")
                logger.info(f"[Rank {self.rank}] Ref-model update at step {step}")


    def _compute_token_metric(
        self,
        name: str,
        values: list[torch.Tensor],
        masks: list[torch.Tensor],
    ) -> float:
        """Compute a per-token metric reduced and normalized according to loss_agg_mode."""
        loss_agg_mode = self.config.algorithm.loss_agg_mode
        device = values[0].device

        metric_sum = torch.tensor(0.0, device=device)
        num_tokens = torch.tensor(0.0, device=device)
        num_seqs = torch.tensor(0.0, device=device)
        for val_i, mask_i in zip(values, masks):
            num_tokens += mask_i.sum()
            if mask_i.any():
                num_seqs += 1
            masked_val = val_i * mask_i
            if loss_agg_mode == "seq-mean-token-mean":
                metric_sum += masked_val.sum() / torch.clamp_min(mask_i.sum(), 1)
            else:
                metric_sum += masked_val.sum()

        raw = {
            name: metric_sum.item(),
            "num_tokens": num_tokens.item(),
            "num_seqs": num_seqs.item(),
        }
        raw = reduce_dict(raw, parallel_state.get_data_parallel_group())
        num_tokens_r = max(raw.pop("num_tokens"), 1)
        num_seqs_r = max(raw.pop("num_seqs"), 1)
        denom = num_seqs_r if loss_agg_mode == "seq-mean-token-mean" else num_tokens_r

        return raw[name] / float(denom)


    @torch.no_grad()
    def _apply_opd_kl_to_advantages(self, rollout_data: RolloutBatch, advantages: list[torch.Tensor]):
        """PG-Style KL penalty: advantages -= pg_ratio * per_token_kl.

        per_token_kl is pre-computed in forward_only via compute_forward_only_outputs.
        """
        if not parallel_state.is_pipeline_last_stage():
            return

        per_token_kl = rollout_data["per_token_kl"]
        pg_ratio = self.config.opd.pg_ratio
        for i in range(len(advantages)):
            advantages[i] = advantages[i] - pg_ratio * per_token_kl[i]

        kl_val = self._compute_token_metric("pg_opd_kl", per_token_kl, rollout_data["loss_masks"])
        rollout_data.setdefault("kl_metrics", {})["train/opd_pg_kl"] = kl_val


    def _free_pg_fields(self, rollout_data: RolloutBatch, gkd_ratio: float):
        """Free PG-only fields after advantages are computed to save memory."""
        rollout_data.pop("per_token_kl", None)
        # GKD off → all teacher_* fields are dead; sweep by the transport prefix
        # so this can never drift from the producer field names.
        if gkd_ratio == 0:
            for k in [k for k in rollout_data if k.startswith("teacher_")]:
                rollout_data.pop(k, None)


    def _init_kl_policies(self):
        """Build KL policy instances and set up teacher lm_heads if needed.

        Only runs on the last PP stage (where hidden states / output_layer live).
        Builds the :class:`TeacherLMHeads` singleton when an active policy needs
        teacher full logits. Returns the policies dict stored as ``self.kl_policies``.
        """
        if not self.config.opd.enable or not parallel_state.is_pipeline_last_stage():
            return {}
        policies = build_kl_policies(self.config)
        if any(p.need_teacher_logits() for p in policies.values()):
            TeacherLMHeads.setup(self.config)
        return policies


    @override
    @torch.no_grad()
    def update_weights(self, channel_meta: ChannelMeta):
        """Update weights"""
        if self.params_offloaded:
            self._onload_model(move_params=True, move_grads=False)
            self.params_offloaded = False
            clear_memory()
        print_memory("before send huggingface weights")
        send_channel = create_sender_channel(channel_meta)
        for name, tensor, in self.bridge.export_hf_weights(self.model):
            send_channel.send((name, tensor))
        send_channel.send(None, flush=True)
        if self.offloaded:
            self._offload_model(move_params=True, move_grads=False)
            self.params_offloaded = True
        clear_memory()
        print_memory("after send huggingface weights")


    @time_tracker("compute_log_probs")
    def _compute_log_probs(
            self,
            step,
            rollout_data: RolloutBatch,
    ) -> dict[str, list[torch.Tensor]]:
        """Compute behavior-policy log-probs for the rollout batch.

        Uses the same mini-batch splitting as ``_train_actor`` to ensure consistent
        micro-batch boundaries across forward-only and training passes.
        When a PG KL policy is active, also returns per-token PG KL.

        ``train`` runs once per step in every mode, before any optimizer step,
        so the live actor weights are exactly the policy last pushed to rollout.
        """
        collected_all = self._forward_only_log_probs(rollout_data, step, pg_policy=self.kl_policies.get("pg"))

        output = {"old_log_probs": collected_all.get("log_probs")}
        if "per_token_kl" in collected_all:
            output["per_token_kl"] = collected_all["per_token_kl"]
        return output


    def _forward_only_log_probs(
            self,
            rollout_data: RolloutBatch,
            step,
            pg_policy=None,
    ) -> dict[str, list[torch.Tensor]]:
        """Run the forward-only pass and collect per-microbatch outputs.

        Shared by :meth:`_compute_log_probs` (current policy) and
        :meth:`_compute_ref_log_probs` (reference policy); the caller controls
        which model weights are live and which ``pg_policy`` (if any) is applied.
        Recovers the original Segment-row order under dynamic / padded batching.
        """
        data_iterator, num_microbatches_list = get_data_iterator(
            self.config.trainer, self.model, rollout_data
        )

        collected_all: dict[str, list[torch.Tensor]] = {}
        for num_microbatches in num_microbatches_list:
            forward_data_store = forward_only(
                self.config, self.model,
                data_iterator=data_iterator,
                num_microbatches=num_microbatches,
                pg_policy=pg_policy,
            )
            for mb in forward_data_store:
                for key, values in mb.items():
                    collected_all.setdefault(key, []).extend(values)

        # Undo the dynamic-batch permutation / -1 padding so outputs align with
        # the Segment rows of rollout_data.
        self._reorder_to_rows(collected_all, data_iterator, len(rollout_data["tokens"]))

        return collected_all


    @torch.no_grad()
    @time_tracker("compute_ref_log_probs")
    def _compute_ref_log_probs(self, step, rollout_data: RolloutBatch) -> dict[str, list[torch.Tensor]]:
        """Compute reference-model log-probs via the forward-only pass.

        Swaps the GPU-resident weights to "ref", runs the forward-only log-prob
        pass (no PG policy), then swaps back to "actor". Eval mode is set by the
        caller.
        """
        self.switch("ref")
        collected_all = self._forward_only_log_probs(rollout_data, step, pg_policy=None)
        self.switch("actor")
        return {"ref_log_probs": collected_all.get("log_probs")}


    @torch.no_grad()
    def _compute_m2po(self, step, rollout_data: RolloutBatch):
        """Pre-compute M2PO-updated loss masks before IS correction."""
        old_log_probs = rollout_data.get("old_log_probs")
        # only last pp stage needs to compute m2po
        if old_log_probs is None:
            return {}
        
        rollout_log_probs = rollout_data.get("rollout_log_probs")
        loss_masks = rollout_data.get("loss_masks")
        raw_loss_masks = rollout_data.get("raw_loss_masks")

        new_loss_masks, m2po_metrics = apply_m2po_masking(
            self.config.algorithm.m2po,
            raw_loss_masks,
            old_log_probs,
            rollout_log_probs,
        )

        m2po_metrics = reduce_dict(
            m2po_metrics, parallel_state.get_data_parallel_group()
        )
        num_tokens = m2po_metrics.pop("valid_tokens")
        if num_tokens <= 0:
            logger.warning(
                "[Rank %s] step=%s M2PO: no valid tokens across all DP ranks after reduce; "
                "metrics may be meaningless",
                self.rank,
                step,
            )
            num_tokens = 1
        m2po_clip_count = m2po_metrics.pop("clip_count", 0.0)
        m2po_metrics = {
            f"train/m2po_{k}": v / float(num_tokens)
            for k, v in m2po_metrics.items()
        }
        m2po_metrics["train/m2po_clip_ratio"] = m2po_clip_count / float(num_tokens)
        return {
            "raw_loss_masks": raw_loss_masks,
            "loss_masks": [
                new_mask * loss_mask
                for new_mask, loss_mask in zip(new_loss_masks, loss_masks, strict=True)
            ],
            "m2po_metrics": m2po_metrics,
        }


    @torch.no_grad()
    def _compute_is_correction(self, step, rollout_data: RolloutBatch):
        """Pre-compute IS correction weights and updated loss masks.  """
        old_log_probs = rollout_data.get("old_log_probs")
        # only last pp stage needs to compute is_correction
        if old_log_probs is None:
            return {}

        rollout_log_probs = rollout_data.get("rollout_log_probs")
        loss_masks = rollout_data.get("loss_masks")
        raw_loss_masks = rollout_data.get("raw_loss_masks")

        # ``raw_loss_masks`` is the pre-IS snapshot taken in ``get_rollout_data``,
        # so OPSM and compute_policy_loss can operate on the original mask.
        is_weights, new_loss_masks, is_metrics = apply_is_correction(
            self.config.algorithm.is_correction,
            raw_loss_masks,
            old_log_probs,
            rollout_log_probs,
            loss_agg_mode=self.config.algorithm.loss_agg_mode,
        )
        is_metrics = reduce_dict(is_metrics, parallel_state.get_data_parallel_group())
        num_tokens = max(is_metrics.pop("num_tokens"), 1)
        num_seqs = max(is_metrics.pop("num_seqs"), 1)
        kl_denom = num_seqs if self.config.algorithm.loss_agg_mode == "seq-mean-token-mean" else num_tokens
        approx_k3_kl = is_metrics.pop("train/is_approx_k3_kl") / float(kl_denom)
        logprob_abs_diff = is_metrics.pop("train/is_logprob_abs_diff") / float(kl_denom)
        for k in list(is_metrics.keys()):
            is_metrics[k] = is_metrics[k] / float(num_tokens)
        is_metrics["train/is_approx_k3_kl"] = approx_k3_kl
        is_metrics["train/is_logprob_abs_diff"] = logprob_abs_diff

        # Always report metrics; only apply weights/masks to loss when enabled
        result = {"is_metrics": is_metrics}
        if self.config.algorithm.is_correction.enable:
            result["is_weights"] = is_weights
            result["loss_masks"] = [
                new_mask * loss_mask
                for new_mask, loss_mask in zip(new_loss_masks, loss_masks, strict=True)
            ]
        return result


    @time_tracker("train_actor")
    def _train_actor(self, step, rollout_data: RolloutBatch):
        """Run training over a rollout consisting of multiple mini batches.

        ``train_minibatch`` is invoked once per mini-batch.

        Args:
            step: Global training step identifier.
            rollout_data: The complete rollout batch for this DP rank.
        """
        data_iterator, num_microbatches = get_data_iterator(
            self.config.trainer, self.model, rollout_data
        )
        global_padding_count, global_padding_ratio = (
            data_iterator[0].get_global_padding_stats()
        )

        avg_metrics: dict[str, float] = {}
        n_minibatches = len(num_microbatches)
        for index in range(n_minibatches):
            is_last_minibatch = index == n_minibatches - 1
            metrics = train_minibatch(
                self.config,
                step,
                data_iterator,
                self.model,
                self.optimizer,
                self.scheduler,
                num_microbatches=num_microbatches[index],
                gkd_policy=self.kl_policies.get("gkd"),
            )
            if is_main_rank():
                for k, v in metrics.items():
                    avg_metrics[k] = avg_metrics.get(k, 0.0) + v
                logger.info(f"[Rank {self.rank}] step: {step}, minibatch: {index}, metrics: {metrics}")
                if not is_last_minibatch:
                    continue
                # Average metrics across minibatches and track
                for k in avg_metrics:
                    avg_metrics[k] /= n_minibatches            
                avg_metrics.update(rollout_data.get("m2po_metrics", {}))
                avg_metrics.update(rollout_data.get("is_metrics", {}))
                avg_metrics.update(rollout_data.get("kl_metrics", {}))
                avg_metrics.update(rollout_data.get("adv_metrics", {}))
                avg_metrics["train/segment_padding_count"] = global_padding_count
                avg_metrics["train/segment_padding_ratio"] = global_padding_ratio
                avg_metrics["perf/train_memory_allocated_max"] = \
                    round(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024), 2)
                avg_metrics["perf/train_memory_reserved_max"] = \
                    round(torch.cuda.max_memory_reserved() / (1024 * 1024 * 1024), 2)
                logger.info(f"[Rank {self.rank}] step: {step}, avg_metrics: {avg_metrics}")
                track(avg_metrics, step)
