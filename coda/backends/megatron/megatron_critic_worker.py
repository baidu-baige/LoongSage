"""Megatron critic worker — the PPO value model.

The critic is a separate Megatron model (own parameters, optimizer, LR
scheduler and checkpoint) that always shares the actor's GPUs: the
:class:`~coda.controller.train_manager.TrainManager` pins every critic rank onto
the bundle of the matching actor rank and alternates the two pools on the GPU
with onload / offload. It uses the actor's ``megatron.model`` /
``megatron.ddp_config`` parallelism, so both pools have the same DP layout and
read the same rollout shard per DP rank.

Per step the critic runs before the actor:

1. forward-only pass -> ``old_values`` (pre-update critic),
2. GAE -> ``advantages`` and ``returns``,
3. value update on ``returns``,
4. return the advantages; the manager hands them to the actor through
   ``Box.critic_ref`` and the actor's ``gae`` advantage estimator reads them.

The value head, the value output closure, GAE and the value loss are the PPO
algorithm part and are filled in separately; the methods marked below raise
``NotImplementedError`` until then.
"""

import logging
from typing import override

import ray
import torch

from megatron.core import parallel_state

from coda.backends.megatron.megatron_train_worker import MegatronTrainWorker
from coda.backends.megatron.data import get_rollout_data, get_data_iterator
from coda.backends.megatron.model import forward_only, train_minibatch, is_main_rank
from coda.utils.types import RolloutBatch
from coda.utils.tracking import time_tracker, track

logger = logging.getLogger(__name__)

def compute_forward_only_values(packed_targets, batch, packed_seq_params, config, pg_policy, output_tensor):
    """``forward_only`` output closure collecting per-response-token values.

    Algorithm part (value-head output -> per-row response values); not
    implemented yet.
    """
    raise NotImplementedError("critic value output closure is part of the PPO algorithm round")


def value_loss_function(config, batch, packed_seq_params, gkd_policy, output_tensor):
    """``train_minibatch`` loss closure for the (clipped) value loss.

    Algorithm part; must follow the ``loss_function`` contract
    ``(loss, denom, metrics)`` with ``num_tokens`` / ``num_sequences`` /
    ``train/loss`` in ``metrics``. Not implemented yet.
    """
    raise NotImplementedError("critic value loss is part of the PPO algorithm round")


class MegatronCriticWorker(MegatronTrainWorker):
    """Per-GPU Ray actor that trains the PPO value model with Megatron-Core."""

    CKPT_DIR_NAME = "critic_dist_ckpt"

    @classmethod
    def validate_config(cls, config):
        """Shared Megatron validation, plus the critic's own optimizer / scheduler.

        ``critic.optimizer`` / ``critic.scheduler`` are configured independently
        of the actor's ``megatron.optimizer`` / ``megatron.scheduler``; only
        their derived fields are filled, from the critic's own lr /
        weight_decay.
        """
        config = super().validate_config(config)
        cls._finalize_optim_sched(config.critic.optimizer, config.critic.scheduler, config)
        return config

    @override
    def init(self, config, master_addr, master_port):
        """Rewrite this process's config for the critic, then run the shared init.

        Ray hands every worker process its own deserialized copy of the config,
        so these rewrites never reach the driver or the actor workers:

        * the shared init builds the optimizer / scheduler from
          ``megatron.optimizer`` / ``megatron.scheduler``, so point them at the
          critic's (already finalized on the driver by :meth:`validate_config`);
        * MoE routing replay reproduces the experts the *rollout policy* picked
          so train-time log-probs match inference; the critic is a different,
          separately trained model, so replaying those routes is meaningless.
        * MTP is a next-tokens prediction head for the policy and reuses
          ``output_layer``, which the critic replaces with the value head, so
          the critic is built without MTP layers (as verl does). The override
          is applied to the provider before ``finalize()``.
        """
        config.megatron.optimizer = config.critic.optimizer
        config.megatron.scheduler = config.critic.scheduler
        config.trainer.use_rollout_routing_replay = False
        config.megatron.model.moe_enable_routing_replay = False
        config.megatron.model.mtp_num_layers = None

        super().init(config, master_addr, master_port)
        logger.info(f"[Rank {self.rank}] MegatronCriticWorker initialization complete")

    @override
    def _configure_provider(self, provider) -> None:
        """Replace the LM head with a hidden->1 value head before DDP wrapping.

        Algorithm part. Must use ``provider.register_pre_wrap_hook`` (appends)
        rather than the ``pre_wrap_hook=`` argument, which would replace the
        hook that loads the HF weights.
        """
        raise NotImplementedError("critic value head is part of the PPO algorithm round")

    @override
    @torch.no_grad()
    def save_model(self, step, checkpoint=True, hf=False):
        """Save ``train_step_{step}/critic_dist_ckpt`` synchronously; no HF export.

        Saving is synchronous because the manager offloads the critic right
        after. The shared tracker is written by whichever model saves last in a
        step: the critic during warmup (``step <= critic_warmup``, the actor is
        not saved), the actor afterwards.
        """
        if step is None or not checkpoint:
            return
        update_tracker = step <= int(self.config.critic.critic_warmup)
        self._save_dist_checkpoint(step, async_save=False, update_tracker=update_tracker)

    @override
    def train(self, step, rollout_data_ref):
        """One critic step; returns this rank's advantages for the actor.

        Returns:
            ``{dp_rank: ObjectRef}`` on the output rank of each DP group (last
            PP stage, TP rank 0, CP rank 0), where the ref resolves to
            ``{"critic_advantages": [cpu tensor per Segment row]}``; ``{}`` on
            every other rank.
        """
        torch.cuda.reset_peak_memory_stats()
        rollout_data = get_rollout_data(rollout_data_ref)

        for m in self.model:
            m.eval()
        rollout_data["old_values"] = self._compute_old_values(step, rollout_data)
        advantages, returns = self._compute_advantages_and_returns(rollout_data)
        rollout_data["advantages"] = advantages
        rollout_data["returns"] = returns

        for m in self.model:
            m.train()
        self._train_critic(step, rollout_data)

        # Advantages come from the pre-update old_values.
        return self._export_advantages(advantages)

    @time_tracker("compute_old_values")
    def _compute_old_values(self, step, rollout_data: RolloutBatch) -> list[torch.Tensor] | None:
        """Forward-only pass of the current critic, aligned to Segment rows."""
        data_iterator, num_microbatches_list = get_data_iterator(
            self.config.trainer, self.model, rollout_data
        )
        collected: dict[str, list[torch.Tensor]] = {}
        for num_microbatches in num_microbatches_list:
            forward_data_store = forward_only(
                self.config, self.model,
                data_iterator=data_iterator,
                num_microbatches=num_microbatches,
                output_fn=compute_forward_only_values,
            )
            for mb in forward_data_store:
                for key, values in mb.items():
                    collected.setdefault(key, []).extend(values)
        self._reorder_to_rows(collected, data_iterator, len(rollout_data["tokens"]))
        return collected.get("values")

    def _compute_advantages_and_returns(self, rollout_data: RolloutBatch):
        """GAE over ``old_values`` -> ``(advantages, returns)`` per Segment row.

        Algorithm part; not implemented yet.
        """
        raise NotImplementedError("GAE is part of the PPO algorithm round")

    @time_tracker("train_critic")
    def _train_critic(self, step, rollout_data: RolloutBatch):
        """Value update over the step's mini-batches; metrics go under ``critic/``."""
        data_iterator, num_microbatches = get_data_iterator(
            self.config.trainer, self.model, rollout_data
        )
        avg_metrics: dict[str, float] = {}
        n_minibatches = len(num_microbatches)
        for index in range(n_minibatches):
            metrics = train_minibatch(
                self.config,
                step,
                data_iterator,
                self.model,
                self.optimizer,
                self.scheduler,
                num_microbatches=num_microbatches[index],
                loss_fn=value_loss_function,
            )
            if is_main_rank():
                for k, v in metrics.items():
                    avg_metrics[k] = avg_metrics.get(k, 0.0) + v

        if is_main_rank():
            report = {
                k.replace("train/", "critic/", 1): v / n_minibatches for k, v in avg_metrics.items()
            }
            report["perf/critic_memory_allocated_max"] = \
                round(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024), 2)
            logger.info(f"[Rank {self.rank}] step: {step}, critic metrics: {report}")
            track(report, step)

    @staticmethod
    def _is_output_rank() -> bool:
        """One rank per DP group publishes the advantages."""
        return (
            parallel_state.is_pipeline_last_stage()
            and parallel_state.get_tensor_model_parallel_rank() == 0
            and parallel_state.get_context_parallel_rank() == 0
        )

    def _export_advantages(self, advantages) -> dict:
        if not self._is_output_rank():
            return {}
        payload = {"critic_advantages": [a.detach().cpu() for a in advantages]}
        return {parallel_state.get_data_parallel_rank(): ray.put(payload)}
