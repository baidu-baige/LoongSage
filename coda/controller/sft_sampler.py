"""SFT sampler: the ``run_mode: sft`` counterpart of ``RolloutSampler``.

No inference engine runs.  Each Trajectory coming out of
``RolloutDataSourceWithBuffer`` carries the full conversation in ``prompt``; this
module tokenizes it, builds the loss mask and fills in the fields
``put_dp_shards_to_ray`` reads, so the rest of the RL training path is reused
unchanged.
"""

from __future__ import annotations

import logging
from typing import Any

from omegaconf import DictConfig

from coda.agentflow.tokenizer_manager import _load_chat_template, load_tokenizer
from coda.agentflow.trajectory_store import Segment, Trajectory, TrajectoryGroup, TrajectoryStatus
from coda.agentflow.loss_mask import LossMaskGenerator, decode_masked_spans, get_response_length
from coda.data_factory.data_source import DataSource

logger = logging.getLogger(__name__)


def validate_sft_config(config: DictConfig) -> None:
    """Reject settings that need rollout data or a reward the SFT path does not produce.

    Raises:
        ValueError: On the first incompatible setting.
    """
    algorithm = config.algorithm
    if algorithm.policy_loss != "sft" or algorithm.advantage_estimator != "sft":
        raise ValueError(
            "run_mode=sft requires algorithm.policy_loss=sft and algorithm.advantage_estimator=sft, "
            f"got {algorithm.policy_loss!r} / {algorithm.advantage_estimator!r}"
        )
    if not config.trainer.use_rollout_log_probs:
        raise ValueError(
            "run_mode=sft requires trainer.use_rollout_log_probs=true (skips the old-log-prob forward pass)"
        )
    disabled = {
        # No rollout engines share the GPUs, so colocate offload/onload is pure overhead.
        "colocate": config.colocate,
        "algorithm.ref_kl.enable": algorithm.ref_kl.enable,
        "algorithm.is_correction.enable": algorithm.is_correction.enable,
        "algorithm.m2po.enable": algorithm.m2po.enable,
        "algorithm.opsm.enable": algorithm.opsm.enable,
        "opd.enable": config.opd.enable,
        "fully_async.enable": config.fully_async.enable,
        "trainer.use_rollout_routing_replay": config.trainer.use_rollout_routing_replay,
    }
    enabled = [key for key, value in disabled.items() if value]
    if enabled:
        raise ValueError(f"run_mode=sft does not support: {', '.join(enabled)}")
    for idx, ds in enumerate(config.data_sources):
        if int(ds.num_trajectories_per_prompt) != 1:
            raise ValueError(
                f"run_mode=sft requires data_sources[{idx}].num_trajectories_per_prompt=1, "
                f"got {ds.num_trajectories_per_prompt}"
            )
    if int(config.rollout.eval.interval) > 0:
        raise ValueError("run_mode=sft does not run eval generation; set rollout.eval.interval<=0")


class SFTSampler:
    """Per-step SFT batch source: the SFT counterpart of ``RolloutSampler``.

    ``__call__(step)`` pulls ``num_prompts_per_step`` conversations from every
    data source and fills in tokens, loss masks and Segments without running
    inference.  Uses the same tokenizer, chat template and DeepSeek encoder
    wrapper as the RL AgentFlow so SFT and RL see identical prompts.
    """

    def __init__(self, config: DictConfig, datasources: list[DataSource], tokenizer: Any = None):
        self.config = config
        self.datasources = datasources
        tokenizer_cfg = config.agentflow.tokenizer
        if tokenizer is None:
            tokenizer = load_tokenizer(
                config.hf_model_path,
                _load_chat_template(tokenizer_cfg.custom_chat_template_path),
            )
        self.tokenizer = tokenizer
        sft_cfg = config.get("sft") or {}
        self.max_seq_len = sft_cfg.get("max_seq_len")
        self.mask_generator = LossMaskGenerator(
            tokenizer,
            sft_cfg.get("loss_mask"),
            dict(tokenizer_cfg.generation_prompt_kwargs or {}),
        )
        self._logged_sample = False
        logger.info(
            "[sft] loss mask preset=%s header=%r end=%r skip_prefixes=%r max_seq_len=%s",
            self.mask_generator.preset,
            self.mask_generator.assistant_header,
            self.mask_generator.end_marker,
            self.mask_generator.skip_prefixes,
            self.max_seq_len,
        )

    def __call__(self, step: int) -> list[TrajectoryGroup]:
        """Return this step's trainable groups from all data sources."""
        groups: list[TrajectoryGroup] = []
        for ds, ds_cfg in zip(self.datasources, self.config.data_sources, strict=True):
            groups += ds.get(int(ds_cfg.num_prompts_per_step), step)
        for group in groups:
            for traj in group.trajectories:
                self._tokenize_trajectory(traj)
        return groups

    def _tokenize_trajectory(self, traj: Trajectory) -> None:
        """Turn one dataset conversation into a trainable Trajectory, in place.

        Renders ``traj.prompt`` with the chat template, builds the assistant-turn
        loss mask and fills the fields the RL path would get from inference:
        ``tokens``, response-space ``loss_masks``, zero ``rollout_log_probs`` /
        ``token_rewards`` placeholders and a single trainable ``Segment``.

        Raises:
            ValueError: If ``traj.prompt`` is not a message list, or it renders
                to fewer than 2 tokens.
        """
        if not isinstance(traj.prompt, list):
            raise ValueError(
                f"SFT trajectory {traj.trajectory_id} needs a message list in dataset.input_key, "
                f"got {type(traj.prompt).__name__}"
            )
        tokens, mask = self.mask_generator.generate(traj.prompt, tools=traj.metadata.get("tools"))
        if self.max_seq_len and len(tokens) > self.max_seq_len:
            tokens, mask = tokens[: self.max_seq_len], mask[: self.max_seq_len]
        if len(tokens) < 2:
            raise ValueError(f"SFT trajectory {traj.trajectory_id} renders to fewer than 2 tokens")

        # The first token has no preceding logit, so at least one token is prompt.
        resp_len = min(get_response_length(mask), len(tokens) - 1)
        if resp_len == 0:
            # Nothing to learn (fully masked or truncated away): keep the sample
            # so DP/mini-batch counts stay fixed, but give it zero loss weight.
            resp_len = 1
            loss_masks = [0]
        else:
            loss_masks = mask[-resp_len:]

        traj.tokens = tokens
        traj.loss_masks = loss_masks
        traj.rollout_log_probs = [0.0] * resp_len
        traj.token_rewards = [0.0] * resp_len
        traj.rollout_weight_versions = [-1] * resp_len
        traj.segments = [
            Segment(token_start=0, token_end=len(tokens), logprob_start=0, logprob_end=resp_len)
        ]
        traj.reward = 0.0
        traj.status = TrajectoryStatus.COMPLETED

        if not self._logged_sample:
            self._logged_sample = True
            logger.info(
                "[sft] first sample %s: %d tokens, %d response tokens, %d trained; trained spans: %r",
                traj.trajectory_id,
                len(tokens),
                resp_len,
                sum(loss_masks),
                decode_masked_spans(self.tokenizer, tokens[-resp_len:], loss_masks),
            )
