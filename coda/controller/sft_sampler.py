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

from omegaconf import DictConfig, OmegaConf

from coda.agentflow.tokenizer_manager import _load_chat_template, load_tokenizer
from coda.agentflow.trajectory_store import Segment, Trajectory, TrajectoryGroup, TrajectoryStatus
from coda.agentflow.sft_tokenizer import SFTTokenizer, decode_trainable_spans, trainable_suffix_length
from coda.data_factory.data_source import DataSource
from coda.utils.tracking import track

logger = logging.getLogger(__name__)

# Values run_mode=sft implies; apply_sft_forced_settings writes them into the config.
SFT_FORCED_SETTINGS = {
    # No rollout engine shares the GPUs, so colocate offload/onload is pure overhead.
    "colocate": False,
    # The sft loss ignores old log-probs; this skips the extra old-policy forward pass.
    "trainer.use_rollout_log_probs": True,
}


def _summary_metrics(trajectories: list[Trajectory], num_truncated: int) -> dict[str, float]:
    """Summarize one step's tokenized trajectories into ``sft/*`` keys.

    Keys that also exist under ``rollout/`` share their suffix and definition with
    ``rollout_sampler._summary_metrics``: ``response_length`` is the response-space
    length ``len(loss_masks)``, equivalent to RL's ``len(rollout_log_probs)``
    (from the first trainable token to the end, so multi-turn context in between
    is included), and ``prompt_length`` is the rest.  A fully masked sample keeps
    a 1-token placeholder response, so it counts as response length 1.
    """
    n = len(trajectories)
    if n == 0:
        return {}
    seq_lengths = [len(t.tokens) for t in trajectories]
    response_lengths = [len(t.loss_masks) for t in trajectories]
    prompt_lengths = [s - r for s, r in zip(seq_lengths, response_lengths, strict=True)]
    trained = [sum(t.loss_masks) for t in trajectories]
    total_tokens = sum(seq_lengths)

    metrics: dict[str, float] = {"sft/completed_count": n}
    for name, values in (("prompt_length", prompt_lengths),
                         ("response_length", response_lengths),
                         ("seq_length", seq_lengths)):
        metrics[f"sft/{name}_mean"] = sum(values) / n
        metrics[f"sft/{name}_max"] = max(values)
        metrics[f"sft/{name}_min"] = min(values)
    # Cut by data_source.sft_max_seq_len over the whole sequence, unlike rollout's
    # response_length_clip_ratio (max_response_len_per_trajectory on the response).
    metrics["sft/seq_length_clip_ratio"] = num_truncated / n
    metrics["sft/trained_tokens"] = sum(trained)
    metrics["sft/trained_token_ratio"] = sum(trained) / total_tokens
    metrics["sft/fully_masked_ratio"] = sum(1 for v in trained if v == 0) / n
    return metrics


def apply_sft_forced_settings(config: DictConfig) -> None:
    """Write ``SFT_FORCED_SETTINGS`` into ``config`` in place and log what changed."""
    applied = []
    for key, value in SFT_FORCED_SETTINGS.items():
        previous = OmegaConf.select(config, key)
        OmegaConf.update(config, key, value)
        applied.append(f"{key}={value}" + ("" if previous == value else f" (was {previous})"))
    logger.info("[sft] run_mode=sft sets: %s", ", ".join(applied))


def validate_sft_config(config: DictConfig) -> None:
    """Reject settings SFT cannot honour.

    Raises:
        ValueError: On the first incompatible setting.
    """
    algorithm = config.algorithm
    if algorithm.policy_loss != "sft" or algorithm.advantage_estimator != "sft":
        raise ValueError(
            "run_mode=sft requires algorithm.policy_loss=sft and algorithm.advantage_estimator=sft, "
            f"got {algorithm.policy_loss!r} / {algorithm.advantage_estimator!r}"
        )
    disabled = {
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
        self.sft_tokenizer = SFTTokenizer(
            tokenizer,
            dict(tokenizer_cfg.generation_prompt_kwargs or {}),
        )
        self._logged_sample = False
        # {step: sft/* summary}; like RolloutSampler.metrics, only the current step is kept.
        self.metrics: dict[int, dict[str, float]] = {}
        logger.info(
            "[sft] chat template probe renders: full=%r gen=%r",
            self.sft_tokenizer.probe_full,
            self.sft_tokenizer.probe_gen,
        )
        logger.info(
            "[sft] markers probed from chat template: header=%r end=%r skip_prefixes=%r",
            self.sft_tokenizer.assistant_header,
            self.sft_tokenizer.end_marker,
            self.sft_tokenizer.skip_prefixes,
        )
        logger.info(
            "[sft] sft_max_seq_len per data source: %s",
            [ds_cfg.sft_max_seq_len for ds_cfg in config.data_sources],
        )

    def __call__(self, step: int) -> list[TrajectoryGroup]:
        """Return this step's trainable groups from all data sources.

        Also stores this batch's ``sft/*`` summary in ``self.metrics[step]`` and
        reports it, as ``RolloutSampler`` does for ``rollout/*``.
        """
        groups: list[TrajectoryGroup] = []
        trajectories: list[Trajectory] = []
        num_truncated = 0
        for ds, ds_cfg in zip(self.datasources, self.config.data_sources, strict=True):
            ds_groups = ds.get(int(ds_cfg.num_prompts_per_step), step)
            max_seq_len = ds_cfg.sft_max_seq_len
            for group in ds_groups:
                for traj in group.trajectories:
                    num_truncated += self._tokenize_trajectory(traj, max_seq_len)
                    trajectories.append(traj)
            groups += ds_groups

        self.metrics = {step: _summary_metrics(trajectories, num_truncated)}
        if self.metrics[step]:
            logger.info("[step %d] sft metrics: %s", step, self.metrics[step])
            track(self.metrics[step], step)
        return groups

    def _tokenize_trajectory(self, traj: Trajectory, max_seq_len: int | None) -> bool:
        """Turn one dataset conversation into a trainable Trajectory, in place.

        Renders ``traj.prompt`` with the chat template, builds the assistant-turn
        loss mask and fills the fields the RL path would get from inference:
        ``tokens``, response-space ``loss_masks``, zero ``rollout_log_probs`` /
        ``token_rewards`` placeholders and a single trainable ``Segment``.

        Returns:
            Whether the conversation was cut to its data source's ``sft_max_seq_len``.

        Raises:
            ValueError: If ``traj.prompt`` is not a message list, or it renders
                to fewer than 2 tokens.
        """
        if not isinstance(traj.prompt, list):
            raise ValueError(
                f"SFT trajectory {traj.trajectory_id} needs a message list in dataset.input_key, "
                f"got {type(traj.prompt).__name__}"
            )
        tokens, mask = self.sft_tokenizer.tokenize(traj.prompt, tools=traj.metadata.get("tools"))
        truncated = bool(max_seq_len and len(tokens) > max_seq_len)
        if truncated:
            tokens, mask = tokens[:max_seq_len], mask[:max_seq_len]
        if len(tokens) < 2:
            raise ValueError(f"SFT trajectory {traj.trajectory_id} renders to fewer than 2 tokens")

        # The first token has no preceding logit, so at least one token is prompt.
        resp_len = min(trainable_suffix_length(mask), len(tokens) - 1)
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
                decode_trainable_spans(self.tokenizer, tokens[-resp_len:], loss_masks),
            )

        return truncated
