"""Score centering for train-inference mismatch (arXiv:2609.20807).

Under a mismatched sampler q, the REINFORCE update at a prefix carries a drift
term ``E_q[R] * E_q[∇log p]`` that pulls the trainer toward the sampler and
compounds across weight syncs. Score centering subtracts the expected score
under q from the sampled token's score, which cancels the drift exactly.

The sampler's top-k head is recorded per response token at rollout time; the
tail beyond it is modelled with the trainer's distribution rescaled to the
sampler's tail mass, so the centering term only touches the k head tokens.
That tail model assumes q is the full softmax. Sampling top_k is the one
truncation supported: when the recorded head holds at least top_k tokens, the
support is its top_k leading tokens, renormalized, with no tail. Other
truncating or logit-editing sampling parameters are rejected.

This module holds the config checks shared by the trainer and the rollout
engine.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from functools import reduce

from omegaconf import DictConfig

logger = logging.getLogger(__name__)

# Sampling parameters that make the sampled distribution differ from softmax(logits / T).
# Each entry maps a parameter to the predicate that flags a non-default value.
_UNSUPPORTED_SAMPLING_PARAMS = {
    # Truncation: SGLang returns log-probs of the full softmax, then samples from a renormalized subset,
    # so the recorded q is not the distribution the token was drawn from. top_k is checked separately
    # in validate_sampling_params: its support can be rebuilt from a large enough recorded head.
    "top_p": lambda v: float(v) < 1.0,  # keeps the smallest head whose cumulative prob reaches top_p
    "min_p": lambda v: float(v) > 0.0,  # drops tokens below min_p * max prob
    # Logit edits: applied before the log-probs are taken, so q is what was sampled, but the trainer's p
    # does not see them. q - p becomes a state-dependent gap rather than numeric error, and masking gives
    # tokens q = 0 that the tail model q ≈ rho * p still assigns mass to.
    "repetition_penalty": lambda v: float(v) != 1.0,  # rescales logits of tokens already in prompt or output
    "frequency_penalty": lambda v: float(v) != 0.0,  # subtracts penalty * occurrence count of each output token
    "presence_penalty": lambda v: float(v) != 0.0,  # subtracts penalty from every token already in the output
    "min_new_tokens": lambda v: int(v) > 0,  # masks EOS and stop tokens until the minimum length
    "logit_bias": lambda v: bool(v),  # adds a fixed bias to the listed token ids
    "json_schema": lambda v: bool(v),  # grammar mask: output must match the JSON schema
    "regex": lambda v: bool(v),  # grammar mask: output must match the regex
    "ebnf": lambda v: bool(v),  # grammar mask: output must match the EBNF grammar
    "structural_tag": lambda v: bool(v),  # grammar mask: tagged spans (e.g. tool-call args) must match a schema
}

# Training-side features that select or drop tokens by the sampled outcome,
# which breaks the expectation the centering term is built on.
_INCOMPATIBLE_FEATURES = {
    "algorithm.m2po.enable": "M2PO masks tokens by their sampled IS ratio",
    "algorithm.opsm.enable": "OPSM drops sequences by their sampled KL and advantage sign",
    "opd.enable": "on-policy distillation is not supported with score centering yet",
}


def validate_train_config(config: DictConfig) -> None:
    """Reject training configs that score centering cannot run with. Call only when it is enabled."""
    algo = config.algorithm
    if algo.policy_loss != "reinforce":
        raise ValueError(f"score centering requires policy_loss=reinforce, got {algo.policy_loss!r}.")
    if int(algo.score_centering.topk) <= 0:
        raise ValueError(f"score_centering.topk must be positive, got {algo.score_centering.topk}.")
    if algo.is_correction.enable and algo.is_correction.level != "token":
        raise ValueError(
            f"score centering requires is_correction.level=token, got {algo.is_correction.level!r}."
        )
    for key, reason in _INCOMPATIBLE_FEATURES.items():
        if reduce(getattr, key.split("."), config):
            raise ValueError(f"score centering is incompatible with {key}: {reason}.")

    if (config.rollout.partial or config.fully_async.enable) and not algo.is_correction.enable:
        logger.warning(
            "score centering without is_correction on stale rollouts is unstable in the paper; "
            "consider is_correction with level=token."
        )


def validate_sampling_params(params: Mapping, where: str, topk: int) -> None:
    """Reject SGLang sampling params under which the sampled q cannot be rebuilt from the recorded top-k.

    ``topk`` is ``algorithm.score_centering.topk``: score centering reads only those leading columns of the
    recorded head, so the sampling support must fit in them.
    """
    unsupported = [
        f"{name}={params[name]!r}"
        for name, flags in _UNSUPPORTED_SAMPLING_PARAMS.items()
        if params.get(name) is not None and flags(params[name])
    ]
    if unsupported:
        raise ValueError(
            f"{where}: score centering needs sampling from softmax(logits / T), optionally truncated "
            f"by top_k, but got {', '.join(unsupported)}. Remove them or disable algorithm.score_centering."
        )

    # top_k = 1 is greedy: the centered score is identically zero. top_k above topk leaves
    # part of the support unrecorded, so the sampled q cannot be renormalized from it.
    top_k = params.get("top_k")
    if top_k is not None and int(top_k) != -1 and not 2 <= int(top_k) <= topk:
        raise ValueError(
            f"{where}: score centering needs top_k=-1 or 2 <= top_k <= algorithm.score_centering.topk "
            f"({topk}), but got top_k={top_k!r}."
        )


def validate_rollout_config(config: DictConfig) -> None:
    """Reject rollout configs whose recorded log-probs would not be the sampled q.

    Call only when score centering is enabled, after the per-replica
    ``sglang_args`` are merged.
    """
    # SGLang returns pre-temperature log-probs when this is set, while the
    # sampler and the trainer both divide by T.
    temperature = float(config.trainer.temperature)
    env_key = "SGLANG_RETURN_ORIGINAL_LOGPROB"
    env_value = config.rollout.env_vars.get(env_key, os.environ.get(env_key))
    # Same parsing as sglang.srt.utils.get_bool_env_var.
    if temperature != 1.0 and str(env_value).lower() in ("true", "1"):
        raise ValueError(
            f"score centering with trainer.temperature={temperature} requires {env_key} unset: "
            "SGLang would return log-probs before temperature scaling."
        )

    for name, replica_cfg in config.rollout.sglang_replicas.items():
        if replica_cfg.num_nodes <= 0:
            continue
        # sglang_args is a ServerArgs passthrough; speculative_algorithm is absent unless set.
        algorithm = replica_cfg.sglang_args.get("speculative_algorithm")
        if algorithm:
            raise ValueError(
                f"rollout.sglang_replicas.{name}: score centering does not support speculative "
                f"decoding (speculative_algorithm={algorithm!r}) yet: the top-k log-probs of "
                "verified tokens are not checked against non-speculative decoding."
            )

    topk = int(config.algorithm.score_centering.topk)
    for idx, ds in enumerate(config.data_sources):
        validate_sampling_params(ds.completion_params or {}, f"data_sources[{idx}].completion_params", topk)
