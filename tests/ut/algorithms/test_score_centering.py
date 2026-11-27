"""Unit tests for the score-centering config checks."""

import logging

import pytest
from omegaconf import OmegaConf

from coda.algorithms.score_centering import (
    validate_rollout_config,
    validate_sampling_params,
    validate_train_config,
)


def make_config(**overrides):
    """Every key the score-centering checks read, with score centering on and nothing else."""
    base = {
        "fully_async": {"enable": False},
        "opd": {"enable": False},
        "trainer": {"temperature": 1.0},
        "rollout": {
            "partial": False,
            "env_vars": {},
            "sglang_replicas": {
                "regular": {"num_nodes": 1, "sglang_args": {}},
                "prefill": {"num_nodes": 0, "sglang_args": {}},
                "decode": {"num_nodes": 0, "sglang_args": {}},
            },
        },
        "data_sources": [{"completion_params": {}}],
        "algorithm": {
            "policy_loss": "reinforce",
            "score_centering": {"enable": True, "topk": 128},
            "is_correction": {"enable": False, "level": "token"},
            "m2po": {"enable": False},
            "opsm": {"enable": False},
        },
    }
    cfg = OmegaConf.create(base)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return cfg


# ---------------------------------------------------------------------------
# validate_train_config
# ---------------------------------------------------------------------------

def test_train_config_accepts_reinforce():
    validate_train_config(make_config())


def test_train_config_accepts_token_level_is_correction():
    validate_train_config(make_config(algorithm={"is_correction": {"enable": True, "level": "token"}}))


@pytest.mark.parametrize("policy_loss", ["grpo", "ppo", "gspo"])
def test_train_config_rejects_clipped_policy_loss(policy_loss):
    with pytest.raises(ValueError, match="policy_loss=reinforce"):
        validate_train_config(make_config(algorithm={"policy_loss": policy_loss}))


@pytest.mark.parametrize("topk", [0, -1])
def test_train_config_rejects_non_positive_topk(topk):
    with pytest.raises(ValueError, match="topk must be positive"):
        validate_train_config(make_config(algorithm={"score_centering": {"topk": topk}}))


@pytest.mark.parametrize("level", ["sequence", "geometric"])
def test_train_config_rejects_sequence_level_is_correction(level):
    cfg = make_config(algorithm={"is_correction": {"enable": True, "level": level}})
    with pytest.raises(ValueError, match="is_correction.level=token"):
        validate_train_config(cfg)


def test_train_config_ignores_level_when_is_correction_disabled():
    validate_train_config(make_config(algorithm={"is_correction": {"enable": False, "level": "sequence"}}))


@pytest.mark.parametrize(
    "overrides, key",
    [
        ({"algorithm": {"m2po": {"enable": True}}}, "algorithm.m2po.enable"),
        ({"algorithm": {"opsm": {"enable": True}}}, "algorithm.opsm.enable"),
        ({"opd": {"enable": True}}, "opd.enable"),
    ],
)
def test_train_config_rejects_incompatible_features(overrides, key):
    with pytest.raises(ValueError, match=key.replace(".", r"\.")):
        validate_train_config(make_config(**overrides))


@pytest.mark.parametrize(
    "overrides",
    [{"rollout": {"partial": True}}, {"fully_async": {"enable": True}}],
)
def test_train_config_warns_on_stale_rollouts_without_is(overrides, caplog):
    with caplog.at_level(logging.WARNING, logger="coda.algorithms.score_centering"):
        validate_train_config(make_config(**overrides))
    assert "without is_correction on stale rollouts" in caplog.text


def test_train_config_does_not_warn_on_stale_rollouts_with_is(caplog):
    cfg = make_config(rollout={"partial": True}, algorithm={"is_correction": {"enable": True}})
    with caplog.at_level(logging.WARNING, logger="coda.algorithms.score_centering"):
        validate_train_config(cfg)
    assert caplog.text == ""


# ---------------------------------------------------------------------------
# validate_sampling_params / validate_rollout_config
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "params",
    [
        {},
        {"top_p": 1.0, "top_k": -1, "min_p": 0.0},
        {"repetition_penalty": 1.0, "frequency_penalty": 0.0, "presence_penalty": 0.0},
        {"stop": ["</think>"], "skip_special_tokens": True, "sampling_seed": 7},
        {"top_k": 20},
        {"top_k": 2},
        {"top_k": 128},
    ],
)
def test_sampling_params_accepts_untruncated_or_covered_top_k(params):
    validate_sampling_params(params, "here", topk=128)


@pytest.mark.parametrize(
    "params, name",
    [
        ({"top_p": 0.95}, "top_p"),
        ({"min_p": 0.05}, "min_p"),
        ({"repetition_penalty": 1.05}, "repetition_penalty"),
        ({"frequency_penalty": 0.1}, "frequency_penalty"),
        ({"presence_penalty": 0.1}, "presence_penalty"),
        ({"min_new_tokens": 8}, "min_new_tokens"),
        ({"logit_bias": {"42": -100.0}}, "logit_bias"),
        ({"json_schema": "{}"}, "json_schema"),
        ({"regex": "[0-9]+"}, "regex"),
        ({"ebnf": "root ::= \"a\""}, "ebnf"),
    ],
)
def test_sampling_params_rejects_truncation_and_logit_edits(params, name):
    with pytest.raises(ValueError, match=f"here: .*{name}="):
        validate_sampling_params(params, "here", topk=128)


@pytest.mark.parametrize("top_k", [129, 1, 0])
def test_sampling_params_rejects_top_k_outside_recorded_head(top_k):
    # Above topk the support is not fully recorded; 1 is greedy, where the centered score is zero.
    with pytest.raises(ValueError, match=rf"score_centering\.topk \(128\).*top_k={top_k}"):
        validate_sampling_params({"top_k": top_k}, "here", topk=128)


def test_sampling_params_rejects_top_k_combined_with_top_p():
    with pytest.raises(ValueError, match="top_p=0.9"):
        validate_sampling_params({"top_k": 20, "top_p": 0.9}, "here", topk=128)


def test_rollout_config_accepts_defaults():
    validate_rollout_config(make_config())


def test_rollout_config_rejects_truncated_data_source():
    cfg = make_config(data_sources=[{"completion_params": {}}, {"completion_params": {"top_p": 0.95, "top_k": 20}}])
    with pytest.raises(ValueError, match=r"data_sources\[1\]\.completion_params: .*top_p=0\.95"):
        validate_rollout_config(cfg)


def test_rollout_config_checks_top_k_against_score_centering_topk():
    validate_rollout_config(make_config(data_sources=[{"completion_params": {"top_k": 20}}]))
    cfg = make_config(
        algorithm={"score_centering": {"topk": 16}},
        data_sources=[{"completion_params": {"top_k": 20}}],
    )
    with pytest.raises(ValueError, match=r"data_sources\[0\]\.completion_params: .*topk \(16\)"):
        validate_rollout_config(cfg)


def test_rollout_config_accepts_null_completion_params():
    validate_rollout_config(make_config(data_sources=[{"completion_params": None}]))


@pytest.mark.parametrize("value", ["1", "true", "True"])
def test_rollout_config_rejects_original_logprob_with_temperature(value):
    cfg = make_config(trainer={"temperature": 0.7}, rollout={"env_vars": {"SGLANG_RETURN_ORIGINAL_LOGPROB": value}})
    with pytest.raises(ValueError, match="SGLANG_RETURN_ORIGINAL_LOGPROB"):
        validate_rollout_config(cfg)


def test_rollout_config_reads_original_logprob_from_process_env(monkeypatch):
    monkeypatch.setenv("SGLANG_RETURN_ORIGINAL_LOGPROB", "1")
    with pytest.raises(ValueError, match="SGLANG_RETURN_ORIGINAL_LOGPROB"):
        validate_rollout_config(make_config(trainer={"temperature": 0.7}))


def test_rollout_config_rollout_env_overrides_process_env(monkeypatch):
    monkeypatch.setenv("SGLANG_RETURN_ORIGINAL_LOGPROB", "1")
    cfg = make_config(trainer={"temperature": 0.7}, rollout={"env_vars": {"SGLANG_RETURN_ORIGINAL_LOGPROB": "0"}})
    validate_rollout_config(cfg)


def test_rollout_config_allows_original_logprob_at_unit_temperature():
    # Pre- and post-temperature log-probs coincide at T = 1.
    cfg = make_config(rollout={"env_vars": {"SGLANG_RETURN_ORIGINAL_LOGPROB": "1"}})
    validate_rollout_config(cfg)


def test_rollout_config_rejects_speculative_decoding():
    cfg = make_config(rollout={"sglang_replicas": {"regular": {"sglang_args": {"speculative_algorithm": "EAGLE"}}}})
    with pytest.raises(ValueError, match=r"sglang_replicas\.regular: .*speculative"):
        validate_rollout_config(cfg)


def test_rollout_config_skips_unused_replica_groups():
    # prefill has no nodes, so its sglang_args never launch an engine.
    cfg = make_config(rollout={"sglang_replicas": {"prefill": {"sglang_args": {"speculative_algorithm": "EAGLE"}}}})
    validate_rollout_config(cfg)
