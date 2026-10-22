"""Unit tests for coda.controller.sft_sampler.

Run: python -m pytest tests/ut/controller/test_sft_sampler.py -v
"""

import logging
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from coda.agentflow.sft_tokenizer import decode_trainable_spans
from coda.agentflow.trajectory_store import Trajectory, TrajectoryGroup, TrajectoryStatus
from coda.controller.sft_sampler import SFTSampler, apply_sft_forced_settings, validate_sft_config
from tests.ut.agentflow.fake_tokenizers import FakeDeepSeekV4Tokenizer, FakeQwenTokenizer

DEFAULT_YAML = Path(__file__).resolve().parents[3] / "conf" / "default.yaml"

SFT_OVERRIDES = {
    "run_mode": "sft",
    "algorithm": {"policy_loss": "sft", "advantage_estimator": "sft"},
    "data_source": {"num_trajectories_per_prompt": 1},
}


def make_sft_config(**overrides):
    cfg = OmegaConf.merge(OmegaConf.load(DEFAULT_YAML), SFT_OVERRIDES, overrides)
    OmegaConf.resolve(cfg)
    # As trainer.normalize_data_sources does (not imported: trainer pulls in Megatron):
    # every data_sources entry inherits the data_source defaults.
    cfg.data_sources = [OmegaConf.merge(cfg.data_source, ds) for ds in cfg.data_sources]
    return cfg


class FakeDataSource:
    """Hands out a fixed list of groups and records every ``get(num, step)`` call."""

    def __init__(self, groups):
        self.groups = groups
        self.calls = []

    def get(self, num, step):
        self.calls.append((num, step))
        return self.groups


@pytest.fixture(autouse=True)
def tracked(monkeypatch):
    """Capture ``track(metrics, step)`` calls; no tracking backend runs in unit tests."""
    calls = []
    monkeypatch.setattr("coda.controller.sft_sampler.track", lambda metrics, step: calls.append((metrics, step)))
    return calls


def make_sampler(groups=(), max_seq_len=None):
    return SFTSampler(
        make_sft_config(data_source={"sft_max_seq_len": max_seq_len}),
        [FakeDataSource(list(groups))],
        tokenizer=FakeQwenTokenizer(),
    )


def make_group(messages, metadata=None, prompt_id="p0", ds_index=0):
    traj = Trajectory(
        trajectory_id=f"{prompt_id}_t0",
        prompt_id=prompt_id,
        prompt=messages,
        metadata=metadata or {},
        ds_index=ds_index,
    )
    return TrajectoryGroup(prompt_id=prompt_id, trajectories=[traj])


def sample_one(group, max_seq_len=None):
    """Run one SFTSampler step over a single group; returns ``(sampler, trajectory)``."""
    sampler = make_sampler([group], max_seq_len=max_seq_len)
    return sampler, sampler(step=1)[0].trajectories[0]


def response_slice(traj):
    """What put_dp_shards_to_ray + gather_and_slice_response pair up for the single Segment."""
    seg = traj.segments[0]
    resp_len = seg.logprob_end - seg.logprob_start
    tokens = traj.tokens[seg.token_start:seg.token_end]
    return tokens[len(tokens) - resp_len:], traj.loss_masks[seg.logprob_start:seg.logprob_end]


def decode(ids):
    """FakeCharTokenizer is one token per character."""
    return "".join(chr(int(t)) for t in ids)


def trained_spans(traj):
    """Contiguous trained runs of the response slice, decoded."""
    return decode_trainable_spans(FakeQwenTokenizer(), *response_slice(traj))


IM_END = "<|im_end|>"
EOS = "<｜end▁of▁sentence｜>"

MESSAGES = [
    {"role": "user", "content": "q1"},
    {"role": "assistant", "content": "a1"},
    {"role": "user", "content": "q2"},
    {"role": "assistant", "content": "a2"},
]

TOOLS = [{"type": "function", "function": {"name": "read"}}]

# A coding-agent episode: reasoning + tool calls, tool outputs in between, one
# failed attempt excluded via ``step_loss_mask: 0`` and a plain final answer.
AGENTIC = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "fix bug"},
    {
        "role": "assistant",
        "content": "look",
        "reasoning_content": "plan",
        "tool_calls": [{"function": {"name": "read"}}],
    },
    {"role": "tool", "content": "TOOL_OUT_1"},
    {
        "role": "assistant",
        "content": "retry",
        "tool_calls": [{"function": {"name": "edit"}}],
        "step_loss_mask": 0,
    },
    {"role": "tool", "content": "TOOL_OUT_2"},
    {"role": "assistant", "content": "done", "tool_calls": [{"function": {"name": "test"}}]},
    {"role": "tool", "content": "TOOL_OUT_3"},
    {"role": "assistant", "content": "fixed"},
]

AGENTIC_SPANS = [
    # The pre-filled "<think>\n" stays in the prompt; the reasoning itself is trained.
    "plan\n</think>\n\nlook\n<tool_call>read</tool_call>" + IM_END,
    "done\n<tool_call>test</tool_call>" + IM_END,
    "fixed" + IM_END,
]

ALL_MASKED = [
    {"role": "user", "content": "q"},
    {"role": "assistant", "content": "a", "step_loss_mask": 0},
]


# ---------------------------------------------------------------------------
# SFTSampler.__call__
# ---------------------------------------------------------------------------

def test_call_pulls_num_prompts_per_step_from_each_datasource():
    groups = [make_group(MESSAGES), make_group(MESSAGES)]
    sampler = make_sampler(groups)

    assert sampler(step=3) == groups
    num_prompts = int(sampler.config.data_sources[0].num_prompts_per_step)
    assert sampler.datasources[0].calls == [(num_prompts, 3)]
    assert all(g.trajectories[0].status == TrajectoryStatus.COMPLETED for g in groups)


def test_call_fills_training_fields():
    sampler, traj = sample_one(make_group(MESSAGES))

    rendered = sampler.tokenizer.apply_chat_template(MESSAGES, tokenize=False)
    resp_len = len(traj.loss_masks)
    assert traj.tokens == sampler.tokenizer.encode(rendered)
    # Response starts at the first trained token.
    assert traj.loss_masks[0] == 1
    assert len(traj.rollout_log_probs) == len(traj.token_rewards) == len(traj.rollout_weight_versions) == resp_len
    assert traj.status == TrajectoryStatus.COMPLETED
    assert traj.reward == 0.0

    assert len(traj.segments) == 1
    seg = traj.segments[0]
    assert (seg.token_start, seg.token_end) == (0, len(traj.tokens))
    assert (seg.logprob_start, seg.logprob_end) == (0, resp_len)
    assert seg.trainable

    tokens, mask = response_slice(traj)
    trained = "".join(chr(t) for t, m in zip(tokens, mask) if m)
    assert trained == "a1<|im_end|>a2<|im_end|>"


def test_call_reads_tools_from_metadata():
    tools = [{"type": "function", "function": {"name": "f"}}]
    sampler, traj = sample_one(make_group(MESSAGES, metadata={"tools": tools}))
    rendered = sampler.tokenizer.apply_chat_template(MESSAGES, tools=tools, tokenize=False)
    assert traj.tokens == sampler.tokenizer.encode(rendered)


def test_truncation_keeps_prefix():
    # 100 tokens: past the first assistant turn (ends at 65), inside the second user turn.
    full = FakeQwenTokenizer().apply_chat_template(MESSAGES)
    _, traj = sample_one(make_group(MESSAGES), max_seq_len=100)

    assert traj.tokens == full[:100]
    assert traj.segments[0].token_end == 100
    assert trained_spans(traj) == ["a1" + IM_END]
    assert traj.loss_masks[-1] == 0  # the cut-off tail is context


def test_max_seq_len_equal_to_length_does_not_truncate():
    full = FakeQwenTokenizer().apply_chat_template(MESSAGES)
    _, traj = sample_one(make_group(MESSAGES), max_seq_len=len(full))

    assert traj.tokens == full
    assert trained_spans(traj) == ["a1" + IM_END, "a2" + IM_END]


@pytest.mark.parametrize(
    "messages, sft_cfg",
    [
        ([], {}),  # empty conversation renders to nothing
        (MESSAGES, {"max_seq_len": 1}),
    ],
)
def test_fewer_than_two_tokens_raises(messages, sft_cfg):
    with pytest.raises(ValueError, match="fewer than 2 tokens"):
        sample_one(make_group(messages), **sft_cfg)


def test_conversation_without_assistant_falls_back_to_zero_weight_row():
    """Pins current behaviour: nothing to train is silent, not an error."""
    _, traj = sample_one(make_group([{"role": "user", "content": "only a question"}]))
    assert traj.loss_masks == [0]


def test_truncated_away_response_falls_back_to_zero_weight_row():
    _, traj = sample_one(make_group(MESSAGES), max_seq_len=10)  # cut before the first assistant turn
    assert len(traj.tokens) == 10
    assert traj.loss_masks == [0]
    assert traj.segments[0].logprob_end == 1


def test_all_turns_masked_falls_back_to_zero_weight_row():
    _, traj = sample_one(make_group(ALL_MASKED))
    assert traj.loss_masks == [0]
    assert len(traj.rollout_log_probs) == 1


def test_string_prompt_is_rejected():
    group = TrajectoryGroup(prompt_id="p0", trajectories=[Trajectory(prompt="plain text")])
    with pytest.raises(ValueError, match="needs a message list"):
        sample_one(group)


# ---------------------------------------------------------------------------
# SFTSampler.metrics
# ---------------------------------------------------------------------------

def test_metrics_describe_the_single_sample(tracked):
    sampler, traj = sample_one(make_group(MESSAGES))
    seq_len = len(traj.tokens)
    resp_len = len(traj.loss_masks)
    trained = sum(traj.loss_masks)

    expected = {"sft/completed_count": 1}
    for name, value in (("prompt_length", seq_len - resp_len),
                        ("response_length", resp_len),
                        ("seq_length", seq_len)):
        expected.update({f"sft/{name}_mean": value, f"sft/{name}_max": value, f"sft/{name}_min": value})
    expected.update({
        "sft/seq_length_clip_ratio": 0.0,
        "sft/trained_tokens": trained,
        "sft/trained_token_ratio": pytest.approx(trained / seq_len),
        "sft/fully_masked_ratio": 0.0,
    })
    assert sampler.metrics == {1: expected}
    assert tracked == [(sampler.metrics[1], 1)]


def test_metrics_cover_the_whole_batch():
    groups = [
        make_group(MESSAGES, prompt_id="long"),
        make_group([{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
                   prompt_id="short"),
    ]
    sampler = make_sampler(groups)
    sampled = [g.trajectories[0] for g in sampler(step=1)]

    lengths = [len(t.tokens) for t in sampled]
    resp_lengths = [len(t.loss_masks) for t in sampled]
    prompt_lengths = [s - r for s, r in zip(lengths, resp_lengths)]
    metrics = sampler.metrics[1]
    assert metrics["sft/completed_count"] == 2
    assert metrics["sft/seq_length_mean"] == pytest.approx(sum(lengths) / 2)
    assert metrics["sft/seq_length_max"] == max(lengths)
    assert metrics["sft/seq_length_min"] == min(lengths)
    assert metrics["sft/response_length_max"] == max(resp_lengths)
    assert metrics["sft/prompt_length_min"] == min(prompt_lengths)
    assert metrics["sft/trained_tokens"] == sum(sum(t.loss_masks) for t in sampled)


def test_metrics_keep_only_the_current_step():
    sampler = make_sampler([make_group(MESSAGES)])
    sampler(step=1)
    sampler(step=2)
    assert list(sampler.metrics) == [2]


def test_metrics_flag_truncated_samples():
    full = FakeQwenTokenizer().apply_chat_template(MESSAGES)
    sampler, _ = sample_one(make_group(MESSAGES), max_seq_len=len(full) - 1)

    assert sampler.metrics[1]["sft/seq_length_max"] == len(full) - 1
    assert sampler.metrics[1]["sft/seq_length_clip_ratio"] == 1.0


def test_metrics_do_not_flag_exact_length_as_truncated():
    full = FakeQwenTokenizer().apply_chat_template(MESSAGES)
    sampler, _ = sample_one(make_group(MESSAGES), max_seq_len=len(full))

    assert sampler.metrics[1]["sft/seq_length_clip_ratio"] == 0.0


def test_metrics_flag_zero_weight_samples():
    sampler, _ = sample_one(make_group(ALL_MASKED))

    metrics = sampler.metrics[1]
    assert metrics["sft/trained_tokens"] == 0
    assert metrics["sft/trained_token_ratio"] == 0.0
    assert metrics["sft/fully_masked_ratio"] == 1.0
    # The zero-weight placeholder response is one token long.
    assert metrics["sft/response_length_min"] == 1


def test_metrics_are_empty_without_samples(tracked):
    sampler = make_sampler([])
    assert sampler(step=1) == []
    assert sampler.metrics == {1: {}}
    assert tracked == []


# ---------------------------------------------------------------------------
# Complex conversations
# ---------------------------------------------------------------------------

def test_agentic_episode_trains_only_kept_assistant_turns():
    tokenizer = FakeQwenTokenizer()
    sampler, traj = sample_one(make_group(AGENTIC, metadata={"tools": TOOLS}))

    rendered = tokenizer.apply_chat_template(AGENTIC, tools=TOOLS, tokenize=False)
    assert traj.tokens == tokenizer.encode(rendered)
    assert trained_spans(traj) == AGENTIC_SPANS

    resp_len = traj.segments[0].logprob_end
    prompt, response = decode(traj.tokens[:-resp_len]), decode(traj.tokens[-resp_len:])
    # Tools block, system and user turns precede the first trained token.
    assert prompt.startswith("<|im_start|>system\n# Tools\n")
    assert prompt.endswith("fix bug" + IM_END + "\n<|im_start|>assistant\n<think>\n")
    # Tool outputs and the excluded attempt sit inside the response slice as context.
    for context in ("TOOL_OUT_1", "retry", "<tool_call>edit</tool_call>", "TOOL_OUT_2", "TOOL_OUT_3"):
        assert context in response
        assert all(context not in span for span in trained_spans(traj))


def test_masked_leading_turn_moves_response_start():
    """A masked first assistant turn becomes prompt; the response starts at the next kept turn."""
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "bad", "step_loss_mask": 0},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "good"},
    ]
    _, traj = sample_one(make_group(messages))

    resp_len = traj.segments[0].logprob_end
    assert "bad" in decode(traj.tokens[:-resp_len])
    assert decode(traj.tokens[-resp_len:]) == "good" + IM_END + "\n"
    assert traj.loss_masks == [1] * len("good" + IM_END) + [0]


def test_masked_trailing_turn_stays_in_response_with_zero_mask():
    messages = MESSAGES[:3] + [{"role": "assistant", "content": "a2", "step_loss_mask": 0}]
    _, traj = sample_one(make_group(messages))

    tokens, mask = response_slice(traj)
    assert decode(tokens).startswith("a1" + IM_END)
    assert decode(tokens).endswith("a2" + IM_END + "\n")
    assert trained_spans(traj) == ["a1" + IM_END]
    assert sum(mask) == len("a1" + IM_END)


# "<|im_start|>user\nq<|im_end|>\n" + "<|im_start|>assistant\n" is 51 characters.
LONG_ANSWER = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "abcdefghij"}]


def test_truncation_mid_turn_trains_kept_head_without_end_marker():
    _, traj = sample_one(make_group(LONG_ANSWER), max_seq_len=56)

    assert len(traj.tokens) == 56
    assert trained_spans(traj) == ["abcde"]
    assert traj.loss_masks == [1] * 5


def test_truncation_right_after_header_falls_back_to_zero_weight_row():
    _, traj = sample_one(make_group(LONG_ANSWER), max_seq_len=51)

    assert len(traj.tokens) == 51
    assert traj.loss_masks == [0]
    assert traj.segments[0].logprob_end == 1


def test_first_token_is_never_trained(monkeypatch):
    """The first token has no preceding logit, so it is forced into the prompt."""
    messages = [{"role": "assistant", "content": "ab"}]
    sampler = make_sampler([make_group(messages)])
    monkeypatch.setattr(
        sampler.sft_tokenizer, "tokenize", lambda messages, tools=None: ([ord("a"), ord("b")], [1, 1])
    )
    traj = sampler(step=1)[0].trajectories[0]

    assert traj.segments[0].logprob_end == len(traj.tokens) - 1
    assert traj.loss_masks == [1]
    assert response_slice(traj)[0] == [ord("b")]
    assert len(traj.rollout_log_probs) == len(traj.tokens) - 1


def test_deepseek_thinking_multi_turn_uses_generation_prompt_kwargs():
    tokenizer = FakeDeepSeekV4Tokenizer()
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2", "reasoning_content": "r2"},
    ]
    config = make_sft_config(
        agentflow={"tokenizer": {"generation_prompt_kwargs": {"thinking": True}}},
    )
    sampler = SFTSampler(config, [FakeDataSource([make_group(messages)])], tokenizer=tokenizer)
    traj = sampler(step=1)[0].trajectories[0]

    assert traj.tokens == tokenizer.apply_chat_template(messages, thinking=True)
    # No tools, so drop_thinking is on: the history turn's reasoning is dropped and
    # it opens with the generation-prompt "</think>"; only the final turn keeps its
    # reasoning, opening with "<think>". Both think tokens are skipped as prompt.
    assert trained_spans(traj) == ["a1" + EOS, "r2</think>a2" + EOS]
    assert decode(traj.tokens[: -len(traj.loss_masks)]) == (
        tokenizer.BOS + "<｜User｜>q1<｜Assistant｜></think>"
    )


# ---------------------------------------------------------------------------
# Batches across steps and data sources
# ---------------------------------------------------------------------------

def test_call_concatenates_datasources_in_config_order():
    ds_a = FakeDataSource([make_group(MESSAGES, prompt_id="a0"), make_group(ALL_MASKED, prompt_id="a1")])
    ds_b = FakeDataSource([make_group(AGENTIC, {"tools": TOOLS}, prompt_id="b0", ds_index=1)])
    config = make_sft_config(data_sources=[
        {"num_prompts_per_step": 2, "num_trajectories_per_prompt": 1},
        {"num_prompts_per_step": 1, "num_trajectories_per_prompt": 1},
    ])
    sampler = SFTSampler(config, [ds_a, ds_b], tokenizer=FakeQwenTokenizer())

    groups = sampler(step=5)

    assert [g.prompt_id for g in groups] == ["a0", "a1", "b0"]
    assert (ds_a.calls, ds_b.calls) == ([(2, 5)], [(1, 5)])
    assert [trained_spans(g.trajectories[0]) for g in groups] == [
        ["a1" + IM_END, "a2" + IM_END], [], AGENTIC_SPANS,
    ]


def test_sft_max_seq_len_applies_per_data_source():
    """Each data source cuts its own conversations; the clip ratio counts both."""
    full = len(FakeQwenTokenizer().apply_chat_template(MESSAGES))
    ds_a = FakeDataSource([make_group(MESSAGES, prompt_id="a0")])
    ds_b = FakeDataSource([make_group(MESSAGES, prompt_id="b0", ds_index=1)])
    config = make_sft_config(data_sources=[
        {"num_prompts_per_step": 1, "num_trajectories_per_prompt": 1, "sft_max_seq_len": full - 1},
        {"num_prompts_per_step": 1, "num_trajectories_per_prompt": 1, "sft_max_seq_len": None},
    ])
    sampler = SFTSampler(config, [ds_a, ds_b], tokenizer=FakeQwenTokenizer())

    groups = sampler(step=1)

    assert [len(g.trajectories[0].tokens) for g in groups] == [full - 1, full]
    assert sampler.metrics[1]["sft/seq_length_clip_ratio"] == 0.5


def test_datasource_count_must_match_config():
    config = make_sft_config(data_sources=[
        {"num_prompts_per_step": 1, "num_trajectories_per_prompt": 1},
        {"num_prompts_per_step": 1, "num_trajectories_per_prompt": 1},
    ])
    sampler = SFTSampler(config, [FakeDataSource([make_group(MESSAGES)])], tokenizer=FakeQwenTokenizer())
    with pytest.raises(ValueError):
        sampler(step=1)


def test_replayed_group_is_retokenized_not_accumulated():
    """The buffer may hand the same group out again; fields are rebuilt from ``prompt``."""
    group = make_group(AGENTIC, metadata={"tools": TOOLS})
    sampler = make_sampler([group])

    traj = sampler(step=1)[0].trajectories[0]
    first = (list(traj.tokens), list(traj.loss_masks), list(traj.segments))
    traj = sampler(step=2)[0].trajectories[0]

    assert (traj.tokens, traj.loss_masks, traj.segments) == first
    assert len(traj.segments) == 1


def test_first_sample_is_logged_once(caplog):
    sampler = make_sampler([make_group(MESSAGES, prompt_id="p0"), make_group(MESSAGES, prompt_id="p1")])
    with caplog.at_level(logging.INFO, logger="coda.controller.sft_sampler"):
        sampler(step=1)
        sampler(step=2)

    logged = [r.getMessage() for r in caplog.records if "first sample" in r.getMessage()]
    assert len(logged) == 1
    assert "p0_t0" in logged[0] and repr("a1" + IM_END) in logged[0]


def test_mixed_batch_flattens_to_one_training_row_per_conversation(monkeypatch):
    """SFTSampler output goes through the unchanged DP split + Ray flatten of the RL path."""
    pytest.importorskip("torch")
    pytest.importorskip("ray")
    from coda.data_factory import data_processor

    captured = []
    monkeypatch.setattr(data_processor.torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(data_processor.ray, "put", lambda data: captured.append(data) or len(captured))

    expected = {
        "multi": ["a1" + IM_END, "a2" + IM_END],
        "agentic": AGENTIC_SPANS,
        "masked": [],
        "long": ["abcdefghij" + IM_END],
    }
    groups = [
        make_group(MESSAGES, prompt_id="multi", ds_index=0),
        make_group(AGENTIC, {"tools": TOOLS}, prompt_id="agentic", ds_index=0),
        make_group(ALL_MASKED, prompt_id="masked", ds_index=1),
        make_group(LONG_ANSWER, prompt_id="long", ds_index=1),
    ]
    sampled = make_sampler(groups)(step=1)

    shards = data_processor.split_traj_group_by_dp(sampled, dp_size=2, num_mini_batch=1)
    data_processor.put_dp_shards_to_ray(shards, dp_size=2)

    assert len(captured) == 2
    rows = {}
    for shard in captured:
        assert len(shard["prompt_id"]) == 2  # one row per conversation, balanced over DP
        for i, prompt_id in enumerate(shard["prompt_id"]):
            tokens, mask = shard["tokens"][i], shard["loss_masks"][i]
            resp_len = shard["response_lengths"][i]
            assert shard["total_lengths"][i] == len(tokens)
            assert resp_len == len(mask) == len(shard["rollout_log_probs"][i]) == len(shard["token_rewards"][i])
            rows[prompt_id] = decode_trainable_spans(
                FakeQwenTokenizer(), tokens[-resp_len:].tolist(), mask.tolist()
            )
    assert rows == expected


# ---------------------------------------------------------------------------
# apply_sft_forced_settings / validate_sft_config
# ---------------------------------------------------------------------------

def test_valid_sft_config_passes():
    validate_sft_config(make_sft_config())


def test_eval_interval_zero_is_allowed():
    validate_sft_config(make_sft_config(rollout={"eval": {"interval": 0}}))


def test_second_datasource_with_multiple_trajectories_raises():
    config = make_sft_config(data_sources=[
        {"num_prompts_per_step": 1, "num_trajectories_per_prompt": 1},
        {"num_prompts_per_step": 1, "num_trajectories_per_prompt": 4},
    ])
    with pytest.raises(ValueError, match=r"data_sources\[1\]\.num_trajectories_per_prompt=1"):
        validate_sft_config(config)


@pytest.mark.parametrize(
    "overrides, logged",
    [
        # default.yaml has colocate=true / use_rollout_log_probs=false
        ({}, "colocate=False (was True), trainer.use_rollout_log_probs=True (was False)"),
        ({"colocate": False, "trainer": {"use_rollout_log_probs": True}},
         "colocate=False, trainer.use_rollout_log_probs=True"),
    ],
)
def test_sft_forced_settings_are_set_and_logged(overrides, logged, caplog):
    config = make_sft_config(**overrides)
    with caplog.at_level(logging.INFO, logger="coda.controller.sft_sampler"):
        apply_sft_forced_settings(config)
    assert config.colocate is False
    assert config.trainer.use_rollout_log_probs is True
    assert any(r.getMessage() == f"[sft] run_mode=sft sets: {logged}" for r in caplog.records)


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"algorithm": {"policy_loss": "grpo"}}, "policy_loss=sft"),
        ({"algorithm": {"advantage_estimator": "grpo"}}, "advantage_estimator=sft"),
        ({"algorithm": {"ref_kl": {"enable": True}}}, "algorithm.ref_kl.enable"),
        ({"algorithm": {"is_correction": {"enable": True}}}, "algorithm.is_correction.enable"),
        ({"algorithm": {"m2po": {"enable": True}}}, "algorithm.m2po.enable"),
        ({"algorithm": {"opsm": {"enable": True}}}, "algorithm.opsm.enable"),
        ({"trainer": {"use_rollout_routing_replay": True}}, "trainer.use_rollout_routing_replay"),
        ({"opd": {"enable": True}}, "opd.enable"),
        ({"fully_async": {"enable": True}}, "fully_async.enable"),
        ({"data_source": {"num_trajectories_per_prompt": 2}}, "num_trajectories_per_prompt=1"),
        ({"rollout": {"eval": {"interval": 10}}}, "rollout.eval.interval<=0"),
    ],
)
def test_invalid_sft_config_raises(overrides, match):
    with pytest.raises(ValueError, match=match):
        validate_sft_config(make_sft_config(**overrides))
