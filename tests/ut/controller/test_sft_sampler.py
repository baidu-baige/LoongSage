"""Unit tests for coda.controller.sft_sampler.

Run: python -m pytest tests/ut/controller/test_sft_sampler.py -v
"""

import logging
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from coda.agentflow.trajectory_store import Trajectory, TrajectoryGroup, TrajectoryStatus
from coda.controller.sft_sampler import (
    DEFAULT_SETTINGS,
    SFTSampler,
    decode_trainable_tokens,
    trainable_suffix_length,
    validate_config,
)
from tests.ut.agentflow.fake_tokenizers import DS_BOS, FakeDeepSeekV4Tokenizer, FakeQwenTokenizer

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
    return decode_trainable_tokens(FakeQwenTokenizer(), *response_slice(traj))


IM_END = "<|im_end|>"
# A ChatML assistant message's own render ends with "\n" after ``<|im_end|>``; as in verl it is trained.
TURN_END = IM_END + "\n"
# A Qwen3.5 turn without reasoning renders an empty think block; after the pre-filled
# "<think>\n" the model emits this.
EMPTY_REASONING = "\n</think>\n\n"
EOS = "<｜end▁of▁sentence｜>"

MESSAGES = [
    {"role": "user", "content": "q1"},
    {"role": "assistant", "content": "a1"},
    {"role": "user", "content": "q2"},
    {"role": "assistant", "content": "a2"},
]

# What SFTSampler renders MESSAGES to.  Each assistant message is rendered on its own,
# so a1 keeps the empty think block the whole render drops from history turns.
MESSAGES_RENDERED = (
    "<|im_start|>user\nq1<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\na1<|im_end|>\n"
    "<|im_start|>user\nq2<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\na2<|im_end|>\n"
)

MESSAGES_SPANS = [EMPTY_REASONING + "a1" + TURN_END, EMPTY_REASONING + "a2" + TURN_END]

TOOLS = [{"type": "function", "function": {"name": "read"}}]

# A coding-agent episode: reasoning + tool calls, tool outputs in between and a
# plain final answer.
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
    },
    {"role": "tool", "content": "TOOL_OUT_2"},
    {"role": "assistant", "content": "done", "tool_calls": [{"function": {"name": "test"}}]},
    {"role": "tool", "content": "TOOL_OUT_3"},
    {"role": "assistant", "content": "fixed"},
]

AGENTIC_SPANS = [
    # The pre-filled "<think>\n" stays in the prompt; the reasoning itself is trained.
    "plan\n</think>\n\nlook\n<tool_call>read</tool_call>" + TURN_END,
    EMPTY_REASONING + "retry\n<tool_call>edit</tool_call>" + TURN_END,
    EMPTY_REASONING + "done\n<tool_call>test</tool_call>" + TURN_END,
    EMPTY_REASONING + "fixed" + TURN_END,
]

# Nothing to train: no assistant message.
NO_ASSISTANT = [{"role": "user", "content": "only a question"}]


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

    resp_len = len(traj.loss_masks)
    assert traj.tokens == sampler.tokenizer.encode(MESSAGES_RENDERED)
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
    assert trained == "".join(MESSAGES_SPANS)


def test_call_reads_tools_from_metadata():
    tools = [{"type": "function", "function": {"name": "f"}}]
    sampler, traj = sample_one(make_group(MESSAGES, metadata={"tools": tools}))
    rendered = f"<|im_start|>system\n# Tools\n{tools}<|im_end|>\n" + MESSAGES_RENDERED
    assert traj.tokens == sampler.tokenizer.encode(rendered)


def test_truncation_keeps_prefix():
    # 100 tokens: past the first assistant turn (ends at 84), inside the second user turn.
    full = FakeQwenTokenizer().encode(MESSAGES_RENDERED)
    _, traj = sample_one(make_group(MESSAGES), max_seq_len=100)

    assert traj.tokens == full[:100]
    assert traj.segments[0].token_end == 100
    assert trained_spans(traj) == MESSAGES_SPANS[:1]
    assert traj.loss_masks[-1] == 0  # the cut-off tail is context


def test_max_seq_len_equal_to_length_does_not_truncate():
    full = FakeQwenTokenizer().encode(MESSAGES_RENDERED)
    _, traj = sample_one(make_group(MESSAGES), max_seq_len=len(full))

    assert traj.tokens == full
    assert trained_spans(traj) == MESSAGES_SPANS


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
    _, traj = sample_one(make_group(NO_ASSISTANT))
    assert traj.loss_masks == [0]
    assert len(traj.rollout_log_probs) == 1


def test_truncated_away_response_falls_back_to_zero_weight_row():
    _, traj = sample_one(make_group(MESSAGES), max_seq_len=10)  # cut before the first assistant turn
    assert len(traj.tokens) == 10
    assert traj.loss_masks == [0]
    assert traj.segments[0].logprob_end == 1


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
    full = FakeQwenTokenizer().encode(MESSAGES_RENDERED)
    sampler, _ = sample_one(make_group(MESSAGES), max_seq_len=len(full) - 1)

    assert sampler.metrics[1]["sft/seq_length_max"] == len(full) - 1
    assert sampler.metrics[1]["sft/seq_length_clip_ratio"] == 1.0


def test_metrics_do_not_flag_exact_length_as_truncated():
    full = FakeQwenTokenizer().encode(MESSAGES_RENDERED)
    sampler, _ = sample_one(make_group(MESSAGES), max_seq_len=len(full))

    assert sampler.metrics[1]["sft/seq_length_clip_ratio"] == 0.0


def test_metrics_flag_zero_weight_samples():
    sampler, _ = sample_one(make_group(NO_ASSISTANT))

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

def test_agentic_episode_trains_every_assistant_turn():
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
    # Tool outputs sit inside the response slice as context.
    for context in ("TOOL_OUT_1", "TOOL_OUT_2", "TOOL_OUT_3"):
        assert context in response
        assert all(context not in span for span in trained_spans(traj))


def test_user_turn_between_assistant_turns_stays_in_response_as_context():
    _, traj = sample_one(make_group(MESSAGES))

    tokens, mask = response_slice(traj)
    assert decode(tokens).startswith(EMPTY_REASONING + "a1" + TURN_END + "<|im_start|>user\nq2")
    assert trained_spans(traj) == MESSAGES_SPANS
    assert sum(mask) == len("".join(MESSAGES_SPANS))


# "<|im_start|>user\nq<|im_end|>\n" + "<|im_start|>assistant\n" is 51 characters; the
# pre-filled "<think>\n" takes the generation prompt to 59.
LONG_ANSWER = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "abcdefghij"}]


def test_truncation_mid_turn_trains_kept_head_without_end_marker():
    max_seq_len = 59 + len(EMPTY_REASONING + "abcde")
    _, traj = sample_one(make_group(LONG_ANSWER), max_seq_len=max_seq_len)

    assert len(traj.tokens) == max_seq_len
    assert trained_spans(traj) == [EMPTY_REASONING + "abcde"]
    assert traj.loss_masks == [1] * len(EMPTY_REASONING + "abcde")


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
        sampler.tokenizer_manager, "tokenize", lambda messages, tools=None: ([ord("a"), ord("b")], [1, 1])
    )
    traj = sampler(step=1)[0].trajectories[0]

    assert traj.segments[0].logprob_end == len(traj.tokens) - 1
    assert traj.loss_masks == [1]
    assert response_slice(traj)[0] == [ord("b")]
    assert len(traj.rollout_log_probs) == len(traj.tokens) - 1


@pytest.mark.parametrize(
    "mask, expected",
    [([0, 0, 1, 0, 1], 3), ([1, 1], 2), ([0, 0], 0), ([], 0)],
)
def test_trainable_suffix_length(mask, expected):
    assert trainable_suffix_length(mask) == expected


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

    # Each message is rendered on its own in thinking mode: every user message ends with
    # the pre-filled "<think>" and every assistant message keeps its reasoning.
    assert tokenizer.decode(traj.tokens) == (
        DS_BOS + "<｜User｜>q1<｜Assistant｜><think>r1</think>a1" + EOS
        + "<｜User｜>q2<｜Assistant｜><think>r2</think>a2" + EOS
    )
    assert decode_trainable_tokens(tokenizer, *response_slice(traj)) == ["r1</think>a1" + EOS, "r2</think>a2" + EOS]
    assert tokenizer.decode(traj.tokens[: -len(traj.loss_masks)]) == DS_BOS + "<｜User｜>q1<｜Assistant｜><think>"


# ---------------------------------------------------------------------------
# Batches across steps and data sources
# ---------------------------------------------------------------------------

def test_call_concatenates_datasources_in_config_order():
    ds_a = FakeDataSource([make_group(MESSAGES, prompt_id="a0"), make_group(NO_ASSISTANT, prompt_id="a1")])
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
        MESSAGES_SPANS, [], AGENTIC_SPANS,
    ]


def test_sft_max_seq_len_applies_per_data_source():
    """Each data source cuts its own conversations; the clip ratio counts both."""
    full = len(FakeQwenTokenizer().encode(MESSAGES_RENDERED))
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
    assert "p0_t0" in logged[0] and repr(MESSAGES_SPANS[0]) in logged[0]


def test_mixed_batch_flattens_to_one_training_row_per_conversation(monkeypatch):
    """SFTSampler output goes through the unchanged DP split + Ray flatten of the RL path."""
    pytest.importorskip("torch")
    pytest.importorskip("ray")
    from coda.data_factory import data_processor

    captured = []
    monkeypatch.setattr(data_processor.torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(data_processor.ray, "put", lambda data: captured.append(data) or len(captured))

    expected = {
        "multi": MESSAGES_SPANS,
        "agentic": AGENTIC_SPANS,
        "no_assistant": [],
        "long": [EMPTY_REASONING + "abcdefghij" + TURN_END],
    }
    groups = [
        make_group(MESSAGES, prompt_id="multi", ds_index=0),
        make_group(AGENTIC, {"tools": TOOLS}, prompt_id="agentic", ds_index=0),
        make_group(NO_ASSISTANT, prompt_id="no_assistant", ds_index=1),
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
            rows[prompt_id] = decode_trainable_tokens(
                FakeQwenTokenizer(), tokens[-resp_len:].tolist(), mask.tolist()
            )
    assert rows == expected


# ---------------------------------------------------------------------------
# validate_config
# ---------------------------------------------------------------------------

def sft_warnings(caplog, config):
    """Run validate_config on ``config``; returns the warnings it logged."""
    with caplog.at_level(logging.WARNING, logger="coda.controller.sft_sampler"):
        validate_config(config)
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


def make_matching_config(**overrides):
    """An sft config that already holds every DEFAULT_SETTINGS value."""
    config = make_sft_config(**overrides)
    for key, value in DEFAULT_SETTINGS.items():
        OmegaConf.update(config, key, value)
    return config


def test_valid_sft_config_passes():
    validate_config(make_sft_config())


def test_every_datasource_is_set_to_one_trajectory_per_prompt(caplog):
    config = make_matching_config(data_sources=[
        {"num_prompts_per_step": 1, "num_trajectories_per_prompt": 1},
        {"num_prompts_per_step": 1, "num_trajectories_per_prompt": 4},
    ])
    assert sft_warnings(caplog, config) == [
        "[sft] run_mode=sft sets data_sources[1].num_trajectories_per_prompt=1 (was 4)",
    ]
    assert [ds.num_trajectories_per_prompt for ds in config.data_sources] == [1, 1]


def test_default_yaml_values_are_overridden_with_warnings(caplog):
    # default.yaml has colocate=true / use_rollout_log_probs=false / rollout.eval.interval=-1
    config = make_sft_config()
    assert sft_warnings(caplog, config) == [
        "[sft] run_mode=sft sets colocate=False (was True), trainer.use_rollout_log_probs=True (was False)",
    ]
    assert all(OmegaConf.select(config, key) == value for key, value in DEFAULT_SETTINGS.items())


def test_config_matching_default_settings_logs_no_warning(caplog):
    assert sft_warnings(caplog, make_matching_config()) == []


@pytest.mark.parametrize(
    "key, configured",
    [
        ("algorithm.ref_kl.enable", True),
        ("algorithm.is_correction.enable", True),
        ("algorithm.m2po.enable", True),
        ("algorithm.opsm.enable", True),
        ("opd.enable", True),
        ("fully_async.enable", True),
        ("trainer.use_rollout_routing_replay", True),
        ("rollout.eval.interval", 10),
    ],
)
def test_conflicting_setting_is_overridden_with_warning(key, configured, caplog):
    config = make_matching_config()
    OmegaConf.update(config, key, configured)
    expected = DEFAULT_SETTINGS[key]
    assert sft_warnings(caplog, config) == [f"[sft] run_mode=sft sets {key}={expected} (was {configured})"]
    assert OmegaConf.select(config, key) == expected


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"algorithm": {"policy_loss": "grpo"}}, "policy_loss=sft"),
        ({"algorithm": {"advantage_estimator": "grpo"}}, "advantage_estimator=sft"),
    ],
)
def test_invalid_sft_config_raises(overrides, match):
    with pytest.raises(ValueError, match=match):
        validate_config(make_sft_config(**overrides))
