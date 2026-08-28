"""Unit tests for ``coda.agentflow.router.parser``."""

import asyncio
import copy
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

# coda.agentflow.router.parser imports six symbols from
# sglang.srt.parser.template_detection, which only exists in the sglang build the
# project's Dockerfile pins (lmsysorg/sglang:v0.5.16). Skip rather than fail
# collection on other sglang versions.
pytest.importorskip(
    "sglang.srt.parser.template_detection",
    reason="sglang.srt.parser.template_detection is unavailable in this sglang build",
)

from coda.agentflow.router.parser import TrajectoryParser, TurnInputContext
from coda.agentflow.router.parser_middleware import ParserMiddleware
from coda.agentflow import tokenizer_manager
from coda.agentflow.tokenizer_manager import (
    _deepseek_v4_thinking_mode,
    _wrap_deepseek_v4_tokenizer,
)
from coda.agentflow.trajectory_store import Segment, Trajectory, TrajectoryStore, TrajectoryStatus, Triplet


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, "chat"),
        ({"enable_thinking": False}, "chat"),
        ({"enable_thinking": True}, "thinking"),
        ({"thinking": True}, "thinking"),
    ],
)
def test_deepseek_v4_thinking_mode(kwargs: dict, expected: str) -> None:
    assert _deepseek_v4_thinking_mode(kwargs) == expected


class FakeDeepSeekV4Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return list(text.encode())

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        return bytes(token_ids).decode()


def test_deepseek_v4_tokenizer_uses_official_encoder() -> None:
    tokenizer = _wrap_deepseek_v4_tokenizer(FakeDeepSeekV4Tokenizer())
    tools = [{
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a command",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
            },
        },
    }]

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "inspect"}],
        tools=tools,
        tokenize=False,
        enable_thinking=False,
    )

    assert prompt.startswith("<｜begin▁of▁sentence｜>")
    assert "<｜DSML｜tool_calls>" in prompt
    assert '"name": "bash"' in prompt
    assert prompt.endswith("<｜Assistant｜></think>")


def test_deepseek_v4_empty_system_is_only_bos() -> None:
    tokenizer = _wrap_deepseek_v4_tokenizer(FakeDeepSeekV4Tokenizer())

    async def apply_chat_template(messages, **kwargs):
        return tokenizer.apply_chat_template(messages, **kwargs)

    parser = TrajectoryParser(
        trajectory_store=TrajectoryStore(),
        tokenizer_manager=SimpleNamespace(
            model_family="deepseek_v4",
            think_tags=("<think>", "</think>"),
            apply_chat_template=apply_chat_template,
        ),
    )

    system_prompt_len = asyncio.run(parser._get_system_prompt_len())

    assert system_prompt_len == len("<｜begin▁of▁sentence｜>".encode())


def test_deepseek_v4_first_turn_encodes_tools_in_system_prompt() -> None:
    tokenizer = _wrap_deepseek_v4_tokenizer(FakeDeepSeekV4Tokenizer())

    async def apply_chat_template(messages, **kwargs):
        return tokenizer.apply_chat_template(messages, **kwargs)

    manager = SimpleNamespace(
        system_prompt_len=len("<｜begin▁of▁sentence｜>".encode()),
        model_family="deepseek_v4",
        think_tags=("<think>", "</think>"),
        apply_chat_template=apply_chat_template,
    )
    parser = TrajectoryParser(
        trajectory_store=TrajectoryStore(),
        tokenizer_manager=manager,
    )
    trajectory = Trajectory(trajectory_id="traj-dsv4-tools", prompt_id="prompt-0")
    tools = [{
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a command",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
            },
        },
    }]

    turn = asyncio.run(parser.build_turn_input(
        trajectory,
        messages=[{"role": "user", "content": "inspect"}],
        tools=tools,
    ))
    prompt = bytes(turn.input_ids).decode()

    assert prompt.count("## Tools") == 1
    assert '"name": "bash"' in prompt
    assert prompt.endswith("<｜Assistant｜></think>")


def test_deepseek_v4_continuation_strips_only_bos() -> None:
    tokenizer = _wrap_deepseek_v4_tokenizer(FakeDeepSeekV4Tokenizer())
    async def apply_chat_template(messages, **kwargs):
        return tokenizer.apply_chat_template(messages, **kwargs)

    manager = SimpleNamespace(
        system_prompt_len=len("<｜begin▁of▁sentence｜>".encode()),
        model_family="deepseek_v4",
        think_tags=("<think>", "</think>"),
        apply_chat_template=apply_chat_template,
    )
    parser = TrajectoryParser(
        trajectory_store=TrajectoryStore(),
        tokenizer_manager=manager,
    )
    messages = [
        {"role": "system", "content": ""},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_0",
                "type": "function",
                "function": {"name": "bash", "arguments": '{"cmd":"pwd"}'},
            }],
        },
        {"role": "tool", "tool_call_id": "call_0", "content": "/repo"},
    ]

    full_ids = asyncio.run(manager.apply_chat_template(messages, add_generation_prompt=True))
    continuation_ids = asyncio.run(parser._tokenize_delta_messages(messages[1:]))

    assert continuation_ids == full_ids[manager.system_prompt_len:]
    assert bytes(continuation_ids).decode().startswith("\n\n<｜DSML｜tool_calls>")
    assert "<tool_result>/repo</tool_result>" in bytes(continuation_ids).decode()


def test_tokenize_delta_falls_back_to_dummy_user_for_qwen35_style_template() -> None:
    """Templates that reject system-only conversations (Qwen3.5) use the verl-style fallback."""
    async def apply_chat_template(messages, add_generation_prompt=True, tools=None, **kwargs):
        if not any(m.get("role") == "user" for m in messages):
            raise ValueError("Qwen3.5 requires at least one user message")
        ids = []
        for message in messages:
            ids.extend([1] if message["role"] == "user" else [2, 3])
        if add_generation_prompt:
            ids.append(9)
        return ids

    parser = TrajectoryParser(
        trajectory_store=TrajectoryStore(),
        tokenizer_manager=SimpleNamespace(apply_chat_template=apply_chat_template),
    )
    delta = [
        {"role": "assistant", "content": "a"},
        {"role": "tool", "content": "r"},
    ]

    delta_ids = asyncio.run(parser._tokenize_delta_messages(delta))

    # dummy user prefix ([1]) is stripped; failure of the system-only probe is cached
    assert delta_ids == [2, 3, 2, 3, 9]
    assert parser._system_prompt_len == -1
    assert parser._dummy_user_prefix_len == 1
    assert asyncio.run(parser._tokenize_delta_messages(delta)) == [2, 3, 2, 3, 9]


def test_deepseek_v4_process_initializer_wraps_worker_tokenizer(tmp_path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({
        "architectures": ["DeepseekV4ForCausalLM"],
    }))
    fake_tokenizer = FakeDeepSeekV4Tokenizer()

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tokenizer):
        tokenizer_manager._init_process_tokenizer(str(tmp_path), None)

    assert tokenizer_manager._PROCESS_TOKENIZER.model_family == "deepseek_v4"
    prompt = tokenizer_manager._PROCESS_TOKENIZER.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        tokenize=False,
    )
    assert prompt.endswith("<｜Assistant｜></think>")


def make_parser() -> TrajectoryParser:
    """Create a parser with the minimum tokenizer stub needed for construction."""
    tokenizer = SimpleNamespace(system_prompt_len=0)
    return TrajectoryParser(
        trajectory_store=TrajectoryStore(),
        tokenizer_manager=tokenizer,
        r3_enabled=True,
    )


@pytest.mark.parametrize(
    ("model_family", "expected_skip_special_tokens"),
    [
        (None, None),
        ("deepseek_v4", False),
    ],
)
def test_parser_sampling_params_keep_special_tokens_only_for_deepseek_v4(
    model_family, expected_skip_special_tokens
) -> None:
    tokenizer_manager = SimpleNamespace(
        system_prompt_len=0,
        model_family=model_family,
    )
    middleware = ParserMiddleware(
        MagicMock(),
        trajectory_store=TrajectoryStore(),
        tokenizer_manager=tokenizer_manager,
    )

    payload = json.loads(
        middleware._build_generate_body(
            {},
            [101],
            request_id="req-0",
        )
    )

    sampling_params = payload.get("sampling_params", {})
    assert sampling_params.get("skip_special_tokens") is expected_skip_special_tokens
    # SGLang must keep trimming the matched EOS: the parsers work on EOS-free text.
    assert sampling_params.get("no_stop_trim") is None


def test_complete_response_is_wrapped_as_openai_stream() -> None:
    response = ParserMiddleware._to_streaming_response({
        "id": "completion-1",
        "object": "chat.completion",
        "created": 123,
        "model": "default",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "reasoning_content": "thinking",
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"cmd":"pwd"}'},
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    })

    async def read_body() -> str:
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
        return "".join(chunks)

    events = [
        json.loads(block.removeprefix("data: "))
        for block in asyncio.run(read_body()).strip().split("\n\n")[:-1]
    ]
    assert events[0]["choices"][0]["delta"]["reasoning_content"] == "thinking"
    assert events[0]["choices"][0]["delta"]["tool_calls"][0]["index"] == 0
    assert events[1]["choices"][0]["finish_reason"] == "tool_calls"
    assert events[1]["usage"]["completion_tokens"] == 2


def test_append_routed_experts_materializes_terminal_row() -> None:
    """The last routed-experts row should reuse the previous token after append."""
    parser = make_parser()
    trajectory = Trajectory(
        trajectory_id="traj-r3",
        prompt_id="prompt-0",
        tokens=[10, 11, 12],
    )
    new_chunk = torch.tensor(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ],
        dtype=torch.int32,
    )

    parser._append_routed_experts(trajectory, new_chunk)

    assert torch.equal(trajectory.rollout_routed_experts[-1], trajectory.rollout_routed_experts[-2])
    assert torch.equal(
        trajectory.rollout_routed_experts,
        torch.tensor(
            [
                [1, 2, 3],
                [4, 5, 6],
                [4, 5, 6],
            ],
            dtype=torch.int32,
        ),
    )


def test_append_routed_experts_keeps_single_row_unchanged() -> None:
    """A one-row routed-experts tensor should stay untouched."""
    parser = make_parser()
    trajectory = Trajectory(
        trajectory_id="traj-r3-short",
        prompt_id="prompt-0",
        tokens=[10],
    )
    new_chunk = torch.tensor([[1, 2, 3]], dtype=torch.int32)

    parser._append_routed_experts(trajectory, new_chunk)

    assert torch.equal(trajectory.rollout_routed_experts, new_chunk)


def test_build_assistant_message_decodes_weight_version() -> None:
    """Request-level SGLang weight_version should be decoded from meta_info."""
    parser = make_parser()

    payload, _, _ = parser.build_assistant_message({
        "text": "hi",
        "meta_info": {
            "weight_version": "3",
            "output_token_logprobs": [[-0.1, 101], [-0.2, 102]],
        },
    })

    assert payload.weight_version == 3


@pytest.fixture
def deepseek_v4_model_dir(tmp_path) -> str:
    """Model dir carrying just enough config for SGLang's architecture detection."""
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "deepseek_v4",
        "architectures": ["DeepseekV4ForCausalLM"],
    }))
    return str(tmp_path)


def make_deepseek_v4_parser(model_dir: str | None) -> TrajectoryParser:
    """Parser for a DeepSeek-V4 tokenizer, which exposes no Jinja chat template."""
    return TrajectoryParser(
        trajectory_store=TrajectoryStore(),
        tokenizer_manager=SimpleNamespace(
            system_prompt_len=1,
            tokenizer=SimpleNamespace(chat_template=None, name_or_path=model_dir),
        ),
    )


def test_build_assistant_message_parses_deepseek_v4_tool_call(deepseek_v4_model_dir: str) -> None:
    parser = make_deepseek_v4_parser(deepseek_v4_model_dir)
    assert parser._tool_call_parser_name == "deepseekv4"
    generated_text = (
        "\n\n<｜DSML｜tool_calls>\n"
        '<｜DSML｜invoke name="bash">\n'
        '<｜DSML｜parameter name="cmd" string="true">pwd</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>"
    )

    _, message, finish_reason = parser.build_assistant_message({
        "text": generated_text,
        "meta_info": {"weight_version": "1", "output_token_logprobs": [[-0.1, 101]]},
    })

    assert message == {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_0",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"cmd": "pwd"}'},
        }],
    }
    assert finish_reason == "tool_calls"


def test_build_assistant_message_parses_deepseek_v4_thinking(deepseek_v4_model_dir: str) -> None:
    """DeepSeek-V4 prefills <think>, so only the end tag reaches the reasoning parser."""
    parser = make_deepseek_v4_parser(deepseek_v4_model_dir)
    assert parser._reasoning_parser_name == "deepseek-v4"

    _, message, finish_reason = parser.build_assistant_message({
        "text": "reasoning</think>answer",
        "meta_info": {"weight_version": "1", "output_token_logprobs": [[-0.1, 101]]},
    })

    assert message == {
        "role": "assistant",
        "content": "answer",
        "reasoning_content": "reasoning",
    }
    assert finish_reason is None


def test_deepseek_v4_parsers_disabled_when_model_config_unavailable(tmp_path) -> None:
    """Architecture detection cannot read a config.json, so parsing degrades to off."""
    parser = make_deepseek_v4_parser(str(tmp_path / "missing"))
    assert parser._reasoning_parser_name is None
    assert parser._tool_call_parser_name is None


def make_sglang_parser(**kwargs) -> TrajectoryParser:
    """Parser wired to the SGLang reasoning/tool-call detectors."""
    return TrajectoryParser(
        trajectory_store=TrajectoryStore(),
        tokenizer_manager=SimpleNamespace(system_prompt_len=0),
        **kwargs,
    )


BASH_TOOLS = [{
    "type": "function",
    "function": {
        "name": "bash",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
        },
    },
}]


@pytest.mark.parametrize(
    ("config", "expected_reasoning", "expected_tool_call"),
    [
        # Config values are SGLang keys verbatim, and the two key spaces differ:
        # "qwen3" only exists as a reasoning key, "qwen25" only as a tool-call key.
        ("qwen3", "qwen3", None),
        ("qwen25", None, "qwen25"),
        ("qwen3_coder", None, "qwen3_coder"),
        (True, None, None),          # no chat template to auto-detect from
        (False, None, None),
        ("none", None, None),
        ("unsupported-model", None, None),
    ],
)
def test_resolve_parser_name_validates_config_against_sglang_keys(
    config, expected_reasoning, expected_tool_call
) -> None:
    parser = make_sglang_parser(reasoning_parser=config, tool_call_parser=config)
    assert parser._reasoning_parser_name == expected_reasoning
    assert parser._tool_call_parser_name == expected_tool_call


# Minimal Qwen3-style template: the enable_thinking toggle defaulting to true is the
# signature SGLang's detection rules key off.
QWEN3_CHAT_TEMPLATE = (
    "{% if not enable_thinking is defined %}{% set enable_thinking = true %}{% endif %}"
    "{% for message in messages %}<|im_start|>{{ message.role }}\n"
    "{{ message.content }}<|im_end|>\n{% endfor %}"
)


def test_unset_parsers_auto_detect_from_chat_template() -> None:
    """Unset config falls back to SGLang's template/vocab-based auto-detection."""
    parser = TrajectoryParser(
        trajectory_store=TrajectoryStore(),
        tokenizer_manager=SimpleNamespace(
            system_prompt_len=0,
            tokenizer=SimpleNamespace(
                chat_template=QWEN3_CHAT_TEMPLATE,
                get_vocab=lambda: {},
            ),
        ),
    )
    assert parser._reasoning_parser_name == "qwen3"
    assert parser._tool_call_parser_name == "qwen"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<think>reason</think>answer", ("reason", "answer")),
        # Chat templates that prefill <think> leave only the end tag in the generation.
        ("reason</think>answer", ("reason", "answer")),
        ("<think>reason", ("reason", "")),
        ("answer", (None, "answer")),
    ],
)
def test_parse_reasoning_covers_prefill_and_plain_generations(text, expected) -> None:
    parser = make_sglang_parser(reasoning_parser="qwen3")
    assert parser._parse_reasoning(text) == expected


def test_parse_reasoning_disabled_returns_text_unchanged() -> None:
    parser = make_sglang_parser(reasoning_parser=False)
    assert parser._parse_reasoning("<think>reason</think>answer") == (
        None, "<think>reason</think>answer"
    )


@pytest.mark.parametrize("tools", [BASH_TOOLS, None])
def test_parse_tool_calls_extracts_qwen_markup_with_or_without_schema(tools) -> None:
    """Calls are forwarded even when the request omitted the tools field."""
    parser = make_sglang_parser(tool_call_parser="qwen25")
    text = 'ok\n<tool_call>\n{"name": "bash", "arguments": {"command": "ls"}}\n</tool_call>'

    assert parser._parse_tool_calls(text, tools) == (
        "ok",
        [{
            "id": "call_0",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command": "ls"}'},
        }],
    )


def test_parse_tool_calls_extracts_qwen3_coder_markup() -> None:
    parser = make_sglang_parser(tool_call_parser="qwen3_coder")
    text = (
        "ok<tool_call>\n<function=bash>\n"
        "<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>"
    )

    assert parser._parse_tool_calls(text, BASH_TOOLS) == (
        "ok",
        [{
            "id": "call_0",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command": "ls"}'},
        }],
    )


def test_parse_tool_calls_without_markup_returns_original_text() -> None:
    parser = make_sglang_parser(tool_call_parser="qwen25")
    assert parser._parse_tool_calls("plain answer", BASH_TOOLS) == ("plain answer", None)


def test_build_turn_input_caches_tools_for_follow_up_turns() -> None:
    """Tools sent on the first turn stay available when later turns omit them."""
    async def apply_chat_template(messages, **kwargs):
        return [1, 2, 3]

    parser = TrajectoryParser(
        trajectory_store=TrajectoryStore(),
        tokenizer_manager=SimpleNamespace(
            system_prompt_len=0,
            apply_chat_template=apply_chat_template,
        ),
    )
    trajectory = Trajectory(trajectory_id="traj-tools-cache", prompt_id="prompt-0")

    first = asyncio.run(parser.build_turn_input(
        trajectory, messages=[{"role": "user", "content": "hi"}], tools=BASH_TOOLS,
    ))
    assert first.tools == BASH_TOOLS

    trajectory.chat_completions[0] = [{"role": "user", "content": "hi"}]
    trajectory.metadata.pop("normalized_history", None)
    second = asyncio.run(parser.build_turn_input(
        trajectory, messages=[{"role": "user", "content": "hi"}], tools=None,
    ))
    assert second.tools == BASH_TOOLS


def test_build_assistant_message_splits_reasoning_and_tool_calls() -> None:
    parser = make_sglang_parser(reasoning_parser="qwen3", tool_call_parser="qwen25")
    generated_text = (
        "<think>plan</think>"
        '<tool_call>\n{"name": "bash", "arguments": {"command": "ls"}}\n</tool_call>'
    )

    _, message, finish_reason = parser.build_assistant_message(
        {
            "text": generated_text,
            "meta_info": {"weight_version": "1", "output_token_logprobs": [[-0.1, 101]]},
        },
        0,
        BASH_TOOLS,
    )

    assert message == {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_0",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command": "ls"}'},
        }],
        "reasoning_content": "plan",
    }
    assert finish_reason == "tool_calls"


def test_update_trajectory_records_weight_versions() -> None:
    """Trajectory bookkeeping should keep token-level versions aligned with logprobs."""
    parser = make_parser()
    trajectory = Trajectory(
        trajectory_id="traj-version",
        prompt_id="prompt-0",
        status=TrajectoryStatus.GENERATING,
    )
    parser.trajectory_store.add(trajectory.trajectory_id, trajectory)
    turn_ctx = TurnInputContext(
        delta_prompt_ids=[1, 2],
        input_ids=[1, 2],
        start_new_segment=True,
        raw_new_messages=[],
        target_segment_id=0,
        new_segment_origin="root",
    )
    payload, assistant_message, _ = parser.build_assistant_message({
        "text": "ok",
        "meta_info": {
            "weight_version": "7",
            "output_token_logprobs": [[-0.3, 3], [-0.4, 4]],
        },
    })

    parser.update_trajectory(trajectory, turn_ctx, payload, assistant_message)

    stored = parser.trajectory_store.get(["traj-version"])["traj-version"][-1]

    assert stored.start_rollout_weight_version == 7
    assert stored.end_rollout_weight_version == 7
    assert stored.rollout_log_probs == [-0.3, -0.4]
    assert stored.rollout_weight_versions == [7, 7]


def test_build_assistant_message_rejects_negative_weight_version() -> None:
    """-1 is the prompt-token / unset sentinel, so it must not come back as a real version."""
    parser = make_parser()

    with pytest.raises(ValueError, match="weight_version must be >= 0"):
        parser.build_assistant_message({
            "text": "ok",
            "meta_info": {
                "weight_version": -1,
                "output_token_logprobs": [[-0.1, 201]],
            },
        })


def test_append_triplet_preserves_zero_weight_version() -> None:
    """Version 0 is a real rollout version and must not be treated as invalid."""
    parser = make_parser()
    trajectory = Trajectory(
        trajectory_id="traj-version-zero",
        prompt_id="prompt-0",
    )

    parser._append_triplet(
        trajectory,
        delta_prompt_ids=[101],
        response_ids=[201],
        logprobs=[-0.1],
        weight_version=0,
        turn_ctx=TurnInputContext(
            delta_prompt_ids=[101],
            input_ids=[101, 201],
            start_new_segment=True,
            raw_new_messages=[],
            target_segment_id=0,
            new_segment_origin="root",
        ),
    )

    assert trajectory.rollout_weight_versions == [0]


@pytest.mark.parametrize("bad_version", [None, "abc"])
def test_build_assistant_message_rejects_non_numeric_weight_version(bad_version: object) -> None:
    """Non-numeric weight_version values should fail fast."""
    parser = make_parser()

    with pytest.raises(ValueError, match="weight_version must be int or str"):
        parser.build_assistant_message({
            "text": "ok",
            "meta_info": {
                "weight_version": bad_version,
                "output_token_logprobs": [[-0.1, 201]],
            },
        })


def test_strip_think_from_ids_returns_answer_slice() -> None:
    """The returned source slice is used for all response-space arrays."""
    parser = make_parser()
    parser.think_end_ids = [99]

    ids, logprobs, keep_start, keep_end = parser._strip_think_from_ids(
        [1, 2, 99, 3, 4],
        [-0.1, -0.2, -0.3, -0.4, -0.5],
    )

    assert ids == [3, 4]
    assert logprobs == [-0.4, -0.5]
    assert keep_start == 3
    assert keep_end == 5


def test_prune_last_response_think_preserves_existing_loss_mask_slice() -> None:
    """Think pruning should not turn pre-masked partial-rollout tokens back on."""
    parser = make_parser()
    parser.think_end_ids = [99]
    trajectory = Trajectory(
        trajectory_id="traj-prune-mask",
        prompt_id="prompt-0",
        tokens=[10, 11, 1, 2, 99, 3, 4],
        loss_masks=[0, 0, 0, 0, 1],
        rollout_log_probs=[-0.1, -0.2, -0.3, -0.4, -0.5],
        rollout_weight_versions=[1, 1, 1, 2, 2],
        segments=[
            Segment(
                token_start=0,
                token_end=7,
                logprob_start=0,
                logprob_end=5,
                triplets=[
                    Triplet(token_start=0, token_end=7, logprob_start=0, logprob_end=5),
                ],
            )
        ],
    )

    asyncio.run(parser._prune_last_response_think(trajectory))

    assert trajectory.tokens == [10, 11, 3, 4]
    assert trajectory.rollout_log_probs == [-0.4, -0.5]
    assert trajectory.loss_masks == [0, 1]
    assert trajectory.rollout_weight_versions == [2, 2]
    assert trajectory.segments[0].triplets[0].logprob_end == 2


def make_prefix_parser():
    """Parser whose tokenizer just returns len(messages)-scaled fake ids, for prefix tests."""
    counter = {"n": 0}

    async def apply_chat_template(messages, **kwargs):
        counter["n"] += 1
        # Distinct, deterministic fake ids per call so segments are easy to tell apart.
        base = counter["n"] * 1000
        return [base + i for i in range(len(messages))]

    return TrajectoryParser(
        trajectory_store=TrajectoryStore(),
        tokenizer_manager=SimpleNamespace(
            system_prompt_len=0,
            apply_chat_template=apply_chat_template,
        ),
    )


def test_build_turn_input_first_turn_sets_root_segment_fields() -> None:
    """The very first turn should describe a root segment targeting id 0."""
    parser = make_prefix_parser()
    trajectory = Trajectory(trajectory_id="traj-root", prompt_id="prompt-0")

    ctx = asyncio.run(parser.build_turn_input(
        trajectory, messages=[{"role": "user", "content": "hi"}],
    ))

    assert ctx.start_new_segment is True
    assert ctx.target_segment_id == 0
    assert ctx.new_segment_parent_id is None
    assert ctx.new_segment_origin == "root"
    assert ctx.new_segment_depth == 0
    assert ctx.is_subagent_placeholder is False


def test_build_turn_input_compaction_opens_new_mainline_segment() -> None:
    """Prefix miss without collab_spawn should open a compact segment (case 3)."""
    parser = make_prefix_parser()
    trajectory = Trajectory(
        trajectory_id="traj-compact",
        prompt_id="prompt-0",
        tokens=[1, 2, 3],
        active_segment_id=0,
        chat_completions={0: [{"role": "user", "content": "old history"}]},
        segments=[Segment(token_start=0, token_end=3, logprob_start=0, logprob_end=1,
                           triplets=[Triplet(token_start=0, token_end=3, logprob_start=0, logprob_end=1)],
                           segment_id=0, origin="root")],
    )

    ctx = asyncio.run(parser.build_turn_input(
        trajectory,
        messages=[{"role": "system", "content": "new summary"}, {"role": "user", "content": "continue"}],
        request_kind=None,
    ))

    assert ctx.start_new_segment is True
    assert ctx.target_segment_id == 1
    assert ctx.new_segment_parent_id == 0
    assert ctx.new_segment_origin == "compact"
    assert ctx.new_segment_depth == 0
    assert ctx.is_subagent_placeholder is False


def test_build_turn_input_matches_reserialized_tool_call_arguments() -> None:
    """A follow-up turn must continue the active segment even when the client
    re-serializes tool_call arguments (JSON.stringify drops the spaces the SGLang
    parser emits). Otherwise every tool-using turn is misread as compaction."""
    parser = make_prefix_parser()
    # Stored history keeps the SGLang-parsed arguments string (space after the colon).
    assistant = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "call_0",
            "type": "function",
            "function": {"name": "read", "arguments": '{"filePath": "/a.go"}'},
        }],
    }
    stored_chat = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        assistant,
    ]
    trajectory = Trajectory(
        trajectory_id="traj-toolarg",
        prompt_id="prompt-0",
        tokens=[1, 2, 3],
        active_segment_id=0,
        chat_completions={0: copy.deepcopy(stored_chat)},
        segments=[Segment(token_start=0, token_end=3, logprob_start=0, logprob_end=1,
                           triplets=[Triplet(token_start=0, token_end=3, logprob_start=0, logprob_end=1)],
                           segment_id=0, origin="root")],
        metadata={"normalized_history": parser._normalize_messages(stored_chat)},
    )

    # Client echoes the assistant back with compact arguments, then appends the tool result.
    incoming = copy.deepcopy(stored_chat)
    incoming[2]["tool_calls"][0]["function"]["arguments"] = '{"filePath":"/a.go"}'
    incoming.append({"role": "tool", "tool_call_id": "call_0", "content": "package main"})

    ctx = asyncio.run(parser.build_turn_input(trajectory, messages=incoming, request_kind=None))

    assert ctx.start_new_segment is False
    assert ctx.target_segment_id == 0


def test_build_turn_input_consecutive_compaction_chains_parent_ids() -> None:
    """A second consecutive mismatch should parent onto the first compact segment."""
    parser = make_prefix_parser()
    trajectory = Trajectory(
        trajectory_id="traj-compact-chain",
        prompt_id="prompt-0",
        tokens=[1, 2, 3],
        active_segment_id=0,
        chat_completions={0: [{"role": "user", "content": "old history"}]},
        segments=[Segment(token_start=0, token_end=3, logprob_start=0, logprob_end=1,
                           triplets=[Triplet(token_start=0, token_end=3, logprob_start=0, logprob_end=1)],
                           segment_id=0, origin="root")],
    )

    first_ctx = asyncio.run(parser.build_turn_input(
        trajectory,
        messages=[{"role": "system", "content": "summary 1"}],
        request_kind="compaction",
    ))
    # Simulate _append_triplet effects for the compact segment (id=1) becoming active.
    trajectory.segments.append(Segment(
        token_start=3, token_end=6, logprob_start=1, logprob_end=2,
        triplets=[Triplet(token_start=3, token_end=6, logprob_start=1, logprob_end=2)],
        segment_id=1, parent_segment_id=0, origin="compact",
    ))
    trajectory.active_segment_id = 1
    trajectory.chat_completions[1] = [{"role": "system", "content": "summary 1"}]
    trajectory.metadata["normalized_history"] = parser._normalize_messages(
        [{"role": "system", "content": "summary 1"}]
    )

    second_ctx = asyncio.run(parser.build_turn_input(
        trajectory,
        messages=[{"role": "system", "content": "summary 2"}],
        request_kind=None,
    ))

    assert first_ctx.new_segment_parent_id == 0
    assert second_ctx.target_segment_id == 2
    assert second_ctx.new_segment_parent_id == 1
    assert second_ctx.new_segment_origin == "compact"


def test_build_turn_input_subagent_fork_keeps_active_segment() -> None:
    """Prefix miss with collab_spawn should open a placeholder without moving active_segment_id."""
    parser = make_prefix_parser()
    trajectory = Trajectory(
        trajectory_id="traj-subagent",
        prompt_id="prompt-0",
        tokens=[1, 2, 3],
        active_segment_id=0,
        chat_completions={0: [{"role": "user", "content": "mainline history"}]},
        segments=[Segment(token_start=0, token_end=3, logprob_start=0, logprob_end=1,
                           triplets=[Triplet(token_start=0, token_end=3, logprob_start=0, logprob_end=1)],
                           segment_id=0, origin="root", depth=0)],
    )

    ctx = asyncio.run(parser.build_turn_input(
        trajectory,
        messages=[{"role": "system", "content": "subtask"}, {"role": "user", "content": "do X"}],
        request_kind="collab_spawn",
    ))

    assert ctx.is_subagent_placeholder is True
    assert ctx.start_new_segment is True
    assert ctx.target_segment_id == 1
    assert ctx.new_segment_parent_id == 0
    assert ctx.new_segment_origin == "subagent"
    assert ctx.new_segment_depth == 1
    # active_segment_id itself is only mutated by _append_triplet; verify it's untouched here.
    assert trajectory.active_segment_id == 0


def test_normalize_messages_strip_think_false_keeps_think_block() -> None:
    """strip_think=False must bypass accumulate_reasoning stripping regardless of setting."""
    parser = make_parser()
    parser.accumulate_reasoning = False
    messages = [{"role": "assistant", "content": "<think>reasoning</think>answer"}]

    normalized = parser._normalize_messages(messages, strip_think=False)

    assert normalized[0]["content"] == "<think>reasoning</think>answer"


def test_normalize_messages_strip_think_default_still_strips() -> None:
    """Default strip_think=True preserves prior accumulate_reasoning=False behavior."""
    parser = make_parser()
    parser.accumulate_reasoning = False
    messages = [{"role": "assistant", "content": "<think>reasoning</think>answer"}]

    normalized = parser._normalize_messages(messages)

    assert normalized[0]["content"] == "answer"


def test_build_turn_input_subagent_fork_does_not_strip_think() -> None:
    """Case 3 (collab_spawn) must leave assistant think content untouched even when
    accumulate_reasoning=False, since subagent turns never rejoin the mainline context."""
    parser = make_prefix_parser()
    parser.accumulate_reasoning = False
    trajectory = Trajectory(
        trajectory_id="traj-subagent-think",
        prompt_id="prompt-0",
        tokens=[1, 2, 3],
        active_segment_id=0,
        chat_completions={0: [{"role": "user", "content": "mainline history"}]},
        segments=[Segment(token_start=0, token_end=3, logprob_start=0, logprob_end=1,
                           triplets=[Triplet(token_start=0, token_end=3, logprob_start=0, logprob_end=1)],
                           segment_id=0, origin="root", depth=0)],
    )

    captured: dict[str, list] = {}

    async def apply_chat_template(messages, **kwargs):
        captured["messages"] = messages
        return [1]

    parser.tokenizer_manager.apply_chat_template = apply_chat_template

    ctx = asyncio.run(parser.build_turn_input(
        trajectory,
        messages=[
            {"role": "system", "content": "subtask"},
            {"role": "assistant", "content": "<think>reasoning</think>answer"},
        ],
        request_kind="collab_spawn",
    ))

    assert ctx.is_subagent_placeholder is True
    assistant_entry = next(m for m in captured["messages"] if m["role"] == "assistant")
    assert assistant_entry["content"] == "<think>reasoning</think>answer"


def test_append_triplet_subagent_placeholder_does_not_touch_flat_arrays() -> None:
    """_append_triplet must skip token/loss writes and active_segment_id for placeholders."""
    parser = make_parser()
    trajectory = Trajectory(
        trajectory_id="traj-subagent-append",
        prompt_id="prompt-0",
        tokens=[1, 2, 3],
        active_segment_id=0,
        segments=[Segment(token_start=0, token_end=3, logprob_start=0, logprob_end=1,
                           triplets=[Triplet(token_start=0, token_end=3, logprob_start=0, logprob_end=1)],
                           segment_id=0, origin="root", depth=0)],
    )
    turn_ctx = TurnInputContext(
        delta_prompt_ids=[],
        input_ids=list(trajectory.tokens),
        start_new_segment=True,
        raw_new_messages=[],
        target_segment_id=1,
        new_segment_parent_id=0,
        new_segment_origin="subagent",
        new_segment_depth=1,
        is_subagent_placeholder=True,
    )

    parser._append_triplet(
        trajectory,
        delta_prompt_ids=[],
        response_ids=[],
        logprobs=[],
        weight_version=5,
        turn_ctx=turn_ctx,
    )

    assert trajectory.tokens == [1, 2, 3]
    assert trajectory.active_segment_id == 0
    assert len(trajectory.segments) == 2
    placeholder = trajectory.segments[1]
    assert placeholder.origin == "subagent"
    assert placeholder.trainable is False
    assert placeholder.token_start == placeholder.token_end == 3
    assert placeholder.triplets == []


def test_update_trajectory_subagent_placeholder_skips_chat_and_weight_version() -> None:
    """Placeholder turns must not enter chat_completions or move weight-version scalars."""
    parser = make_parser()
    trajectory = Trajectory(
        trajectory_id="traj-subagent-update",
        prompt_id="prompt-0",
        status=TrajectoryStatus.GENERATING,
        tokens=[1, 2, 3],
        active_segment_id=0,
        chat_completions={0: [{"role": "user", "content": "mainline"}]},
        segments=[Segment(token_start=0, token_end=3, logprob_start=0, logprob_end=1,
                           triplets=[Triplet(token_start=0, token_end=3, logprob_start=0, logprob_end=1)],
                           segment_id=0, origin="root", depth=0)],
    )
    parser.trajectory_store.add(trajectory.trajectory_id, trajectory)
    turn_ctx = TurnInputContext(
        delta_prompt_ids=[],
        input_ids=list(trajectory.tokens),
        start_new_segment=True,
        raw_new_messages=[{"role": "user", "content": "subtask"}],
        target_segment_id=1,
        new_segment_parent_id=0,
        new_segment_origin="subagent",
        new_segment_depth=1,
        is_subagent_placeholder=True,
    )
    payload, assistant_message, _ = parser.build_assistant_message({
        "text": "subtask result",
        "meta_info": {"weight_version": "9", "output_token_logprobs": [[-0.1, 501]]},
    })

    parser.update_trajectory(trajectory, turn_ctx, payload, assistant_message)

    stored = parser.trajectory_store.get(["traj-subagent-update"])["traj-subagent-update"][-1]
    assert stored.start_rollout_weight_version == -1
    assert stored.end_rollout_weight_version == -1
    assert stored.chat_completions == {0: [{"role": "user", "content": "mainline"}]}
    assert stored.active_segment_id == 0
    assert len(stored.segments) == 2
    assert stored.segments[1].origin == "subagent"


def test_build_turn_input_mainline_resumes_after_subagent_placeholder() -> None:
    """After a subagent branch closes, the next request should hit the active segment (case 1)."""
    parser = make_prefix_parser()
    mainline_chat = [{"role": "user", "content": "mainline history"}]
    trajectory = Trajectory(
        trajectory_id="traj-resume",
        prompt_id="prompt-0",
        tokens=[1, 2, 3],
        active_segment_id=0,
        chat_completions={0: mainline_chat},
        metadata={"normalized_history": parser._normalize_messages(mainline_chat)},
        segments=[
            Segment(token_start=0, token_end=3, logprob_start=0, logprob_end=1,
                    triplets=[Triplet(token_start=0, token_end=3, logprob_start=0, logprob_end=1)],
                    segment_id=0, origin="root", depth=0),
            Segment(token_start=3, token_end=3, logprob_start=1, logprob_end=1,
                    triplets=[], segment_id=1, parent_segment_id=0, origin="subagent",
                    depth=1, trainable=False),
        ],
    )

    ctx = asyncio.run(parser.build_turn_input(
        trajectory,
        messages=[*mainline_chat, {"role": "assistant", "content": "subtask done"}],
        request_kind=None,
    ))

    assert ctx.start_new_segment is False
    assert ctx.target_segment_id == 0
    assert ctx.is_subagent_placeholder is False
