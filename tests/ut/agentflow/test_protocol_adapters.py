"""Pure protocol conversion tests for the black-box agent adapters."""

from __future__ import annotations

import asyncio
import json

import pytest
from starlette.requests import Request

from coda.agentflow.router.protocols import (
    OpenAIChatCompletionsAdapter as ChatCompletionsAdapter,
    OpenAIResponsesAdapter as CodexAdapter,
)
from coda.agentflow.agent.codex.codex_agent import (
    _CONFIG_TOML,
    _MODEL_INSTRUCTIONS,
    _build_config,
)


def _codex_stream(response: dict) -> str:
    streaming_response = CodexAdapter().build_streaming_response(response)

    async def _read() -> str:
        chunks = []
        async for chunk in streaming_response.body_iterator:
            chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
        return "".join(chunks)

    return asyncio.run(_read())


def _codex_events(response: dict) -> list[dict]:
    return [
        json.loads(block.split("data: ", 1)[1])
        for block in _codex_stream(response).strip().split("\n\n")
    ]


def _request(path: str, headers: dict[str, str] | None = None) -> Request:
    return Request({
        "type": "http",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [
            (key.lower().encode(), value.encode())
            for key, value in (headers or {}).items()
        ],
        "server": ("testserver", 80),
        "client": ("testclient", 50000),
    })


@pytest.mark.parametrize(
    ("adapter", "path"),
    [
        (ChatCompletionsAdapter(), "/trajectory/0/v1/chat/completions"),
        (CodexAdapter(), "/trajectory/0/v1/responses/compact"),
    ],
)
def test_adapter_parse_route_returns_trajectory_identity(adapter, path) -> None:
    assert adapter.parse_route(path) == ("trajectory", 0)


def test_codex_request_kind_uses_its_own_headers() -> None:
    adapter = CodexAdapter()
    path = "/traj/0/v1/responses"
    compact_path = "/traj/0/v1/responses/compact"
    assert adapter.normalize_headers(_request(path, {"request_kind": "collab_spawn"}))[
        "request_kind"
    ] == "collab_spawn"
    assert adapter.normalize_headers(_request(path, {"x-openai-subagent": "review"}))[
        "request_kind"
    ] == "collab_spawn"
    assert adapter.normalize_headers(
        _request(path, {"x-codex-turn-metadata": '{"request_kind":"compaction"}'}),
    )["request_kind"] == "compaction"
    assert adapter.normalize_headers(
        _request(path, {
            "x-openai-subagent": "collab_spawn",
            "x-codex-turn-metadata": '{"request_kind":"compaction"}',
        }),
    )["request_kind"] == "collab_spawn"
    assert adapter.normalize_headers(_request(compact_path))[
        "request_kind"
    ] == "compaction"
    assert adapter.normalize_headers(_request(path)).get("request_kind") is None


def test_codex_request_preserves_empty_assistant_content_and_image_placeholder() -> None:
    messages, _ = CodexAdapter().parse_request({
        "input": [
            {"role": "assistant", "content": ""},
            {"role": "user", "content": [{"type": "input_image", "image_url": "x"}]},
        ]
    })
    assert messages == [
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "[image omitted]"},
    ]


def test_codex_compact_route_and_history_item() -> None:
    adapter = CodexAdapter()
    assert adapter.parse_route("/traj/0/v1/responses/compact") == ("traj", 0)
    messages, _ = adapter.parse_request(
        {
            "input": [
                {"type": "compaction", "encrypted_content": "preserve this summary"},
                {"role": "user", "content": "continue"},
            ]
        }
    )
    assert messages == [
        {"role": "assistant", "content": "preserve this summary"},
        {"role": "user", "content": "continue"},
    ]
    messages, _ = adapter.parse_request({
        "input": [
            {"type": "context_compaction", "encrypted_content": "alias summary"},
            {"role": "user", "content": "continue"},
        ]
    })
    assert messages[0] == {"role": "assistant", "content": "alias summary"}


def test_codex_compaction_trigger_builds_streamed_v2_response() -> None:
    response = CodexAdapter().build_response(
        {
            "model": "qwen",
            "input": [
                {"role": "developer", "content": "instructions"},
                {"role": "user", "content": "solve this"},
                {"type": "function_call", "name": "bash", "call_id": "call-1"},
                {"type": "compaction_trigger"},
            ],
        },
        {"role": "assistant", "content": "short summary"},
        "stop",
        {"prompt_tokens": 2, "completion_tokens": 3},
    )
    assert response["object"] == "response"
    assert response["status"] == "completed"
    assert response["output"] == [
        {"type": "compaction", "encrypted_content": "short summary"},
    ]
    assert response["usage"] == {
        "input_tokens": 2,
        "output_tokens": 3,
        "total_tokens": 5,
    }
    stream = _codex_stream(response)
    events = _codex_events(response)
    assert [event["type"] for event in events] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.output_item.done",
        "response.completed",
    ]
    assert stream.count("event: response.output_item.done") == 1
    assert '"type": "compaction"' in stream
    assert "event: response.completed" in stream


def test_codex_context_length_error_is_streamed_as_response_failed() -> None:
    response = CodexAdapter().build_error_response(
        {"model": "qwen", "stream": True},
        error_type="context_length_exceeded",
        message="context_length=65536 exhausted (input_len=67161)",
        status_code=400,
    )
    async def _read() -> str:
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
        return "".join(chunks)

    stream = asyncio.run(_read())
    events = [
        json.loads(block.split("data: ", 1)[1])
        for block in stream.strip().split("\n\n")
    ]

    assert response.status_code == 200
    assert [event["type"] for event in events] == [
        "response.created",
        "response.in_progress",
        "response.failed",
    ]
    event = events[-1]
    assert event["type"] == "response.failed"
    assert event["response"]["status"] == "failed"
    assert event["response"]["error"] == {
        "code": "context_length_exceeded",
        "message": "context_length=65536 exhausted (input_len=67161)",
    }


@pytest.mark.parametrize(
    ("adapter", "body", "expected"),
    [
        (CodexAdapter(), {"max_output_tokens": 17}, 17),
        (CodexAdapter(), {"max_new_tokens": 21}, 21),
        (CodexAdapter(), {"max_tokens": 5}, None),
        (ChatCompletionsAdapter(), {"max_completion_tokens": 17}, 17),
        (ChatCompletionsAdapter(), {"max_tokens": 19}, 19),
        # max_completion_tokens wins: max_tokens is deprecated by OpenAI.
        (ChatCompletionsAdapter(), {"max_tokens": 19, "max_completion_tokens": 17}, 17),
        (ChatCompletionsAdapter(), {"max_output_tokens": 5}, None),
        (ChatCompletionsAdapter(), {}, None),
    ],
)
def test_adapter_normalizes_max_new_tokens(adapter, body, expected) -> None:
    """Each adapter normalizes its own protocol's output-token field."""
    body = dict(body)
    if isinstance(adapter, CodexAdapter):
        body["input"] = "solve"
    else:
        body["messages"] = [{"role": "user", "content": "solve"}]
    adapter.parse_request(body)
    assert body.get("max_new_tokens") == expected


def test_protocol_errors_use_each_adapter_wire_format() -> None:
    message = "context_length=100 exhausted (input_len=100)"

    chat = ChatCompletionsAdapter().build_error_response(
        {},
        error_type="context_length_exceeded",
        message=message,
        status_code=400,
    )
    assert chat.status_code == 400
    assert json.loads(chat.body) == {
        "error": {"message": message, "type": "context_length_exceeded"},
    }

    codex = CodexAdapter().build_error_response(
        {},
        error_type="context_length_exceeded",
        message=message,
        status_code=400,
    )
    assert codex.status_code == 400
    assert json.loads(codex.body) == {
        "error": {"message": message, "type": "context_length_exceeded"},
    }

def test_codex_legacy_compact_response_retains_user_messages() -> None:
    adapter = CodexAdapter()
    response = adapter.build_response(
        {
            "model": "qwen",
            "input": [
                {"role": "developer", "content": "instructions"},
                {"role": "user", "content": "solve this"},
            ],
        },
        {"role": "assistant", "content": "short summary"},
        "stop",
        {"prompt_tokens": 2, "completion_tokens": 3},
        path="/traj/0/v1/responses/compact",
    )
    assert response["output"] == [
        {"type": "message", "role": "developer", "content": "instructions"},
        {"type": "message", "role": "user", "content": "solve this"},
        {"type": "compaction", "encrypted_content": "short summary"},
    ]


def test_codex_function_and_namespace_tools_round_trip() -> None:
    body = {
        "tools": [
            {
                "type": "function",
                "name": "update_plan",
                "description": "update the plan",
                "parameters": {"type": "object", "properties": {}},
                "strict": True,
            },
            {
                "type": "namespace",
                "name": "multi_agent_v1",
                "description": "collaboration tools",
                "tools": [{
                    "type": "function",
                    "name": "spawn_agent",
                    "description": "spawn an agent",
                    "parameters": {"type": "object", "properties": {}},
                }],
            },
        ]
    }
    _, tools = CodexAdapter().parse_request({**body, "input": "solve"})
    assert [tool["function"]["name"] for tool in tools or []] == [
        "update_plan",
        "multi_agent_v1__spawn_agent",
    ]
    assert tools is not None
    assert tools[0]["function"]["strict"] is True

    assistant_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call-plan",
                "type": "function",
                "function": {"name": "update_plan", "arguments": '{"plan":[]}'},
            },
            {
                "id": "call-spawn",
                "type": "function",
                "function": {
                    "name": "multi_agent_v1__spawn_agent",
                    "arguments": '{"message":"inspect"}',
                },
            },
        ],
    }
    response = CodexAdapter().build_response(
        body,
        assistant_message,
        "tool_calls",
        {"prompt_tokens": 1, "completion_tokens": 2},
    )
    assert response["output"][0]["type"] == "function_call"
    assert response["output"][1]["name"] == "spawn_agent"
    assert response["output"][1]["namespace"] == "multi_agent_v1"

    messages, _ = CodexAdapter().parse_request({"input": response["output"]})
    assert messages == [assistant_message]


def test_codex_function_strict_preserves_false_and_absent() -> None:
    body = {
        "tools": [
            {
                "type": "function",
                "name": "exec_command",
                "parameters": {"type": "object", "properties": {}},
                "strict": False,
            },
            {
                "type": "function",
                "name": "write_stdin",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "type": "namespace",
                "name": "multi_agent_v1",
                "tools": [{
                    "type": "function",
                    "name": "spawn_agent",
                    "parameters": {"type": "object", "properties": {}},
                    "strict": True,
                }],
            },
        ]
    }

    _, tools = CodexAdapter().parse_request({**body, "input": "solve"})

    assert tools is not None
    assert tools[0]["function"]["strict"] is False
    assert "strict" not in tools[1]["function"]
    assert tools[2]["function"]["strict"] is True


def test_codex_tool_conversion_uses_only_function_tools() -> None:
    body = {
        "input": "solve",
        "tools": [
            {"type": "function", "name": "future_function"},
            {"type": "custom", "name": "future_custom"},
            {
                "type": "namespace",
                "name": "future_namespace",
                "tools": [{"type": "function", "name": "future_child"}],
            },
            {"type": "web_search"},
        ],
    }

    _, tools = CodexAdapter().parse_request(body)

    assert [tool["function"]["name"] for tool in tools or []] == [
        "future_function",
        "future_namespace__future_child",
    ]

    messages, _ = CodexAdapter().parse_request({
        "input": [
            {
                "type": "custom_tool_call",
                "name": "future_custom",
                "call_id": "call-custom",
                "input": "freeform input",
            },
            {
                "type": "custom_tool_call_output",
                "call_id": "call-custom",
                "output": "ignored",
            },
        ],
    })
    assert messages == []


def test_codex_config_uses_fallback_model_metadata() -> None:
    config = _build_config("http://router:8000", 100000)
    assert 'model = "coda-actor"' in config
    assert "model_context_window = 100000" in config
    assert "model_auto_compact_token_limit = 90000" in config
    assert "model_catalog_json" not in config
    assert 'name = "Coda"' in config
    assert "goals = false" in config
    assert "developer_instructions =" not in config
    assert "include_instructions = false" in config
    assert 'model_instructions_file = "/root/.codex/coda_prompt.md"' in config
    assert "x-coda-force-strict-tools" not in config
    for redundant_setting in (
        "unified_exec = true",
        "multi_agent = true",
        'wire_api = "responses"',
        "supports_websockets = false",
        "requires_openai_auth = false",
    ):
        assert redundant_setting not in config


def test_codex_prompt_matches_available_tools() -> None:
    assert "`apply_patch`" not in _MODEL_INSTRUCTIONS
    assert "`cat`" not in _MODEL_INSTRUCTIONS
    assert "complete capability set" in _MODEL_INSTRUCTIONS
    assert "outer tool name remains `exec_command`" in _MODEL_INSTRUCTIONS
    assert "The `plan` argument must be an array" in _MODEL_INSTRUCTIONS
    assert "only an `agent_id` returned" in _MODEL_INSTRUCTIONS


def test_codex_empty_assistant_content_is_preserved_in_wire_blocks() -> None:
    codex = CodexAdapter().build_response(
        {"model": "qwen"},
        {"role": "assistant", "content": ""},
        "stop",
        {"prompt_tokens": 1, "completion_tokens": 0},
    )
    assert codex["output"][0]["content"] == [{
        "type": "output_text",
        "text": "",
        "annotations": [],
        "logprobs": [],
    }]


def test_codex_response_stream_uses_standard_item_lifecycles() -> None:
    response = CodexAdapter().build_response(
        {
            "model": "qwen",
            "tools": [{
                "type": "function",
                "name": "exec_command",
                "parameters": {"type": "object"},
            }],
        },
        {
            "role": "assistant",
            "content": "done",
            "reasoning_content": "plan",
            "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "exec_command",
                    "arguments": '{"cmd":"pwd"}',
                },
            }],
        },
        "tool_calls",
        {"prompt_tokens": 1, "completion_tokens": 2},
    )
    events = _codex_events(response)
    assert [event["type"] for event in events] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.reasoning_summary_part.added",
        "response.reasoning_summary_text.delta",
        "response.reasoning_summary_text.done",
        "response.reasoning_summary_part.done",
        "response.output_item.done",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert [event["sequence_number"] for event in events] == list(range(len(events)))

    reasoning_delta = events[4]
    text_delta = events[10]
    arguments_delta = events[15]
    assert reasoning_delta["delta"] == "plan"
    assert text_delta["delta"] == "done"
    assert arguments_delta["delta"] == '{"cmd":"pwd"}'

    for output_index, done_event in enumerate(
        event for event in events if event["type"] == "response.output_item.done"
    ):
        assert done_event["output_index"] == output_index
        assert done_event["item"] == response["output"][output_index]


def test_codex_response_stream_emits_incomplete_for_length_limit() -> None:
    response = CodexAdapter().build_response(
        {"model": "qwen"},
        {"role": "assistant", "content": "partial"},
        "length",
        {"prompt_tokens": 1, "completion_tokens": 2},
    )
    assert response["status"] == "incomplete"
    assert response["incomplete_details"] == {"reason": "max_output_tokens"}
    assert response["output"][0]["status"] == "incomplete"
    events = _codex_events(response)
    assert events[-1]["type"] == "response.incomplete"


def test_codex_assistant_message_round_trip() -> None:
    assistant_message = {
        "role": "assistant",
        "content": "done",
        "reasoning_content": "plan",
        "tool_calls": [{
            "id": "call-1",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"cmd":"pwd"}'},
        }],
    }
    response = CodexAdapter().build_response(
        {"model": "qwen"},
        assistant_message,
        "tool_calls",
        {"prompt_tokens": 3, "completion_tokens": 4},
    )

    messages, _ = CodexAdapter().parse_request({"input": response["output"]})

    assert messages == [assistant_message]


def test_mid_list_system_messages_are_folded_without_mutating_input() -> None:
    input_items = [
        {"role": "user", "content": "before"},
        {"role": "developer", "content": "reminder"},
        {"role": "user", "content": "after"},
    ]
    messages, _ = CodexAdapter().parse_request({"input": input_items})
    assert input_items[1]["role"] == "developer"
    assert len(messages) == 2
    assert "<system-reminder>" in messages[0]["content"]
    assert messages[1] == input_items[2]


def test_system_suffix_without_user_is_kept_as_a_placeholder_turn() -> None:
    messages, _ = CodexAdapter().parse_request({
        "input": [
            {"role": "assistant", "content": "answer"},
            {"role": "developer", "content": "reminder"},
        ]
    })
    assert messages[-1]["role"] == "user"
    assert "reminder" in messages[-1]["content"]


def test_chat_response_preserves_existing_metadata() -> None:
    response = ChatCompletionsAdapter().build_response(
        {"model": "qwen", "metadata": {"weight_version": 7}},
        {"role": "assistant", "content": "ok"},
        "stop",
        {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )
    assert response["metadata"] == {"weight_version": 7}


def test_chat_response_preserves_generate_metadata() -> None:
    response = ChatCompletionsAdapter().build_response(
        {"model": "qwen"},
        {"role": "assistant", "content": "ok"},
        "stop",
        {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        upstream_response={
            "meta_info": {
                "id": "rid-1",
                "weight_version": 9,
                "finish_reason": {"type": "stop", "matched": "<eos>"},
            }
        },
    )
    assert response["id"] == "rid-1"
    assert response["choices"][0]["matched_stop"] == "<eos>"
    assert response["metadata"] == {"weight_version": 9}


def test_chat_request_rejects_multiple_choices() -> None:
    with pytest.raises(ValueError, match="n > 1"):
        ChatCompletionsAdapter().parse_request({"messages": [{"role": "user", "content": "hi"}], "n": 2})
