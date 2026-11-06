"""Unit tests for the mini-swe agent's litellm think-extraction patch."""

import pytest

pytest.importorskip("minisweagent")
litellm = pytest.importorskip("litellm")

from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (  # noqa: E402
    convert_to_model_response_object,
)

import coda.agentflow.agent.swe.mini_swe_agent  # noqa: E402,F401  (applies the patch on import)


def _convert(message: dict) -> litellm.Message:
    raw = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "default",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", **message}}],
    }
    response = convert_to_model_response_object(
        response_object=raw,
        model_response_object=litellm.ModelResponse(),
        response_type="completion",
    )
    return response.choices[0].message


def test_leading_think_block_stays_in_content():
    content = "<think>plan the fix</think>\n\nTHOUGHT: run the tests"
    message = _convert({"content": content})
    assert message.content == content
    assert not getattr(message, "reasoning_content", None)
    assert message.model_dump()["content"] == content


def test_explicit_reasoning_content_is_kept():
    message = _convert({"content": "THOUGHT: run the tests", "reasoning_content": "plan the fix"})
    assert message.content == "THOUGHT: run the tests"
    assert message.reasoning_content == "plan the fix"


def test_plain_content_unchanged():
    message = _convert({"content": "THOUGHT: run the tests"})
    assert message.content == "THOUGHT: run the tests"
