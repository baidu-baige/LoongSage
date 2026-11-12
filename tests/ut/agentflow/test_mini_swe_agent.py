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


def _env(result: dict):
    from coda.agentflow.agent.swe.mini_swe_agent import CodaSandboxEnvironment

    return CodaSandboxEnvironment(lambda command, **kwargs: result, cwd="/testbed")


def _model():
    from coda.agentflow.agent.swe.mini_swe_agent import CodaLitellmModel

    return CodaLitellmModel(
        model_name="openai/default",
        cost_tracking="ignore_errors",
        model_kwargs={"api_base": "http://127.0.0.1:1/v1", "api_key": "not-needed"},
    )


def test_generate_seconds_accumulates_across_calls(monkeypatch):
    monkeypatch.setattr(litellm, "completion", lambda **kwargs: "ok")
    model = _model()

    assert model.generate_seconds == 0.0
    model._query([])
    after_first = model.generate_seconds
    model._query([])

    assert after_first > 0.0
    assert model.generate_seconds > after_first


def test_generate_seconds_counts_failed_calls(monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("upstream down")

    monkeypatch.setattr(litellm, "completion", boom)
    model = _model()

    with pytest.raises(RuntimeError):
        model._query([])

    assert model.generate_seconds > 0.0


def test_tool_seconds_counts_normal_and_failing_commands():
    from coda.agentflow.agent.swe.mini_swe_agent import CodaSandboxEnvironment

    ok = _env({"stdout": "ok\n", "stderr": "", "exit_code": 0})
    ok.execute("echo ok")
    assert (ok.tool_calls, ok.tool_seconds > 0) == (1, True)

    def raising(command, **kwargs):
        raise RuntimeError("sandbox gone")

    broken = CodaSandboxEnvironment(raising, cwd="/testbed")
    output = broken.execute("ls")

    assert output["exception_info"] == "sandbox gone"
    assert (broken.tool_calls, broken.tool_seconds > 0) == (1, True)


def test_phase_metrics_parse_into_rollout_metric_names():
    """The five time_* keys plus the two counters cover a whole trajectory."""
    from coda.utils import eval_utils

    extra = {
        "metric/mean,max/time_sandbox_setup": 48.0,
        "metric/mean,max/time_generate": 110.0,
        "metric/mean,max/time_tool_calls": 95.0,
        "metric/mean,max/time_agent_other": 40.0,
        "metric/mean,max/time_grade": 2.0,
        "metric/mean/tool_calls": 25.0,
        "metric/mean,max/attempts": 1.0,
        "actual": {"passed": 1},  # non-numeric reward payload must be ignored
    }

    specs = eval_utils.metadata_metrics(extra)
    reduced = eval_utils.reduce_metadata_metrics({s: [v] for s, v in specs.items()})

    assert reduced["time_sandbox_setup_mean"] == 48.0
    assert reduced["time_generate_max"] == 110.0
    assert reduced["time_grade_mean"] == 2.0
    assert reduced["attempts_max"] == 1.0
    assert reduced["tool_calls_mean"] == 25.0
    assert "tool_calls_max" not in reduced
    assert sum(reduced[f"{k}_mean"] for k in
               ("time_sandbox_setup", "time_generate", "time_tool_calls",
                "time_agent_other", "time_grade")) == 295.0
