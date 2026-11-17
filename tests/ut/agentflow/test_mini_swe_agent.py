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


def test_timed_out_command_is_reported_to_the_model():
    output = _env({
        "stdout": "", "stderr": "", "exit_code": -1, "success": False,
        "timed_out": True, "error": "Command timed out after 600s",
    }).execute("python -c 'while True: pass'")

    assert output["exception_info"] == "Command timed out after 600s"
    assert output["returncode"] == -1


def test_outer_timeout_with_partial_output_still_reports_timeout(monkeypatch):
    """Regression: bytes stdout from TimeoutExpired used to raise TypeError and hide the reason."""
    import subprocess

    from coda.agentflow.agent.swe.mini_swe_agent import CodaSandboxEnvironment
    from coda.agentflow.sandbox.docker_sandbox import DockerSandboxClient

    def raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="exec", timeout=1, output=b"partial\n")

    monkeypatch.setattr(subprocess, "run", raise_timeout)
    client = DockerSandboxClient.from_config({"command_exec_timeout_seconds": 1})
    env = CodaSandboxEnvironment(lambda command, **kwargs: client.execute("sandbox-1", command, **kwargs))

    output = env.execute("sleep 999")

    assert output["exception_info"] == "Command timed out after 1s"
    assert output["output"] == "partial\n"


def test_normal_command_carries_no_exception_info():
    output = _env({
        "stdout": "ok\n", "stderr": "", "exit_code": 0, "success": True,
        "timed_out": False, "error": "",
    }).execute("echo ok")

    assert output == {"output": "ok\n", "returncode": 0, "exception_info": ""}


def test_sandbox_client_without_timeout_keys_still_works():
    """Custom sandbox backends may omit the optional keys."""
    output = _env({"stdout": "", "stderr": "", "exit_code": -1, "success": False}).execute("ls")

    assert output["exception_info"] == ""


def test_timeout_renders_as_exception_tag_in_the_observation():
    """Render with the swebench.yaml observation template the agent uses at runtime."""
    import jinja2

    from coda.agentflow.agent.swe.mini_swe_agent import _SWEBENCH_CONFIG

    output = _env({
        "stdout": "", "stderr": "", "exit_code": -1, "success": False,
        "timed_out": True, "error": "Command timed out after 600s",
    }).execute("python -c 'while True: pass'")

    rendered = jinja2.Template(_SWEBENCH_CONFIG["model"]["observation_template"]).render(output=output)

    assert "<exception>Command timed out after 600s</exception>" in rendered
