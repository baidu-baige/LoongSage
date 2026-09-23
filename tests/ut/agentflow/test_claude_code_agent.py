"""Unit tests for the Claude Code black-box agent (stateless sandbox contract)."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from coda.agentflow.agent.claude_code.claude_code_agent import ClaudeCodeAgent
from coda.agentflow.sandbox.base import SandboxClient
from coda.reward.reward import Reward


class _FakeClient(SandboxClient):
    """Stateless fake backend: records calls, returns exit_code 0 by default.

    ``add_override`` lets a test make a specific command (matched by substring)
    return a custom response, so the happy-path setup commands can succeed while
    only the Claude invocation fails.
    """

    def __init__(self) -> None:
        self.created_with: list[dict[str, Any]] = []
        self.executed: list[tuple[str, str, dict[str, Any]]] = []
        self.deleted: list[str] = []
        self._overrides: list[tuple[str, dict[str, Any]]] = []

    def add_override(self, substring: str, response: dict[str, Any]) -> None:
        self._overrides.append((substring, response))

    def create(self, **kwargs: Any) -> str:
        self.created_with.append(kwargs)
        return f"sbx-{len(self.created_with)}"

    def execute(self, sandbox_id: str, command: str, **kwargs: Any) -> dict[str, Any]:
        self.executed.append((sandbox_id, command, kwargs))
        for substring, response in self._overrides:
            if substring in command:
                return response
        return {"exit_code": 0, "stdout": "", "stderr": ""}

    def delete(self, sandbox_id: str, **kwargs: Any) -> None:
        self.deleted.append(sandbox_id)


def _make_agent(
    client: SandboxClient | None,
    *,
    sandbox_id: str | None = None,
    reward_fn: Any = None,
) -> ClaudeCodeAgent:
    return ClaudeCodeAgent(
        router_url="http://router:8000",
        reward_fn=reward_fn or MagicMock(return_value=Reward(final_reward=1.0, is_valid=True)),
        sandbox_client=client,
        sandbox_id=sandbox_id,
        context_length=1000,
    )


def test_requires_sandbox_client() -> None:
    with pytest.raises(ValueError, match="requires data_source.sandbox"):
        ClaudeCodeAgent(
            router_url="http://router:8000",
            reward_fn=MagicMock(),
            sandbox_client=None,
        )


@pytest.mark.asyncio
async def test_creates_sandbox_and_passes_stateless_context_to_reward() -> None:
    client = _FakeClient()
    reward = Reward(final_reward=1.0, is_valid=True)
    reward_fn = MagicMock(return_value=reward)
    agent = _make_agent(client, sandbox_id=None, reward_fn=reward_fn)

    result = await agent.run_trajectory(
        {
            "prompt": "fix the bug",
            "label": "L",
            "metadata": {"docker_image": "img:1", "repo_path": "/testbed"},
        }
    )

    # create() was called once and the returned id is retained on the agent.
    assert client.created_with == [{"image": "img:1"}]
    assert agent.sandbox_id == "sbx-1"
    assert result is reward
    # Every command targets the same, agent-owned sandbox id (stateless execute).
    assert client.executed and all(sid == "sbx-1" for sid, _, _ in client.executed)
    # reward_fn receives the stateless (client, id) pair via the context arg.
    args, _ = reward_fn.call_args
    assert args[0] == []
    assert args[1] == "L"
    assert args[2]["sandbox_client"] is client
    assert args[2]["sandbox_id"] == "sbx-1"


@pytest.mark.asyncio
async def test_reuses_existing_sandbox_id_without_recreating() -> None:
    client = _FakeClient()
    agent = _make_agent(client, sandbox_id="sbx-existing")

    await agent.run_trajectory(
        {"prompt": "hi", "label": None, "metadata": {"docker_image": "img:1"}}
    )

    assert client.created_with == []
    assert agent.sandbox_id == "sbx-existing"
    assert all(sid == "sbx-existing" for sid, _, _ in client.executed)


@pytest.mark.asyncio
async def test_requires_docker_image() -> None:
    client = _FakeClient()
    agent = _make_agent(client, sandbox_id=None)

    with pytest.raises(ValueError, match="docker_image"):
        await agent.run_trajectory({"prompt": "hi", "label": None, "metadata": {}})


@pytest.mark.asyncio
async def test_context_length_rejection_returns_valid_zero_reward() -> None:
    client = _FakeClient()
    # The Claude invocation exits non-zero; diagnostics surface a context error.
    client.add_override("--print", {"exit_code": 1, "stdout": "", "stderr": ""})
    client.add_override(
        "ps -eo",
        {"exit_code": 1, "stdout": "Error: context length exceeded", "stderr": ""},
    )
    reward_fn = MagicMock()
    agent = _make_agent(client, sandbox_id="sbx-1", reward_fn=reward_fn)

    result = await agent.run_trajectory(
        {"prompt": "big task", "label": None, "metadata": {"docker_image": "img:1"}}
    )

    assert result.is_valid is True
    assert result.final_reward == 0.0
    assert result.is_correct is False
    # A graceful context rejection must not run the reward function.
    reward_fn.assert_not_called()


@pytest.mark.asyncio
async def test_non_context_failure_raises() -> None:
    client = _FakeClient()
    client.add_override("--print", {"exit_code": 2, "stdout": "", "stderr": ""})
    client.add_override(
        "ps -eo", {"exit_code": 0, "stdout": "boom traceback", "stderr": ""}
    )
    agent = _make_agent(client, sandbox_id="sbx-1")

    with pytest.raises(RuntimeError, match="Claude Code exited with code 2"):
        await agent.run_trajectory(
            {"prompt": "x", "label": None, "metadata": {"docker_image": "img:1"}}
        )


@pytest.mark.asyncio
async def test_close_interrupts_running_process() -> None:
    client = _FakeClient()
    agent = _make_agent(client, sandbox_id="sbx-9")
    agent._claude_running = True

    await agent.close()

    assert any(
        sid == "sbx-9" and "pkill" in command for sid, command, _ in client.executed
    )


@pytest.mark.asyncio
async def test_close_noop_when_not_running() -> None:
    client = _FakeClient()
    agent = _make_agent(client, sandbox_id="sbx-9")

    await agent.close()

    assert client.executed == []