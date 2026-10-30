"""Partial-resume behaviour shared by the black-box CLI agents (OpenCode, Claude Code, Codex).

On resume AgentFlow sets ``resume=True`` and passes the active-segment history as
``prompt``. A black-box CLI cannot ingest that history, so the agent must reload
the CLI's own session persisted in the retained sandbox instead of starting a
fresh run.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest

from coda.agentflow.agent.claude_code.claude_code_agent import ClaudeCodeAgent
from coda.agentflow.agent.codex.codex_agent import CodexAgent
from coda.agentflow.agent.opencode.opencode_agent import OpenCodeAgent
from coda.agentflow.sandbox.base import SandboxClient
from coda.agentflow.utils import RESUME_PROMPT
from coda.reward.reward import Reward

_TASK = "fix the bug in foo.py"
_HISTORY = [
    {"role": "system", "content": "cli system prompt"},
    {"role": "user", "content": "<system-reminder>injected context</system-reminder>"},
    {"role": "user", "content": _TASK},
    {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{}"}},
    ]},
    {"role": "tool", "tool_call_id": "c1", "content": "tool output"},
]

# (agent class, CLI substring identifying the launch command, resume flag, marker path)
_AGENTS = [
    pytest.param(OpenCodeAgent, " run ", "run --continue ", "/tmp/coda-opencode.started", id="opencode"),
    pytest.param(ClaudeCodeAgent, "--print", "--print --continue ", "/tmp/coda-claude-code.started", id="claude_code"),
    pytest.param(CodexAgent, " exec ", " resume --last ", "/tmp/coda-codex.started", id="codex"),
]


class _FakeClient(SandboxClient):
    """Stateless fake backend; ``existing_files`` drives ``test -f`` results."""

    def __init__(self, existing_files: set[str] | None = None) -> None:
        self.existing_files = existing_files or set()
        self.executed: list[str] = []

    def create(self, **kwargs: Any) -> str:
        return "sbx-new"

    def execute(self, sandbox_id: str, command: str, **kwargs: Any) -> dict[str, Any]:
        self.executed.append(command)
        if command.startswith("test -f "):
            path = command[len("test -f "):].strip("'")
            return {"exit_code": 0 if path in self.existing_files else 1, "stdout": "", "stderr": ""}
        return {"exit_code": 0, "stdout": "", "stderr": ""}

    def delete(self, sandbox_id: str, **kwargs: Any) -> None:
        pass


def _make_agent(agent_cls: type, client: SandboxClient) -> Any:
    return agent_cls(
        router_url="http://router:8000/t1/0",
        reward_fn=MagicMock(return_value=Reward(final_reward=1.0, is_valid=True)),
        sandbox_client=client,
        sandbox_id="sbx-1",
        context_length=1000,
    )


def _launch_command(client: _FakeClient, cli_marker: str) -> str:
    commands = [c for c in client.executed if cli_marker in c and "touch " in c]
    assert len(commands) == 1, client.executed
    return commands[0]


async def _run(agent: Any, prompt: Any, initial_prompt: Any = None, *, resume: bool = False) -> None:
    trajectory = {
        "prompt": prompt,
        "label": None,
        "metadata": {"docker_image": "img:1", "repo_path": "/testbed"},
        "resume": resume,
    }
    if initial_prompt is not None:
        trajectory["initial_prompt"] = initial_prompt
    await agent.run_trajectory(trajectory)


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_cls,cli_marker,resume_flag,marker_path", _AGENTS)
async def test_fresh_run_uses_task_and_writes_marker(agent_cls, cli_marker, resume_flag, marker_path) -> None:
    client = _FakeClient()
    await _run(_make_agent(agent_cls, client), [{"role": "user", "content": _TASK}])

    command = _launch_command(client, cli_marker)
    assert f"touch {marker_path}" in command
    assert _TASK in command
    assert resume_flag not in command
    assert RESUME_PROMPT not in command
    # A non-resume run never triggers the session probe.
    assert not any(c.startswith("test -f ") for c in client.executed)


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_cls,cli_marker,resume_flag,marker_path", _AGENTS)
async def test_history_without_resume_flag_starts_fresh_without_probe(
    agent_cls, cli_marker, resume_flag, marker_path,
) -> None:
    """E.g. a retry on a fresh sandbox: the prompt still carries the pre-abort
    history, but AgentFlow did not mark the attempt as a resume."""
    client = _FakeClient(existing_files={marker_path})
    await _run(_make_agent(agent_cls, client), _HISTORY, initial_prompt=[{"role": "user", "content": _TASK}])

    command = _launch_command(client, cli_marker)
    assert resume_flag not in command
    assert _TASK in command
    assert "injected context" not in command
    assert not any(c.startswith("test -f ") for c in client.executed)


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_cls,cli_marker,resume_flag,marker_path", _AGENTS)
async def test_partial_resume_continues_cli_session(agent_cls, cli_marker, resume_flag, marker_path) -> None:
    client = _FakeClient(existing_files={marker_path})
    await _run(
        _make_agent(agent_cls, client), _HISTORY,
        initial_prompt=[{"role": "user", "content": _TASK}], resume=True,
    )

    command = _launch_command(client, cli_marker)
    assert resume_flag in command
    assert RESUME_PROMPT in command
    # The task is already in the CLI's session; it must not be re-sent, and the
    # flattened history (injected reminders, tool output) must not leak in.
    assert _TASK not in command
    assert "injected context" not in command
    assert "tool output" not in command


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_cls,cli_marker,resume_flag,marker_path", _AGENTS)
async def test_resume_without_cli_session_starts_fresh_from_initial_prompt(
    agent_cls, cli_marker, resume_flag, marker_path,
) -> None:
    """Sandbox retained but the abort came before the CLI started: no session to continue."""
    client = _FakeClient()
    await _run(
        _make_agent(agent_cls, client), _HISTORY,
        initial_prompt=[{"role": "user", "content": _TASK}], resume=True,
    )

    command = _launch_command(client, cli_marker)
    assert resume_flag not in command
    assert _TASK in command
    assert "injected context" not in command
    assert "tool output" not in command


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "agent_cls,running_flag,pattern",
    [
        pytest.param(OpenCodeAgent, "_opencode_running", r"/root/\.opencode/bin/[o]pencode( |$)", id="opencode"),
        pytest.param(ClaudeCodeAgent, "_claude_running", "(^|/)[c]laude( |$)", id="claude_code"),
        pytest.param(CodexAgent, "_codex_running", "(^|/)[c]odex( |$)", id="codex"),
    ],
)
async def test_close_kills_cli_process_tree_only_while_running(agent_cls, running_flag, pattern) -> None:
    client = _FakeClient()
    agent = _make_agent(agent_cls, client)
    await agent.close()
    assert client.executed == []

    setattr(agent, running_flag, True)
    await agent.close()
    assert len(client.executed) == 1
    command = client.executed[0]
    assert command.startswith("bash -c ")
    assert "coda-kill" in command and pattern in command
    assert getattr(agent, running_flag) is False
