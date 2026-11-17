"""Unit tests for the stateless SandboxClient contract."""

import subprocess
from typing import Any
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from coda.agentflow.agent.bcp.bcp_agent import BCPAgent
from coda.agentflow.agent.gsm8k.gsm8k_agent import GSM8KAgent
from coda.agentflow.sandbox.base import SandboxClient
from coda.reward.functions.r2e_gym import R2EGymReward


class _FakeClient(SandboxClient):
    def __init__(self) -> None:
        self.created_with: list[dict[str, Any]] = []
        self.executed: list[tuple[str, str, dict[str, Any]]] = []
        self.deleted: list[str] = []

    def create(self, **kwargs: Any) -> str:
        self.created_with.append(kwargs)
        return f"sbx-{len(self.created_with)}"

    def execute(
        self, sandbox_id: str, command: str, **kwargs: Any
    ) -> dict[str, Any]:
        self.executed.append((sandbox_id, command, kwargs))
        return {"exit_code": 0, "stdout": sandbox_id, "stderr": ""}

    def delete(self, sandbox_id: str, **kwargs: Any) -> None:
        self.deleted.append(sandbox_id)


def test_one_client_manages_independent_instances() -> None:
    client = _FakeClient()

    first = client.create(image="img:1")
    second = client.create(image="img:2")
    first_result = client.execute(first, "pwd", workdir="/one")
    second_result = client.execute(second, "pwd", workdir="/two")
    client.delete(first)
    client.delete(second)

    assert (first, second) == ("sbx-1", "sbx-2")
    assert first_result["stdout"] == first
    assert second_result["stdout"] == second
    assert client.executed == [
        (first, "pwd", {"workdir": "/one"}),
        (second, "pwd", {"workdir": "/two"}),
    ]
    assert client.deleted == [first, second]
    assert not hasattr(client, "_sandbox_id")


def test_docker_config_uses_standard_command_timeout() -> None:
    from coda.agentflow.sandbox.docker_sandbox import DockerSandboxClient

    client = DockerSandboxClient.from_config({"command_exec_timeout_seconds": 321})

    assert client.timeout == 321


def test_gsm8k_agent_has_no_backend_behavior() -> None:
    client = MagicMock()

    agent = GSM8KAgent(
        router_url="http://router",
        reward_fn=MagicMock(),
        sandbox_client=client,
        sandbox_id="sandbox-1",
    )

    assert not hasattr(agent, "execute")
    client.create.assert_not_called()
    client.execute.assert_not_called()
    client.delete.assert_not_called()


def test_bcp_agent_has_no_backend_behavior() -> None:
    client = MagicMock()

    with pytest.MonkeyPatch.context() as monkeypatch:
        http_client = MagicMock()
        monkeypatch.setattr("coda.agentflow.agent.bcp.bcp_agent.httpx.AsyncClient", http_client)
        agent = BCPAgent(
            router_url="http://router",
            retrieval_service_url="http://retrieval",
            reward_fn=MagicMock(),
            sandbox_client=client,
            sandbox_id="sandbox-1",
        )

    assert not hasattr(agent, "execute")
    assert hasattr(agent, "_execute_tool")
    client.create.assert_not_called()
    client.execute.assert_not_called()
    client.delete.assert_not_called()


def test_r2e_prepare_returns_digest_for_client_and_id() -> None:
    digest = "a" * 64
    client = MagicMock()
    client.execute.return_value = {
        "exit_code": 0,
        "stdout": f"setup complete\nCODA_R2E_ASSET_SHA256={digest}\n",
        "stderr": "",
    }

    result = R2EGymReward(OmegaConf.create({})).prepare_sandbox(
        client, "sandbox-3", metadata={"repo_path": "/workspace"}
    )

    assert result == {"_coda_r2e_asset_sha256": digest}
    args, kwargs = client.execute.call_args
    assert args[0] == "sandbox-3"
    assert "CODA_R2E_ASSET_SHA256" in args[1]
    assert kwargs == {"workdir": "/workspace"}


def _docker_client(timeout_seconds: int = 600):
    from coda.agentflow.sandbox.docker_sandbox import DockerSandboxClient

    return DockerSandboxClient.from_config({"command_exec_timeout_seconds": timeout_seconds})


def _k8s_client(timeout_seconds: int = 600):
    from coda.agentflow.sandbox.k8s_sandbox import K8sSandboxClient

    return K8sSandboxClient.from_config({"command_exec_timeout_seconds": timeout_seconds})


def _completed(returncode: int, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


@pytest.mark.parametrize("make_client", [_docker_client, _k8s_client])
def test_inner_timeout_wrapper_is_reported(monkeypatch, make_client) -> None:
    """`timeout` exits 124 when it kills the command; that is the usual timeout path."""
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed(124))

    result = make_client().execute("sandbox-1", "sleep 999")

    assert result["timed_out"] is True
    assert result["error"] == "Command timed out after 600s"
    assert result["success"] is False


@pytest.mark.parametrize("make_client", [_docker_client, _k8s_client])
def test_outer_subprocess_timeout_is_reported(monkeypatch, make_client) -> None:
    def raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="exec", timeout=kwargs.get("timeout", 0))

    monkeypatch.setattr(subprocess, "run", raise_timeout)

    result = make_client(120).execute("sandbox-1", "sleep 999")

    assert result["timed_out"] is True
    assert result["error"] == "Command timed out after 120s"
    assert result["exit_code"] == -1


@pytest.mark.parametrize("make_client", [_docker_client, _k8s_client])
def test_outer_subprocess_timeout_returns_text_output(monkeypatch, make_client) -> None:
    """TimeoutExpired carries bytes (stderr None) even when run() used text=True."""

    def raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="exec", timeout=kwargs.get("timeout", 0), output=b"partial\n")

    monkeypatch.setattr(subprocess, "run", raise_timeout)

    result = make_client(120).execute("sandbox-1", "sleep 999")

    assert result["stdout"] == "partial\n"
    assert result["stderr"] == ""


def test_decode_output_handles_bytes_str_and_none() -> None:
    from coda.agentflow.sandbox.base import decode_output

    assert decode_output(b"caf\xc3\xa9") == "café"
    assert decode_output(b"bad\xff") == "bad\ufffd"
    assert decode_output("text") == "text"
    assert decode_output(None) == ""


@pytest.mark.parametrize("make_client", [_docker_client, _k8s_client])
def test_normal_command_is_not_marked_timed_out(monkeypatch, make_client) -> None:
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed(0, stdout="hello\n"))

    result = make_client().execute("sandbox-1", "echo hello")

    assert result["timed_out"] is False
    assert result["error"] == ""
    assert (result["exit_code"], result["success"]) == (0, True)


@pytest.mark.parametrize("make_client", [_docker_client, _k8s_client])
def test_nonzero_exit_is_not_a_timeout(monkeypatch, make_client) -> None:
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed(2, stderr="boom\n"))

    result = make_client().execute("sandbox-1", "false")

    assert result["timed_out"] is False
    assert result["error"] == ""
    assert (result["exit_code"], result["success"]) == (2, False)
