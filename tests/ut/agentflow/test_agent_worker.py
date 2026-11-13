"""Unit tests for out-of-process agent execution."""

import asyncio

import pytest

ray = pytest.importorskip("ray")

from coda.agentflow.agent_worker import (  # noqa: E402
    AgentWorker,
    AttemptRequest,
    AttemptResult,
    _DataSourceContext,
    forwarded_env_vars,
)
from coda.reward.reward import Reward  # noqa: E402


class _FakeAgent:
    """Minimal agent: records what it was given and whether it was closed."""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.sandbox_id = kwargs.get("sandbox_id")
        self.closed = False
        self.cancelled = False
        _FakeAgent.instances.append(self)

    async def run_trajectory(self, payload):
        self.payload = payload
        return Reward(final_reward=1.0)

    async def close(self):
        self.closed = True


class _CreatingAgent(_FakeAgent):
    """Agent that creates its own sandbox, as agents without prepare_sandbox do."""

    async def run_trajectory(self, payload):
        self.sandbox_id = "sandbox-made-by-agent"
        return Reward(final_reward=0.5)


class _FailingAgent(_FakeAgent):
    async def run_trajectory(self, payload):
        raise ValueError("boom")


class _HangingAgent(_FakeAgent):
    async def run_trajectory(self, payload):
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            self.cancelled = True
            raise


def _worker(agent_cls) -> AgentWorker:
    worker = AgentWorker(ds_configs={0: {}}, max_concurrency=4)
    worker._contexts[0] = _DataSourceContext(
        agent_class=agent_cls, reward_fn=object(), sandbox_client=None, agent_cfg={},
    )
    return worker


def _request(sandbox_id="sandbox-from-driver") -> AttemptRequest:
    return AttemptRequest(
        ds_index=0, trajectory_id="traj0", attempt_id=0,
        router_url="http://127.0.0.1:1/traj0/0", payload={"prompt": []},
        max_response_len_per_trajectory=128, sandbox_id=sandbox_id,
    )


@pytest.fixture(autouse=True)
def _clear_instances():
    _FakeAgent.instances.clear()
    yield


def test_run_attempt_returns_reward_and_closes_agent():
    result = asyncio.run(_worker(_FakeAgent).run_attempt(_request()))
    assert isinstance(result, AttemptResult)
    assert result.error is None
    assert result.reward.final_reward == 1.0
    assert result.sandbox_id == "sandbox-from-driver"
    agent = _FakeAgent.instances[0]
    assert agent.closed
    assert agent.kwargs["router_url"] == "http://127.0.0.1:1/traj0/0"
    assert agent.payload == {"prompt": []}


def test_sandbox_created_by_agent_is_reported_back():
    result = asyncio.run(_worker(_CreatingAgent).run_attempt(_request(sandbox_id=None)))
    assert result.sandbox_id == "sandbox-made-by-agent"


def test_failure_becomes_error_string_not_exception():
    result = asyncio.run(_worker(_FailingAgent).run_attempt(_request()))
    assert result.reward is None
    assert result.error.startswith("ValueError: boom")
    assert "Traceback" in result.error
    assert _FakeAgent.instances[0].closed


def test_cancel_runs_agent_cleanup_then_propagates():
    async def scenario():
        worker = _worker(_HangingAgent)
        task = asyncio.ensure_future(worker.run_attempt(_request()))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    agent = _FakeAgent.instances[0]
    assert agent.cancelled
    assert agent.closed


def test_forwarded_env_includes_configured_prefixes(monkeypatch):
    monkeypatch.setenv("MYSANDBOX_SERVER", "http://sandbox:8081")
    monkeypatch.setenv("MYSANDBOX_TOKEN", "secret")
    monkeypatch.setenv("UNRELATED_VAR", "x")
    monkeypatch.setenv("PYTHONPATH", "/repo")

    env = forwarded_env_vars(["MYSANDBOX_"])

    assert env["MYSANDBOX_SERVER"] == "http://sandbox:8081"
    assert env["MYSANDBOX_TOKEN"] == "secret"
    assert env["PYTHONPATH"] == "/repo"
    assert "UNRELATED_VAR" not in env


def test_forwarded_env_without_prefixes_only_carries_pythonpath(monkeypatch):
    monkeypatch.setenv("MYSANDBOX_TOKEN", "secret")
    monkeypatch.setenv("PYTHONPATH", "/repo")

    assert forwarded_env_vars() == {"PYTHONPATH": "/repo"}


def _flow_stub(agent_workers, partial, prepare_sandbox):
    """Enough of an AgentFlow for _init_agent_worker_pool, without starting a router."""
    import contextlib
    from types import SimpleNamespace

    from omegaconf import OmegaConf

    reward_fn = SimpleNamespace()
    if prepare_sandbox:
        reward_fn.prepare_sandbox = lambda *a, **k: {}
    return SimpleNamespace(
        config=OmegaConf.create({
            "agentflow": {"agent_workers": agent_workers, "router": {"abort_timeout_seconds": 600}},
        }),
        partial_rollout_enabled=partial,
        ds_configs={0: OmegaConf.create({"agent": {"name": "mini-swe"}})},
        ds_sandbox_clients={0: object()},
        ds_reward_fns={0: reward_fn},
        _resources=contextlib.AsyncExitStack(),
    )


def test_pool_is_not_created_when_workers_is_zero():
    from coda.agentflow.agent_flow import AgentFlow

    flow = _flow_stub(agent_workers=0, partial=False, prepare_sandbox=True)
    AgentFlow._init_agent_worker_pool(flow)
    assert flow.agent_worker_pool is None


def test_partial_rollout_without_prepare_sandbox_is_rejected():
    from coda.agentflow.agent_flow import AgentFlow

    flow = _flow_stub(agent_workers=8, partial=True, prepare_sandbox=False)
    with pytest.raises(ValueError, match="prepare_sandbox"):
        AgentFlow._init_agent_worker_pool(flow)
