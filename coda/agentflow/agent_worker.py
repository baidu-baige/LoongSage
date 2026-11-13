"""Out-of-process agent execution.

Agent work (request building, jinja rendering, json/pydantic serialization) is pure
Python and runs under the driver's GIL, so rollout throughput saturates no matter how
many trajectories are in flight. This module moves one attempt's
``agent.run_trajectory`` into a Ray actor so that work runs in parallel across processes.

The driver keeps everything that needs its own in-process state: the router (and the
trajectory store the router writes into), retries, sandbox lifetime and trajectory dumps.
A worker only needs plain data plus HTTP access to the router and the sandbox backend.
"""

from __future__ import annotations

import asyncio
import logging
import os
import traceback
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import ray

from coda.agentflow.agent import get_agent_class
from coda.agentflow.sandbox import create_sandbox_client
from coda.reward import create_reward_fn
from coda.reward.reward import Reward

logger = logging.getLogger(__name__)

# Ray actors do not inherit the driver environment, so entries a worker needs are
# forwarded through runtime_env. PYTHONPATH is always forwarded because workers import
# the same tree as the driver; sandbox backends that read credentials from the
# environment declare their prefixes via agentflow.agent_worker_env_prefixes.
_FORWARDED_ENV_KEYS = ("PYTHONPATH",)


@dataclass
class AttemptRequest:
    """Inputs for one multi-turn attempt, as plain picklable data."""

    ds_index: int
    trajectory_id: str
    attempt_id: int
    router_url: str
    payload: dict[str, Any]
    max_response_len_per_trajectory: int
    sandbox_id: str | None = None


@dataclass
class AttemptResult:
    """Outcome of one attempt. ``error`` set means the attempt failed."""

    reward: Reward | None = None
    sandbox_id: str | None = None
    error: str | None = None


@dataclass
class _DataSourceContext:
    """Per-data-source objects rebuilt once inside a worker."""

    agent_class: Any
    reward_fn: Any
    sandbox_client: Any
    agent_cfg: dict[str, Any] = field(default_factory=dict)


def forwarded_env_vars(env_prefixes: Sequence[str] = ()) -> dict[str, str]:
    """Return the driver environment entries workers need."""
    prefixes = tuple(env_prefixes)
    env = {k: v for k, v in os.environ.items() if prefixes and k.startswith(prefixes)}
    env.update({k: os.environ[k] for k in _FORWARDED_ENV_KEYS if k in os.environ})
    return env


class AgentWorker:
    """Runs agent attempts for one or more data sources.

    Built as a Ray actor by :class:`AgentWorkerPool`. Per-data-source objects are
    rebuilt here from config rather than shipped from the driver: sandbox clients and
    reward functions are stateless (a few strings plus lazily created HTTP calls), so
    rebuilding is cheaper and safer than pickling driver-owned instances.
    """

    def __init__(self, ds_configs: dict[int, Any], max_concurrency: int) -> None:
        self._ds_configs = ds_configs
        self._contexts: dict[int, _DataSourceContext] = {}
        self._max_concurrency = int(max_concurrency)
        self._executor_expanded = False

    def _context(self, ds_index: int) -> _DataSourceContext:
        """Return (and cache) the agent class, reward function and sandbox client."""
        if (context := self._contexts.get(ds_index)) is not None:
            return context
        ds_cfg = self._ds_configs[ds_index]
        agent_cfg = dict(ds_cfg.get("agent") or {})
        sandbox_config = dict(ds_cfg.get("sandbox") or {})
        sandbox_config.pop("max_concurrency", None)
        context = _DataSourceContext(
            agent_class=get_agent_class(agent_cfg["name"]),
            reward_fn=create_reward_fn(ds_cfg.reward),
            sandbox_client=create_sandbox_client(sandbox_config),
            agent_cfg=agent_cfg,
        )
        self._contexts[ds_index] = context
        return context

    async def run_attempt(self, request: AttemptRequest) -> AttemptResult:
        """Execute one attempt and return its reward, or the failure as a string.

        Exceptions become ``AttemptResult.error`` because the driver's retry loop only
        logs the message and retries; it never branches on the exception type. Agent-level
        signals (format errors, context-length exhaustion) are handled inside the agent,
        in this process. ``CancelledError`` propagates after ``agent.close()`` so the
        driver sees the cancellation it requested.
        """
        if not self._executor_expanded:
            # Agents run their sync work through asyncio.to_thread; size the pool to this
            # worker's share of trajectories instead of the driver's global pool.
            asyncio.get_running_loop().set_default_executor(
                ThreadPoolExecutor(max_workers=self._max_concurrency)
            )
            self._executor_expanded = True

        context = self._context(request.ds_index)
        agent = None
        try:
            agent = context.agent_class(
                router_url=request.router_url,
                max_response_len_per_trajectory=request.max_response_len_per_trajectory,
                **context.agent_cfg,
                reward_fn=context.reward_fn,
                sandbox_client=context.sandbox_client,
                sandbox_id=request.sandbox_id,
            )
            reward = await agent.run_trajectory(request.payload)
            return AttemptResult(reward=reward, sandbox_id=agent.sandbox_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return AttemptResult(
                sandbox_id=getattr(agent, "sandbox_id", request.sandbox_id),
                error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            )
        finally:
            if agent is not None:
                with_close = getattr(agent, "close", None)
                if with_close is not None:
                    await agent.close()


class AgentWorkerPool:
    """Driver-side handle to a fixed set of :class:`AgentWorker` actors."""

    def __init__(
        self,
        ds_configs: dict[int, Any],
        num_workers: int,
        num_cpus: float,
        max_concurrency: int,
        cancel_timeout_seconds: float,
        env_prefixes: Sequence[str] = (),
    ) -> None:
        self._ds_configs = ds_configs
        self._num_workers = int(num_workers)
        self._max_concurrency = int(max_concurrency)
        self._cancel_timeout_seconds = float(cancel_timeout_seconds)
        self._worker_cls = ray.remote(num_cpus=num_cpus)(AgentWorker).options(
            runtime_env={"env_vars": forwarded_env_vars(env_prefixes)}
        )
        self._workers = [self._new_worker() for _ in range(self._num_workers)]
        self._next = 0
        logger.info(
            "[init] agent worker pool: workers=%d num_cpus=%s max_concurrency=%d",
            self._num_workers, num_cpus, self._max_concurrency,
        )

    def _new_worker(self):
        """Create one worker actor."""
        return self._worker_cls.remote(self._ds_configs, self._max_concurrency)

    def _claim_worker(self) -> tuple[int, Any]:
        """Return the next worker in round-robin order."""
        index = self._next % self._num_workers
        self._next += 1
        return index, self._workers[index]

    async def run_attempt(self, request: AttemptRequest) -> AttemptResult:
        """Run one attempt on a worker.

        A dead worker is replaced and the attempt reported as failed, so the driver's
        existing per-trajectory retry handles it. On cancellation the remote task is
        cancelled and briefly waited on, giving the agent's cleanup a chance to release
        its sandbox before the driver moves on.
        """
        index, worker = self._claim_worker()
        ref = worker.run_attempt.remote(request)
        try:
            return await ref
        except asyncio.CancelledError:
            ray.cancel(ref)
            try:
                await asyncio.wait_for(
                    asyncio.shield(ref), timeout=self._cancel_timeout_seconds
                )
            except (asyncio.TimeoutError, asyncio.CancelledError, ray.exceptions.RayError):
                pass
            except Exception as exc:
                logger.warning("[agent-worker] cleanup after cancel failed: %s", exc)
            raise
        except ray.exceptions.RayActorError as exc:
            logger.error("[agent-worker] worker %d died, replacing it: %s", index, exc)
            self._workers[index] = self._new_worker()
            return AttemptResult(sandbox_id=request.sandbox_id, error=f"agent worker died: {exc}")

    def shutdown(self) -> None:
        """Kill all worker actors."""
        for worker in self._workers:
            try:
                ray.kill(worker)
            except Exception as exc:
                logger.warning("[agent-worker] kill failed: %s", exc)
        self._workers = []

