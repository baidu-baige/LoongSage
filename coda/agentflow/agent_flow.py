#!/usr/bin/env python3
"""AgentFlow: orchestrator for RL agent trajectories."""
import asyncio
import contextlib
import copy
import json
import logging
import os
import socket
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

import httpx
from fastapi import status

from coda.agentflow.agent.base_agent import BaseAgent
from coda.reward.reward import Reward
from coda.agentflow.trajectory_store import (
    Trajectory,
    TrajectoryStore,
    TrajectoryStatus,
)
from coda.agentflow.trajectory_queue import TrajQueue
from coda.agentflow.tokenizer_manager import (
    BaseTokenizerManager,
    create_tokenizer_manager,
)
from coda.agentflow.agent import get_agent_class
from coda.agentflow.sandbox import create_sandbox_client
from coda.agentflow.utils import CONTEXT_LENGTH_EXCEEDED
from coda.reward import create_reward_fn

logger = logging.getLogger(__name__)

_ROLLOUT_SINGLE_TURN = "single_turn"
_ROLLOUT_MULTI_TURN = "multi_turn"


def get_rollout_mode(ds_unit) -> str:
    """Determine rollout mode from a data source unit config.

    Returns 'multi_turn' if agent.name is set, otherwise 'single_turn'.
    """
    agent_name = ds_unit.get("agent", {}).get("name")
    if agent_name:
        return _ROLLOUT_MULTI_TURN
    return _ROLLOUT_SINGLE_TURN


def _active_chat(traj: Trajectory) -> list[dict[str, Any]] | Any:
    """Return the chat-completions segment at active_segment_id, for resume/reward use.

    Falls back to traj.prompt when there is no stored chat yet (fresh trajectory).
    """
    active_id = traj.active_segment_id
    if active_id in traj.chat_completions:
        return traj.chat_completions[active_id]
    return traj.prompt


class AgentFlow:
    """Orchestrator: Router in background thread, TrajectoryStore shared with Router, one Agent per trajectory.

    Supports multiple data sources, each with its own agent/reward/token config.

    Usage: af = AgentFlow(config); rewards = await af.generate(trajectories)
    """

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def __init__(self, config: Any):
        self.config = config
        group_sizes: dict[int, int] = {}
        for ds_index, ds_cfg in enumerate(self.config.data_sources):
            group_sizes[ds_index] = int(ds_cfg.num_trajectories_per_prompt)
        self.traj_queue = TrajQueue(group_sizes=group_sizes) 
        self.retry_limit = self.config.rollout.retry_limit
        self.accumulate_reasoning = self.config.agentflow.router.accumulate_reasoning
        self.r3_enabled = self.config.trainer.use_rollout_routing_replay
        self.max_response_len_per_trajectory = int(self.config.data_sources[0].max_response_len_per_trajectory)
        self.partial_rollout_enabled = self.config.rollout.partial
        self.mask_offpolicy_in_partial_rollout = self.config.rollout.mask_offpolicy_in_partial_rollout
        self.trajectory_store = TrajectoryStore()
        # Sandbox pool: trajectory_id -> live sandbox client. Entries survive a
        # partial-rollout abort so a resumed trajectory reuses its sandbox (and
        # the tool state inside) instead of recreating one from scratch.
        self._sandbox_pool: dict[str, Any] = {}
        self.router = None
        self.tokenizer_manager: BaseTokenizerManager | None = None
        self._active_tasks: set[asyncio.Task] = set()
        self._inflight_queue_write_futures: set[asyncio.Future] = set()
        self._resources = contextlib.AsyncExitStack()

        self._max_connections = int(self.config.agentflow.router.max_connections)
        # Per-request generation + shared bounded control client, both gated by a
        # semaphore -> each pool's C~=1 so httpcore's O(C^2) _assign stays cheap.
        self._generate_sem = asyncio.Semaphore(self._max_connections)
        self._control_sem = asyncio.Semaphore(self._max_connections)
        self._control_client: httpx.AsyncClient | None = None

        # Per-data-source agent classes and reward functions (keyed by ds index)
        self.ds_agent_classes: dict[int, type[BaseAgent] | None] = {}
        self.ds_reward_fns: dict[int, Any] = {}
        self.ds_configs: dict[int, Any] = {}  # per-data-source unit configs

        self._init_per_datasource()

        logger.info("[init] retry_limit=%s data_sources=%d", self.retry_limit, len(self.ds_configs))
        self._run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._init_tokenizer()
        self._init_router()
        logger.info("[init] AgentFlow ready")

    def _init_tokenizer(self) -> None:
        """Initialize tokenizer from config."""
        tokenizer_cfg = self.config.agentflow.tokenizer
        self.tokenizer_manager = create_tokenizer_manager(tokenizer_cfg, self.config.hf_model_path)
        self._resources.callback(self.tokenizer_manager.shutdown)
        logger.info(
            "[init] tokenizer=%s manager=%s workers=%d",
            self.tokenizer_manager.tokenizer.__class__.__name__,
            self.tokenizer_manager.mode,
            self.tokenizer_manager.num_workers,
        )

    @staticmethod
    def _resolve_local_ip() -> str:
        """Resolve the local machine's routable IP address."""
        try:
            # Connect to an external address to determine which local interface is used.
            # No data is actually sent; this just lets the OS pick the right source IP.
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"

    @staticmethod
    def _find_free_port() -> int:
        """Find and return a free TCP port on the local machine."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    def _init_router(self) -> None:
        """Initialize and start Router in a background thread."""
        from coda.agentflow.router.router import Router

        router_config = self.config.agentflow.router
        router_ip = router_config.ip
        router_port = router_config.port
        # Auto-resolve local IP when not explicitly configured, so that rollout
        # workers on other nodes can reach this router without manual setup.
        if not router_ip:
            router_ip = self._resolve_local_ip()
            router_config.ip = router_ip
            logger.info("[init] auto-resolved router IP to %s", router_ip)
        # Auto-assign a free port when configured as 0.
        if not router_port:
            router_port = self._find_free_port()
            router_config.port = router_port
            logger.info("[init] auto-assigned router port to %d", router_port)
        self.router_url = f"http://{router_ip}:{router_port}"

        self._control_client = httpx.AsyncClient(
            timeout=router_config.proxy_timeout_seconds,
            limits=httpx.Limits(max_connections=self._max_connections),
        )

        async def _shutdown_router() -> None:
            await self._control_client.aclose()
            await self.router.shutdown()
            self.router = None

        self._resources.push_async_callback(_shutdown_router)
        logger.info("[init] router at %s", self.router_url)

        self.router = Router(
            config=router_config,
            middleware_kwargs={k: v for k, v in vars(self).items() if not k.startswith("_")},
        )
        self.router.start_background(host="0.0.0.0", port=router_port)
        logger.info("[init] router ready at %s", self.router_url)

    def _init_per_datasource(self) -> None:
        """Initialize agent classes and reward functions for each data source unit."""

        for idx, ds_cfg in enumerate(self.config.data_sources):
            self.ds_configs[idx] = ds_cfg
            rollout_mode = get_rollout_mode(ds_cfg)

            # Agent
            if rollout_mode == _ROLLOUT_SINGLE_TURN:
                self.ds_agent_classes[idx] = None
            else:
                agent_name = ds_cfg.agent.name
                if not agent_name:
                    raise ValueError(f"data_sources[{idx}].agent.name is required for multi_turn mode")
                self.ds_agent_classes[idx] = get_agent_class(agent_name)
                logger.info("[init] ds[%d] agent=%s", idx, agent_name)

            # Reward
            reward_fn = create_reward_fn(ds_cfg.reward)
            self.ds_reward_fns[idx] = reward_fn
            if reward_fn:
                logger.info("[init] ds[%d] reward_fn=%s", idx, reward_fn.__class__.__name__)
            else:
                logger.info("[init] ds[%d] no reward_fn configured", idx)

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    async def generate(self, trajectories: list[Trajectory]) -> list[Reward]:
        """Run trajectories concurrently and return rewards in the same order."""
        # Expand the default asyncio thread pool so that concurrent pod-creation
        # (asyncio.to_thread) calls don't starve agent.run() inference threads.
        if not hasattr(self, '_executor_expanded'):
            asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=8192))
            self._executor_expanded = True
            logger.info("Expanded default thread pool executor to max_workers=%d", 8192)
        logger.info("[generate] trajectories=%d", len(trajectories))
        tasks: list[asyncio.Task] = []
        for trajectory in trajectories:
            ds_index = trajectory.ds_index
            if ds_index not in self.ds_configs:
                raise ValueError(
                    f"[{trajectory.trajectory_id}] invalid ds_index={ds_index} "
                    f"(is_eval={trajectory.is_eval})"
                )
            if trajectory.status in (TrajectoryStatus.COMPLETED, TrajectoryStatus.FAILED):
                tasks.append(asyncio.create_task(self._reemit_existing(trajectory)))
                continue
            if trajectory.status == TrajectoryStatus.GENERATING:
                raise ValueError(
                    f"[{trajectory.trajectory_id}] unsupported input status for generate(): "
                    f"{trajectory.status}"
                )
            # Pre-register so abort() can mark this trajectory even before
            # _run_trajectory reaches its own store.add().
            if not self.trajectory_store.trajectory_data.get(trajectory.trajectory_id):
                self.trajectory_store.add(trajectory.trajectory_id, trajectory)
            tasks.append(asyncio.create_task(self._run_trajectory(trajectory)))
        for task in tasks:
            self._active_tasks.add(task)
            task.add_done_callback(self._active_tasks.discard)
        results = await asyncio.gather(*tasks, return_exceptions=True)
        rewards: list[Reward] = []
        for index, reward in enumerate(results):
            if isinstance(reward, BaseException):
                logger.warning("[generate] trajectory %d failed unexpectedly: %s", index, reward)
                rewards.append(Reward(final_reward=0.0, is_valid=False))
            else:
                rewards.append(reward)
        positive = sum(1 for r in rewards if r.final_reward and r.final_reward > 0)
        logger.info(
            "[generate] done rewards=%s positive(>0)=%d/%d (%.2f%%)",
            [r.final_reward for r in rewards],
            positive, len(rewards),
            positive / len(rewards) * 100 if rewards else 0.0,
        )
        return rewards

    async def abort(self) -> None:
        """Abort active rollouts and wait until no stale trajectory writes remain.

        Abort every active trajectory.
        """

        # 1. Abort LLM worker requests via Router.
        try:
            async with self._control_sem:
                resp = await self._control_client.post(f"{self.router_url}/abort_all_workers")
            resp.raise_for_status()
            logger.info("[abort] abort_all_workers: %s", resp.json())
        except Exception as exc:
            logger.warning("[abort] abort_all_workers failed: %s", exc)
            raise

        # 2. Cancel active asyncio tasks.
        tasks = list(self._active_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # 3. Wait for pending trajectory queue writes.
        if self._inflight_queue_write_futures:
            await asyncio.gather(*list(self._inflight_queue_write_futures), return_exceptions=True)

        # 4. Wait for Router middleware writes to finish.
        await self._wait_inflight_requests_complete()

        # 5. Mark remaining active trajectories as ABORTED.
        aborted: list[Trajectory] = []
        for tid in list(self.trajectory_store.trajectory_data.keys()):
            attempts = self.trajectory_store.trajectory_data.get(tid)
            if not attempts:
                continue
            latest = attempts[-1]
            if latest.status in (TrajectoryStatus.COMPLETED, TrajectoryStatus.ABORTED):
                continue
            logger.info("[abort] Marking trajectory=%s attempt=%d as ABORTED", tid, latest.attempt_id)
            latest.status = TrajectoryStatus.ABORTED
            self.trajectory_store.update(tid, latest)
            aborted.append(latest)
        if aborted:
            await asyncio.gather(*[self._emit_terminal_trajectory(t) for t in aborted])
        logger.info("[abort] complete aborted=%d", len(aborted))

    async def _wait_inflight_requests_complete(self) -> None:
        """Wait for Router middleware handlers after rollout tasks are cancelled."""
        try:
            async with self._control_sem:
                resp = await self._control_client.post(
                    f"{self.router_url}/wait_inflight_requests_complete",
                )
            resp.raise_for_status()
            logger.info("[abort] inflight requests complete: %s", resp.json())
        except Exception as exc:
            logger.warning("[abort] wait inflight requests complete failed: %s", exc)
            raise

    async def clear(self) -> None:
        """Release all resources: trajectory store, router, tokenizer."""
        logger.info("[clear] clearing AgentFlow")
        self.trajectory_store.clear()
        for trajectory_id in list(self._sandbox_pool.keys()):
            await self._release_sandbox(trajectory_id)
        await self._resources.aclose()

    # -------------------------------------------------------------------------
    # Trajectory execution
    # -------------------------------------------------------------------------

    async def _run_trajectory(self, traj: Trajectory) -> Reward:
        """Run one trajectory with retry, dispatching to single- or multi-turn mode."""
        ds_index = traj.ds_index
        rollout_mode = get_rollout_mode(self.ds_configs[ds_index])
        attempt_traj_template = self._prepare_attempt_template(traj)
        base_attempt_id = traj.attempt_id
        for local_retry_idx in range(self.retry_limit):
            attempt_id = base_attempt_id + local_retry_idx
            attempt_traj = attempt_traj_template.model_copy(deep=True)
            attempt_traj.attempt_id = attempt_id
            attempt_traj.status = TrajectoryStatus.GENERATING
            if local_retry_idx == 0:
                self.trajectory_store.update(traj.trajectory_id, attempt_traj)
            else:
                self.trajectory_store.add(traj.trajectory_id, attempt_traj)

            try:
                if rollout_mode == _ROLLOUT_SINGLE_TURN:
                    reward = await self._execute_single_turn(traj, attempt_id)
                else:
                    reward = await self._execute_multi_turn(traj, attempt_id)
            except Exception as exc:
                # CancelledError (BaseException) is intentionally not caught — abort() step 4 handles it.
                logger.warning(
                    "[%s] Attempt %d/%d failed: %s\n%s",
                    traj.trajectory_id, local_retry_idx + 1, self.retry_limit, exc, traceback.format_exc(),
                )
                attempt_traj.status = TrajectoryStatus.FAILED
                self.trajectory_store.update(traj.trajectory_id, attempt_traj)
                await self._router_session_request(
                    "POST",
                    "/abort_session",
                    traj.trajectory_id,
                    attempt_id,
                )
                if local_retry_idx == self.retry_limit - 1:
                    logger.error("[%s] All %d attempts failed", traj.trajectory_id, self.retry_limit)
                    if dump_path := self.config.agentflow.dump_trajectory_path:
                        self._dump_trajectory(attempt_traj, dump_path)
                    await self._emit_terminal_trajectory(attempt_traj)
                    return Reward(final_reward=0.0, is_valid=False)
            # else: _execute_*_turn returned successfully without raising exception — process the reward.
            else:
                terminal_traj = self._post_process_reward(traj.trajectory_id, attempt_id, reward)
                if terminal_traj:
                    await self._emit_terminal_trajectory(terminal_traj)
                else:
                    logger.info(
                        "[%s] attempt %d: trajectory already emitted by abort(), skipping",
                        traj.trajectory_id, attempt_id,
                    )
                await self._router_session_request(
                    "DELETE",
                    "/release_session",
                    traj.trajectory_id,
                    attempt_id,
                )
                logger.info("[%s] Attempt %d complete, reward=%s", traj.trajectory_id, attempt_id, reward.final_reward)
                return reward

    def _prepare_attempt_template(self, traj: Trajectory) -> Trajectory:
        """Prepare the mutable template used for all attempts of one submitted trajectory."""
        attempt_traj_template = traj.model_copy(deep=True)
        existing_response_len = len(attempt_traj_template.rollout_log_probs)
        if (
            self.partial_rollout_enabled
            and self.mask_offpolicy_in_partial_rollout
            and existing_response_len > 0
        ):
            if len(attempt_traj_template.loss_masks) != existing_response_len:
                raise ValueError(
                    f"[{traj.trajectory_id}] loss_masks length {len(attempt_traj_template.loss_masks)} "
                    f"!= rollout_log_probs length {existing_response_len}"
                )
            if len(attempt_traj_template.rollout_weight_versions) != existing_response_len:
                raise ValueError(
                    f"[{traj.trajectory_id}] rollout_weight_versions length "
                    f"{len(attempt_traj_template.rollout_weight_versions)} "
                    f"!= rollout_log_probs length {existing_response_len}"
                )
            masked_tokens = sum(1 for mask in attempt_traj_template.loss_masks if mask)
            zero_tokens = existing_response_len - masked_tokens
            attempt_traj_template.loss_masks = [0] * existing_response_len
            logger.info(
                "[partial_resume] partial rollout resume: [%s] masked %d existing loss tokens "
                "(response_len=%d pre_zero=%d rollout_weight_versions_len=%d)",
                traj.trajectory_id,
                masked_tokens,
                existing_response_len,
                zero_tokens,
                len(attempt_traj_template.rollout_weight_versions),
            )
        return attempt_traj_template

    async def _execute_multi_turn(self, traj: Trajectory, attempt_id: int) -> Reward:
        """Execute one multi-turn attempt via an Agent instance."""
        ds_index = traj.ds_index
        ds_cfg = self.ds_configs[ds_index]
        agent_class = self.ds_agent_classes[ds_index]
        reward_fn = self.ds_reward_fns[ds_index]
        agent_cfg = ds_cfg.agent if hasattr(ds_cfg, 'agent') else ds_cfg.get("agent", {})

        completion_params = dict(ds_cfg.completion_params)  # copy to avoid mutating shared config
        max_response_len_per_trajectory = int(ds_cfg.max_response_len_per_trajectory)

        # Eval rounds use rollout.eval.temperature when set; completion_params still wins.
        eval_temperature = self.config.rollout.eval.temperature
        temperature = float(
            eval_temperature if traj.is_eval and eval_temperature is not None
            else self.config.trainer.temperature
        )

        init_kwargs: dict[str, Any] = {
            "router_url": f"{self.router_url}/{traj.trajectory_id}/{attempt_id}",
            "completion_params": completion_params,
            "max_response_len_per_trajectory": max_response_len_per_trajectory,
            "temperature": temperature,
            **agent_cfg,
        }
        sandbox_client = self._sandbox_pool.get(traj.trajectory_id)
        if sandbox_client is not None:
            sandbox_id = getattr(sandbox_client, "sandbox_id", None)
            if sandbox_id:
                logger.info(
                    "[%s] Reusing pooled sandbox %s (partial-rollout resume)",
                    traj.trajectory_id, sandbox_id,
                )
            else:
                logger.info(
                    "[%s] Pooled sandbox has no live instance (aborted before "
                    "creation); a new one will be created (partial-rollout resume)",
                    traj.trajectory_id,
                )
        else:
            sandbox_client = create_sandbox_client(self.config.agentflow.sandbox)

        if sandbox_client is not None:
            self._sandbox_pool[traj.trajectory_id] = sandbox_client
            init_kwargs["sandbox_env_client"] = sandbox_client

        if reward_fn:
            init_kwargs["reward_fn"] = reward_fn

        agent = agent_class(**init_kwargs)
        keep_sandbox = False
        try:
            prepare_sandbox = getattr(reward_fn, "prepare_sandbox", None)
            if sandbox_client is not None and prepare_sandbox is not None:
                if sandbox_client.sandbox_id is None:
                    image = str(traj.metadata.get("docker_image") or "")
                    if not image:
                        raise ValueError("trajectory metadata['docker_image'] is missing")
                    await asyncio.to_thread(sandbox_client.create, image=image)
                await asyncio.to_thread(
                    prepare_sandbox,
                    sandbox_client,
                    metadata=copy.deepcopy(traj.metadata),
                )

            reward = await agent.run_trajectory({
                "prompt": copy.deepcopy(_active_chat(traj)),
                "label": copy.deepcopy(traj.label),
                "metadata": copy.deepcopy(traj.metadata),
            })

            attempts = self.trajectory_store.get(
                [traj.trajectory_id], attempt_id=attempt_id).get(traj.trajectory_id) or []
            completed_attempt = attempts[-1] if attempts else None
            if completed_attempt is None:
                raise RuntimeError(
                    f"[{traj.trajectory_id}] attempt {attempt_id} missing from trajectory store "
                    "after agent completed"
                )

            if not completed_attempt.rollout_log_probs:
                raise RuntimeError(f"[{traj.trajectory_id}] attempt {attempt_id} completed without rollout log probs")

            if not any(segment.trainable for segment in completed_attempt.segments):
                raise RuntimeError(f"[{traj.trajectory_id}] attempt {attempt_id} completed without trainable Segment")

            return reward
        except asyncio.CancelledError:
            # Partial-rollout abort: keep the sandbox in the pool so the resumed
            # trajectory continues with its accumulated tool state.
            keep_sandbox = self.partial_rollout_enabled
            raise
        finally:
            # Every other exit deletes the sandbox: terminal state (reward
            # computed), or failed attempt whose sandbox state is untrustworthy
            # for the retry (commands may have partially executed).
            if not keep_sandbox:
                await self._release_sandbox(traj.trajectory_id)
            await agent.clear()

    async def _release_sandbox(self, trajectory_id: str) -> None:
        """Delete and drop the pooled sandbox of a trajectory, if any."""
        sandbox_client = self._sandbox_pool.pop(trajectory_id, None)
        if sandbox_client is None:
            return
        try:
            await asyncio.to_thread(sandbox_client.delete)
        except Exception as exc:
            logger.warning("[%s] Failed to delete sandbox: %s", trajectory_id, exc)

    async def _execute_single_turn(self, traj: Trajectory, attempt_id: int) -> Reward:
        """Execute one single-turn attempt directly through the Router (no agent)."""
        ds_index = traj.ds_index
        ds_cfg = self.ds_configs[ds_index]
        reward_fn = self.ds_reward_fns[ds_index]

        async with self._generate_sem:
            messages = copy.deepcopy(_active_chat(traj))  # already normalized to list[dict] by _build_messages
            # Eval rounds use rollout.eval.temperature when set; completion_params still wins.
            eval_temperature = self.config.rollout.eval.temperature
            temperature = float(
                eval_temperature if traj.is_eval and eval_temperature is not None
                else self.config.trainer.temperature
            )
            generate_client = httpx.AsyncClient(timeout=float(self.config.agentflow.router.proxy_timeout_seconds))
            try:
                response = await generate_client.post(
                    f"{self.router_url}/{traj.trajectory_id}/{attempt_id}/v1/chat/completions",
                    json={
                        "model": "default",
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": int(ds_cfg.max_response_len_per_trajectory),
                        **ds_cfg.completion_params,
                    },
                )
            finally:
                await generate_client.aclose()
        is_context_length_exceeded = False
        if response.status_code == httpx.codes.BAD_REQUEST:
            try:
                err = response.json().get("error", {})
            except Exception:
                err = {}
            is_context_length_exceeded = err.get("type") == CONTEXT_LENGTH_EXCEEDED
        if is_context_length_exceeded:
            logger.info("[%s] max_response_len_per_trajectory exhausted, using partial response", traj.trajectory_id)
        else:
            response.raise_for_status()

        trajs = self.trajectory_store.get([traj.trajectory_id], attempt_id=attempt_id).get(traj.trajectory_id) or []
        traj_out = trajs[-1] if trajs else None
        if not traj_out:
            logger.warning("[%s] Trajectory not found in store after single-turn", traj.trajectory_id)
        if reward_fn:
            reward_messages = copy.deepcopy(_active_chat(traj_out) if traj_out else messages)
            return reward_fn(
                reward_messages, traj.label,
                trajectory=traj_out.model_dump() if traj_out else {},
                max_tokens=int(ds_cfg.max_response_len_per_trajectory),
            )
        logger.warning("[%s] No reward_fn configured, defaulting to 0.0", traj.trajectory_id)
        return Reward(final_reward=0.0, is_valid=False)

    def _post_process_reward(self, trajectory_id: str, attempt_id: int, reward: Reward) -> Trajectory | None:
        """Finalize reward-related trajectory state and persist it."""
        trajs = self.trajectory_store.get([trajectory_id], attempt_id=attempt_id)
        if not trajs.get(trajectory_id):
            logger.warning("[%s] Trajectory not found in store", trajectory_id)
            return None
        traj = trajs[trajectory_id][-1]
        traj.reward = reward.final_reward
        # An unjudged reward falls back to the sign of the reward; explicit verdicts win.
        traj.is_correct = (
            reward.is_correct if reward.is_correct is not None else reward.final_reward > 0
        )
        traj.status = TrajectoryStatus.COMPLETED
        if not reward.is_valid:
            traj.masked_out = True
        if reward.extra_info:
            traj.metadata.update(reward.extra_info)
        if reward.completion_rewards:
            all_triplets = [trip for seg in traj.segments for trip in seg.triplets]
            if len(reward.completion_rewards) != len(all_triplets):
                raise ValueError(
                    f"[{trajectory_id}] completion_rewards length ({len(reward.completion_rewards)}) "
                    f"does not match triplet count ({len(all_triplets)})"
                )
            for triplet, completion_reward in zip(all_triplets, reward.completion_rewards):
                triplet.reward = completion_reward
        n = len(traj.rollout_log_probs)
        if n > 0:
            token_rewards = [0.0] * n
            token_rewards[-1] = reward.final_reward
            traj.token_rewards = token_rewards
        self.trajectory_store.update(trajectory_id, traj)
        logger.debug("[%s] Stored reward=%s", trajectory_id, reward.final_reward)
        if dump_path := self.config.agentflow.dump_trajectory_path:
            self._dump_trajectory(traj, dump_path)
        return traj

    def _dump_trajectory(self, traj: Trajectory, base_dir: str) -> None:
        """Save trajectory as JSON to <base_dir>/<run_timestamp>/ for offline analysis."""
        try:
            dump_dir = os.path.join(base_dir, self._run_timestamp)
            os.makedirs(dump_dir, exist_ok=True)
            data = traj.model_dump()
            fname = f"{traj.trajectory_id}_a{traj.attempt_id}.json"
            path = os.path.join(dump_dir, fname)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            logger.info("[%s] Trajectory dumped to %s", traj.trajectory_id, path)
        except Exception as exc:
            logger.warning("[%s] Failed to dump trajectory: %s", traj.trajectory_id, exc)

    async def _emit_terminal_trajectory(self, trajectory: Trajectory) -> None:
        """Push a terminal trajectory to the queue and remove it from the store.
        Eval trajectories share this queue; RolloutSampler splits them out by is_eval.
        """
        write_task = asyncio.create_task(
            asyncio.to_thread(self.traj_queue.add, trajectory.model_copy(deep=True))
        )
        self._inflight_queue_write_futures.add(write_task)
        write_task.add_done_callback(self._inflight_queue_write_futures.discard)
        try:
            await write_task
        finally:
            # Delete unconditionally: even if the queue write fails the trajectory
            # should not remain in the store (the write error is already logged by the caller).
            self.trajectory_store.delete([trajectory.trajectory_id])

    async def _reemit_existing(self, traj: Trajectory) -> Reward:
        """Re-emit an already-terminal trajectory (partial rollout resume)."""
        await self._release_sandbox(traj.trajectory_id)
        await self._emit_terminal_trajectory(traj)
        return Reward(final_reward=traj.reward, is_valid=traj.status is not TrajectoryStatus.FAILED)

    # -------------------------------------------------------------------------
    # Router session helpers
    # -------------------------------------------------------------------------

    async def _router_session_request(
        self, method: str, path: str, trajectory_id: str, attempt_id: int
    ) -> None:
        try:
            async with self._control_sem:
                resp = await self._control_client.request(
                    method, f"{self.router_url}{path}/{trajectory_id}/{attempt_id}"
                )
            if resp.status_code not in (status.HTTP_200_OK, status.HTTP_404_NOT_FOUND):
                logger.warning(
                    "[%s#%d] Unexpected status on router session %s %s: %s",
                    trajectory_id, attempt_id, method, path, resp.status_code,
                )
        except Exception as exc:
            logger.warning(
                "[%s#%d] Failed router session %s %s: %s",
                trajectory_id, attempt_id, method, path, exc,
            )
