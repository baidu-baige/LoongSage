"""Unit tests for AgentFlow.generate() and _run_trajectory()."""

import asyncio
import tempfile
import os
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch, call

import httpx
import pytest
from omegaconf import OmegaConf

from coda.agentflow.agent_flow import AgentFlow
from coda.agentflow.trajectory_store import Trajectory, TrajectoryStatus, Segment, Triplet, TrajectoryStore
from coda.agentflow.utils import CONTEXT_LENGTH_EXCEEDED
from coda.reward.reward import Reward


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_config(rollout_mode: str = "single_turn", retry_limit: int = 1) -> MagicMock:
    """Return a minimal mock config accepted by AgentFlow.__init__."""
    cfg = MagicMock()
    cfg.rollout.retry_limit = retry_limit
    cfg.rollout.partial = False
    cfg.rollout.mask_offpolicy_in_partial_rollout = False
    cfg.rollout.eval.temperature = None
    cfg.agentflow.router.accumulate_reasoning = False
    cfg.agentflow.router.ip = "127.0.0.1"
    cfg.agentflow.router.port = 9999
    cfg.agentflow.router.proxy_timeout_seconds = 30
    cfg.agentflow.router.max_connections = 100
    cfg.agentflow.router.abort_timeout_seconds = 600
    cfg.agentflow.agent.name = "dummy_agent" if rollout_mode == "multi_turn" else ""
    cfg.agentflow.tokenizer = MagicMock()
    cfg.agentflow.dump_trajectory_path = ""
    cfg.agentflow.sandbox = MagicMock()
    cfg.hf_model_path = "/fake/model"
    cfg.trainer.use_rollout_routing_replay = False
    cfg.trainer.temperature = 1.0

    # data_sources[0] must expose real values for .get()/getattr access.
    ds = OmegaConf.create({
        "agent": {"name": "dummy_agent" if rollout_mode == "multi_turn" else ""},
        "reward": {},
        "max_response_len_per_trajectory": 512,
        "completion_params": {},
        "num_trajectories_per_prompt": 8,
    })
    cfg.data_sources = [ds]
    return cfg


def _make_trajectory(tid: str, status: TrajectoryStatus = TrajectoryStatus.PENDING) -> Trajectory:
    return Trajectory(
        trajectory_id=tid,
        prompt=[{"role": "user", "content": "hello"}],
        label="correct",
        attempt_id=0,
        status=status,
    )


def _make_agent_flow(rollout_mode: str = "single_turn", retry_limit: int = 1,
                     reward_fn=None) -> AgentFlow:
    """Construct an AgentFlow with all heavy side-effects patched out."""
    cfg = _make_config(rollout_mode=rollout_mode, retry_limit=retry_limit)

    with (
        patch("coda.agentflow.agent_flow.create_tokenizer_manager") as mock_tok,
        patch("coda.agentflow.agent_flow.create_reward_fn", return_value=reward_fn),
        patch("coda.agentflow.agent_flow.get_agent_class", return_value=MagicMock()),
        patch("coda.agentflow.router.router.Router") as mock_router_cls,
    ):
        mock_tok_inst = MagicMock()
        mock_tok_inst.tokenizer.__class__.__name__ = "FakeTok"
        mock_tok_inst.mode = "sync"
        mock_tok_inst.num_workers = 1
        mock_tok.return_value = mock_tok_inst

        mock_router_cls.return_value = MagicMock()

        af = AgentFlow(cfg)
        af.traj_queue = MagicMock()

    # Control-plane requests (abort / session / wait) go through the shared
    # `_control_client`; replace it with an AsyncMock so those assertions are
    # transparent. The generation hot path builds a fresh per-request client
    # inline (see `_patch_generation_client`), so it is patched per-test.
    af._control_client = AsyncMock()
    return af


@contextmanager
def _patch_generation_client(post=None):
    """Patch the per-request ``httpx.AsyncClient`` created inline by
    ``_execute_single_turn`` for the generation call.

    Yields the mock client so tests can set/assert on ``.post``. ``.aclose`` is
    an AsyncMock so the ``finally: await client.aclose()`` path works.
    """
    client = AsyncMock()
    client.aclose = AsyncMock()
    if post is not None:
        client.post = post
    with patch("coda.agentflow.agent_flow.httpx.AsyncClient", return_value=client):
        yield client


# ---------------------------------------------------------------------------
# __init__: invalid rollout_mode
# ---------------------------------------------------------------------------

class TestInit:
    def test_missing_agent_name_in_multi_turn_raises(self):
        """multi_turn mode without agent.name should raise ValueError."""
        # Force get_rollout_mode to return multi_turn (via ds.get("agent",{}).get("name"))
        # but leave ds.agent.name empty — impl raises when it re-checks agent.name.
        cfg = _make_config(rollout_mode="single_turn")
        # Manually construct a ds where the dict path returns a truthy name but attr is empty.
        # Simpler path: assert ValueError propagates from _init_per_datasource by patching
        # get_rollout_mode to return multi_turn.
        with (
            patch("coda.agentflow.agent_flow.create_tokenizer_manager") as mock_tok,
            patch("coda.agentflow.agent_flow.create_reward_fn", return_value=None),
            patch("coda.agentflow.agent_flow.get_agent_class", return_value=MagicMock()),
            patch("coda.agentflow.agent_flow.get_rollout_mode", return_value="multi_turn"),
            patch("coda.agentflow.router.router.Router"),
        ):
            mock_tok_inst = MagicMock()
            mock_tok_inst.tokenizer.__class__.__name__ = "FakeTok"
            mock_tok_inst.mode = "sync"
            mock_tok_inst.num_workers = 1
            mock_tok.return_value = mock_tok_inst
            with pytest.raises(ValueError, match="agent.name is required"):
                AgentFlow(cfg)

    def test_multi_turn_initializes_agent_class(self):
        """multi_turn mode should call get_agent_class and store it per data source."""
        af = _make_agent_flow(rollout_mode="multi_turn")
        assert af.ds_agent_classes[0] is not None

    def test_reward_fn_stored_when_configured(self):
        """If create_reward_fn returns a non-None value, it should be stored per data source."""
        fake_fn = MagicMock()
        af = _make_agent_flow(reward_fn=fake_fn)
        assert af.ds_reward_fns[0] is fake_fn


# ---------------------------------------------------------------------------
# generate(): routing by status
# ---------------------------------------------------------------------------

class TestGenerateRouting:
    """generate() should dispatch tasks correctly based on trajectory status."""

    @pytest.mark.asyncio
    async def test_pending_trajectory_calls_run_trajectory(self):
        af = _make_agent_flow()
        traj = _make_trajectory("t1", TrajectoryStatus.PENDING)
        expected_reward = Reward(final_reward=1.0, is_valid=True)

        with patch.object(af, "_run_trajectory", new_callable=AsyncMock, return_value=expected_reward):
            rewards = await af.generate([traj])

        assert len(rewards) == 1
        assert rewards[0].final_reward == 1.0

    @pytest.mark.asyncio
    async def test_pending_already_in_store_skips_add(self):
        """If trajectory is already registered in the store, generate() should not re-add it."""
        af = _make_agent_flow()
        traj = _make_trajectory("t1", TrajectoryStatus.PENDING)
        af.trajectory_store.add(traj.trajectory_id, traj)

        with patch.object(af.trajectory_store, "add") as mock_add:
            with patch.object(af, "_run_trajectory", new_callable=AsyncMock,
                              return_value=Reward(final_reward=0.5)):
                await af.generate([traj])
        mock_add.assert_not_called()

    @pytest.mark.asyncio
    async def test_completed_trajectory_calls_reemit(self):
        af = _make_agent_flow()
        traj = _make_trajectory("t1", TrajectoryStatus.COMPLETED)
        traj.reward = 0.8
        expected_reward = Reward(final_reward=0.8, is_valid=True)

        with patch.object(af, "_reemit_existing", new_callable=AsyncMock, return_value=expected_reward):
            rewards = await af.generate([traj])

        assert rewards[0].final_reward == 0.8

    @pytest.mark.asyncio
    async def test_failed_trajectory_calls_reemit(self):
        af = _make_agent_flow()
        traj = _make_trajectory("t1", TrajectoryStatus.FAILED)
        traj.reward = 0.0

        with patch.object(af, "_reemit_existing", new_callable=AsyncMock,
                          return_value=Reward(final_reward=0.0, is_valid=False)):
            rewards = await af.generate([traj])

        assert rewards[0].is_valid is False

    @pytest.mark.asyncio
    async def test_generating_status_raises(self):
        af = _make_agent_flow()
        traj = _make_trajectory("t1", TrajectoryStatus.GENERATING)

        with pytest.raises(ValueError, match="unsupported input status"):
            await af.generate([traj])

    @pytest.mark.asyncio
    async def test_task_exception_returns_zero_reward(self):
        """If _run_trajectory raises, generate() should return Reward(0, invalid)."""
        af = _make_agent_flow()
        traj = _make_trajectory("t1")

        async def _boom(_traj):
            raise RuntimeError("boom")

        with patch.object(af, "_run_trajectory", side_effect=_boom):
            rewards = await af.generate([traj])

        assert rewards[0].final_reward == 0.0
        assert rewards[0].is_valid is False


# ---------------------------------------------------------------------------
# generate(): multiple trajectories
# ---------------------------------------------------------------------------

class TestGenerateMultiple:
    @pytest.mark.asyncio
    async def test_rewards_order_preserved(self):
        """Rewards must be returned in the same order as input trajectories."""
        af = _make_agent_flow()
        trajs = [_make_trajectory(f"t{i}") for i in range(3)]
        rewards_map = {
            "t0": Reward(final_reward=0.1),
            "t1": Reward(final_reward=0.5),
            "t2": Reward(final_reward=0.9),
        }

        async def _fake_run(traj: Trajectory) -> Reward:
            return rewards_map[traj.trajectory_id]

        with patch.object(af, "_run_trajectory", side_effect=_fake_run):
            rewards = await af.generate(trajs)

        assert [r.final_reward for r in rewards] == [0.1, 0.5, 0.9]

    @pytest.mark.asyncio
    async def test_empty_trajectories_returns_empty(self):
        af = _make_agent_flow()
        rewards = await af.generate([])
        assert rewards == []


# ---------------------------------------------------------------------------
# _run_trajectory()
# ---------------------------------------------------------------------------

class TestRunTrajectory:
    @pytest.mark.asyncio
    async def test_receives_trajectory_directly(self):
        """_run_trajectory must accept a Trajectory object (not index+list)."""
        af = _make_agent_flow()
        af.ds_configs[0] = {"agent": {}}
        traj = _make_trajectory("t1")

        expected = Reward(final_reward=0.7, is_valid=True)
        with patch.object(af, "_execute_single_turn", new_callable=AsyncMock, return_value=expected):
            with patch.object(af, "_post_process_reward", return_value=traj):
                with patch.object(af, "_emit_terminal_trajectory", new_callable=AsyncMock):
                    with patch.object(af, "_router_session_request", new_callable=AsyncMock):
                        af.trajectory_store.add(traj.trajectory_id, traj)
                        reward = await af._run_trajectory(traj)

        assert reward.final_reward == 0.7

    @pytest.mark.asyncio
    async def test_multi_turn_dispatches_to_execute_multi_turn(self):
        """In multi_turn mode, _run_trajectory calls _execute_multi_turn."""
        af = _make_agent_flow(rollout_mode="multi_turn")
        af.ds_configs[0] = {"agent": {"name": "dummy_agent"}}
        traj = _make_trajectory("t1")

        expected = Reward(final_reward=0.5, is_valid=True)
        with patch.object(af, "_execute_multi_turn", new_callable=AsyncMock, return_value=expected):
            with patch.object(af, "_post_process_reward", return_value=traj):
                with patch.object(af, "_emit_terminal_trajectory", new_callable=AsyncMock):
                    with patch.object(af, "_router_session_request", new_callable=AsyncMock):
                        af.trajectory_store.add(traj.trajectory_id, traj)
                        reward = await af._run_trajectory(traj)

        assert reward.final_reward == 0.5

    @pytest.mark.asyncio
    async def test_terminal_traj_none_skips_emit(self):
        """When _post_process_reward returns None, _emit_terminal_trajectory must NOT be called."""
        af = _make_agent_flow()
        af.ds_configs[0] = {"agent": {}}
        traj = _make_trajectory("t1")

        with patch.object(af, "_execute_single_turn", new_callable=AsyncMock,
                          return_value=Reward(final_reward=0.3)):
            with patch.object(af, "_post_process_reward", return_value=None):
                with patch.object(af, "_emit_terminal_trajectory", new_callable=AsyncMock) as mock_emit:
                    with patch.object(af, "_router_session_request", new_callable=AsyncMock):
                        af.trajectory_store.add(traj.trajectory_id, traj)
                        await af._run_trajectory(traj)

        mock_emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """_run_trajectory should retry up to retry_limit times on exception."""
        af = _make_agent_flow(retry_limit=3)
        af.ds_configs[0] = {"agent": {}}
        traj = _make_trajectory("t1")
        call_count = 0

        async def _flaky(_traj, _attempt_id):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("transient error")
            return Reward(final_reward=1.0, is_valid=True)

        with patch.object(af, "_execute_single_turn", side_effect=_flaky):
            with patch.object(af, "_post_process_reward", return_value=traj):
                with patch.object(af, "_emit_terminal_trajectory", new_callable=AsyncMock):
                    with patch.object(af, "_router_session_request", new_callable=AsyncMock) as mock_session:
                        af.trajectory_store.add(traj.trajectory_id, traj)
                        reward = await af._run_trajectory(traj)

        assert reward.final_reward == 1.0
        assert call_count == 3
        assert mock_session.await_args_list == [
            call("POST", "/abort_session", "t1", 0),
            call("POST", "/abort_session", "t1", 1),
            call("DELETE", "/release_session", "t1", 2),
        ]

    @pytest.mark.asyncio
    async def test_all_retries_exhausted_returns_zero(self):
        """When every attempt fails, _run_trajectory returns Reward(0, invalid)."""
        af = _make_agent_flow(retry_limit=2)
        af.ds_configs[0] = {"agent": {}}
        traj = _make_trajectory("t1")

        async def _always_fail(_traj, _attempt_id):
            raise ValueError("always fails")

        with patch.object(af, "_execute_single_turn", side_effect=_always_fail):
            with patch.object(af, "_emit_terminal_trajectory", new_callable=AsyncMock):
                with patch.object(af, "_router_session_request", new_callable=AsyncMock) as mock_session:
                    af.trajectory_store.add(traj.trajectory_id, traj)
                    reward = await af._run_trajectory(traj)

        assert reward.final_reward == 0.0
        assert reward.is_valid is False
        assert mock_session.await_args_list == [
            call("POST", "/abort_session", "t1", 0),
            call("POST", "/abort_session", "t1", 1),
        ]

    def test_prepare_attempt_template_masks_partial_resume_tokens(self):
        """Partial rollout resume should exclude old generated tokens when configured."""
        af = AgentFlow.__new__(AgentFlow)
        af.partial_rollout_enabled = True
        af.mask_offpolicy_in_partial_rollout = True
        traj = _make_trajectory("t1", TrajectoryStatus.PENDING)
        traj.loss_masks = [1, 1, 1, 1]
        traj.rollout_log_probs = [-0.1, -0.2, 0.0, -0.3]
        traj.rollout_weight_versions = [0, 0, -1, 0]

        attempt = af._prepare_attempt_template(traj)

        assert attempt.loss_masks == [0, 0, 0, 0]
        assert len(attempt.loss_masks) == len(attempt.rollout_log_probs)
        assert traj.loss_masks == [1, 1, 1, 1]

    def test_prepare_attempt_template_rejects_misaligned_partial_masks(self):
        """Partial rollout mask rewrite requires response-space arrays to be aligned."""
        af = AgentFlow.__new__(AgentFlow)
        af.partial_rollout_enabled = True
        af.mask_offpolicy_in_partial_rollout = True
        traj = _make_trajectory("t1", TrajectoryStatus.PENDING)
        traj.loss_masks = [1]
        traj.rollout_log_probs = [-0.1, -0.2]

        with pytest.raises(ValueError, match="loss_masks length"):
            af._prepare_attempt_template(traj)

    def test_prepare_attempt_template_rejects_misaligned_weight_versions(self):
        """Misaligned rollout_weight_versions should raise ValueError."""
        af = AgentFlow.__new__(AgentFlow)
        af.partial_rollout_enabled = True
        af.mask_offpolicy_in_partial_rollout = True
        traj = _make_trajectory("t1", TrajectoryStatus.PENDING)
        traj.loss_masks = [1, 1]
        traj.rollout_log_probs = [-0.1, -0.2]
        traj.rollout_weight_versions = [0]  # length mismatch

        with pytest.raises(ValueError, match="rollout_weight_versions length"):
            af._prepare_attempt_template(traj)

    def test_prepare_attempt_template_keeps_masks_when_switch_disabled(self):
        """Default behavior should not change existing partial-rollout masks."""
        af = AgentFlow.__new__(AgentFlow)
        af.partial_rollout_enabled = True
        af.mask_offpolicy_in_partial_rollout = False
        traj = _make_trajectory("t1", TrajectoryStatus.ABORTED)
        traj.loss_masks = [1, 1, 1]

        attempt = af._prepare_attempt_template(traj)

        assert attempt.loss_masks == [1, 1, 1]


# ---------------------------------------------------------------------------
# abort()
# ---------------------------------------------------------------------------

class TestAbort:
    @pytest.mark.asyncio
    async def test_abort_all_calls_abort_all_workers(self):
        """abort() with no args should abort workers, then wait for Router requests."""
        af = _make_agent_flow()
        traj = _make_trajectory("t1")
        af.trajectory_store.add(traj.trajectory_id, traj)

        af._control_client.post = AsyncMock(return_value=MagicMock(status_code=200, json=lambda: {}))
        await af.abort()

        af._control_client.post.assert_has_awaits([
            call(f"{af.router_url}/abort_all_workers"),
            call(f"{af.router_url}/wait_inflight_requests_complete"),
        ])

    @pytest.mark.asyncio
    async def test_abort_all_marks_generating_as_aborted(self):
        """abort() should mark GENERATING trajectories as ABORTED."""
        af = _make_agent_flow()
        traj = _make_trajectory("t1", TrajectoryStatus.GENERATING)
        af.trajectory_store.add(traj.trajectory_id, traj)

        af._control_client.post = AsyncMock(return_value=MagicMock(status_code=200, json=lambda: {}))
        with patch.object(af, "_emit_terminal_trajectory", new_callable=AsyncMock) as mock_emit:
            await af.abort()

        # _emit_terminal_trajectory is mocked so the store is not deleted, but the
        # trajectory must have been marked ABORTED and passed to emit.
        mock_emit.assert_called_once()
        emitted: Trajectory = mock_emit.call_args[0][0]
        assert emitted.status == TrajectoryStatus.ABORTED

    @pytest.mark.asyncio
    async def test_abort_all_skips_completed(self):
        """abort() should not re-emit already COMPLETED trajectories."""
        af = _make_agent_flow()
        traj = _make_trajectory("t1", TrajectoryStatus.COMPLETED)
        af.trajectory_store.add(traj.trajectory_id, traj)

        af._control_client.post = AsyncMock(return_value=MagicMock(status_code=200, json=lambda: {}))
        with patch.object(af, "_emit_terminal_trajectory", new_callable=AsyncMock) as mock_emit:
            await af.abort()

        mock_emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_abort_all_workers_exception_is_raised(self):
        """abort() should fail fast if the router abort-all call fails."""
        af = _make_agent_flow()
        af._control_client.post = AsyncMock(side_effect=Exception("network error"))

        with pytest.raises(Exception, match="network error"):
            await af.abort()

    @pytest.mark.asyncio
    async def test_abort_all_workers_http_error_is_raised(self):
        """abort() should fail fast if the router reports abort-all failure."""
        af = _make_agent_flow()
        request = httpx.Request("POST", f"{af.router_url}/abort_all_workers")
        response = httpx.Response(
            status_code=503,
            json={"status": "failed"},
            request=request,
        )
        af._control_client.post = AsyncMock(return_value=response)

        with pytest.raises(httpx.HTTPStatusError):
            await af.abort()

    @pytest.mark.asyncio
    async def test_cancels_active_tasks(self):
        """abort() must cancel every task registered in _active_tasks."""
        af = _make_agent_flow()
        af._control_client.post = AsyncMock(return_value=MagicMock(status_code=200, json=lambda: {}))

        async def _never_ending():
            await asyncio.sleep(3600)

        task = asyncio.create_task(_never_ending())
        af._active_tasks.add(task)

        await af.abort()

        assert task.cancelled()

    @pytest.mark.asyncio
    async def test_waits_for_inflight_queue_write_futures(self):
        """abort() must await any in-flight trajectory-queue write futures before returning."""
        af = _make_agent_flow()
        af._control_client.post = AsyncMock(return_value=MagicMock(status_code=200, json=lambda: {}))

        write_done = False

        async def _slow_write():
            nonlocal write_done
            await asyncio.sleep(0.01)
            write_done = True

        fut = asyncio.ensure_future(_slow_write())
        af._inflight_queue_write_futures.add(fut)

        await af.abort()

        assert write_done is True

    @pytest.mark.asyncio
    async def test_calls_wait_inflight_requests_complete(self):
        """abort() step 4 must call the router's wait_inflight_requests_complete endpoint."""
        af = _make_agent_flow()
        af._control_client.post = AsyncMock(return_value=MagicMock(status_code=200, json=lambda: {}))

        await af.abort()

        posted_urls = [c.args[0] for c in af._control_client.post.await_args_list]
        assert f"{af.router_url}/wait_inflight_requests_complete" in posted_urls

    @pytest.mark.asyncio
    async def test_wait_inflight_requests_complete_failure_is_raised(self):
        """A failure waiting for inflight Router requests must propagate."""
        af = _make_agent_flow()

        async def _post(url, *args, **kwargs):
            if url.endswith("/abort_all_workers"):
                return MagicMock(status_code=200, json=lambda: {})
            raise RuntimeError("router unreachable")

        af._control_client.post = AsyncMock(side_effect=_post)

        with pytest.raises(RuntimeError, match="router unreachable"):
            await af.abort()

    @pytest.mark.asyncio
    async def test_skips_already_aborted_trajectories(self):
        """abort() should not re-emit trajectories already marked ABORTED."""
        af = _make_agent_flow()
        traj = _make_trajectory("t1", TrajectoryStatus.ABORTED)
        af.trajectory_store.add(traj.trajectory_id, traj)

        af._control_client.post = AsyncMock(return_value=MagicMock(status_code=200, json=lambda: {}))
        with patch.object(af, "_emit_terminal_trajectory", new_callable=AsyncMock) as mock_emit:
            await af.abort()

        mock_emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_marks_multiple_pending_trajectories_aborted(self):
        """abort() must mark every non-terminal trajectory in the store as ABORTED."""
        af = _make_agent_flow()
        traj1 = _make_trajectory("t1", TrajectoryStatus.GENERATING)
        traj2 = _make_trajectory("t2", TrajectoryStatus.PENDING)
        af.trajectory_store.add(traj1.trajectory_id, traj1)
        af.trajectory_store.add(traj2.trajectory_id, traj2)

        af._control_client.post = AsyncMock(return_value=MagicMock(status_code=200, json=lambda: {}))
        with patch.object(af, "_emit_terminal_trajectory", new_callable=AsyncMock) as mock_emit:
            await af.abort()

        assert mock_emit.await_count == 2
        emitted_ids = {c.args[0].trajectory_id for c in mock_emit.await_args_list}
        assert emitted_ids == {"t1", "t2"}


# ---------------------------------------------------------------------------
# clear()
# ---------------------------------------------------------------------------

class TestClear:
    @pytest.mark.asyncio
    async def test_clear_empties_store(self):
        af = _make_agent_flow()
        traj = _make_trajectory("t1")
        af.trajectory_store.add(traj.trajectory_id, traj)

        with patch.object(af._resources, "aclose", new_callable=AsyncMock):
            await af.clear()

        assert af.trajectory_store.trajectory_data == {}


# ---------------------------------------------------------------------------
# _post_process_reward()
# ---------------------------------------------------------------------------

class TestPostProcessReward:
    def _setup(self, rollout_log_probs=None):
        af = _make_agent_flow()
        traj = _make_trajectory("t1")
        if rollout_log_probs is not None:
            traj.rollout_log_probs = rollout_log_probs
        af.trajectory_store.add(traj.trajectory_id, traj)
        return af, traj

    def test_sets_reward_and_completed(self):
        af, traj = self._setup()
        reward = Reward(final_reward=0.9, is_valid=True)
        result = af._post_process_reward(traj.trajectory_id, 0, reward)
        assert result is not None
        assert result.reward == 0.9
        assert result.status == TrajectoryStatus.COMPLETED

    def test_invalid_reward_sets_masked_out(self):
        af, traj = self._setup()
        reward = Reward(final_reward=0.0, is_valid=False)
        result = af._post_process_reward(traj.trajectory_id, 0, reward)
        assert result.masked_out is True

    @pytest.mark.parametrize("final_reward,expected", [(1.0, True), (0.0, False), (-1.0, False)])
    def test_is_correct_falls_back_to_reward_sign(self, final_reward, expected):
        """An unjudged reward derives correctness from the sign of final_reward."""
        af, traj = self._setup()
        result = af._post_process_reward(traj.trajectory_id, 0, Reward(final_reward=final_reward))
        assert result.is_correct is expected

    @pytest.mark.parametrize("final_reward,is_correct", [(-1.0, True), (5.0, False)])
    def test_explicit_is_correct_overrides_reward_sign(self, final_reward, is_correct):
        """Shaping can make the sign misleading, so an explicit verdict must win."""
        af, traj = self._setup()
        reward = Reward(final_reward=final_reward, is_correct=is_correct)
        result = af._post_process_reward(traj.trajectory_id, 0, reward)
        assert result.is_correct is is_correct

    def test_extra_info_merged_into_metadata(self):
        af, traj = self._setup()
        reward = Reward(final_reward=1.0, is_valid=True, extra_info={"key": "val"})
        result = af._post_process_reward(traj.trajectory_id, 0, reward)
        assert result.metadata.get("key") == "val"

    def test_token_rewards_set_from_log_probs(self):
        af, traj = self._setup(rollout_log_probs=[-0.1, -0.2, -0.3])
        reward = Reward(final_reward=0.5, is_valid=True)
        result = af._post_process_reward(traj.trajectory_id, 0, reward)
        assert len(result.token_rewards) == 3
        assert result.token_rewards[-1] == 0.5
        assert result.token_rewards[0] == 0.0

    def test_completion_rewards_assigned_to_triplets(self):
        af, traj = self._setup()
        seg = Segment(triplets=[
            Triplet(token_start=0, token_end=2, logprob_start=0, logprob_end=2),
            Triplet(token_start=0, token_end=4, logprob_start=2, logprob_end=4),
        ])
        traj.segments = [seg]
        af.trajectory_store.update(traj.trajectory_id, traj)
        reward = Reward(final_reward=1.0, is_valid=True, completion_rewards=[0.3, 0.7])
        result = af._post_process_reward(traj.trajectory_id, 0, reward)
        assert result.segments[0].triplets[0].reward == 0.3
        assert result.segments[0].triplets[1].reward == 0.7

    def test_completion_rewards_length_mismatch_raises(self):
        af, traj = self._setup()
        seg = Segment(triplets=[
            Triplet(token_start=0, token_end=2, logprob_start=0, logprob_end=2),
        ])
        traj.segments = [seg]
        af.trajectory_store.update(traj.trajectory_id, traj)
        reward = Reward(final_reward=1.0, is_valid=True, completion_rewards=[0.3, 0.7])
        with pytest.raises(ValueError, match="completion_rewards length"):
            af._post_process_reward(traj.trajectory_id, 0, reward)

    def test_returns_none_when_not_in_store(self):
        af = _make_agent_flow()
        reward = Reward(final_reward=1.0, is_valid=True)
        result = af._post_process_reward("nonexistent", 0, reward)
        assert result is None

    def test_dumps_trajectory_when_path_configured(self):
        af, traj = self._setup()
        af.config.agentflow.dump_trajectory_path = "/tmp/fake_dump"
        reward = Reward(final_reward=1.0, is_valid=True)
        with patch.object(af, "_dump_trajectory") as mock_dump:
            af._post_process_reward(traj.trajectory_id, 0, reward)
        mock_dump.assert_called_once()


# ---------------------------------------------------------------------------
# _dump_trajectory()
# ---------------------------------------------------------------------------

class TestDumpTrajectory:
    def test_writes_json_file(self):
        af = _make_agent_flow()
        traj = _make_trajectory("t1")
        with tempfile.TemporaryDirectory() as tmpdir:
            af._dump_trajectory(traj, tmpdir)
            files = os.listdir(os.path.join(tmpdir, af._run_timestamp))
            assert any("t1" in f for f in files)

    def test_dump_failure_is_logged_not_raised(self, caplog):
        """A dump failure must never abort the rollout, but must leave a trace."""
        import logging

        af = _make_agent_flow()
        traj = _make_trajectory("t1")

        with caplog.at_level(logging.WARNING, logger="coda.agentflow.agent_flow"):
            af._dump_trajectory(traj, "/nonexistent/\x00/path")

        assert caplog.records, "a failed dump must be logged"


# ---------------------------------------------------------------------------
# _emit_terminal_trajectory()
# ---------------------------------------------------------------------------

class TestEmitTerminalTrajectory:
    @pytest.mark.asyncio
    async def test_calls_queue_add_and_deletes_from_store(self):
        af = _make_agent_flow()
        traj = _make_trajectory("t1")
        af.trajectory_store.add(traj.trajectory_id, traj)
        af.traj_queue.add = MagicMock()

        await af._emit_terminal_trajectory(traj)

        af.traj_queue.add.assert_called_once()
        assert "t1" not in af.trajectory_store.trajectory_data


# ---------------------------------------------------------------------------
# _reemit_existing()
# ---------------------------------------------------------------------------

class TestReemitExisting:
    @pytest.mark.asyncio
    async def test_returns_reward_from_trajectory(self):
        af = _make_agent_flow()
        traj = _make_trajectory("t1", TrajectoryStatus.COMPLETED)
        traj.reward = 0.6

        with patch.object(af, "_emit_terminal_trajectory", new_callable=AsyncMock):
            reward = await af._reemit_existing(traj)

        assert reward.final_reward == 0.6
        assert reward.is_valid is True

    @pytest.mark.asyncio
    async def test_failed_status_is_not_valid(self):
        af = _make_agent_flow()
        traj = _make_trajectory("t1", TrajectoryStatus.FAILED)
        traj.reward = 0.0

        with patch.object(af, "_emit_terminal_trajectory", new_callable=AsyncMock):
            reward = await af._reemit_existing(traj)

        assert reward.is_valid is False

    @pytest.mark.asyncio
    async def test_releases_pooled_sandbox(self):
        """Re-emitting a terminal trajectory must release any pooled sandbox left by abort()."""
        af = _make_agent_flow()
        traj = _make_trajectory("t1", TrajectoryStatus.COMPLETED)
        traj.reward = 1.0
        fake_sandbox = MagicMock()
        fake_sandbox.delete = MagicMock()
        af._sandbox_pool[traj.trajectory_id] = fake_sandbox

        with patch.object(af, "_emit_terminal_trajectory", new_callable=AsyncMock):
            await af._reemit_existing(traj)

        fake_sandbox.delete.assert_called_once()
        assert traj.trajectory_id not in af._sandbox_pool


# ---------------------------------------------------------------------------
# _router_session_request()
# ---------------------------------------------------------------------------

class TestRouterSessionRequest:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("status_code", [200, 404])
    async def test_expected_status_codes_log_no_warning(self, caplog, status_code):
        """200 and 404 are both normal for release_session (404 = already gone)."""
        import logging

        af = _make_agent_flow()
        resp = MagicMock()
        resp.status_code = status_code
        af._control_client.request = AsyncMock(return_value=resp)

        with caplog.at_level(logging.WARNING, logger="coda.agentflow.agent_flow"):
            await af._router_session_request("DELETE", "/release_session", "t1", 1)

        af._control_client.request.assert_awaited_once()
        assert caplog.records == []

    @pytest.mark.asyncio
    async def test_unexpected_status_code_logs_warning(self, caplog):
        af = _make_agent_flow()
        resp = MagicMock()
        resp.status_code = 500
        af._control_client.request = AsyncMock(return_value=resp)
        import logging
        with caplog.at_level(logging.WARNING, logger="coda.agentflow.agent_flow"):
            await af._router_session_request("DELETE", "/release_session", "t1", 1)
        assert any("Unexpected status" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_connection_error_is_logged_not_raised(self, caplog):
        """Session cleanup is best-effort; a dead router must not fail the attempt."""
        import logging

        af = _make_agent_flow()
        af._control_client.request = AsyncMock(side_effect=Exception("conn refused"))

        with caplog.at_level(logging.WARNING, logger="coda.agentflow.agent_flow"):
            await af._router_session_request("DELETE", "/release_session", "t1", 1)

        assert any("conn refused" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# _execute_single_turn(): completion_params forwarding
# ---------------------------------------------------------------------------

class TestExecuteMultiTurn:
    """_execute_multi_turn: sandbox pool reuse/release, keep_sandbox on abort, validation."""

    def _setup(self, partial_rollout_enabled: bool = False):
        af = _make_agent_flow(rollout_mode="multi_turn")
        af.partial_rollout_enabled = partial_rollout_enabled
        ds = OmegaConf.create({
            "agent": {"name": "dummy_agent"},
            "reward": {},
            "max_response_len_per_trajectory": 512,
            "completion_params": {},
        })
        af.ds_configs[0] = ds
        af.ds_reward_fns[0] = None
        traj = _make_trajectory("t1", TrajectoryStatus.PENDING)
        return af, traj

    def _agent_completing(self, af: AgentFlow, traj: Trajectory, attempt_id: int,
                           reward: Reward, trainable: bool = True):
        """Wire af.ds_agent_classes[0] so run_trajectory() stores a valid completed attempt."""
        completed = traj.model_copy(deep=True)
        completed.rollout_log_probs = [-0.1] if trainable is not None else []
        completed.segments = [Segment(trainable=trainable)] if trainable is not None else []
        af.trajectory_store.add(traj.trajectory_id, completed)

        agent_instance = MagicMock()
        agent_instance.run_trajectory = AsyncMock(return_value=reward)
        agent_instance.clear = AsyncMock()
        af.ds_agent_classes[0] = MagicMock(return_value=agent_instance)
        return agent_instance

    @pytest.mark.asyncio
    async def test_creates_new_sandbox_when_none_pooled(self):
        af, traj = self._setup()
        expected = Reward(final_reward=1.0, is_valid=True)
        self._agent_completing(af, traj, 0, expected)
        fake_sandbox = MagicMock(sandbox_id="sbx-1")
        fake_sandbox.delete = MagicMock()

        with patch("coda.agentflow.agent_flow.create_sandbox_client", return_value=fake_sandbox) as mock_create:
            reward = await af._execute_multi_turn(traj, 0)

        mock_create.assert_called_once()
        # Terminal success releases the sandbox in the finally block.
        assert traj.trajectory_id not in af._sandbox_pool
        assert reward.final_reward == 1.0

    def test_does_not_precreate_sandbox_without_prepare_hook(self):
        af, traj = self._setup()
        af.ds_reward_fns[0] = object()
        self._agent_completing(af, traj, 0, Reward(final_reward=1.0))
        fake_sandbox = MagicMock(sandbox_id=None)

        with patch(
            "coda.agentflow.agent_flow.create_sandbox_client",
            return_value=fake_sandbox,
        ):
            asyncio.run(af._execute_multi_turn(traj, 0))

        fake_sandbox.create.assert_not_called()

    def test_prepares_sandbox_when_reward_defines_prepare_hook(self):
        af, traj = self._setup()
        traj.metadata["docker_image"] = "example/image:latest"
        reward_fn = MagicMock()
        af.ds_reward_fns[0] = reward_fn
        self._agent_completing(af, traj, 0, Reward(final_reward=1.0))
        fake_sandbox = MagicMock(sandbox_id=None)

        with patch(
            "coda.agentflow.agent_flow.create_sandbox_client",
            return_value=fake_sandbox,
        ):
            asyncio.run(af._execute_multi_turn(traj, 0))

        fake_sandbox.create.assert_called_once_with(image="example/image:latest")
        reward_fn.prepare_sandbox.assert_called_once_with(
            fake_sandbox,
            metadata={"docker_image": "example/image:latest"},
        )

    @pytest.mark.asyncio
    async def test_reuses_pooled_sandbox_without_creating_new_one(self):
        af, traj = self._setup()
        expected = Reward(final_reward=1.0, is_valid=True)
        self._agent_completing(af, traj, 0, expected)
        pooled_sandbox = MagicMock(sandbox_id="sbx-pooled")
        pooled_sandbox.delete = MagicMock()
        af._sandbox_pool[traj.trajectory_id] = pooled_sandbox

        with patch("coda.agentflow.agent_flow.create_sandbox_client") as mock_create:
            await af._execute_multi_turn(traj, 0)

        mock_create.assert_not_called()
        pooled_sandbox.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_releases_sandbox_on_normal_completion(self):
        """Terminal success must delete the sandbox and drop it from the pool."""
        af, traj = self._setup()
        expected = Reward(final_reward=1.0, is_valid=True)
        self._agent_completing(af, traj, 0, expected)
        fake_sandbox = MagicMock(sandbox_id="sbx-1")
        fake_sandbox.delete = MagicMock()

        with patch("coda.agentflow.agent_flow.create_sandbox_client", return_value=fake_sandbox):
            await af._execute_multi_turn(traj, 0)

        fake_sandbox.delete.assert_called_once()
        assert traj.trajectory_id not in af._sandbox_pool

    @pytest.mark.asyncio
    async def test_releases_sandbox_on_failure(self):
        """A raised exception (e.g. missing rollout_log_probs) must still release the sandbox."""
        af, traj = self._setup()
        # trainable=None -> _agent_completing stores an attempt with empty rollout_log_probs,
        # which triggers the "completed without rollout log probs" RuntimeError.
        self._agent_completing(af, traj, 0, Reward(final_reward=1.0), trainable=None)
        fake_sandbox = MagicMock(sandbox_id="sbx-1")
        fake_sandbox.delete = MagicMock()

        with patch("coda.agentflow.agent_flow.create_sandbox_client", return_value=fake_sandbox):
            with pytest.raises(RuntimeError, match="without rollout log probs"):
                await af._execute_multi_turn(traj, 0)

        fake_sandbox.delete.assert_called_once()
        assert traj.trajectory_id not in af._sandbox_pool

    @pytest.mark.asyncio
    async def test_keeps_sandbox_on_cancel_when_partial_rollout_enabled(self):
        """CancelledError with partial_rollout_enabled=True must keep the sandbox pooled."""
        af, traj = self._setup(partial_rollout_enabled=True)
        fake_sandbox = MagicMock(sandbox_id="sbx-1")
        fake_sandbox.delete = MagicMock()
        af._sandbox_pool[traj.trajectory_id] = fake_sandbox

        agent_instance = MagicMock()
        agent_instance.run_trajectory = AsyncMock(side_effect=asyncio.CancelledError())
        agent_instance.clear = AsyncMock()
        af.ds_agent_classes[0] = MagicMock(return_value=agent_instance)

        with pytest.raises(asyncio.CancelledError):
            await af._execute_multi_turn(traj, 0)

        fake_sandbox.delete.assert_not_called()
        assert af._sandbox_pool[traj.trajectory_id] is fake_sandbox
        agent_instance.clear.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_releases_sandbox_on_cancel_when_partial_rollout_disabled(self):
        """CancelledError with partial_rollout_enabled=False must still release the sandbox."""
        af, traj = self._setup(partial_rollout_enabled=False)
        fake_sandbox = MagicMock(sandbox_id="sbx-1")
        fake_sandbox.delete = MagicMock()
        af._sandbox_pool[traj.trajectory_id] = fake_sandbox

        agent_instance = MagicMock()
        agent_instance.run_trajectory = AsyncMock(side_effect=asyncio.CancelledError())
        agent_instance.clear = AsyncMock()
        af.ds_agent_classes[0] = MagicMock(return_value=agent_instance)

        with pytest.raises(asyncio.CancelledError):
            await af._execute_multi_turn(traj, 0)

        fake_sandbox.delete.assert_called_once()
        assert traj.trajectory_id not in af._sandbox_pool

    @pytest.mark.asyncio
    async def test_missing_attempt_in_store_raises(self):
        """If the agent completes but the store has no matching attempt, raise RuntimeError."""
        af, traj = self._setup()
        agent_instance = MagicMock()
        agent_instance.run_trajectory = AsyncMock(return_value=Reward(final_reward=1.0))
        agent_instance.clear = AsyncMock()
        af.ds_agent_classes[0] = MagicMock(return_value=agent_instance)
        fake_sandbox = MagicMock(sandbox_id="sbx-1")
        fake_sandbox.delete = MagicMock()

        with patch("coda.agentflow.agent_flow.create_sandbox_client", return_value=fake_sandbox):
            with pytest.raises(RuntimeError, match="missing from trajectory store"):
                await af._execute_multi_turn(traj, 0)

    @pytest.mark.asyncio
    async def test_no_trainable_segment_raises(self):
        """A completed attempt whose segments are all non-trainable must raise RuntimeError."""
        af, traj = self._setup()
        completed = traj.model_copy(deep=True)
        completed.rollout_log_probs = [-0.1]
        completed.segments = [Segment(trainable=False)]
        af.trajectory_store.add(traj.trajectory_id, completed)

        agent_instance = MagicMock()
        agent_instance.run_trajectory = AsyncMock(return_value=Reward(final_reward=1.0))
        agent_instance.clear = AsyncMock()
        af.ds_agent_classes[0] = MagicMock(return_value=agent_instance)
        fake_sandbox = MagicMock(sandbox_id="sbx-1")
        fake_sandbox.delete = MagicMock()

        with patch("coda.agentflow.agent_flow.create_sandbox_client", return_value=fake_sandbox):
            with pytest.raises(RuntimeError, match="without trainable Segment"):
                await af._execute_multi_turn(traj, 0)

    @pytest.mark.asyncio
    async def test_agent_clear_called_even_on_exception(self):
        """agent.clear() must run in the finally block regardless of outcome."""
        af, traj = self._setup()
        agent_instance = MagicMock()
        agent_instance.run_trajectory = AsyncMock(side_effect=ValueError("boom"))
        agent_instance.clear = AsyncMock()
        af.ds_agent_classes[0] = MagicMock(return_value=agent_instance)
        fake_sandbox = MagicMock(sandbox_id="sbx-1")
        fake_sandbox.delete = MagicMock()

        with patch("coda.agentflow.agent_flow.create_sandbox_client", return_value=fake_sandbox):
            with pytest.raises(ValueError, match="boom"):
                await af._execute_multi_turn(traj, 0)

        agent_instance.clear.assert_awaited_once()


# ---------------------------------------------------------------------------
# _release_sandbox()
# ---------------------------------------------------------------------------

class TestReleaseSandbox:
    @pytest.mark.asyncio
    async def test_noop_when_no_pooled_sandbox(self):
        af = _make_agent_flow()
        # Should not raise even though no sandbox was ever pooled for this id.
        await af._release_sandbox("unknown-traj")

    @pytest.mark.asyncio
    async def test_deletes_and_drops_from_pool(self):
        af = _make_agent_flow()
        fake_sandbox = MagicMock()
        fake_sandbox.delete = MagicMock()
        af._sandbox_pool["t1"] = fake_sandbox

        await af._release_sandbox("t1")

        fake_sandbox.delete.assert_called_once()
        assert "t1" not in af._sandbox_pool

    @pytest.mark.asyncio
    async def test_delete_exception_is_logged_not_raised(self, caplog):
        import logging

        af = _make_agent_flow()
        fake_sandbox = MagicMock()
        fake_sandbox.delete = MagicMock(side_effect=Exception("delete failed"))
        af._sandbox_pool["t1"] = fake_sandbox

        with caplog.at_level(logging.WARNING, logger="coda.agentflow.agent_flow"):
            await af._release_sandbox("t1")

        assert "t1" not in af._sandbox_pool
        assert any("delete failed" in r.getMessage() for r in caplog.records)


class TestExecuteSingleTurn:
    """completion_params from data-source config must be merged into the POST body."""

    def _setup(self, completion_params=None):
        af = _make_agent_flow()
        # completion_params live per data source; construct a fresh DictConfig so
        # OmegaConf can spread it via ** when the impl builds the JSON body.
        ds = OmegaConf.create({
            "agent": {"name": ""},
            "reward": {},
            "max_response_len_per_trajectory": 512,
            "completion_params": completion_params if completion_params is not None else {},
        })
        af.ds_configs[0] = ds
        # Stub reward_fn so _execute_single_turn returns quickly
        af.ds_reward_fns[0] = MagicMock(return_value=Reward(final_reward=1.0, is_valid=True))
        # Stub trajectory store lookup
        traj = _make_trajectory("t1")
        af.trajectory_store.add(traj.trajectory_id, traj)
        return af, traj

    @pytest.mark.asyncio
    async def test_default_no_completion_params(self):
        """With empty completion_params the body contains only the standard fields."""
        af, traj = self._setup(completion_params={})
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        with _patch_generation_client() as client:
            client.post = AsyncMock(return_value=mock_resp)
            await af._execute_single_turn(traj, 0)
            body = client.post.call_args.kwargs["json"]
        assert set(body.keys()) == {"model", "messages", "temperature", "max_tokens"}
        assert body["max_tokens"] == 512

    @pytest.mark.asyncio
    async def test_completion_params_forwarded(self):
        """Extra sampling params must appear in the POST body."""
        af, traj = self._setup(completion_params={"top_p": 0.9, "top_k": 50})
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        with _patch_generation_client() as client:
            client.post = AsyncMock(return_value=mock_resp)
            await af._execute_single_turn(traj, 0)
            body = client.post.call_args.kwargs["json"]
        assert body["top_p"] == 0.9
        assert body["top_k"] == 50

    @pytest.mark.asyncio
    async def test_completion_params_override_temperature(self):
        """A temperature key inside completion_params should override the default."""
        af, traj = self._setup(completion_params={"temperature": 0.3})
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        with _patch_generation_client() as client:
            client.post = AsyncMock(return_value=mock_resp)
            await af._execute_single_turn(traj, 0)
            body = client.post.call_args.kwargs["json"]
        assert body["temperature"] == 0.3

    @pytest.mark.asyncio
    async def test_completion_params_none_treated_as_empty(self):
        """None completion_params must not raise and produce a clean body."""
        af, traj = self._setup(completion_params=None)
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        with _patch_generation_client() as client:
            client.post = AsyncMock(return_value=mock_resp)
            await af._execute_single_turn(traj, 0)
            body = client.post.call_args.kwargs["json"]
        assert "model" in body
        assert "messages" in body

    @pytest.mark.asyncio
    async def test_context_length_exceeded_uses_partial_response_for_reward(self):
        af = AgentFlow.__new__(AgentFlow)
        af.router_url = "http://router"
        af._active_tasks = set()
        af._generate_sem = asyncio.Semaphore(1)
        af.trajectory_store = TrajectoryStore()
        ds_cfg = MagicMock()
        ds_cfg.max_response_len_per_trajectory = 128
        ds_cfg.completion_params = {}
        af.ds_configs = {0: ds_cfg}
        reward_fn = MagicMock(return_value=Reward(final_reward=0.25, is_valid=True))
        af.ds_reward_fns = {0: reward_fn}
        af.config = MagicMock()
        af.config.trainer.temperature = 1.0
        af.config.agentflow.router.proxy_timeout_seconds = 30

        traj = _make_trajectory("t-overlong", TrajectoryStatus.GENERATING)
        partial = traj.model_copy(deep=True)
        partial.chat_completions = {0: [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "partial"},
        ]}
        af.trajectory_store.add(traj.trajectory_id, partial)
        response = httpx.Response(
            status_code=400,
            json={"error": {"type": CONTEXT_LENGTH_EXCEEDED, "message": "exhausted"}},
            request=httpx.Request("POST", "http://router/t-overlong/0/v1/chat/completions"),
        )
        with _patch_generation_client() as client:
            client.post = AsyncMock(return_value=response)
            reward = await af._execute_single_turn(traj, 0)

        assert reward.final_reward == 0.25
        reward_fn.assert_called_once()
        assert reward_fn.call_args.args[0][-1]["content"] == "partial"

    @pytest.mark.asyncio
    async def test_execute_single_turn_uses_active_segment_not_last_chat(self):
        """messages must come from active_segment_id's chat, even if it isn't chat_completions[-1].

        Simulates a trajectory that forked a subagent branch (appended a later chat_completions
        entry for the child) while active_segment_id remained on the parent/mainline segment.
        """
        af, traj = self._setup()
        traj.active_segment_id = 0
        traj.chat_completions = {
            0: [{"role": "user", "content": "mainline"}],
            1: [{"role": "user", "content": "subagent branch, must not be used"}],
        }
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        with _patch_generation_client() as client:
            client.post = AsyncMock(return_value=mock_resp)
            await af._execute_single_turn(traj, 0)
            body = client.post.call_args.kwargs["json"]

        assert body["messages"] == [{"role": "user", "content": "mainline"}]
