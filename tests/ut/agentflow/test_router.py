"""Unit tests for Router worker management, routing logic, and proxy behavior."""

import asyncio
import importlib.util
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import status
from omegaconf import OmegaConf

from coda.agentflow.router.router import Router
from coda.agentflow.trajectory_store import Trajectory, TrajectoryStore
from coda.agentflow.utils import build_request_id


def _has_sglang_template_detection() -> bool:
    """coda.agentflow.router.parser needs sglang.srt.parser.template_detection.

    That module only exists in the sglang build the project's Dockerfile pins
    (lmsysorg/sglang:v0.5.16); on other versions the parser middleware cannot be
    imported at all, so the middleware-backed proxy tests must skip.
    """
    try:
        return importlib.util.find_spec("sglang.srt.parser.template_detection") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


requires_parser_middleware = pytest.mark.skipif(
    not _has_sglang_template_detection(),
    reason="sglang.srt.parser.template_detection is unavailable in this sglang build",
)


def make_store(*specs: tuple[str, int]) -> TrajectoryStore:
    """Create a TrajectoryStore pre-populated with minimal trajectories."""
    store = TrajectoryStore()
    for tid, aid in specs:
        store.add(tid, Trajectory(trajectory_id=tid, prompt_id="p0", attempt_id=aid))
    return store


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

ROUTER_BASE_CONFIG = {
    "ip": "0.0.0.0",
    "port": None,
    "proxy_timeout_seconds": 600.0,
    "abort_timeout_seconds": 600.0,
    "accumulate_reasoning": False,
    "rollout_worker_load_threshold": 4,
    "max_connections": 512,
    "middleware": {},
}


def make_router(*, middleware_kwargs: dict | None = None, **kwargs) -> Router:
    """Create a bare Router without middleware for core routing tests."""
    config = dict(ROUTER_BASE_CONFIG)
    config.update(kwargs)
    return Router(config=OmegaConf.create(config), middleware_kwargs=middleware_kwargs)


def make_client(router: Router, upstream_body: dict | None = None) -> httpx.AsyncClient:
    """Create an in-process HTTP client for the Router ASGI app.

    Args:
        router: The router instance.
        upstream_body: Optional dict to inject as request.state.upstream_body.
                       If provided, wraps the ASGI app to simulate parser middleware behavior.
    """
    if upstream_body is not None:
        import json

        class InjectBodyMiddleware:
            """Middleware that injects upstream_body into request.state."""

            def __init__(self, app, body_bytes: bytes):
                self.app = app
                self.body_bytes = body_bytes

            async def __call__(self, scope, receive, send):
                if scope["type"] == "http":
                    # Ensure state dict exists
                    if "state" not in scope:
                        scope["state"] = {}
                    # Set upstream_body before any request body is read
                    scope["state"]["upstream_body"] = self.body_bytes
                return await self.app(scope, receive, send)

        body_bytes = json.dumps(upstream_body).encode("utf-8")
        app_with_middleware = InjectBodyMiddleware(router.app, body_bytes)
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app_with_middleware),
            base_url="http://testserver",
        )
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=router.app),
        base_url="http://testserver",
    )


def make_proxy_router(trajectory_store: TrajectoryStore | None = None) -> Router:
    """Create a Router configured like the session-scoped chat route expects."""
    return make_router(
        middleware={"parser": None},
        middleware_kwargs={
            "trajectory_store": trajectory_store or TrajectoryStore(),
            "tokenizer_manager": FakeTokenizerManager(),
            "accumulate_reasoning": False,
            # Keyed by Trajectory.ds_index, which defaults to 0.
            "ds_configs": {0: OmegaConf.create({"max_response_len_per_trajectory": 1024})},
        },
    )


@asynccontextmanager
async def proxy_client(router: Router, *, upstream_body: dict, **request_kwargs):
    """Yield ``(asgi_client, generate_request_mock)`` for proxy forwarding tests.

    ``Router.proxy`` builds a throwaway ``httpx.AsyncClient`` per forward, so the
    seam for tests is the constructor rather than a router attribute. The ASGI
    test client is created *before* the patch goes live, so it keeps the real
    httpx implementation and only the proxy's client is mocked.

    Args:
        router: The router under test.
        upstream_body: Body injected as ``request.state.upstream_body``.
        request_kwargs: Forwarded to the ``AsyncMock`` standing in for
            ``AsyncClient.request`` (e.g. ``return_value`` or ``side_effect``).
    """
    async with make_client(router, upstream_body=upstream_body) as client:
        request_mock = AsyncMock(**request_kwargs)
        generate_client = MagicMock()
        generate_client.request = request_mock
        generate_client.aclose = AsyncMock()
        with patch.object(httpx, "AsyncClient", return_value=generate_client):
            yield client, request_mock


# ---------------------------------------------------------------------------
# Worker management
# ---------------------------------------------------------------------------


class TestWorkerManagement:
    """Tests for add/exclude/include worker management endpoints."""

    @pytest.mark.asyncio
    async def test_add_worker(self):
        """Add worker returns 200 and worker appears in active list."""
        router = make_router()
        async with make_client(router) as client:
            result = await client.post(
                "/add_worker", json={"worker_url": "http://worker1:8888"}
            )
            workers = await client.get("/list_workers")
        assert result.status_code == status.HTTP_200_OK
        assert result.json() == {
            "status": "success",
            "worker": "http://worker1:8888",
        }
        assert workers.status_code == status.HTTP_200_OK
        assert "http://worker1:8888" in workers.json()["active_workers"]

    @pytest.mark.asyncio
    async def test_add_worker_missing_url_returns_422(self):
        """Add worker returns 422 when worker_url is missing (Pydantic validation)."""
        router = make_router()
        async with make_client(router) as client:
            result = await client.post("/add_worker", json={})

        # Pydantic validates required field, returns 422
        assert result.status_code == 422

    @pytest.mark.asyncio
    async def test_add_worker_invalid_json_returns_422(self):
        """Add worker returns 422 when the request body is invalid JSON."""
        router = make_router()
        async with make_client(router) as client:
            result = await client.post(
                "/add_worker",
                content=b"{",
                headers={"content-type": "application/json"},
            )

        # FastAPI/Pydantic returns 422 for invalid JSON that fails validation
        assert result.status_code == 422

    @pytest.mark.asyncio
    async def test_add_worker_duplicate_returns_409(self):
        """Add worker returns 409 when worker already exists, and deduplicates."""
        router = make_router()
        async with make_client(router) as client:
            first = await client.post(
                "/add_worker", json={"worker_url": "http://worker1:8888"}
            )
            second = await client.post(
                "/add_worker", json={"worker_url": "http://worker1:8888"}
            )
            workers = await client.get("/list_workers")

        assert first.status_code == status.HTTP_200_OK
        assert second.status_code == status.HTTP_409_CONFLICT
        assert workers.json()["active_workers"].count("http://worker1:8888") == 1

    @pytest.mark.asyncio
    async def test_list_workers_empty_on_new_router(self):
        """List workers returns empty lists for a freshly created router."""
        router = make_router()
        async with make_client(router) as client:
            workers = await client.get("/list_workers")

        assert workers.status_code == status.HTTP_200_OK
        body = workers.json()
        assert body["active_workers"] == []
        assert body["dead_workers"] == []

    @pytest.mark.asyncio
    async def test_exclude_worker_moves_to_dead(self):
        """Exclude worker removes it from active list and adds it to dead list."""
        router = make_router()
        async with make_client(router) as client:
            await client.post("/add_worker", json={"worker_url": "http://worker1:8888"})
            result = await client.put(
                "/exclude_worker", json={"worker_url": "http://worker1:8888"}
            )
            workers = await client.get("/list_workers")

        assert result.status_code == status.HTTP_200_OK
        assert result.json()["status"] == "success"
        body = workers.json()
        assert "http://worker1:8888" not in body["active_workers"]
        assert "http://worker1:8888" in body["dead_workers"]

    @pytest.mark.asyncio
    async def test_exclude_worker_missing_url_returns_422(self):
        """Exclude worker returns 422 when worker_url is missing (Pydantic validation)."""
        router = make_router()
        async with make_client(router) as client:
            result = await client.put("/exclude_worker", json={})

        assert result.status_code == 422

    @pytest.mark.asyncio
    async def test_exclude_nonexistent_returns_404(self):
        """Exclude a worker that was never registered returns 404."""
        router = make_router()
        async with make_client(router) as client:
            result = await client.put(
                "/exclude_worker", json={"worker_url": "http://notexist:8888"}
            )

        assert result.status_code == status.HTTP_404_NOT_FOUND
        assert result.json()["error"] == "Worker not found"

    @pytest.mark.asyncio
    async def test_exclude_already_dead_returns_400(self):
        """Exclude a worker that is already in the dead pool returns 400."""
        router = make_router()
        async with make_client(router) as client:
            await client.post("/add_worker", json={"worker_url": "http://w1:8888"})
            _ = await client.put("/exclude_worker", json={"worker_url": "http://w1:8888"})
            result = await client.put(
                "/exclude_worker", json={"worker_url": "http://w1:8888"}
            )

        assert result.status_code == status.HTTP_400_BAD_REQUEST
        assert result.json()["error"] == "Worker is already excluded"

    @pytest.mark.asyncio
    async def test_include_never_registered_returns_404(self):
        """Include a worker that was never registered returns 404 with 'never registered'."""
        router = make_router()
        async with make_client(router) as client:
            result = await client.put(
                "/include_worker", json={"worker_url": "http://notexist:8888"}
            )

        assert result.status_code == status.HTTP_404_NOT_FOUND
        assert "never registered" in result.json()["error"]

    @pytest.mark.asyncio
    async def test_include_already_active_returns_400(self):
        """Include a registered but non-excluded worker returns 400."""
        router = make_router()
        async with make_client(router) as client:
            await client.post("/add_worker", json={"worker_url": "http://w1:8888"})
            result = await client.put(
                "/include_worker", json={"worker_url": "http://w1:8888"}
            )

        assert result.status_code == status.HTTP_400_BAD_REQUEST
        assert result.json()["error"] == "Worker is already active"

    @pytest.mark.asyncio
    async def test_include_dead_worker_restores_to_active(self):
        """Include a dead worker moves it back to the active pool."""
        router = make_router()
        async with make_client(router) as client:
            await client.post("/add_worker", json={"worker_url": "http://w1:8888"})
            await client.put("/exclude_worker", json={"worker_url": "http://w1:8888"})
            result = await client.put(
                "/include_worker", json={"worker_url": "http://w1:8888"}
            )
            workers = await client.get("/list_workers")

        assert result.status_code == status.HTTP_200_OK
        body = workers.json()
        assert "http://w1:8888" in body["active_workers"]
        assert "http://w1:8888" not in body["dead_workers"]

    @pytest.mark.asyncio
    async def test_exclude_keeps_session_routing(self):
        """Excluding a worker does not proactively delete existing sticky sessions."""
        router = make_router()
        async with make_client(router) as client:
            await client.post("/add_worker", json={"worker_url": "http://w1:8888"})
            router.session_id_to_worker[build_request_id("traj-001", 0)] = "http://w1:8888"
            router.session_id_to_worker[build_request_id("traj-002", 0)] = "http://w1:8888"
            await client.put("/exclude_worker", json={"worker_url": "http://w1:8888"})

        assert router.session_id_to_worker[build_request_id("traj-001", 0)] == "http://w1:8888"
        assert router.session_id_to_worker[build_request_id("traj-002", 0)] == "http://w1:8888"

    @pytest.mark.asyncio
    async def test_exclude_preserves_other_sessions(self):
        """Excluding worker A does not affect sticky sessions bound to worker B."""
        router = make_router()
        async with make_client(router) as client:
            await client.post("/add_worker", json={"worker_url": "http://w1:8888"})
            await client.post("/add_worker", json={"worker_url": "http://w2:8888"})
            router.session_id_to_worker[build_request_id("traj-001", 0)] = "http://w1:8888"
            router.session_id_to_worker[build_request_id("traj-002", 0)] = "http://w2:8888"
            await client.put("/exclude_worker", json={"worker_url": "http://w1:8888"})

        assert router.session_id_to_worker[build_request_id("traj-001", 0)] == "http://w1:8888"
        assert router.session_id_to_worker[build_request_id("traj-002", 0)] == "http://w2:8888"

    @pytest.mark.asyncio
    async def test_release_session_removes_one_mapping(self):
        """Releasing one attempt deletes only that sticky binding and returns ids."""
        router = make_router()
        async with make_client(router) as client:
            await client.post("/add_worker", json={"worker_url": "http://w1:8888"})
            router.session_id_to_worker[build_request_id("traj-001", 0)] = "http://w1:8888"
            router.session_id_to_worker[build_request_id("traj-001", 1)] = "http://w1:8888"
            result = await client.delete("/release_session/traj-001/0")

        assert result.status_code == status.HTTP_200_OK
        assert result.json() == {
            "status": "success",
            "trajectory_id": "traj-001",
            "attempt_id": 0,
        }
        assert build_request_id("traj-001", 0) not in router.session_id_to_worker
        assert router.session_id_to_worker[build_request_id("traj-001", 1)] == "http://w1:8888"

    @pytest.mark.asyncio
    async def test_release_session_not_found_returns_404(self):
        """Releasing an unknown attempt returns 404."""
        router = make_router()
        async with make_client(router) as client:
            result = await client.delete("/release_session/traj-404/7")

        assert result.status_code == status.HTTP_404_NOT_FOUND
        body = result.json()
        assert body["error"] == "No sticky session found"
        assert body["trajectory_id"] == "traj-404"
        assert body["attempt_id"] == 7

    @pytest.mark.asyncio
    async def test_abort_session_aborts_worker_request_and_clears_session(self):
        """Aborting one attempt sends its rid to the pinned worker and clears sticky routing."""
        router = make_router()
        request_id = build_request_id("traj-001", 0)
        router.session_id_to_worker[request_id] = "http://w1:8888"
        router._control_client.post = AsyncMock(return_value=httpx.Response(200, json={}))

        async with make_client(router) as client:
            result = await client.post("/abort_session/traj-001/0")

        assert result.status_code == status.HTTP_200_OK
        assert result.json() == {
            "status": "success",
            "trajectory_id": "traj-001",
            "attempt_id": 0,
        }
        router._control_client.post.assert_awaited_once_with(
            "http://w1:8888/abort_request",
            json={"rid": request_id, "abort_all": False},
        )
        assert request_id not in router.session_id_to_worker

    @pytest.mark.asyncio
    async def test_abort_session_not_found_returns_404(self):
        """Aborting an unknown attempt returns 404 and does not contact workers."""
        router = make_router()
        router._control_client.post = AsyncMock()

        async with make_client(router) as client:
            result = await client.post("/abort_session/traj-404/7")

        assert result.status_code == status.HTTP_404_NOT_FOUND
        body = result.json()
        assert body["error"] == "No active session found"
        assert body["trajectory_id"] == "traj-404"
        assert body["attempt_id"] == 7
        router._control_client.post.assert_not_called()


class TestAbortAllWorkers:
    """Tests for aborting every active worker."""

    @pytest.mark.asyncio
    async def test_abort_all_success_clears_sessions_and_counts(self):
        """Successful abort-all clears sticky sessions and in-flight counts."""
        router = make_router()
        router.worker_request_counts["http://w1:8888"] = 2
        router.worker_request_counts["http://w2:8888"] = 1
        router.session_id_to_worker[build_request_id("traj-001", 0)] = "http://w1:8888"
        router.session_id_to_worker[build_request_id("traj-002", 0)] = "http://w2:8888"
        router._control_client.post = AsyncMock(
            side_effect=[httpx.Response(200, json={}), httpx.Response(200, json={})]
        )

        async with make_client(router) as client:
            result = await client.post("/abort_all_workers")

        assert result.status_code == status.HTTP_200_OK
        assert result.json() == {
            "status": "success",
            "workers_aborted": 2,
            "sessions_cleared": 2,
        }
        assert router.session_id_to_worker == {}
        assert router.worker_request_counts == {"http://w1:8888": 0, "http://w2:8888": 0}

    @pytest.mark.asyncio
    async def test_abort_all_http_failure_preserves_sessions_and_counts(self):
        """A non-2xx worker response fails abort-all without clearing router state."""
        router = make_router()
        router.worker_request_counts["http://w1:8888"] = 2
        router.worker_request_counts["http://w2:8888"] = 1
        router.session_id_to_worker[build_request_id("traj-001", 0)] = "http://w1:8888"
        router._control_client.post = AsyncMock(
            side_effect=[
                httpx.Response(200, json={}),
                httpx.Response(500, text="busy"),
                httpx.Response(500, text="busy"),
                httpx.Response(500, text="busy"),
            ]
        )

        async with make_client(router) as client:
            result = await client.post("/abort_all_workers")

        assert result.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        body = result.json()
        assert body["status"] == "failed"
        assert body["workers_aborted"] == 1
        assert body["sessions_cleared"] == 0
        assert body["failed_workers"] == [
            {"worker": "http://w2:8888", "status_code": 500, "error": "busy"}
        ]
        assert router.session_id_to_worker == {
            build_request_id("traj-001", 0): "http://w1:8888"
        }
        assert router.worker_request_counts == {"http://w1:8888": 2, "http://w2:8888": 1}

    @pytest.mark.asyncio
    async def test_abort_all_request_exception_preserves_sessions_and_counts(self):
        """A worker request exception fails abort-all without clearing router state."""
        router = make_router()
        router.worker_request_counts["http://w1:8888"] = 2
        router.session_id_to_worker[build_request_id("traj-001", 0)] = "http://w1:8888"
        router._control_client.post = AsyncMock(side_effect=httpx.ConnectError("unreachable"))

        async with make_client(router) as client:
            result = await client.post("/abort_all_workers")

        assert result.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        body = result.json()
        assert body["status"] == "failed"
        assert body["workers_aborted"] == 0
        assert body["sessions_cleared"] == 0
        assert body["failed_workers"] == [
            {"worker": "http://w1:8888", "error": "unreachable"}
        ]
        assert router.session_id_to_worker == {
            build_request_id("traj-001", 0): "http://w1:8888"
        }
        assert router.worker_request_counts == {"http://w1:8888": 2}


# ---------------------------------------------------------------------------
# ParserMiddleware inflight tracking
# ---------------------------------------------------------------------------


class TestWaitInflightRequestsComplete:
    """Tests for waiting on in-flight ParserMiddleware requests."""

    @pytest.mark.asyncio
    async def test_wait_inflight_requests_complete_waits_until_request_ends(self):
        """wait_inflight_requests_complete returns success after all tracked requests exit."""
        # A generous deadline: the endpoint must succeed because the request
        # finished, not because the timeout was racing it.
        router = make_router(abort_timeout_seconds=10.0)
        request_id = build_request_id("traj-001", 0)
        router.inflight_requests.add(request_id)

        async def end_request() -> None:
            await asyncio.sleep(0.01)
            await router.inflight_requests.delete(request_id)

        end_task = asyncio.create_task(end_request())
        async with make_client(router) as client:
            result = await client.post("/wait_inflight_requests_complete")
        await end_task

        assert result.status_code == status.HTTP_200_OK
        assert result.json() == {
            "status": "success",
            "completed": 1,
        }

    @pytest.mark.asyncio
    async def test_wait_inflight_requests_complete_times_out_with_unfinished_counts(self):
        """wait_inflight_requests_complete returns 504 if tracked requests stay active."""
        router = make_router(abort_timeout_seconds=0.2)
        request_id = build_request_id("traj-001", 0)
        router.inflight_requests.add(request_id)

        async with make_client(router) as client:
            result = await client.post("/wait_inflight_requests_complete")
        await router.inflight_requests.delete(request_id)

        assert result.status_code == status.HTTP_504_GATEWAY_TIMEOUT
        assert result.json()["status"] == "failed"
        assert result.json()["error"] == "wait inflight requests complete timeout"
        assert result.json()["unfinished"] == {request_id: 1}

# ---------------------------------------------------------------------------
# Routing selection logic
# ---------------------------------------------------------------------------


class TestRoutingLogic:
    """Tests for attempt-scoped sticky + least-loaded routing logic."""

    def _make_router_with_workers(self, workers: list[str]) -> Router:
        """Create a router with pre-registered workers (count=0 each)."""
        router = make_router()
        for w in workers:
            router.worker_request_counts[w] = 0
        return router

    def test_sticky_routing_pins_to_same_worker(self):
        """Sticky routing returns the pinned worker and increments its count."""
        router = self._make_router_with_workers(["http://w1:8888", "http://w2:8888"])
        router.session_id_to_worker[build_request_id("traj-001", 0)] = "http://w1:8888"

        selected = router._select_worker(build_request_id("traj-001", 0))

        assert selected == "http://w1:8888"
        assert router.worker_request_counts["http://w1:8888"] == 1
        assert router.worker_request_counts["http://w2:8888"] == 0

    def test_sticky_routing_skipped_when_worker_dead(self):
        """Sticky routing falls back to least-loaded when pinned worker is dead."""
        router = self._make_router_with_workers(["http://w1:8888", "http://w2:8888"])
        router.session_id_to_worker[build_request_id("traj-001", 0)] = "http://w1:8888"
        router.dead_workers.add("http://w1:8888")

        selected = router._select_worker(build_request_id("traj-001", 0))

        assert selected == "http://w2:8888"
        assert router.session_id_to_worker[build_request_id("traj-001", 0)] == "http://w2:8888"

    def test_sticky_routing_skipped_when_overloaded(self):
        """Sticky routing falls back when pinned worker exceeds the load threshold."""
        router = self._make_router_with_workers(["http://w1:8888", "http://w2:8888"])
        router.session_id_to_worker[build_request_id("traj-001", 0)] = "http://w1:8888"
        router.worker_request_counts["http://w1:8888"] = router.rollout_worker_load_threshold + 1

        selected = router._select_worker(build_request_id("traj-001", 0))

        assert selected == "http://w2:8888"
        assert router.session_id_to_worker[build_request_id("traj-001", 0)] == "http://w2:8888"

    def test_least_loaded_worker_selected(self):
        """Without a sticky pin, the least-loaded worker is selected."""
        router = self._make_router_with_workers(["http://w1:8888", "http://w2:8888"])
        router.worker_request_counts["http://w1:8888"] = 3
        router.worker_request_counts["http://w2:8888"] = 1

        selected = router._select_worker(build_request_id("traj-001", 0))

        assert selected == "http://w2:8888"

    def test_retry_attempt_uses_independent_sticky_session(self):
        """A new attempt can be rebound independently from the previous attempt."""
        router = self._make_router_with_workers(["http://w1:8888", "http://w2:8888"])
        router.session_id_to_worker[build_request_id("traj-001", 0)] = "http://w1:8888"
        router.worker_request_counts["http://w1:8888"] = router.rollout_worker_load_threshold + 1

        selected = router._select_worker(build_request_id("traj-001", 1))

        assert selected == "http://w2:8888"
        assert router.session_id_to_worker[build_request_id("traj-001", 0)] == "http://w1:8888"
        assert router.session_id_to_worker[build_request_id("traj-001", 1)] == "http://w2:8888"

    def test_no_workers_raises_runtime_error(self):
        """_select_worker raises RuntimeError when no workers are registered."""
        router = make_router()
        with pytest.raises(RuntimeError, match="No healthy workers available"):
            router._select_worker(build_request_id("traj-001", 0))

    def test_all_workers_dead_raises_runtime_error(self):
        """_select_worker raises RuntimeError when all workers are in the dead pool."""
        router = self._make_router_with_workers(["http://w1:8888"])
        router.dead_workers.add("http://w1:8888")
        with pytest.raises(RuntimeError, match="No healthy workers available"):
            router._select_worker(build_request_id("traj-001", 0))

    def test_accumulate_reasoning_default_false(self):
        """accumulate_reasoning defaults to False from base config."""
        router = make_router()
        assert router.accumulate_reasoning is False

    def test_accumulate_reasoning_propagated(self):
        """accumulate_reasoning is propagated correctly from config."""
        router = make_router(accumulate_reasoning=True)
        assert router.accumulate_reasoning is True


# ---------------------------------------------------------------------------
# Proxy and request forwarding
# ---------------------------------------------------------------------------


@requires_parser_middleware
class TestProxyForwarding:
    """Tests for proxy request forwarding to workers."""

    @pytest.mark.asyncio
    async def test_proxy_requires_parser_middleware(self):
        """Session-scoped chat route requires parser middleware to prepare upstream body."""
        router = make_router()
        async with make_client(router) as client:
            response = await client.post(
                "/req-001/0/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert response.json()["error"] == "Router failed to prepare upstream request"

    @pytest.mark.asyncio
    async def test_proxy_no_available_workers_returns_503(self):
        """Proxy returns 503 when no healthy workers are registered."""
        router = make_proxy_router(make_store(("req-001", 0)))
        async with make_client(router, upstream_body={"messages": [{"role": "user", "content": "hi"}]}) as client:
            response = await client.post(
                "/req-001/0/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )

        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert response.json() == {"error": "No healthy workers available"}

    @pytest.mark.asyncio
    async def test_proxy_forwards_worker_4xx_unchanged(self):
        """A worker-side rejection must reach the caller with its status and body.

        The request itself is well-formed so it clears parser-middleware
        validation and actually reaches the worker; the router adds no judgement
        of its own and only relays whatever the worker decides.
        """
        router = make_proxy_router(make_store(("req-001", 0)))
        router.worker_request_counts["http://w1:8888"] = 0

        mock_response = MagicMock()
        mock_response.status_code = status.HTTP_400_BAD_REQUEST
        mock_response.headers = {"content-type": "application/json"}
        mock_response.aread = AsyncMock(return_value=b'{"error": "worker rejected it"}')

        async with proxy_client(
            router,
            upstream_body={"messages": [{"role": "user", "content": "hello"}]},
            return_value=mock_response,
        ) as (client, request_mock):
            response = await client.post(
                "/req-001/0/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hello"}]},
            )

        request_mock.assert_awaited_once()
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert response.json() == {"error": "worker rejected it"}

    @pytest.mark.asyncio
    async def test_proxy_forwards_request_to_worker(self):
        """Proxy forwards the request to selected worker and returns its response."""
        router = make_proxy_router(make_store(("req-001", 0)))
        router.worker_request_counts["http://w1:8888"] = 0

        mock_response = MagicMock()
        mock_response.status_code = status.HTTP_200_OK
        mock_response.headers = {"content-type": "application/json"}
        mock_response.aread = AsyncMock(
            return_value=json.dumps(
                {
                    "text": "ok",
                    "meta_info": {
                        "prompt_tokens": 3,
                        "completion_tokens": 1,
                        "weight_version": "0",
                        "output_token_logprobs": [[-0.1, 201]],
                    },
                }
            ).encode("utf-8")
        )

        async with proxy_client(
            router,
            upstream_body={
                "messages": [{"role": "user", "content": "hello"}],
                "temperature": 0.7,
            },
            return_value=mock_response,
        ) as (client, _):
            response = await client.post(
                "/req-001/0/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hello"}],
                    "temperature": 0.7,
                },
            )

        assert response.status_code == status.HTTP_200_OK
        # Parser middleware transforms the response to OpenAI format
        response_json = response.json()
        assert "choices" in response_json
        assert response_json["choices"][0]["message"]["content"] == "ok"
        assert router.worker_request_counts["http://w1:8888"] == 0  # released after request

    @pytest.mark.asyncio
    async def test_proxy_sticky_routing_by_trajectory_id(self):
        """Proxy pins one trajectory/attempt pair to the selected worker."""
        router = make_proxy_router(make_store(("traj-abc", 0)))
        router.worker_request_counts["http://w1:8888"] = 0

        mock_response = MagicMock()
        mock_response.status_code = status.HTTP_200_OK
        mock_response.headers = {"content-type": "application/json"}
        mock_response.aread = AsyncMock(return_value=json.dumps({
            "text": "hi",
            "meta_info": {"prompt_tokens": 3, "completion_tokens": 1,
                          "weight_version": "0", "output_token_logprobs": [[-0.1, 201]]},
        }).encode())

        async with proxy_client(
            router,
            upstream_body={"messages": [{"role": "user", "content": "hi"}]},
            return_value=mock_response,
        ) as (client, _):
            await client.post(
                "/traj-abc/0/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )

        assert router.session_id_to_worker.get(build_request_id("traj-abc", 0)) == "http://w1:8888"

    @pytest.mark.asyncio
    async def test_proxy_attempts_keep_independent_sticky_sessions(self):
        """Different attempts keep independent sticky-session bindings."""
        router = make_proxy_router(make_store(("traj-abc", 0), ("traj-abc", 1)))
        router.worker_request_counts["http://w1:8888"] = 0
        router.worker_request_counts["http://w2:8888"] = 0

        mock_response = MagicMock()
        mock_response.status_code = status.HTTP_200_OK
        mock_response.headers = {"content-type": "application/json"}
        mock_response.aread = AsyncMock(return_value=json.dumps({
            "text": "hi",
            "meta_info": {"prompt_tokens": 3, "completion_tokens": 1,
                          "weight_version": "0", "output_token_logprobs": [[-0.1, 201]]},
        }).encode())

        async with proxy_client(
            router,
            upstream_body={"messages": [{"role": "user", "content": "hi"}]},
            return_value=mock_response,
        ) as (client, _):
            await client.post(
                "/traj-abc/0/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )
            # Set load above threshold to force the second attempt to use a different worker
            router.worker_request_counts["http://w1:8888"] = router.rollout_worker_load_threshold + 1
            await client.post(
                "/traj-abc/1/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )

        assert router.session_id_to_worker.get(build_request_id("traj-abc", 0)) == "http://w1:8888"
        assert router.session_id_to_worker.get(build_request_id("traj-abc", 1)) == "http://w2:8888"

    @pytest.mark.asyncio
    async def test_proxy_increments_request_count_during_request(self):
        """Worker request count is 1 while the upstream request is in-flight."""
        router = make_proxy_router(make_store(("req-001", 0)))
        router.worker_request_counts["http://w1:8888"] = 0
        request_id = build_request_id("req-001", 0)

        valid_body = json.dumps({
            "text": "ok",
            "meta_info": {"prompt_tokens": 3, "completion_tokens": 1,
                          "weight_version": "0", "output_token_logprobs": [[-0.1, 201]]},
        }).encode()
        mock_response = MagicMock()
        mock_response.status_code = status.HTTP_200_OK
        mock_response.headers = {"content-type": "application/json"}
        mock_response.aread = AsyncMock(return_value=valid_body)

        async def check_count(*args, **kwargs):
            assert router.worker_request_counts["http://w1:8888"] == 1
            assert router.inflight_requests.get() == {request_id: 1}
            return mock_response

        async with proxy_client(
            router,
            upstream_body={"messages": [{"role": "user", "content": "hi"}]},
            side_effect=check_count,
        ) as (client, _):
            await client.post(
                "/req-001/0/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )

        assert router.worker_request_counts["http://w1:8888"] == 0  # decremented after
        assert router.inflight_requests.get() == {}

    @pytest.mark.asyncio
    async def test_proxy_admission_gate_bounds_concurrent_forwards(self):
        """proxy() never forwards more than max_connections requests at once."""
        gate = 2
        router = make_proxy_router(make_store(*[(f"req-{i}", 0) for i in range(5)]))
        router.worker_request_counts["http://w1:8888"] = 0
        router.rollout_worker_load_threshold = 100  # keep routing from blocking admission
        # Shrink the admission gate so saturation is reachable with a handful of requests.
        router._generate_sem = asyncio.Semaphore(gate)

        concurrent = 0
        max_concurrent = 0
        saturated = asyncio.Event()
        release = asyncio.Event()

        valid_body = json.dumps({
            "text": "ok",
            "meta_info": {"prompt_tokens": 3, "completion_tokens": 1,
                          "weight_version": "0", "output_token_logprobs": [[-0.1, 201]]},
        }).encode()
        mock_response = MagicMock()
        mock_response.status_code = status.HTTP_200_OK
        mock_response.headers = {"content-type": "application/json"}
        mock_response.aread = AsyncMock(return_value=valid_body)

        async def blocking_request(*args, **kwargs):
            nonlocal concurrent, max_concurrent
            concurrent += 1
            max_concurrent = max(max_concurrent, concurrent)
            if concurrent >= gate:
                saturated.set()
            await release.wait()
            concurrent -= 1
            return mock_response

        async with proxy_client(
            router,
            upstream_body={"messages": [{"role": "user", "content": "hi"}]},
            side_effect=blocking_request,
        ) as (client, _):
            tasks = [
                asyncio.create_task(client.post(
                    f"/req-{i}/0/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "hi"}]},
                ))
                for i in range(5)
            ]
            # Wait until the gate saturates, then confirm nothing slips past it.
            await asyncio.wait_for(saturated.wait(), timeout=2.0)
            await asyncio.sleep(0.05)
            assert max_concurrent == gate
            release.set()
            responses = await asyncio.gather(*tasks)

        assert max_concurrent == gate  # never exceeded across the whole run
        assert all(r.status_code == status.HTTP_200_OK for r in responses)

    @pytest.mark.asyncio
    async def test_proxy_handles_non_json_response(self):
        """Proxy returns raw bytes for non-JSON content-type responses."""
        router = make_proxy_router(make_store(("req-001", 0)))
        router.worker_request_counts["http://w1:8888"] = 0

        mock_response = MagicMock()
        mock_response.status_code = status.HTTP_200_OK
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.aread = AsyncMock(return_value=b"plain text response")

        async with proxy_client(
            router,
            upstream_body={"messages": [{"role": "user", "content": "hi"}]},
            return_value=mock_response,
        ) as (client, _):
            response = await client.post(
                "/req-001/0/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )

        assert response.status_code == status.HTTP_200_OK
        assert response.text == "plain text response"

    @pytest.mark.asyncio
    async def test_proxy_worker_5xx_is_forwarded(self):
        """Proxy forwards 5xx responses from the worker without modification.

        ``upstream_body`` must be injected: without it the router returns its own
        500 ("failed to prepare upstream request") and the assertion could not
        distinguish a forwarded 500 from a never-forwarded one.
        """
        router = make_proxy_router(make_store(("req-001", 0)))
        router.worker_request_counts["http://w1:8888"] = 0

        mock_response = MagicMock()
        mock_response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        mock_response.headers = {"content-type": "application/json"}
        mock_response.aread = AsyncMock(return_value=b'{"error": "internal"}')

        upstream_body = {
            "input_ids": [1, 2, 3],
            "rid": build_request_id("req-001", 0),
            "sampling_params": {},
        }

        async with proxy_client(
            router, upstream_body=upstream_body, return_value=mock_response
        ) as (client, request_mock):
            response = await client.post(
                "/req-001/0/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )

        request_mock.assert_awaited_once()
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert response.json() == {"error": "internal"}
        assert router.worker_request_counts["http://w1:8888"] == 0  # count still released

    @pytest.mark.asyncio
    async def test_proxy_network_error_returns_502(self):
        """Proxy returns 502 when the upstream connection fails."""
        router = make_proxy_router(make_store(("req-001", 0)))
        router.worker_request_counts["http://w1:8888"] = 0

        async with proxy_client(
            router,
            upstream_body={"messages": [{"role": "user", "content": "hi"}]},
            side_effect=httpx.RequestError("Connection refused"),
        ) as (client, _):
            response = await client.post(
                "/req-001/0/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )

        assert response.status_code == status.HTTP_502_BAD_GATEWAY
        assert router.worker_request_counts["http://w1:8888"] == 0  # count still released

    @pytest.mark.asyncio
    async def test_proxy_targets_generate_endpoint(self):
        """Proxy always forwards chat-completions requests to the worker /generate endpoint."""
        router = make_proxy_router(make_store(("req-001", 0)))
        router.worker_request_counts["http://w1:8888"] = 0

        mock_response = MagicMock()
        mock_response.status_code = status.HTTP_200_OK
        mock_response.headers = {"content-type": "application/json"}
        mock_response.aread = AsyncMock(return_value=json.dumps({
            "text": "ok",
            "meta_info": {"prompt_tokens": 3, "completion_tokens": 1,
                          "weight_version": "0", "output_token_logprobs": [[-0.1, 201]]},
        }).encode())

        async with proxy_client(
            router,
            upstream_body={"messages": [{"role": "user", "content": "hi"}]},
            return_value=mock_response,
        ) as (client, request_mock):
            await client.post(
                "/req-001/0/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]},
            )

        assert request_mock.await_args is not None
        assert request_mock.await_args.kwargs["url"] == "http://w1:8888/generate"


class FakeTokenizerManager:
    """Minimal tokenizer manager stub for parser-middleware integration tests."""

    def __init__(self) -> None:
        self.tokenizer = object()
        self.system_prompt_len = 0

    async def apply_chat_template(self, messages, *, add_generation_prompt=True, tools=None):
        return [101, 102, 103]

    async def encode(self, text):
        return [201]

    async def decode(self, token_ids, skip_special_tokens=True):
        return "decoded text"


@requires_parser_middleware
class TestChatCompletionIntegration:
    """Tests for parser middleware + proxy integration."""

    @pytest.mark.asyncio
    async def test_chat_completion_response_is_converted_to_openai_format(self):
        """The proxy hits /generate and the middleware maps the reply back to OpenAI.

        The upstream body is injected rather than produced by the middleware, so
        this covers the response direction and the endpoint rewrite only. The
        request-direction transform (messages -> input_ids, max_tokens ->
        sampling_params.max_new_tokens) is covered in test_parser.py.
        """
        router = make_proxy_router(make_store(("traj-1", 0)))
        router.worker_request_counts["http://w1:8888"] = 0

        mock_response = MagicMock()
        mock_response.status_code = status.HTTP_200_OK
        mock_response.headers = {"content-type": "application/json"}
        mock_response.aread = AsyncMock(
            return_value=json.dumps(
                {
                    "text": "ok",
                    "meta_info": {
                        "prompt_tokens": 3,
                        "completion_tokens": 1,
                        "weight_version": "0",
                        "output_token_logprobs": [[-0.1, 201]],
                    },
                }
            ).encode("utf-8")
        )

        upstream_body = {
            "input_ids": [101, 102, 103],
            "rid": build_request_id("traj-1", 0),
            "sampling_params": {"temperature": 0.3, "max_new_tokens": 8},
        }

        async with proxy_client(
            router, upstream_body=upstream_body, return_value=mock_response
        ) as (client, request_mock):
            response = await client.post(
                "/traj-1/0/v1/chat/completions",
                json={
                    "model": "default",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 8,
                    "temperature": 0.3,
                },
            )

        assert response.status_code == status.HTTP_200_OK
        response_json = response.json()
        assert response_json["choices"][0]["message"]["content"] == "ok"
        # The chat-completions path must be rewritten to the worker's /generate.
        assert request_mock.await_args.kwargs["url"] == "http://w1:8888/generate"
        forwarded = json.loads(request_mock.await_args.kwargs["content"])
        assert "messages" not in forwarded


# ---------------------------------------------------------------------------
# Worker request count management
# ---------------------------------------------------------------------------


class TestWorkerRequestCountManagement:
    """Tests for _finish_worker in-flight count tracking."""

    def test_finish_worker_clamps_at_zero(self):
        """_finish_worker does not go below zero (max(0, count-1))."""
        router = make_router()
        router.worker_request_counts["http://w1:8888"] = 0

        router._finish_worker("http://w1:8888")

        assert router.worker_request_counts["http://w1:8888"] == 0

    def test_finish_worker_does_not_affect_other_workers(self):
        """_finish_worker only decrements the targeted worker."""
        router = make_router()
        router.worker_request_counts["http://w1:8888"] = 2
        router.worker_request_counts["http://w2:8888"] = 4

        router._finish_worker("http://w1:8888")

        assert router.worker_request_counts["http://w1:8888"] == 1
        assert router.worker_request_counts["http://w2:8888"] == 4

    def test_select_and_finish_cycle(self):
        """Multiple selects followed by matching finishes returns count to 0."""
        router = make_router()
        router.worker_request_counts["http://w1:8888"] = 0

        worker = router._select_worker(build_request_id("t1", 0))
        assert worker == "http://w1:8888"
        router._select_worker(build_request_id("t2", 0))
        router._select_worker(build_request_id("t3", 0))
        assert router.worker_request_counts["http://w1:8888"] == 3

        for _ in range(3):
            router._finish_worker("http://w1:8888")
        assert router.worker_request_counts["http://w1:8888"] == 0
