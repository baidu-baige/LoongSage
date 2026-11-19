"""Unit tests for PD-disaggregated routing in the Router.

The pre-PD suite lives in ``test_router.py`` and still covers the single-form
(``regular``) topology; everything here is about the prefill/decode pair: slot
selection per worker_type, the bootstrap triple, fan-out failure handling, the
two-sided abort, and the control-plane endpoints that now traverse two levels.

Deliberately built on a bare ``make_router()`` rather than the parser-middleware
router: ``proxy()`` only reads ``request.state.upstream_body``, so these tests run
without sglang installed instead of being skipped away.
"""

import asyncio
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import status
from omegaconf import OmegaConf

from coda.agentflow.router.router import (
    WORKER_TYPE_DECODE,
    WORKER_TYPE_PREFILL,
    Router,
    WorkerSlot,
)
from coda.agentflow.utils import build_request_id

PD_THRESHOLDS = {"prefill": 32, "decode": 0}

ROUTER_PD_CONFIG = {
    "ip": "0.0.0.0",
    "port": None,
    "proxy_timeout_seconds": 600.0,
    "abort_timeout_seconds": 600.0,
    "accumulate_reasoning": False,
    "rollout_worker_load_threshold": PD_THRESHOLDS,
    "max_connections": 512,
    "middleware": {},
}


def make_pd_router(**overrides) -> Router:
    """A router configured for PD, with no workers registered yet."""
    config = dict(ROUTER_PD_CONFIG)
    config.update(overrides)
    return Router(config=OmegaConf.create(config))


def make_client(
    router: Router,
    upstream_body: dict | bytes | None = None,
    state_sink: dict | None = None,
) -> httpx.AsyncClient:
    """In-process ASGI client, optionally pre-seeding ``request.state.upstream_body``.

    ``state_sink`` receives a copy of the ASGI scope state after the call, which is
    how the handover of the prefill body through ``request.state`` is observed.
    """
    app = router.app
    if upstream_body is not None or state_sink is not None:
        body_bytes = (
            upstream_body
            if isinstance(upstream_body, bytes) or upstream_body is None
            else json.dumps(upstream_body).encode("utf-8")
        )

        class InjectBody:
            def __init__(self, inner):
                self.inner = inner

            async def __call__(self, scope, receive, send):
                if scope["type"] == "http":
                    scope.setdefault("state", {})
                    if body_bytes is not None:
                        scope["state"]["upstream_body"] = body_bytes
                try:
                    return await self.inner(scope, receive, send)
                finally:
                    if state_sink is not None and scope["type"] == "http":
                        state_sink.update(scope.get("state") or {})

        app = InjectBody(app)
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    )


async def register_pair(
    router: Router,
    *,
    prefill: str = "http://p1:2000",
    decode: str = "http://d1:2000",
    bootstrap_port: int = 2002,
    prefill_dp_size: int = 1,
) -> None:
    """Register one prefill and one decode worker through the real endpoint."""
    async with make_client(router) as client:
        result = await client.post(
            "/add_worker",
            json={
                "worker_url": prefill,
                "worker_type": "prefill",
                "bootstrap_port": bootstrap_port,
                "dp_size": prefill_dp_size,
            },
        )
        assert result.status_code == status.HTTP_200_OK, result.text
        result = await client.post(
            "/add_worker", json={"worker_url": decode, "worker_type": "decode"}
        )
        assert result.status_code == status.HTTP_200_OK, result.text


def worker_response(status_code: int = 200, body: bytes = b'{"text": "ok"}') -> MagicMock:
    """A stand-in for the httpx response of one worker's /generate."""
    response = MagicMock()
    response.status_code = status_code
    response.headers = {"content-type": "application/json"}
    response.aread = AsyncMock(return_value=body)
    return response


@asynccontextmanager
async def pd_proxy_client(router: Router, *, upstream_body, replies: dict):
    """Yield ``(client, calls, state)`` with per-worker ``/generate`` replies.

    ``replies`` maps a URL prefix to a response, an exception to raise, or an async
    callable. ``Router._post_generate`` builds a throwaway client per forward, so the
    seam is the constructor; the ASGI client is created before the patch so it keeps
    the real httpx.
    """
    state: dict = {}
    async with make_client(router, upstream_body=upstream_body, state_sink=state) as client:
        calls: list[tuple[str, bytes]] = []

        async def _request(*, method, url, headers, content):
            calls.append((url, content))
            for prefix, reply in replies.items():
                if url.startswith(prefix):
                    if isinstance(reply, BaseException):
                        raise reply
                    if asyncio.iscoroutinefunction(reply):
                        return await reply()
                    return reply
            raise AssertionError(f"unexpected upstream url {url}")

        generate_client = MagicMock()
        generate_client.request = AsyncMock(side_effect=_request)
        generate_client.aclose = AsyncMock()
        with patch.object(httpx, "AsyncClient", return_value=generate_client):
            yield client, calls, state


# ---------------------------------------------------------------------------
# Threshold configuration
# ---------------------------------------------------------------------------


class TestLoadThresholdConfig:
    """The threshold is read per worker_type, with no scalar or default fallback."""

    def test_mapping_threshold_is_read_per_worker_type(self):
        """Each worker_type reads its own value."""
        router = make_pd_router()
        assert router.rollout_worker_load_threshold[WORKER_TYPE_PREFILL] == 32
        assert router.rollout_worker_load_threshold[WORKER_TYPE_DECODE] == 0

    def test_missing_key_raises_instead_of_defaulting(self):
        """No ``default`` fallback: a missing key is a configuration error."""
        router = make_pd_router(rollout_worker_load_threshold={"prefill": 32})
        with pytest.raises(KeyError):
            router.rollout_worker_load_threshold[WORKER_TYPE_DECODE]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestPdRegistration:
    """/add_worker fills the per-worker_type tables."""

    @pytest.mark.asyncio
    async def test_registration_fills_the_three_tables(self):
        router = make_pd_router()
        await register_pair(router)
        assert router.worker_dp_size["prefill"] == {"http://p1:2000": 1}
        assert router.worker_dp_size["decode"] == {"http://d1:2000": 1}
        assert router.prefill_bootstrap_port == {"http://p1:2000": 2002}
        assert set(router.slot_request_counts) == {"prefill", "decode"}

    @pytest.mark.asyncio
    async def test_known_url_always_returns_409(self):
        """Live or excluded, a URL already in the tables is a duplicate."""
        router = make_pd_router()
        await register_pair(router)
        async with make_client(router) as client:
            live = await client.post(
                "/add_worker", json={"worker_url": "http://d1:2000", "worker_type": "decode"}
            )
            await client.put("/exclude_worker", json={"worker_url": "http://d1:2000"})
            excluded = await client.post(
                "/add_worker", json={"worker_url": "http://d1:2000", "worker_type": "decode"}
            )
        assert live.status_code == status.HTTP_409_CONFLICT
        assert excluded.status_code == status.HTTP_409_CONFLICT


# ---------------------------------------------------------------------------
# Slot selection
# ---------------------------------------------------------------------------


class TestPdSlotSelection:
    """One slot per worker_type, committed atomically."""

    @pytest.mark.asyncio
    async def test_selection_returns_one_slot_per_worker_type(self):
        router = make_pd_router()
        await register_pair(router)

        slots = router._select_slots(build_request_id("traj-001", 0))

        assert slots == {
            "prefill": WorkerSlot("http://p1:2000", 0),
            "decode": WorkerSlot("http://d1:2000", 0),
        }
        assert router.slot_request_counts["prefill"][slots["prefill"]] == 1
        assert router.slot_request_counts["decode"][slots["decode"]] == 1

    @pytest.mark.asyncio
    async def test_prefill_stays_sticky_while_decode_rebalances(self):
        """prefill keeps its prefix cache; decode's 0 threshold releases a busy slot."""
        router = make_pd_router()
        await register_pair(router)
        async with make_client(router) as client:
            await client.post(
                "/add_worker", json={"worker_url": "http://d2:2000", "worker_type": "decode"}
            )
        request_id = build_request_id("traj-001", 0)

        first = router._select_slots(request_id)
        # First turn still in flight (count 1 > decode threshold 0), so the next turn
        # of the same attempt keeps prefill but moves to the idle decode worker.
        second = router._select_slots(request_id)

        assert second["prefill"] == first["prefill"]
        assert second["decode"] != first["decode"]

    @pytest.mark.asyncio
    async def test_decode_is_reused_once_idle(self):
        """Threshold 0 does not disable decode stickiness, it only skips a busy slot."""
        router = make_pd_router()
        await register_pair(router)
        request_id = build_request_id("traj-001", 0)

        first = router._select_slots(request_id)
        router._finish_slot(first)
        second = router._select_slots(request_id)

        assert second == first

    @pytest.mark.asyncio
    async def test_empty_decode_pool_leaves_both_pools_untouched(self):
        """Two-phase commit: raising must not leak prefill's increment."""
        router = make_pd_router()
        await register_pair(router)
        async with make_client(router) as client:
            await client.put("/exclude_worker", json={"worker_url": "http://d1:2000"})
        prefill_slot = WorkerSlot("http://p1:2000", 0)

        for _ in range(3):
            with pytest.raises(RuntimeError, match="No healthy workers available"):
                router._select_slots(build_request_id("traj-001", 0))

        assert router.slot_request_counts["prefill"][prefill_slot] == 0
        assert router.session_id_to_slots == {}


# ---------------------------------------------------------------------------
# Body injection
# ---------------------------------------------------------------------------


class TestBootstrapInjection:
    """Both halves must receive the same rendezvous values."""

    @pytest.mark.asyncio
    async def test_both_halves_get_the_same_bootstrap_triple(self):
        router = make_pd_router()
        await register_pair(router)
        slots = router._select_slots(build_request_id("traj-001", 0))
        body = json.dumps({"input_ids": [1, 2, 3], "rid": "r"}).encode()

        prefill_body = json.loads(
            router._inject(body, slots, worker_type="prefill", room=12345)
        )
        decode_body = json.loads(router._inject(body, slots, worker_type="decode", room=12345))

        for payload in (prefill_body, decode_body):
            assert payload["bootstrap_host"] == "p1"
            assert payload["bootstrap_port"] == 2002
            assert payload["bootstrap_room"] == 12345
        # prefill has a single DP group here, so decode needs no DP hint.
        assert "disagg_prefill_dp_rank" not in decode_body

    @pytest.mark.asyncio
    async def test_decode_is_told_which_prefill_dp_group_holds_the_kv(self):
        router = make_pd_router()
        await register_pair(router, prefill_dp_size=2)
        slots = router._select_slots(build_request_id("traj-001", 0))
        body = json.dumps({"input_ids": [1]}).encode()

        decode_body = json.loads(router._inject(body, slots, worker_type="decode", room=7))
        prefill_body = json.loads(router._inject(body, slots, worker_type="prefill", room=7))

        assert decode_body["disagg_prefill_dp_rank"] == slots["prefill"].dp_rank
        assert prefill_body["routed_dp_rank"] == slots["prefill"].dp_rank
        assert "disagg_prefill_dp_rank" not in prefill_body

    @pytest.mark.asyncio
    async def test_decode_only_asks_for_the_generated_routed_experts(self):
        """Prefill owns the prompt rows; decode re-fetching them doubles the payload."""
        router = make_pd_router()
        await register_pair(router)
        slots = router._select_slots(build_request_id("traj-001", 0))
        body = json.dumps(
            {
                "input_ids": [1, 2, 3, 4, 5],
                "rid": "r",
                "return_routed_experts": True,
                "routed_experts_start_len": 2,
            }
        ).encode()

        decode_body = json.loads(router._inject(body, slots, worker_type="decode", room=1))
        prefill_body = json.loads(router._inject(body, slots, worker_type="prefill", room=1))

        assert decode_body["routed_experts_start_len"] == 5
        assert prefill_body["routed_experts_start_len"] == 2

    @pytest.mark.asyncio
    async def test_unparseable_body_raises_under_pd(self):
        """Without the triple the request would hang, so a warning is not enough."""
        router = make_pd_router()
        await register_pair(router)
        slots = router._select_slots(build_request_id("traj-001", 0))
        with pytest.raises(RuntimeError, match="cannot parse upstream body"):
            router._inject(b"not json", slots, worker_type="decode", room=1)


# ---------------------------------------------------------------------------
# Fan-out and merge
# ---------------------------------------------------------------------------


class TestPdFanOut:
    """Two upstream requests, one response, and no silent decode hangs."""

    @pytest.mark.asyncio
    async def test_both_halves_are_called_and_only_decode_is_returned(self):
        router = make_pd_router()
        await register_pair(router)
        replies = {
            "http://p1:2000": worker_response(body=b'{"text": "prefill"}'),
            "http://d1:2000": worker_response(body=b'{"text": "decode"}'),
        }
        async with pd_proxy_client(
            router, upstream_body={"input_ids": [1, 2, 3]}, replies=replies
        ) as (client, calls, state):
            result = await client.post("/traj-001/0/v1/chat/completions")

        assert result.status_code == status.HTTP_200_OK
        assert result.content == b'{"text": "decode"}'
        assert sorted(url for url, _ in calls) == [
            "http://d1:2000/generate",
            "http://p1:2000/generate",
        ]
        # The prefill reply is handed to the parser middleware, not to the agent.
        assert state["prefill_body"] == b'{"text": "prefill"}'

    @pytest.mark.asyncio
    async def test_prefill_non_2xx_returns_502_and_aborts_decode(self):
        """A 4xx/5xx from prefill is not an exception, and decode must not wait 300s."""
        router = make_pd_router()
        await register_pair(router)
        router._control_client.post = AsyncMock(return_value=httpx.Response(200, json={}))
        decode_started = asyncio.Event()

        async def _hang():
            decode_started.set()
            await asyncio.sleep(3600)

        replies = {
            "http://p1:2000": worker_response(status_code=500, body=b"oom"),
            "http://d1:2000": _hang,
        }
        async with pd_proxy_client(
            router, upstream_body={"input_ids": [1]}, replies=replies
        ) as (client, _calls, _state):
            result = await asyncio.wait_for(
                client.post("/traj-001/0/v1/chat/completions"), timeout=5
            )

        assert result.status_code == status.HTTP_502_BAD_GATEWAY
        assert "Prefill worker failed" in result.json()["error"]
        assert decode_started.is_set()
        router._control_client.post.assert_awaited_once_with(
            "http://d1:2000/abort_request",
            json={"rid": build_request_id("traj-001", 0), "abort_all": False},
        )
        assert router.pd_prefill_failures == 1

    @pytest.mark.asyncio
    async def test_prefill_connection_error_returns_502(self):
        router = make_pd_router()
        await register_pair(router)
        router._control_client.post = AsyncMock(return_value=httpx.Response(200, json={}))
        replies = {
            "http://p1:2000": httpx.ConnectError("refused"),
            "http://d1:2000": worker_response(),
        }
        async with pd_proxy_client(
            router, upstream_body={"input_ids": [1]}, replies=replies
        ) as (client, _calls, _state):
            result = await client.post("/traj-001/0/v1/chat/completions")

        assert result.status_code == status.HTTP_502_BAD_GATEWAY
        router._control_client.post.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_decode_network_error_returns_502(self):
        """The 502 for decode now comes out of _merge, not proxy's except clause."""
        router = make_pd_router()
        await register_pair(router)
        replies = {
            "http://p1:2000": worker_response(),
            "http://d1:2000": httpx.ConnectTimeout("timed out"),
        }
        async with pd_proxy_client(
            router, upstream_body={"input_ids": [1]}, replies=replies
        ) as (client, _calls, _state):
            result = await client.post("/traj-001/0/v1/chat/completions")

        assert result.status_code == status.HTTP_502_BAD_GATEWAY
        assert "Upstream request failed" in result.json()["error"]
        assert router.pd_decode_timeouts == 1

    @pytest.mark.asyncio
    async def test_counts_are_released_on_every_path(self):
        router = make_pd_router()
        await register_pair(router)
        replies = {
            "http://p1:2000": worker_response(),
            "http://d1:2000": worker_response(),
        }
        async with pd_proxy_client(
            router, upstream_body={"input_ids": [1]}, replies=replies
        ) as (client, _calls, _state):
            await client.post("/traj-001/0/v1/chat/completions")

        assert router.slot_request_counts["prefill"] == {WorkerSlot("http://p1:2000", 0): 0}
        assert router.slot_request_counts["decode"] == {WorkerSlot("http://d1:2000", 0): 0}

    @pytest.mark.asyncio
    async def test_injection_failure_leaks_no_counts(self):
        """Raising between commit and dispatch must not leave a charge behind."""
        router = make_pd_router()
        await register_pair(router)

        async with make_client(router, upstream_body=b"not json") as client:
            result = await client.post("/traj-001/0/v1/chat/completions")

        assert result.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert router.slot_request_counts["prefill"] == {WorkerSlot("http://p1:2000", 0): 0}
        assert router.slot_request_counts["decode"] == {WorkerSlot("http://d1:2000", 0): 0}

    @pytest.mark.asyncio
    async def test_empty_decode_pool_returns_503(self):
        router = make_pd_router()
        await register_pair(router)
        async with make_client(router) as control:
            await control.put("/exclude_worker", json={"worker_url": "http://d1:2000"})

        async with make_client(router, upstream_body={"input_ids": [1]}) as client:
            result = await client.post("/traj-001/0/v1/chat/completions")

        assert result.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert result.json() == {"error": "No healthy workers available"}


# ---------------------------------------------------------------------------
# Abort paths
# ---------------------------------------------------------------------------


class TestPdAbort:
    """Both halves of a pair get aborted, decode first."""

    @pytest.mark.asyncio
    async def test_abort_session_hits_both_ends_decode_first(self):
        router = make_pd_router()
        await register_pair(router)
        request_id = build_request_id("traj-001", 0)
        router._select_slots(request_id)
        router._control_client.post = AsyncMock(return_value=httpx.Response(200, json={}))

        async with make_client(router) as client:
            result = await client.post("/abort_session/traj-001/0")

        assert result.status_code == status.HTTP_200_OK
        assert result.json()["aborted"] == ["http://d1:2000", "http://p1:2000"]
        assert [c.args[0] for c in router._control_client.post.await_args_list] == [
            "http://d1:2000/abort_request",
            "http://p1:2000/abort_request",
        ]
        assert request_id not in router.session_id_to_slots

    @pytest.mark.asyncio
    async def test_partial_abort_failure_returns_503_and_keeps_the_pair(self):
        router = make_pd_router()
        await register_pair(router)
        request_id = build_request_id("traj-001", 0)
        router._select_slots(request_id)
        router._control_client.post = AsyncMock(
            side_effect=[
                httpx.Response(200, json={}),  # decode ok
                httpx.Response(500, text="busy"),  # prefill, 3 attempts
                httpx.Response(500, text="busy"),
                httpx.Response(500, text="busy"),
            ]
        )

        async with make_client(router) as client:
            result = await client.post("/abort_session/traj-001/0")

        assert result.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert result.json()["worker"] == ["http://p1:2000"]
        assert request_id in router.session_id_to_slots

    @pytest.mark.asyncio
    async def test_abort_all_workers_covers_both_pools(self):
        router = make_pd_router()
        await register_pair(router)
        router._select_slots(build_request_id("traj-001", 0))
        router._control_client.post = AsyncMock(return_value=httpx.Response(200, json={}))

        async with make_client(router) as client:
            result = await client.post("/abort_all_workers")

        assert result.status_code == status.HTTP_200_OK
        assert result.json() == {
            "status": "success",
            "workers_aborted": 2,
            "sessions_cleared": 1,
        }
        assert router.session_id_to_slots == {}
        assert router.slot_request_counts["prefill"] == {WorkerSlot("http://p1:2000", 0): 0}
        assert router.slot_request_counts["decode"] == {WorkerSlot("http://d1:2000", 0): 0}

    @pytest.mark.asyncio
    async def test_excluding_a_prefill_aborts_its_pinned_sessions(self):
        """Its decode peer is parked on a KV push that will never arrive."""
        router = make_pd_router()
        await register_pair(router)
        router._select_slots(build_request_id("traj-001", 0))
        router._select_slots(build_request_id("traj-002", 0))
        router._control_client.post = AsyncMock(return_value=httpx.Response(200, json={}))

        async with make_client(router) as client:
            result = await client.put(
                "/exclude_worker", json={"worker_url": "http://p1:2000"}
            )

        assert result.status_code == status.HTTP_200_OK
        assert result.json()["aborted_sessions"] == 2
        assert result.json()["worker_type"] == "prefill"
        assert router.session_id_to_slots == {}
        assert {c.args[0] for c in router._control_client.post.await_args_list} == {
            "http://d1:2000/abort_request"
        }

    @pytest.mark.asyncio
    async def test_excluding_a_decode_aborts_nothing(self):
        router = make_pd_router()
        await register_pair(router)
        router._select_slots(build_request_id("traj-001", 0))
        router._control_client.post = AsyncMock(return_value=httpx.Response(200, json={}))

        async with make_client(router) as client:
            result = await client.put(
                "/exclude_worker", json={"worker_url": "http://d1:2000"}
            )

        assert result.json()["aborted_sessions"] == 0
        router._control_client.post.assert_not_called()


# ---------------------------------------------------------------------------
# Control plane
# ---------------------------------------------------------------------------


class TestPdControlPlane:
    """Endpoints that walk the state tables now walk two levels."""

    @pytest.mark.asyncio
    async def test_list_workers_keeps_the_flat_shape_and_adds_worker_meta(self):
        router = make_pd_router()
        await register_pair(router)

        async with make_client(router) as client:
            body = (await client.get("/list_workers")).json()

        assert sorted(body["active_workers"]) == ["http://d1:2000", "http://p1:2000"]
        assert body["load_stats"] == {"http://p1:2000": 0, "http://d1:2000": 0}
        assert body["slot_load_stats"] == {"http://p1:2000": [0], "http://d1:2000": [0]}
        assert body["worker_meta"] == {
            "http://p1:2000": {
                "worker_type": "prefill",
                "dp_size": 1,
                "bootstrap_port": 2002,
            },
            "http://d1:2000": {
                "worker_type": "decode",
                "dp_size": 1,
                "bootstrap_port": None,
            },
        }

    @pytest.mark.asyncio
    async def test_excluded_worker_drops_out_of_active_workers(self):
        """A rebuilt engine usually returns on a new port; both coexist in the tables."""
        router = make_pd_router()
        await register_pair(router)
        router._control_client.post = AsyncMock(return_value=httpx.Response(200, json={}))

        async with make_client(router) as client:
            await client.put("/exclude_worker", json={"worker_url": "http://d1:2000"})
            await client.post(
                "/add_worker", json={"worker_url": "http://d9:2000", "worker_type": "decode"}
            )
            body = (await client.get("/list_workers")).json()

        assert sorted(body["active_workers"]) == ["http://d9:2000", "http://p1:2000"]
        assert body["dead_workers"] == ["http://d1:2000"]
        # Routing never picks it again: candidates are filtered by dead_workers.
        slots = router._select_slots(build_request_id("traj-001", 0))
        assert slots["decode"] == WorkerSlot("http://d9:2000", 0)

    @pytest.mark.asyncio
    async def test_include_worker_restores_the_slot(self):
        router = make_pd_router()
        await register_pair(router)
        router._control_client.post = AsyncMock(return_value=httpx.Response(200, json={}))

        async with make_client(router) as client:
            await client.put("/exclude_worker", json={"worker_url": "http://p1:2000"})
            result = await client.put(
                "/include_worker", json={"worker_url": "http://p1:2000"}
            )

        assert result.status_code == status.HTTP_200_OK
        assert result.json()["worker_type"] == "prefill"
        assert router.slot_request_counts["prefill"] == {WorkerSlot("http://p1:2000", 0): 0}
        slots = router._select_slots(build_request_id("traj-001", 0))
        assert slots["prefill"] == WorkerSlot("http://p1:2000", 0)

    @pytest.mark.asyncio
    async def test_unknown_worker_still_returns_404(self):
        router = make_pd_router()
        await register_pair(router)
        async with make_client(router) as client:
            excluded = await client.put(
                "/exclude_worker", json={"worker_url": "http://nope:2000"}
            )
            included = await client.put(
                "/include_worker", json={"worker_url": "http://nope:2000"}
            )

        assert excluded.status_code == status.HTTP_404_NOT_FOUND
        assert excluded.json() == {"error": "Worker not found"}
        assert included.status_code == status.HTTP_404_NOT_FOUND
        assert included.json() == {"error": "Worker not found (never registered)"}

    @pytest.mark.asyncio
    async def test_release_session_drops_the_whole_pair(self):
        router = make_pd_router()
        await register_pair(router)
        router._select_slots(build_request_id("traj-001", 0))

        async with make_client(router) as client:
            result = await client.delete("/release_session/traj-001/0")

        assert result.status_code == status.HTTP_200_OK
        assert router.session_id_to_slots == {}
