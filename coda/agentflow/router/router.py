"""Router for load balancing LLM requests across SGLang workers."""

import asyncio
import inspect
import json
import logging
import random
import ssl
import threading
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlsplit

import httpx
import uvicorn
from fastapi import APIRouter, FastAPI, Request, status
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel as PydanticModel

from coda.agentflow.router import resolve_middleware_chain
from coda.agentflow.utils import build_request_id

# The three ``rollout.sglang_replicas`` group names: ``regular`` is the single-form
# topology, ``prefill`` + ``decode`` is PD disaggregation.
WORKER_TYPE_REGULAR = "regular"
WORKER_TYPE_PREFILL = "prefill"
WORKER_TYPE_DECODE = "decode"
# Abort order inside one P/D pair: decode before prefill.
PD_ABORT_ORDER = (WORKER_TYPE_DECODE, WORKER_TYPE_PREFILL, WORKER_TYPE_REGULAR)
# SGLang accepts bootstrap_room in [0, 2^63).
BOOTSTRAP_ROOM_BITS = 63
# httpx builds a fresh ``ssl.SSLContext`` (re-reading the CA bundle) inside every
# ``AsyncClient``, synchronously on the event loop. ``_post_generate`` keeps one
# client per request and shares only this context.
_SSL_CONTEXT = ssl.create_default_context()


class WorkerPayload(PydanticModel):
    """Worker management payload."""
    worker_url: str
    # Number of attention-DP groups inside the worker (SGLang ``--attn-dp-size``).
    # The router uses it to assign each request to
    # a specific DP group via ``routed_dp_rank`` instead of SGLang load-balancing
    # internally. 1 means "no DP attention" (nothing to route).
    dp_size: int = 1
    # Which replica group the engine belongs to: regular, prefill or decode.
    worker_type: str = WORKER_TYPE_REGULAR
    # KV handshake port of a prefill engine (``disaggregation_bootstrap_port``);
    # unused for the other worker types.
    bootstrap_port: int | None = None


@dataclass(frozen=True, slots=True)
class WorkerSlot:
    """One dispatch target: a single attention-DP group of one SGLang worker.

    ``dp_rank`` is the DP group inside the worker (always 0 when the worker does
    not run DP attention). Worker selection, sticky sessions and in-flight
    accounting are all keyed on this one structure.
    """

    worker_url: str
    dp_rank: int = 0

    def __str__(self) -> str:
        return f"{self.worker_url}:{self.dp_rank}"

@dataclass(frozen=True, slots=True)
class UpstreamResult:
    """One worker's reply to a forwarded ``/generate``, already fully read."""

    status_code: int
    headers: dict[str, str]
    content: bytes


def _upstream_ok(result: Any) -> bool:
    """True only when a forward produced a 2xx reply rather than an exception."""
    return isinstance(result, UpstreamResult) and 200 <= result.status_code < 300


logger = logging.getLogger("Router")

HOP_BY_HOP = {
    "content-length",
    "transfer-encoding",
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "upgrade",
}

# FastAPI needs concrete paths at startup. Protocol selection and request-kind
# detection remain in the adapters through ProtocolAdapter.match().
SESSION_PROXY_ROUTES = (
    "/v1/chat/completions",
    "/v1/messages",
    "/v1/responses",
    "/v1/responses/compact",
)


def filter_headers(headers: dict[str, str]) -> dict[str, str]:
    """Drop connection-specific upstream headers from proxied responses."""
    return {k: v for k, v in headers.items() if k.lower() not in HOP_BY_HOP}


class InflightRequests:
    """Track middleware handlers so abort can wait out stale writes."""

    def __init__(self) -> None:
        self._counts: Counter[str] = Counter()
        self._cond = asyncio.Condition()

    def add(self, request_id: str) -> None:
        """Record a parser middleware request before it may write trajectory state."""
        self._counts[request_id] += 1

    async def delete(self, request_id: str) -> None:
        """Record a parser middleware request leaving and wake abort waiters."""
        async with self._cond:
            self._counts[request_id] -= 1
            if self._counts[request_id] <= 0:
                self._counts.pop(request_id, None)
            self._cond.notify_all()

    def get(self) -> dict[str, int]:
        """Return active parser middleware request counts by request ID."""
        return {request_id: count for request_id, count in self._counts.items() if count > 0}

    async def wait_complete(self, timeout: float) -> None:
        """Wait until all parser middleware requests have finished."""
        async def _wait() -> None:
            async with self._cond:
                await self._cond.wait_for(lambda: not self._counts)

        await asyncio.wait_for(_wait(), timeout)


class Router:
    """Reverse proxy with sticky + least-loaded routing across SGLang workers.

    Routing is driven by ``worker_type``: a request is dispatched to one slot per
    registered worker type. The single-form topology registers only ``regular`` and
    therefore fans out to one worker; PD disaggregation registers
    ``prefill`` + ``decode`` and fans the same body out to both, with the bootstrap
    triple injected so the two halves rendezvous over RDMA.
    """

    def __init__(
        self,
        config: Any,
        *,
        middleware_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize routing state, HTTP client, and FastAPI app."""
        self.config = config
        # worker_type -> in-flight requests a pinned slot may carry before sticky
        # routing re-balances
        self.rollout_worker_load_threshold = config.rollout_worker_load_threshold
        self.accumulate_reasoning = config.accumulate_reasoning

        # worker_type -> WorkerSlot -> in-flight request count, one pool per type
        self.slot_request_counts: dict[str, dict[WorkerSlot, int]] = {}
        # Worker URLs quarantined from the routing pool (URL granularity)
        self.dead_workers: set[str] = set()
        # request_id (trajectory_id#attempt_id) -> worker_type -> pinned WorkerSlot
        # (sticky routing)
        self.session_id_to_slots: dict[str, dict[str, WorkerSlot]] = {}
        # worker_type -> URL -> attention-DP group count reported by the worker
        self.worker_dp_size: dict[str, dict[str, int]] = {}
        # URL -> KV handshake port, prefill only
        self.prefill_bootstrap_port: dict[str, int] = {}
        self.inflight_requests = InflightRequests()
        # PD failure counters, one per side
        self.pd_prefill_failures = 0
        self.pd_decode_timeouts = 0

        self._max_connections = int(config.max_connections)
        # Dedicated control client + semaphore for worker-abort requests, off the
        # forward path so an abort storm can't inflate it and pin the router loop.
        # A PD pair spends one permit, not two.
        self._generate_sem = asyncio.Semaphore(self._max_connections)
        self._control_sem = asyncio.Semaphore(self._max_connections)
        self._control_client = httpx.AsyncClient(
            timeout=config.proxy_timeout_seconds,
            limits=httpx.Limits(max_connections=self._max_connections),
        )
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._startup_error: BaseException | None = None
        self.trajectory_store = (middleware_kwargs or {}).get("trajectory_store")
        self.app = FastAPI()
        self._setup_routes()
        # Callbacks run on shutdown to release middleware-owned resources (e.g. R3 decode pool).
        self._shutdown_callbacks: list[Callable[[], None]] = []
        middleware_context = dict(middleware_kwargs or {})
        middleware_context["inflight_requests"] = self.inflight_requests
        middleware_context["shutdown_callbacks"] = self._shutdown_callbacks
        self._setup_middlewares(
            resolve_middleware_chain(config.middleware),
            middleware_context,
        )

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------

    def _setup_routes(self) -> None:
        """Register management and session-scoped proxy routes."""
        management_router = APIRouter()
        management_router.post("/add_worker", response_model=None)(self.add_worker)
        management_router.get("/list_workers", response_model=None)(self.list_workers)
        management_router.put("/exclude_worker", response_model=None)(self.exclude_worker)
        management_router.put("/include_worker", response_model=None)(self.include_worker)
        management_router.delete(
            "/release_session/{trajectory_id}/{attempt_id}",
            response_model=None,
        )(self.release_session)
        management_router.post(
            "/abort_session/{trajectory_id}/{attempt_id}",
            response_model=None,
        )(self.abort_session)
        management_router.post("/abort_all_workers", response_model=None)(
            self.abort_all_workers
        )
        management_router.post("/wait_inflight_requests_complete", response_model=None)(
            self.wait_inflight_requests_complete
        )

        # ``/{trajectory_id}/{attempt_id}`` is treated as the session-scoped base URL.
        session_router = APIRouter(prefix="/{trajectory_id}/{attempt_id}")
        for route in SESSION_PROXY_ROUTES:
            session_router.post(route, response_model=None)(self.proxy)
        # Token counting bypasses generation: report the trajectory's stored token
        # length straight from the store instead of proxying to a worker.
        session_router.post("/v1/messages/count_tokens", response_model=None)(
            self.count_tokens
        )

        self.app.include_router(management_router)
        self.app.include_router(session_router)

    def _setup_middlewares(
        self,
        middleware_specs: list[dict[str, Any]],
        middleware_kwargs: dict[str, Any],
    ) -> None:
        """Install configured middlewares in FastAPI wrapping order."""
        for spec in reversed(middleware_specs):
            logger.info("[setup] loading middleware: %s", spec["name"])
            init_kwargs = self._filter_middleware_context(spec["middleware_cls"], middleware_kwargs)
            init_kwargs.update(spec["params"])
            self.app.add_middleware(
                spec["middleware_cls"], **init_kwargs
            )

    @staticmethod
    def _filter_middleware_context(
        middleware_cls: type[Any],
        middleware_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Keep only kwargs accepted by a middleware constructor."""
        init_params = inspect.signature(middleware_cls.__init__).parameters
        if any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in init_params.values()
        ):
            return dict(middleware_kwargs)

        allowed = {
            name
            for name, param in init_params.items()
            if name not in {"self", "app"}
            and param.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }
        return {k: v for k, v in middleware_kwargs.items() if k in allowed}

    # -------------------------------------------------------------------------
    # Worker lifecycle (synchronous — no await, safe in single asyncio loop)
    # -------------------------------------------------------------------------

    def _slots_of(self, worker_type: str, worker_url: str) -> list[WorkerSlot]:
        """All DP-group slots of a worker (a single slot when DP attention is off)."""
        dp_size = self.worker_dp_size[worker_type][worker_url]
        return [WorkerSlot(worker_url, dp) for dp in range(dp_size)]

    def _worker_type_of(self, worker_url: str) -> str | None:
        """Reverse-lookup a URL's worker_type, or None when never registered.

        A URL lives under exactly one worker_type: an engine process has one
        ``disaggregation_mode``, and mixed registration is rejected by add_worker.
        """
        for worker_type, urls in self.worker_dp_size.items():
            if worker_url in urls:
                return worker_type
        return None

    def _select_slots(self, request_id: str) -> dict[str, WorkerSlot]:
        """Pick a (worker, DP group) slot per worker_type and increment its in-flight count.

        For each registered worker_type: sticky per session first (the same slot for
        every turn of a trajectory attempt, which keeps SGLang's prefix cache on one DP
        group), else the least-loaded healthy slot. A pinned slot is abandoned once it
        carries more than that type's ``rollout_worker_load_threshold``.
        """
        if not self.worker_dp_size:
            raise RuntimeError("No healthy workers available")
        # Select a slot for every type first, then commit the writes together.
        selected: dict[str, WorkerSlot] = {}
        for worker_type in sorted(self.worker_dp_size):
            counts = self.slot_request_counts[worker_type]
            pinned = self.session_id_to_slots.get(request_id, {}).get(worker_type)
            if (
                pinned is not None
                and pinned.worker_url not in self.dead_workers
                and counts[pinned] <= self.rollout_worker_load_threshold[worker_type]
            ):
                selected[worker_type] = pinned
                logger.info("[route] sticky %s -> %s (%s)", request_id, pinned, worker_type)
                continue
            available = [s for s in counts if s.worker_url not in self.dead_workers]
            if not available:
                raise RuntimeError("No healthy workers available")
            slot = min(available, key=counts.__getitem__)
            selected[worker_type] = slot
            logger.info("[route] new %s -> %s (%s)", request_id, slot, worker_type)

        self.session_id_to_slots[request_id] = dict(selected)
        for worker_type, slot in selected.items():
            self.slot_request_counts[worker_type][slot] += 1
        return selected

    def _finish_slot(self, slots: dict[str, WorkerSlot]) -> None:
        """Decrement the in-flight count of every slot charged for one request."""
        for worker_type, slot in slots.items():
            counts = self.slot_request_counts.get(worker_type, {})
            if slot not in counts:
                logger.error(
                    "[worker] _finish_slot: unknown slot %s (%s) — skipping decrement",
                    slot, worker_type,
                )
                continue
            counts[slot] = max(0, counts[slot] - 1)

    # -------------------------------------------------------------------------
    # Management API
    # -------------------------------------------------------------------------

    async def add_worker(self, payload: WorkerPayload) -> JSONResponse:
        """Register a worker URL with its worker_type, DP group count, and bootstrap port for routing."""
        worker_url = payload.worker_url
        worker_type = payload.worker_type
        if self._worker_type_of(worker_url) is not None:
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={"error": "Worker already exists", "worker": worker_url},
            )

        dp_size = max(1, int(payload.dp_size or 1))
        self.worker_dp_size.setdefault(worker_type, {})[worker_url] = dp_size
        counts = self.slot_request_counts.setdefault(worker_type, {})
        for slot in self._slots_of(worker_type, worker_url):
            counts[slot] = 0
        if worker_type == WORKER_TYPE_PREFILL:
            self.prefill_bootstrap_port[worker_url] = payload.bootstrap_port
        logger.info(
            "[worker] added %s (worker_type=%s dp_size=%d bootstrap_port=%s)",
            worker_url, worker_type, dp_size, payload.bootstrap_port,
        )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "worker": worker_url,
                "worker_type": worker_type,
            },
        )

    async def list_workers(self) -> JSONResponse:
        """Return active, dead, and per-slot load state."""
        slot_load: dict[str, list[int]] = {}
        worker_meta: dict[str, dict[str, Any]] = {}
        for worker_type, dp_sizes in self.worker_dp_size.items():
            counts = self.slot_request_counts[worker_type]
            for url, dp_size in dp_sizes.items():
                slot_load[url] = [counts[s] for s in self._slots_of(worker_type, url)]
                worker_meta[url] = {
                    "worker_type": worker_type,
                    "dp_size": dp_size,
                    "bootstrap_port": self.prefill_bootstrap_port.get(url),
                }
        active_workers = [url for url in worker_meta if url not in self.dead_workers]
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "active_workers": active_workers,
                "dead_workers": list(self.dead_workers),
                "load_stats": {url: sum(counts) for url, counts in slot_load.items()},
                "slot_load_stats": slot_load,
                "worker_meta": worker_meta,
            },
        )

    async def exclude_worker(self, payload: WorkerPayload) -> JSONResponse:
        """Remove a worker from the active routing pool.

        Excluding a prefill worker also aborts every session pinned to it, whose
        decode peer is waiting for a KV push that will never arrive.
        """
        worker_url = payload.worker_url
        worker_type = self._worker_type_of(worker_url)
        if worker_type is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"error": "Worker not found"},
            )
        if worker_url in self.dead_workers:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": "Worker is already excluded"},
            )
        self.dead_workers.add(worker_url)
        aborted_sessions = 0
        if worker_type == WORKER_TYPE_PREFILL:
            aborted_sessions = await self._abort_sessions_pinned_to_prefill(worker_url)
        logger.info(
            "[worker] excluded %s (worker_type=%s aborted_sessions=%d)",
            worker_url, worker_type, aborted_sessions,
        )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "worker": worker_url,
                "worker_type": worker_type,
                "aborted_sessions": aborted_sessions,
            },
        )

    async def include_worker(self, payload: WorkerPayload) -> JSONResponse:
        """Restore an excluded worker to the active pool."""
        worker_url = payload.worker_url
        worker_type = self._worker_type_of(worker_url)
        if worker_type is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"error": "Worker not found (never registered)"},
            )
        if worker_url not in self.dead_workers:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": "Worker is already active"},
            )
        self.dead_workers.discard(worker_url)
        logger.info("[worker] included %s (worker_type=%s)", worker_url, worker_type)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "worker": worker_url,
                "worker_type": worker_type,
            },
        )

    async def release_session(self, trajectory_id: str, attempt_id: int) -> JSONResponse:
        """Release sticky routing state for one trajectory attempt."""
        request_id = build_request_id(trajectory_id, attempt_id)
        released = self.session_id_to_slots.pop(request_id, None) is not None
        if released:
            logger.info("[session] released %s", request_id)
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "status": "success",
                    "trajectory_id": trajectory_id,
                    "attempt_id": attempt_id,
                },
            )
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "error": "No sticky session found",
                "trajectory_id": trajectory_id,
                "attempt_id": attempt_id,
            },
        )

    async def _send_worker_abort_request(self, worker_url: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Send /abort_request with a small retry budget."""
        max_attempts = 3
        failure = None
        target_url = f"{worker_url.rstrip('/')}/abort_request"
        for attempt in range(1, max_attempts + 1):
            try:
                async with self._control_sem:
                    response = await self._control_client.post(target_url, json=payload)
                if 200 <= response.status_code < 300:
                    return None
                failure = {
                    "worker": worker_url,
                    "status_code": response.status_code,
                    "error": response.text,
                }
            except Exception as exc:
                failure = {"worker": worker_url, "error": str(exc)}

            logger.warning(
                "[abort] worker %s abort failed attempt %d/%d: %s",
                worker_url, attempt, max_attempts, failure,
            )
            if attempt < max_attempts:
                await asyncio.sleep(1)

        return failure

    async def _wait_inflight_requests_complete(self) -> JSONResponse | None:
        """Wait for parser middleware to finish before marking trajectories aborted."""
        timeout = float(self.config.abort_timeout_seconds)
        try:
            await self.inflight_requests.wait_complete(timeout)
            return None
        except asyncio.TimeoutError:
            active_requests = self.inflight_requests.get()
            logger.warning(
                "[abort] wait inflight requests complete timeout timeout=%s unfinished=%s",
                timeout, active_requests,
            )
            return JSONResponse(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                content={
                    "status": "failed",
                    "error": "wait inflight requests complete timeout",
                    "timeout": timeout,
                    "unfinished": active_requests,
                },
            )

    async def _abort_sessions_pinned_to_prefill(self, worker_url: str) -> int:
        """Abort the decode half of every session pinned to a dead prefill worker."""
        victims = [
            (request_id, slots[WORKER_TYPE_DECODE])
            for request_id, slots in self.session_id_to_slots.items()
            if (pinned := slots.get(WORKER_TYPE_PREFILL)) is not None
            and pinned.worker_url == worker_url
        ]
        await asyncio.gather(
            *(
                self._send_worker_abort_request(
                    decode_slot.worker_url, {"rid": request_id, "abort_all": False}
                )
                for request_id, decode_slot in victims
            ),
        )
        for request_id, _ in victims:
            self.session_id_to_slots.pop(request_id, None)
        return len(victims)

    async def abort_session(self, trajectory_id: str, attempt_id: int) -> JSONResponse:
        """Abort the worker request(s) pinned to one failed trajectory attempt.

        Under PD both halves get an ``/abort_request``, decode first.
        """
        request_id = build_request_id(trajectory_id, attempt_id)
        slots = self.session_id_to_slots.get(request_id)
        if not slots:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "error": "No active session found",
                    "trajectory_id": trajectory_id,
                    "attempt_id": attempt_id,
                },
            )
        ordered_types = [wt for wt in PD_ABORT_ORDER if wt in slots]
        aborted = [slots[wt].worker_url for wt in ordered_types]
        failures = []
        for worker_type in ordered_types:
            failure = await self._send_worker_abort_request(
                slots[worker_type].worker_url,
                {"rid": request_id, "abort_all": False},
            )
            if failure:
                failures.append(failure)
        if failures:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "failed",
                    "trajectory_id": trajectory_id,
                    "attempt_id": attempt_id,
                    "worker": [f["worker"] for f in failures],
                    "error": failures,
                },
            )
        self.session_id_to_slots.pop(request_id, None)
        logger.info("[session] aborted failed attempt %s (workers=%s)", request_id, aborted)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "trajectory_id": trajectory_id,
                "attempt_id": attempt_id,
                "aborted": aborted,
            },
        )

    async def abort_all_workers(self) -> JSONResponse:
        """Abort all active workers and clear sticky session state."""
        active_workers = [
            w
            for w in dict.fromkeys(
                slot.worker_url
                for counts in self.slot_request_counts.values()
                for slot in counts
            )
            if w not in self.dead_workers
        ]
        if not active_workers:
            logger.info("[abort] no active workers")
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={"status": "success", "workers_aborted": 0, "sessions_cleared": 0},
            )

        results = await asyncio.gather(
            *(
                self._send_worker_abort_request(w, {"abort_all": True})
                for w in active_workers
            ),
        )
        failed_workers = [r for r in results if r is not None]
        success_count = len(active_workers) - len(failed_workers)

        if failed_workers:
            logger.warning(
                "[abort] workers=%d/%d failed=%s",
                success_count,
                len(active_workers),
                failed_workers,
            )
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "failed",
                    "workers_aborted": success_count,
                    "failed_workers": failed_workers,
                    "sessions_cleared": 0,
                },
            )

        sessions_cleared = len(self.session_id_to_slots)
        self.session_id_to_slots.clear()

        for counts in self.slot_request_counts.values():
            for slot in counts:
                counts[slot] = 0

        logger.info(
            "[abort] workers=%d/%d sessions_cleared=%d",
            success_count,
            len(active_workers),
            sessions_cleared,
        )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "workers_aborted": success_count,
                "sessions_cleared": sessions_cleared,
            },
        )

    async def wait_inflight_requests_complete(self) -> JSONResponse:
        """Wait for all tracked middleware requests to finish."""
        tracked_count = len(self.inflight_requests.get())
        wait_error = await self._wait_inflight_requests_complete()
        if wait_error:
            return wait_error
        logger.info("[abort] inflight requests complete count=%d", tracked_count)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "completed": tracked_count,
            },
        )

    # -------------------------------------------------------------------------
    # Proxy
    # -------------------------------------------------------------------------

    async def count_tokens(
        self,
        trajectory_id: str,
        attempt_id: int,
        request: Request,
    ) -> Response:
        """Report the trajectory's stored token length for client-side accounting.

        Bypasses generation entirely: the client's context / auto-compact
        accounting reflects real usage. Falls back to 0 when the trajectory
        cannot be resolved (e.g. before its first turn).
        """
        input_tokens = 0
        if self.trajectory_store is not None:
            entries = self.trajectory_store.get(
                [trajectory_id], attempt_id=attempt_id
            ).get(trajectory_id)
            if entries:
                input_tokens = len(entries[-1].tokens)
        return JSONResponse(content={"input_tokens": input_tokens})

    async def proxy(
        self,
        trajectory_id: str,
        attempt_id: int,
        request: Request,
    ) -> Response:
        """Proxy a prepared chat request to SGLang /generate.

        One target in the regular topology, a prefill/decode pair under PD.
        """
        request_id = build_request_id(trajectory_id, attempt_id)
        body_bytes = getattr(request.state, "upstream_body", None)
        if body_bytes is None:
            logger.error("[proxy] upstream_body not set for %s", request_id)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": "Router failed to prepare upstream request"},
            )

        headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
        slots = None
        async with self._generate_sem:
            try:
                # Inside the try: _select_slots has already charged every end.
                slots = self._select_slots(request_id)
                # Per-request rendezvous id shared by both halves; unrelated to rid.
                room = random.getrandbits(BOOTSTRAP_ROOM_BITS)
                # Build every body before creating any task.
                bodies = {
                    worker_type: self._inject(body_bytes, slots, worker_type=worker_type, room=room)
                    for worker_type in slots
                }
                tasks = {
                    worker_type: asyncio.create_task(
                        self._post_generate(
                            slots[worker_type], body, method=request.method, headers=headers
                        )
                    )
                    for worker_type, body in bodies.items()
                }
                results = await self._dispatch(tasks)
                return await self._merge(results, request, request_id, slots)
            except RuntimeError as exc:
                return JSONResponse(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    content={"error": str(exc)},
                )
            finally:
                if slots is not None:
                    self._finish_slot(slots)

    async def _dispatch(
        self, tasks: dict[str, asyncio.Task]
    ) -> dict[str, UpstreamResult | Exception]:
        """Await each upstream, prefill first, keeping its response or exception.

        A failed prefill stops the wait; decode is then cancelled.
        """
        results: dict[str, UpstreamResult | Exception] = {}
        collect_order = sorted(tasks, key=lambda t: t != WORKER_TYPE_PREFILL)
        try:
            for worker_type in collect_order:
                try:
                    results[worker_type] = await tasks[worker_type]
                except Exception as exc:  # noqa: BLE001 — reported by _merge
                    results[worker_type] = exc
                if worker_type == WORKER_TYPE_PREFILL and not _upstream_ok(
                    results[worker_type]
                ):
                    break
        finally:
            # Drop the halves left unawaited by a bail-out or a cancelled request.
            for task in tasks.values():
                task.cancel()
        return results

    async def _merge(
        self,
        results: dict[str, UpstreamResult | Exception],
        request: Request,
        request_id: str,
        slots: dict[str, WorkerSlot],
    ) -> Response:
        """Turn the fan-out results into one response.

        decode's reply becomes the response; prefill's goes to the parser middleware
        through ``request.state``.
        """
        is_pd = WORKER_TYPE_PREFILL in slots
        if is_pd:
            prefill_result = results[WORKER_TYPE_PREFILL]
            if not _upstream_ok(prefill_result):
                # A 4xx/5xx from prefill is not an exception, so both forms count.
                self.pd_prefill_failures += 1
                detail = (
                    f"{type(prefill_result).__name__}: {prefill_result}"
                    if isinstance(prefill_result, Exception)
                    else f"status_code={prefill_result.status_code}"
                )
                logger.error(
                    "[proxy] prefill failed for %s (%s): %s",
                    request_id, slots[WORKER_TYPE_PREFILL], detail,
                )
                # Located by rid: AbortReq has no bootstrap_room field.
                await self._send_worker_abort_request(
                    slots[WORKER_TYPE_DECODE].worker_url,
                    {"rid": request_id, "abort_all": False},
                )
                return JSONResponse(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    content={"error": f"Prefill worker failed: {detail}"},
                )

        primary = WORKER_TYPE_DECODE if is_pd else WORKER_TYPE_REGULAR
        result = results[primary]
        if isinstance(result, Exception):
            if isinstance(result, httpx.RequestError):
                # httpx.TimeoutException derives from RequestError, so timeouts too.
                if is_pd:
                    self.pd_decode_timeouts += 1
                logger.error("[proxy] %s -> %s: %s", request_id, slots[primary], result)
                return JSONResponse(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    content={"error": f"Upstream request failed: {result}"},
                )
            raise result

        if is_pd:
            request.state.prefill_body = results[WORKER_TYPE_PREFILL].content

        return Response(
            content=result.content,
            status_code=result.status_code,
            headers=filter_headers(result.headers),
            media_type=result.headers.get("content-type") or None,
        )

    async def _post_generate(
        self,
        slot: WorkerSlot,
        body_bytes: bytes,
        *,
        method: str,
        headers: dict[str, str],
    ) -> UpstreamResult:
        """Forward one prepared body to one worker's ``/generate`` and read it fully."""
        target_url = f"{slot.worker_url.rstrip('/')}/generate"
        # Shares only the pre-built SSL context; the client itself stays per-request so
        # one dispatch still owns exactly one connection (see ``_SSL_CONTEXT``).
        generate_client = httpx.AsyncClient(
            timeout=self.config.proxy_timeout_seconds, verify=_SSL_CONTEXT
        )
        try:
            worker_response = await generate_client.request(
                method=method,
                url=target_url,
                headers=headers,
                content=body_bytes,
            )
            content = await worker_response.aread()
        finally:
            await generate_client.aclose()
        return UpstreamResult(
            status_code=worker_response.status_code,
            headers=dict(worker_response.headers),
            content=content,
        )

    def _inject(
        self,
        body_bytes: bytes,
        slots: dict[str, WorkerSlot],
        *,
        worker_type: str,
        room: int,
    ) -> bytes:
        """Add the per-target routing fields to one copy of the ``/generate`` body.

        ``routed_dp_rank`` pins the request to ``slot``'s DP group instead of letting
        SGLang load-balance internally. Under PD both halves also get the bootstrap
        triple, and decode the DP group of the prefill holding the cache.

        No-op when the worker has a single DP group and PD is off.
        """
        slot = slots[worker_type]
        is_pd = WORKER_TYPE_PREFILL in slots
        dp_size = self.worker_dp_size.get(worker_type, {}).get(slot.worker_url, 1)
        if dp_size <= 1 and not is_pd:
            return body_bytes
        try:
            payload = json.loads(body_bytes)
        except (TypeError, ValueError) as exc:
            if is_pd:
                # Without the triple the request would hang, so fail instead of warn.
                raise RuntimeError(
                    "cannot parse upstream body to inject PD bootstrap fields "
                    f"(worker={slot.worker_url})"
                ) from exc
            logger.warning(
                "[route] cannot parse upstream body to set routed_dp_rank (worker=%s)", slot.worker_url
            )
            return body_bytes
        if dp_size > 1:
            payload["routed_dp_rank"] = slot.dp_rank
        if is_pd:
            prefill = slots[WORKER_TYPE_PREFILL]
            payload["bootstrap_host"] = urlsplit(prefill.worker_url).hostname
            payload["bootstrap_port"] = self.prefill_bootstrap_port[prefill.worker_url]
            payload["bootstrap_room"] = room
            prefill_dp_size = self.worker_dp_size[WORKER_TYPE_PREFILL][prefill.worker_url]
            if worker_type == WORKER_TYPE_DECODE and prefill_dp_size > 1:
                # Hint telling decode which prefill DP worker holds the KV cache.
                payload["disagg_prefill_dp_rank"] = prefill.dp_rank
            if worker_type == WORKER_TYPE_DECODE and payload.get("return_routed_experts"):
                # Prefill covers [start_len, prompt_len); decode only needs
                # [prompt_len, seqlen - 1). Asking decode for the wider range
                # transports every prompt row a second time, only for the parser
                # to discard that copy.
                payload["routed_experts_start_len"] = len(payload["input_ids"])
        return json.dumps(payload).encode("utf-8")

    # -------------------------------------------------------------------------
    # Server lifecycle
    # -------------------------------------------------------------------------

    def start_background(self, host: str, port: int) -> None:
        """Start uvicorn in a background thread."""
        self._server = uvicorn.Server(
            uvicorn.Config(self.app, host=host, port=port, log_level="warning")
        )
        self._startup_error = None

        def _run() -> None:
            """Run uvicorn and capture startup/runtime failures."""
            try:
                self._server.run()
            except BaseException as exc:
                self._startup_error = exc
                logger.exception("[startup] background server failed")

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        self._wait_ready()

    def _wait_ready(self, timeout: float = 30.0) -> None:
        """Block until the background uvicorn server is ready."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._server is not None and self._server.started:
                return
            if (exc := self._startup_error) is not None:
                self._server = self._thread = None
                raise RuntimeError(f"Router startup failed: {exc}") from exc
            if self._thread is not None and not self._thread.is_alive():
                self._server = self._thread = None
                raise RuntimeError("Router exited before becoming ready")
            time.sleep(0.05)

        self._server = self._thread = None
        raise RuntimeError(f"Router did not become ready within {timeout}s")

    async def shutdown(self) -> None:
        """Stop uvicorn and close the upstream HTTP client."""
        await self._control_client.aclose()
        if self._server is not None:
            self._server.should_exit = True
        _join_timeout = 10
        if self._thread is not None and self._thread.is_alive():
            await asyncio.to_thread(self._thread.join, _join_timeout)
            if self._thread.is_alive():
                logger.warning("[shutdown] router thread did not stop within %ss", _join_timeout)
        self._server = self._thread = None
        for callback in self._shutdown_callbacks:
            try:
                callback()
            except Exception as e:
                logger.warning("[shutdown] shutdown callback %r failed: %s", callback, e)
