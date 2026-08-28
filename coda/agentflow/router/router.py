"""Router for load balancing LLM requests across SGLang workers."""

import asyncio
import inspect
import logging
import threading
import time
from collections import Counter
from typing import Any

import httpx
import uvicorn
from fastapi import APIRouter, FastAPI, Request, status
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel as PydanticModel

from coda.agentflow.router import resolve_middleware_chain
from coda.agentflow.utils import build_request_id


class WorkerPayload(PydanticModel):
    """Worker management payload."""
    worker_url: str


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
    """Reverse proxy with sticky + least-loaded routing across SGLang workers."""

    def __init__(
        self,
        config: Any,
        *,
        middleware_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize routing state, HTTP client, and FastAPI app."""
        self.config = config
        self.rollout_worker_load_threshold = config.rollout_worker_load_threshold
        self.accumulate_reasoning = config.accumulate_reasoning

        # URL -> in-flight request count
        self.worker_request_counts: dict[str, int] = {}
        # Workers quarantined from the routing pool
        self.dead_workers: set[str] = set()
        # request_id (trajectory_id#attempt_id) -> target worker URL for sticky routing
        self.session_id_to_worker: dict[str, str] = {}
        self.inflight_requests = InflightRequests()

        self._max_connections = int(config.max_connections)
        # Dedicated control client + semaphore for worker-abort requests, off the
        # forward path so an abort storm can't inflate it and pin the router loop.
        self._generate_sem = asyncio.Semaphore(self._max_connections)
        self._control_sem = asyncio.Semaphore(self._max_connections)
        self._control_client = httpx.AsyncClient(
            timeout=config.proxy_timeout_seconds,
            limits=httpx.Limits(max_connections=self._max_connections),
        )
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._startup_error: BaseException | None = None
        self.app = FastAPI()
        self._setup_routes()
        middleware_context = dict(middleware_kwargs or {})
        middleware_context["inflight_requests"] = self.inflight_requests
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
        session_router.post(
            "/v1/chat/completions",
            response_model=None,
        )(self.proxy)

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

    def _select_worker(self, request_id: str) -> str:
        """Pick a healthy worker and increment its in-flight count."""
        if request_id in self.session_id_to_worker:
            target = self.session_id_to_worker[request_id]
            load = self.worker_request_counts.get(target, 0)
            if target not in self.dead_workers and load <= self.rollout_worker_load_threshold:
                logger.info("[route] sticky %s -> %s (load=%d)", request_id, target, load)
                self.worker_request_counts[target] = load + 1
                return target

        available = [w for w in self.worker_request_counts if w not in self.dead_workers]
        if not available:
            raise RuntimeError("No healthy workers available")
        target = min(available, key=self.worker_request_counts.__getitem__)

        self.session_id_to_worker[request_id] = target
        logger.info("[route] new %s -> %s", request_id, target)
        self.worker_request_counts[target] += 1
        return target

    def _finish_worker(self, worker_url: str) -> None:
        """Decrement worker in-flight count after request completion."""
        if worker_url not in self.worker_request_counts:
            logger.error("[worker] _finish_worker: unknown worker %s — skipping decrement", worker_url)
            return
        self.worker_request_counts[worker_url] = max(0, self.worker_request_counts[worker_url] - 1)

    # -------------------------------------------------------------------------
    # Management API
    # -------------------------------------------------------------------------

    async def add_worker(self, payload: WorkerPayload) -> JSONResponse:
        """Register a worker URL for future routing."""
        worker_url = payload.worker_url
        if worker_url in self.worker_request_counts:
            return JSONResponse(
                status_code=status.HTTP_409_CONFLICT,
                content={"error": "Worker already exists", "worker": worker_url},
            )
        self.worker_request_counts[worker_url] = 0
        logger.info("[worker] added %s", worker_url)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "success", "worker": worker_url},
        )

    async def list_workers(self) -> JSONResponse:
        """Return active, dead, and per-worker load state."""
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "active_workers": [w for w in self.worker_request_counts if w not in self.dead_workers],
                "dead_workers": list(self.dead_workers),
                "load_stats": dict(self.worker_request_counts),
            },
        )

    async def exclude_worker(self, payload: WorkerPayload) -> JSONResponse:
        """Remove a worker from the active routing pool."""
        worker_url = payload.worker_url
        if worker_url not in self.worker_request_counts:
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
        logger.info("[worker] excluded %s", worker_url)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "success", "worker": worker_url},
        )

    async def include_worker(self, payload: WorkerPayload) -> JSONResponse:
        """Restore an excluded worker to the active pool."""
        worker_url = payload.worker_url
        if worker_url not in self.worker_request_counts:
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
        logger.info("[worker] included %s", worker_url)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "success", "worker": worker_url},
        )

    async def release_session(self, trajectory_id: str, attempt_id: int) -> JSONResponse:
        """Release sticky routing state for one trajectory attempt."""
        request_id = build_request_id(trajectory_id, attempt_id)
        released = self.session_id_to_worker.pop(request_id, None) is not None
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

    async def abort_session(self, trajectory_id: str, attempt_id: int) -> JSONResponse:
        """Abort the worker request pinned to one failed trajectory attempt."""
        request_id = build_request_id(trajectory_id, attempt_id)
        worker_url = self.session_id_to_worker.get(request_id)
        if worker_url is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "error": "No active session found",
                    "trajectory_id": trajectory_id,
                    "attempt_id": attempt_id,
                },
            )
        failure = await self._send_worker_abort_request(
            worker_url,
            {"rid": request_id, "abort_all": False},
        )
        if failure:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "failed",
                    "trajectory_id": trajectory_id,
                    "attempt_id": attempt_id,
                    "worker": worker_url,
                    "error": failure,
                },
            )
        self.session_id_to_worker.pop(request_id, None)
        logger.info("[session] aborted failed attempt %s (worker=%s)", request_id, worker_url)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "trajectory_id": trajectory_id,
                "attempt_id": attempt_id,
            },
        )

    async def abort_all_workers(self) -> JSONResponse:
        """Abort all active workers and clear sticky session state."""
        active_workers = [
            w for w in self.worker_request_counts if w not in self.dead_workers
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

        sessions_cleared = len(self.session_id_to_worker)
        self.session_id_to_worker.clear()

        for w in self.worker_request_counts:
            self.worker_request_counts[w] = 0

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

    async def proxy(
        self,
        trajectory_id: str,
        attempt_id: int,
        request: Request,
    ) -> Response:
        """Proxy a prepared chat request to SGLang /generate."""
        request_id = build_request_id(trajectory_id, attempt_id)
        body_bytes = getattr(request.state, "upstream_body", None)
        if body_bytes is None:
            logger.error("[proxy] upstream_body not set for %s", request_id)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": "Router failed to prepare upstream request"},
            )

        headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
        worker_url = None
        async with self._generate_sem:
            try:
                worker_url = self._select_worker(request_id)
                target_url = f"{worker_url.rstrip('/')}/generate"

                generate_client = httpx.AsyncClient(timeout=self.config.proxy_timeout_seconds)
                try:
                    worker_response = await generate_client.request(
                        method=request.method,
                        url=target_url,
                        headers=headers,
                        content=body_bytes,
                    )
                    content = await worker_response.aread()
                finally:
                    await generate_client.aclose()
                filtered_headers = filter_headers(dict(worker_response.headers))
                return Response(
                    content=content,
                    status_code=worker_response.status_code,
                    headers=filtered_headers,
                    media_type=worker_response.headers.get("content-type") or None,
                )
            except RuntimeError as exc:
                return JSONResponse(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    content={"error": str(exc)},
                )
            except httpx.RequestError as exc:
                logger.error("[proxy] %s -> %s: %s", request_id, worker_url, exc)
                return JSONResponse(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    content={"error": f"Upstream request failed: {exc}"},
                )
            finally:
                if worker_url is not None:
                    self._finish_worker(worker_url)

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
