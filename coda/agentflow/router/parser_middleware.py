"""HTTP middleware for adapting client requests to SGLang and saving trajectories."""

import asyncio
import http
import json
import logging
import traceback
from collections import Counter, defaultdict
from typing import Any, Callable, cast

from fastapi import status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from coda.agentflow.tokenizer_manager import DEEPSEEK_OFFICIAL_ENCODER_MODULES
from coda.agentflow.trajectory_store import TrajectoryStore
from coda.agentflow.utils import CONTEXT_LENGTH_EXCEEDED, build_request_id
from .parser import TrajectoryParser, TurnInputContext
from .protocols import PROTOCOL_ADAPTERS, ProtocolAdapter

logger = logging.getLogger(__name__)
ParserConfig = str | bool | dict[str, Any] | None

_ADAPTERS: tuple[ProtocolAdapter, ...] = PROTOCOL_ADAPTERS


class ParserMiddleware(BaseHTTPMiddleware):
    """Adapt supported client protocols and write token-aligned trajectories."""

    _filter_headers: Callable[[dict[str, str]], dict[str, str]]

    def __init__(
        self,
        app: ASGIApp,
        config: Any,
        trajectory_store: TrajectoryStore | None = None,
        tokenizer_manager: Any = None,
        accumulate_reasoning: bool = False,
        r3_enabled: bool = False,
        reasoning_parser: ParserConfig = None,
        tool_call_parser: ParserConfig = None,
        inflight_requests: Any = None,
        ds_configs: dict[int, Any] | None = None,
    ):
        super().__init__(app)
        from coda.agentflow.router.router import filter_headers
        self._filter_headers = filter_headers
        self.train_temperature = config.trainer.temperature
        self.eval_temperature = config.rollout.eval.temperature
        self.r3_enabled = r3_enabled
        self.inflight_requests = inflight_requests
        self.ds_configs = ds_configs
        self._trajectory_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._trajectory_lock_refs: Counter[str] = Counter()
        self.parser = TrajectoryParser(
            trajectory_store=trajectory_store,
            tokenizer_manager=tokenizer_manager,
            accumulate_reasoning=accumulate_reasoning,
            r3_enabled=r3_enabled,
            reasoning_parser=reasoning_parser,
            tool_call_parser=tool_call_parser,
        )

    async def dispatch(self, request: Request, call_next):
        """Intercept a recognized-protocol request; pass all others through."""
        path = request.url.path
        adapter: ProtocolAdapter | None = None
        for candidate in _ADAPTERS:
            if candidate.parse_route(path) is not None:
                adapter = candidate
                break
        if adapter is None:
            return await call_next(request)

        trajectory_id, attempt_id = adapter.parse_route(path)
        request_id = build_request_id(trajectory_id, attempt_id)
        normalized_headers = adapter.normalize_headers(request)

        # 1. Parse and validate request
        try:
            request_body = cast(dict[str, Any], json.loads(await request.body()))
            messages, tools = adapter.parse_request(request_body)
            if not messages:
                raise ValueError("messages is empty")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.error("Failed to parse request: %s\n%s", e, traceback.format_exc())
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": str(e)},
            )

        trajectory = self.parser.get_trajectory(trajectory_id, attempt_id)
        assert trajectory is not None, f"trajectory {trajectory_id} not found"
        request_kind = normalized_headers.get("request_kind")

        # Serialize concurrent requests (e.g. litellm retries) for the same trajectory
        # to prevent build_turn_input/prune_think and update_trajectory from racing.
        # Increment ref count and fetch the lock before the first await so both steps
        # are atomic in asyncio's single-threaded model.
        self._trajectory_lock_refs[trajectory_id] += 1
        if self.inflight_requests is not None:
            self.inflight_requests.add(request_id)
        lock = self._trajectory_locks[trajectory_id]
        try:
            async with lock:
                turn_ctx = await self.parser.build_turn_input(
                    trajectory, messages=messages, tools=tools, request_kind=request_kind,
                    mask_tool_call_args=adapter.rewrites_history_tool_args,
                )
                # Enforce max_response_len_per_trajectory: compare response-area length against the budget.
                # response_area = everything after the initial prompt (LLM replies + tool responses).
                seg = (
                    trajectory.segments[trajectory.active_segment_id]
                    if trajectory.segments and not turn_ctx.start_new_segment
                    else None
                )
                if seg and seg.triplets:
                    t0 = seg.triplets[0]
                    # Triplet token offsets index the trajectory-wide token array, while
                    # turn_ctx.input_ids is local to the active segment. Convert the
                    # first turn to a segment-local length before removing its response.
                    initial_prompt_len = (t0.token_end - t0.token_start) - (t0.logprob_end - t0.logprob_start)
                    response_area = len(turn_ctx.input_ids) - initial_prompt_len
                else:
                    response_area = 0
                ds_config = self.ds_configs[trajectory.ds_index]
                max_response_len = int(ds_config.max_response_len_per_trajectory)
                context_length = int(ds_config.get("agent", {}).get("context_length", 0))
                # Generation is capped by the tighter of the two trajectory budgets.
                # The message reports both so logs show which one bound it.
                budget = max_response_len - response_area
                if context_length > 0:
                    budget = min(budget, context_length - len(turn_ctx.input_ids))
                budget_exhausted = (
                    "generation budget exhausted: "
                    f"max_response_len_per_trajectory={max_response_len} "
                    f"(response_area={response_area}), "
                    f"context_length={context_length} "
                    f"(input_len={len(turn_ctx.input_ids)})"
                )
                if budget <= 0:
                    return adapter.build_error_response(
                        request_body,
                        error_type=CONTEXT_LENGTH_EXCEEDED,
                        message=budget_exhausted,
                        status_code=http.HTTPStatus.BAD_REQUEST,
                    )
                # The client's own cap applies only when tighter than the budget.
                request_max_new_tokens = request_body.get("max_new_tokens")
                max_new_tokens = (
                    request_max_new_tokens
                    if request_max_new_tokens is not None and 0 < request_max_new_tokens < budget
                    else budget
                )
                # A "length" finish is only a failure when the router shrank the cap
                # below what the client asked for; getting exactly the requested
                # length back is a normal stop.
                budget_capped = not request_max_new_tokens or max_new_tokens < request_max_new_tokens

                request.state.upstream_body = self._build_generate_body(
                    turn_ctx.input_ids,
                    completion_params=ds_config.get("completion_params", {}),
                    request_id=request_id,
                    max_new_tokens=max_new_tokens,
                    routed_experts_start_len=turn_ctx.routed_experts_start_len,
                    is_eval=trajectory.is_eval,
                )

                # 2. Forward to LLM worker and parse response
                response = await call_next(request)
                response_body = await self._read_response_body(response)
                parsed_response = self._parse_generate_response(
                    response, response_body, turn_ctx, trajectory_id
                )
                if isinstance(parsed_response, Response):
                    return parsed_response
                payload, response_json, assistant_message, finish_reason_override = parsed_response
                finish_reason = self._finish_reason(
                    response_json, finish_reason_override
                )

                final_content = adapter.build_response(
                    request_body,
                    assistant_message,
                    finish_reason,
                    self._usage(response_json),
                    path=path,
                    upstream_response=response_json,
                )

                # 3. Persist trajectory state.
                # Write errors are logged but swallowed to preserve the response.
                try:
                    self.parser.update_trajectory(
                        trajectory=trajectory,
                        turn_ctx=turn_ctx,
                        payload=payload,
                        assistant_message=assistant_message,
                    )
                except Exception as e:
                    logger.critical(
                        "[update] trajectory write failed for %s, state may be inconsistent: %s",
                        trajectory_id, e, exc_info=True,
                    )
                if finish_reason == "length" and budget_capped:
                    return adapter.build_error_response(
                        request_body,
                        error_type=CONTEXT_LENGTH_EXCEEDED,
                        message=budget_exhausted,
                        status_code=http.HTTPStatus.BAD_REQUEST,
                    )
        finally:
            if self.inflight_requests is not None:
                # Always remove this request from inflight, even if handling is cancelled.
                # Otherwise abort() may wait for all parser requests until timeout.
                await asyncio.shield(self.inflight_requests.delete(request_id))
            self._release_trajectory_lock(trajectory_id)

        if request_body.get("stream") is True:
            return adapter.build_streaming_response(final_content)

        return Response(
            content=json.dumps(final_content).encode("utf-8"),
            status_code=response.status_code,
            headers={"content-type": "application/json"},
            media_type="application/json",
        )

    @staticmethod
    def _finish_reason(
        response_json: dict[str, Any], finish_reason_override: str | None
    ) -> str:
        if finish_reason_override:
            return finish_reason_override
        meta = response_json.get("meta_info") or {}
        finish_reason_meta = meta.get("finish_reason") or {}
        if isinstance(finish_reason_meta, dict):
            return finish_reason_meta.get("type", "stop")
        return "stop"

    @staticmethod
    def _usage(response_json: dict[str, Any]) -> dict[str, Any]:
        meta = response_json.get("meta_info") or {}
        usage = response_json.get("usage") or {}
        prompt_tokens = meta.get("prompt_tokens", usage.get("prompt_tokens", 0))
        completion_tokens = meta.get(
            "completion_tokens", usage.get("completion_tokens", 0)
        )
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def _release_trajectory_lock(self, trajectory_id: str) -> None:
        """Decrement the per-trajectory lock ref count and remove the lock when no longer needed."""
        self._trajectory_lock_refs[trajectory_id] -= 1
        if self._trajectory_lock_refs[trajectory_id] <= 0:
            self._trajectory_locks.pop(trajectory_id, None)
            del self._trajectory_lock_refs[trajectory_id]

    def _build_generate_body(
        self,
        input_ids: list[int],
        *,
        completion_params: dict[str, Any] | None = None,
        request_id: str,
        max_new_tokens: int,
        routed_experts_start_len: int = 0,
        is_eval: bool = False,
    ) -> bytes:
        sampling_params = dict(completion_params or {})
        sampling_params["temperature"] = (
            self.eval_temperature
            if is_eval and self.eval_temperature is not None
            else self.train_temperature
        )
        if self.parser.model_family in DEEPSEEK_OFFICIAL_ENCODER_MODULES:
            # DeepSeek-V4/V4.1 mark their think and DSML tool-call tags as special tokens,
            # so the detokenizer must keep them for the reasoning / tool-call parsers to
            # see them. no_stop_trim stays off so SGLang still strips the trailing EOS.
            sampling_params["skip_special_tokens"] = False

        # The Router owns the remaining trajectory budget regardless of config.
        sampling_params["max_new_tokens"] = max_new_tokens
        payload: dict[str, Any] = {
            "input_ids": input_ids,
            "rid": request_id,
            "return_logprob": True,
            "sampling_params": sampling_params,
        }

        if self.r3_enabled:
            payload["return_routed_experts"] = True
            payload["routed_experts_start_len"] = routed_experts_start_len

        return json.dumps(payload).encode("utf-8")

    async def _read_response_body(self, response: Response) -> bytes:
        # Buffer all chunks from the LLM worker's StreamingResponse. Currently we wait for
        # the full body before returning to the agent; isolating chunk collection here
        # prepares for future streaming support (tee the iterator to forward tokens in real time).
        if hasattr(response, "body_iterator"):
            chunks: list[bytes] = []
            async for chunk in response.body_iterator:
                chunks.append(chunk)
            return b"".join(chunks)
        return response.body

    def _parse_generate_response(
        self,
        response: Response,
        body: bytes,
        turn_ctx: TurnInputContext,
        trajectory_id: str,
    ) -> tuple[Any, dict[str, Any], dict[str, Any], str | None] | Response:
        """Decode LLM worker response and build the assistant message.

        Returns (payload, response_json, assistant_message, finish_reason_override) on success,
        or a raw Response passthrough on error/non-2xx.
        """
        try:
            response_json = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            logger.warning("[parse] upstream payload non-JSON for %s: %s", trajectory_id, body[:200])
            return Response(content=body, status_code=response.status_code,
                            headers=self._filter_headers(dict(response.headers)),
                            media_type=response.media_type)

        if response.status_code >= status.HTTP_400_BAD_REQUEST or (
            "error" in response_json and not response_json.get("text")
        ):
            return Response(content=body, status_code=response.status_code,
                            headers=self._filter_headers(dict(response.headers)),
                            media_type=response.media_type)

        # Abort passthrough: covers both waiting-queue and running-batch aborts triggered by
        # /abort_request. In both cases SGLang returns HTTP 200 with finish_reason.type == "abort"
        # (running-batch: FINISH_ABORT(status_code=None), waiting-queue: _handle_abort_req path).
        # Skip writing to TrajectoryStore to avoid partial/empty data.
        _finish_reason = (response_json.get("meta_info") or {}).get("finish_reason") or {}
        if isinstance(_finish_reason, dict) and _finish_reason.get("type") == "abort":
            logger.warning(
                "[parse] upstream aborted request for %s: %s",
                trajectory_id, _finish_reason.get("message", ""),
            )
            return Response(content=body, status_code=response.status_code,
                            headers=self._filter_headers(dict(response.headers)),
                            media_type=response.media_type)

        payload, assistant_message, finish_reason_override = self.parser.build_assistant_message(
            response_json, turn_ctx.routed_experts_start_len, turn_ctx.tools
        )
        if response_json.get("choices"):
            response_json["choices"][0]["message"] = assistant_message
            if finish_reason_override == "tool_calls":
                response_json["choices"][0]["finish_reason"] = "tool_calls"

        return payload, response_json, assistant_message, finish_reason_override
