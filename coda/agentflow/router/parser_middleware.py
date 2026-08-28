"""HTTP Middleware for proxying Chat Completion requests to an LLM worker and saving Trajectories."""

import asyncio
import http
import json
import logging
import re
import time
import traceback
from collections import Counter, defaultdict
from typing import Any, Callable, cast

from fastapi import status
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from coda.agentflow.tokenizer_manager import DEEPSEEK_V4_MODEL_FAMILY
from coda.agentflow.trajectory_store import TrajectoryStore
from coda.agentflow.utils import CONTEXT_LENGTH_EXCEEDED, build_request_id
from .parser import TrajectoryParser, TurnInputContext

logger = logging.getLogger(__name__)
ParserConfig = str | bool | dict[str, Any] | None

_CHAT_COMPLETIONS_ROUTE_RE = re.compile(
    r"^/(?P<trajectory_id>[^/]+)/(?P<attempt_id>\d+)/v1/chat/completions$"
)

# OpenAI chat/completions field → SGLang sampling_params field.
# Mirrors ChatCompletionRequest.to_sampling_params() in SGLang.
# None values are dropped to avoid overriding server-side defaults.
_SAMPLING_PARAM_MAP: tuple[tuple[str, str], ...] = (
    # ── Standard OpenAI ──────────────────────────────────────────────────
    ("temperature",        "temperature"),
    ("top_p",              "top_p"),
    ("presence_penalty",   "presence_penalty"),
    ("frequency_penalty",  "frequency_penalty"),
    ("stop",               "stop"),
    ("seed",               "sampling_seed"),
    ("logit_bias",         "logit_bias"),
    # ── SGLang extensions (present in ChatCompletionRequest) ──────────────
    ("top_k",              "top_k"),
    ("min_p",              "min_p"),
    ("min_tokens",         "min_new_tokens"),
    ("repetition_penalty", "repetition_penalty"),
    ("stop_token_ids",     "stop_token_ids"),
    ("stop_regex",         "stop_regex"),
    ("no_stop_trim",       "no_stop_trim"),
    ("ignore_eos",         "ignore_eos"),
    ("skip_special_tokens", "skip_special_tokens"),
    ("regex",              "regex"),
    ("ebnf",               "ebnf"),
    ("custom_params",      "custom_params"),
)

# OpenAI chat/completions field → SGLang GenerateReqInput top-level field.
# These live outside sampling_params in the /generate request body.
# Uses tuple-of-tuples (same as _SAMPLING_PARAM_MAP) to support field renames.
_GENERATE_FIELD_MAP: tuple[tuple[str, str], ...] = (
    ("lora_path",              "lora_path"),
    ("logprob_start_len",      "logprob_start_len"),
    ("top_logprobs",           "top_logprobs_num"),    # OpenAI name → SGLang name
    ("return_hidden_states",   "return_hidden_states"),
    ("custom_logit_processor", "custom_logit_processor"),
    ("priority",               "priority"),
    ("extra_key",              "extra_key"),
    ("data_parallel_rank",     "data_parallel_rank"),
    ("bootstrap_host",         "bootstrap_host"),
    ("bootstrap_port",         "bootstrap_port"),
    ("bootstrap_room",         "bootstrap_room"),
)


class ParserMiddleware(BaseHTTPMiddleware):
    """Intercept OpenAI chat-completions requests and write training trajectories."""

    _filter_headers: Callable[[dict[str, str]], dict[str, str]]

    def __init__(
        self,
        app: ASGIApp,
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
        """Intercept chat-completions requests; pass all others through."""
        match = _CHAT_COMPLETIONS_ROUTE_RE.fullmatch(request.url.path)
        if match is None:
            return await call_next(request)
        trajectory_id, attempt_id = match.group("trajectory_id"), int(match.group("attempt_id"))
        request_id = build_request_id(trajectory_id, attempt_id)

        # 1. Parse and validate request
        try:
            request_body = cast(dict[str, Any], json.loads(await request.body()))
            messages = cast(list[dict[str, Any]], request_body["messages"])
            tools = request_body.get("tools")
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
        # Distinguishes a context-compaction prefix mismatch from a subagent fork;
        request_kind = request.headers.get("request_kind")

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
                remaining_response = max_response_len - response_area
                remaining_context = (
                    context_length - len(turn_ctx.input_ids) if context_length > 0 else None
                )
                remaining = (
                    min(remaining_response, remaining_context)
                    if remaining_context is not None
                    else remaining_response
                )
                if remaining <= 0:
                    if remaining_context is not None and remaining_context <= remaining_response:
                        message = (
                            f"context_length={context_length} exhausted "
                            f"(input_len={len(turn_ctx.input_ids)})"
                        )
                    else:
                        message = (
                            "max_response_len_per_trajectory="
                            f"{max_response_len} exhausted "
                            f"(response_area={response_area})"
                        )
                    return JSONResponse(
                        {"error": {"message": message, "type": CONTEXT_LENGTH_EXCEEDED}},
                        status_code=http.HTTPStatus.BAD_REQUEST,
                    )
                req_max = (request_body.get("max_tokens")
                           or request_body.get("max_completion_tokens")
                           or request_body.get("max_new_tokens") or 0)
                if req_max > remaining:
                    request_body["max_tokens"] = remaining
                    request_body.pop("max_completion_tokens", None)
                    request_body.pop("max_new_tokens", None)

                request.state.upstream_body = self._build_generate_body(
                    request_body,
                    turn_ctx.input_ids,
                    request_id=build_request_id(trajectory_id, attempt_id),
                    routed_experts_start_len=turn_ctx.routed_experts_start_len,
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

                final_content = self._to_openai_response(
                    request_body,
                    response_json,
                    assistant_message,
                    payload.weight_version,
                    finish_reason_override,
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
        finally:
            if self.inflight_requests is not None:
                # Always remove this request from inflight, even if handling is cancelled.
                # Otherwise abort() may wait for all parser requests until timeout.
                await asyncio.shield(self.inflight_requests.delete(request_id))
            self._release_trajectory_lock(trajectory_id)

        if request_body.get("stream") is True:
            return self._to_streaming_response(final_content)

        return Response(
            content=json.dumps(final_content).encode("utf-8"),
            status_code=response.status_code,
            headers={"content-type": "application/json"},
            media_type="application/json",
        )

    @staticmethod
    def _to_streaming_response(response: dict[str, Any]) -> StreamingResponse:
        """Wrap one complete chat completion in OpenAI-compatible SSE events."""
        choice = response["choices"][0]
        message = choice["message"]
        delta = {
            key: message[key]
            for key in ("content", "reasoning_content")
            if message.get(key) is not None
        }
        if message.get("tool_calls"):
            delta["tool_calls"] = [
                {**tool_call, "index": index}
                for index, tool_call in enumerate(message["tool_calls"])
            ]

        chunk = {
            "id": response.get("id", ""),
            "object": "chat.completion.chunk",
            "created": response.get("created", int(time.time())),
            "model": response.get("model", "default"),
        }
        events = [
            {
                **chunk,
                "choices": [
                    {
                        "index": choice.get("index", 0),
                        "delta": delta,
                        "finish_reason": None,
                    }
                ],
            },
            {
                **chunk,
                "choices": [
                    {
                        "index": choice.get("index", 0),
                        "delta": {},
                        "finish_reason": choice.get("finish_reason"),
                    }
                ],
                "usage": response.get("usage"),
            },
        ]
        body = [f"data: {json.dumps(event)}\n\n" for event in events]
        body.append("data: [DONE]\n\n")
        return StreamingResponse(body, media_type="text/event-stream")

    def _release_trajectory_lock(self, trajectory_id: str) -> None:
        """Decrement the per-trajectory lock ref count and remove the lock when no longer needed."""
        self._trajectory_lock_refs[trajectory_id] -= 1
        if self._trajectory_lock_refs[trajectory_id] <= 0:
            self._trajectory_locks.pop(trajectory_id, None)
            del self._trajectory_lock_refs[trajectory_id]

    def _build_generate_body(
        self,
        body: dict[str, Any],
        input_ids: list[int],
        *,
        request_id: str,
        routed_experts_start_len: int = 0,
    ) -> bytes:
        # Merge sampling params: explicit sampling_params dict takes precedence,
        # then top-level OpenAI fields as fallback.
        sampling_params = dict(body.get("sampling_params") or {})
        for chat_field, gen_field in _SAMPLING_PARAM_MAP:
            if chat_field in body and body[chat_field] is not None:
                sampling_params.setdefault(gen_field, body[chat_field])
        if self.parser.model_family == DEEPSEEK_V4_MODEL_FAMILY:
            # DeepSeek-V4 marks its think and DSML tool-call tags as special tokens, so the
            # detokenizer must keep them for the reasoning / tool-call parsers to see them.
            # no_stop_trim stays off so SGLang still strips the trailing EOS.
            sampling_params["skip_special_tokens"] = False

        payload: dict[str, Any] = {
            "input_ids": input_ids,
            "rid": request_id,
            "return_logprob": True,
        }

        # max_completion_tokens (preferred) / max_tokens (deprecated) / max_new_tokens (native)
        max_new_tokens = (
            body.get("max_new_tokens")
            or body.get("max_completion_tokens")
            or body.get("max_tokens")
        )
        if max_new_tokens is not None:
            sampling_params["max_new_tokens"] = max_new_tokens

        if sampling_params:
            payload["sampling_params"] = sampling_params

        for chat_field, gen_field in _GENERATE_FIELD_MAP:
            if chat_field in body and body[chat_field] is not None:
                payload[gen_field] = body[chat_field]

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

    def _to_openai_response(
        self,
        request_body: dict[str, Any],
        worker_json: dict[str, Any],
        assistant_message: dict[str, Any],
        weight_version: int,
        finish_reason_override: str | None = None,
    ) -> dict[str, Any]:
        if "choices" in worker_json:
            return worker_json
        meta = worker_json.get("meta_info", {})
        finish_reason_meta = meta.get("finish_reason") or {}
        finish_reason = (
            finish_reason_meta.get("type", "stop")
            if isinstance(finish_reason_meta, dict)
            else "stop"
        )
        if finish_reason_override:
            finish_reason = finish_reason_override
        matched_stop = (
            finish_reason_meta.get("matched") if isinstance(finish_reason_meta, dict) else None
        )
        if finish_reason_override == "tool_calls":
            matched_stop = None
        prompt_tokens = meta.get("prompt_tokens", 0)
        completion_tokens = meta.get("completion_tokens", 0)
        return {
            "id": meta.get("id", ""),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request_body.get("model", "default"),
            "choices": [
                {
                    "index": 0,
                    "message": assistant_message,
                    "logprobs": None,
                    "finish_reason": finish_reason,
                    "matched_stop": matched_stop,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "metadata": {"weight_version": weight_version},
        }
