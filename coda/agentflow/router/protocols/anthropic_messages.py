"""Anthropic Messages protocol adapter."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
import uuid
from typing import Any

from fastapi.responses import JSONResponse, StreamingResponse

from .base import ProtocolAdapter, _route_re

_TOTAL_TOKENS_REMINDER_RE = re.compile(
    r"^\s*<total_tokens>(?:\d+|Infinite) tokens left</total_tokens>\s*$"
)

_SYSTEM_REMINDER_PREFIX = "<system-reminder>\n"
_SYSTEM_REMINDER_SUFFIX = "\n</system-reminder>\n"

logger = logging.getLogger(__name__)

_CLAUDE_CODE_AGENT_ID_HEADER = "x-claude-code-agent-id"


def _copy_headers(headers: Any) -> dict[str, str]:
    return {str(k).lower(): str(v) for k, v in (headers or {}).items()}


def flatten_content(content: Any) -> str:
    """Convert Anthropic content blocks into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(flatten_content(x.get("text", "") if isinstance(x, dict) else x) for x in content)
    return str(content)


def _to_openai_tool(name: str, description: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": schema}}


def fold_mid_list_system_messages(
    messages: list[dict[str, Any]],
    adjacent_tool_pattern: Any = None,
) -> list[dict[str, Any]]:
    """Fold non-leading ``role=="system"`` messages into a neighboring message.

    Claude Code (mid-conversation-system beta) appends per-turn ``system``
    messages. Chat templates expect system text only at index 0, so each such
    message is wrapped in a ``<system-reminder>`` block and folded into the
    nearest preceding user message (or prepended to the nearest following one
    when there is no earlier user turn).

    When the system text is adjacent to a tool result — the shape of Claude
    Code's per-turn trailing reminders (occasional "available agent types /
    skills" refreshes) — the tool message is preferred as the fold target:
    the reminder lands on the newest tool result (itself new this turn) and
    never rewrites an older user turn. ``adjacent_tool_pattern`` recognizes
    the pure ``<total_tokens>...`` counter, which is DROPPED outright (see
    the loop body). Does not mutate the input list.
    """
    if not messages:
        return messages
    system_indices = [
        i for i, m in enumerate(messages)
        if i > 0 and isinstance(m, dict) and m.get("role") == "system"
    ]
    if not system_indices:
        return messages

    result = [dict(m) if isinstance(m, dict) else m for m in messages]
    dropped: set[int] = set()

    def _find_user(rng: range) -> int | None:
        return next(
            (
                j
                for j in rng
                if j not in dropped
                and isinstance(result[j], dict)
                and result[j].get("role") == "user"
            ),
            None,
        )

    for i in system_indices:
        system_text = flatten_content(result[i].get("content"))
        is_tool_reminder = bool(adjacent_tool_pattern and adjacent_tool_pattern.fullmatch(system_text))
        if is_tool_reminder:
            # Ephemeral per-turn token counter: Claude Code rewrites it every
            # turn AND prunes stale copies from history in some builds. Any
            # fold target makes the stored history depend on text that a
            # later request may drop, breaking the prefix comparison. Drop it
            # instead — the rule is stateless per request, so it is robust
            # whether the client keeps or prunes old reminders.
            dropped.add(i)
            continue
        wrapped = _SYSTEM_REMINDER_PREFIX + system_text + _SYSTEM_REMINDER_SUFFIX
        target_idx = None
        prepend = False
        if (
            i > 0
            and result[i - 1].get("role") == "tool"
            and (i + 1 >= len(result) or result[i + 1].get("role") != "user")
        ):
            # Trailing reminder after a completed tool call (next turn is
            # another assistant or end of history): fold onto that tool
            # result instead of rewriting an older user turn.
            target_idx = i - 1
        if target_idx is None:
            target_idx = _find_user(range(i - 1, -1, -1))
        if target_idx is None:
            target_idx = _find_user(range(i + 1, len(result)))
            prepend = True
        if target_idx is None:
            # Synthetic histories and some compaction payloads can end with a
            # system message and no neighboring user turn. Preserve it as a
            # user reminder instead of indexing ``result[None]``.
            result[i] = {"role": "user", "content": wrapped}
            continue
        existing = flatten_content(result[target_idx].get("content"))
        result[target_idx] = {
            **result[target_idx],
            "content": (wrapped + "\n" + existing) if prepend else (existing + "\n" + wrapped),
        }
        dropped.add(i)

    return [m for idx, m in enumerate(result) if idx not in dropped]


def _content_shape(content: Any) -> dict[str, Any]:
    """Summarize content structure without logging content text."""
    if isinstance(content, str):
        return {"type": "str", "length": len(content)}
    if not isinstance(content, list):
        return {"type": type(content).__name__}

    blocks = []
    for block in content:
        if not isinstance(block, dict):
            blocks.append({"type": type(block).__name__})
            continue
        block_summary = {"type": block.get("type", "unknown")}
        for field in ("text", "thinking", "input", "content"):
            value = block.get(field)
            if isinstance(value, str):
                block_summary[f"{field}_length"] = len(value)
            elif isinstance(value, (dict, list)):
                block_summary[f"{field}_type"] = type(value).__name__
        if isinstance(block.get("name"), str):
            block_summary["name"] = block["name"]
        blocks.append(block_summary)
    return {"type": "list", "count": len(content), "blocks": blocks}


def _request_shape(body: dict[str, Any]) -> dict[str, Any]:
    """Build a content-free request shape for correlating request classes."""
    messages = body.get("messages") or []
    return {
        "top_level_keys": sorted(body),
        "model": body.get("model"),
        "max_tokens": body.get("max_tokens"),
        "message_count": len(messages),
        "messages": [
            {
                "role": message.get("role"),
                "content": _content_shape(message.get("content")),
            }
            for message in messages
            if isinstance(message, dict)
        ],
        "system": _content_shape(body.get("system")),
        "tools": [
            {
                "name": tool.get("name"),
                "input_schema_keys": sorted(tool["input_schema"])
                if isinstance(tool.get("input_schema"), dict)
                else [],
            }
            for tool in body.get("tools") or []
            if isinstance(tool, dict)
        ],
        "metadata_keys": sorted(body.get("metadata", {}))
        if isinstance(body.get("metadata"), dict)
        else [],
    }


def _request_marker_hits(body: dict[str, Any]) -> list[str]:
    """Return diagnostic keyword hits without logging request text."""
    values: list[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, str):
            values.append(value.lower())
        elif isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, dict):
            for key in ("text", "thinking", "content"):
                if key in value:
                    collect(value[key])

    collect(body.get("system"))
    collect(body.get("messages"))
    text = "\n".join(values)
    markers = (
        "subagent",
        "sub-agent",
        "compaction",
        "compact",
        "context window",
        "summarize",
        "summary",
    )
    return [marker for marker in markers if marker in text]


class AnthropicMessagesAdapter(ProtocolAdapter):
    """Anthropic Messages API (``/v1/messages``, ``/v1/messages/count_tokens``).

    Serves Claude Code running in a sandbox: the CLI's ``ANTHROPIC_BASE_URL``
    points at the Router's ``/{trajectory_id}/{attempt_id}`` session prefix and
    the SDK appends ``/v1/messages``. Requests arrive in Anthropic wire format
    and are converted to coda's internal OpenAI-shaped messages before the
    parser tokenizes them, so sampled token ids / log probs stay exact.

    Subagent vs compaction routing (the two request classes that break naive
    history tracking):
    - Claude Code Task sub-agents are identified only by the non-empty
      ``x-claude-code-agent-id`` header and reported as
      ``request_kind=collab_spawn``; the parser then serves them on an isolated
      placeholder segment excluded from the training data. A collab_spawn
      request that still prefix-matches the active segment is a normal
      mainline turn — parser topology wins, not this classification.
    - Auto-compaction has no request-kind header. It is inferred by the parser
      from a mainline history-prefix miss and opens a new compact segment.
    """

    routes = ("/v1/messages",)
    _MESSAGES_ROUTE_RE = _route_re(r"v1/messages")

    # Claude Code rewrites old tool_use arguments when echoing history back, so
    # tool-call arguments must be excluded from the prefix comparison.
    rewrites_history_tool_args: bool = True

    def parse_route(self, path: str) -> tuple[str, int] | None:
        """Return (trajectory_id, attempt_id) for a /v1/messages route, or None."""
        match = self.match(path)
        if match is None:
            return None
        return match.group("trajectory_id"), int(match.group("attempt_id"))

    def match(self, path: str) -> "re.Match[str] | None":
        """Match the /v1/messages route."""
        return self._MESSAGES_ROUTE_RE.fullmatch(path)

    def normalize_headers(self, request: Any) -> dict[str, str]:
        """Normalize the Claude Code subagent id header for the parser."""
        request_obj = request if hasattr(request, "headers") else None
        headers = _copy_headers(request.headers if request_obj is not None else request)
        path = ""
        try:
            path = request_obj.url.path if request_obj is not None else str(request.get("path", ""))
        except Exception:
            path = ""
        request_kind = self.classify_request(path, {}, headers)
        if request_kind is None:
            headers.pop("request_kind", None)
        else:
            headers["request_kind"] = request_kind
        return headers

    def classify_request(
        self,
        path: str,
        body: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> str | None:
        """Return parser request_kind from the Claude Code agent-id header.

        Claude Code uses the regular ``/v1/messages`` route for both normal turns
        and auto-compaction. Unknown requests therefore remain ``None`` so the
        parser can use its history-prefix check to identify compaction safely.
        """
        normalized = _copy_headers(headers)
        request_kind = (
            "collab_spawn"
            if normalized.get(_CLAUDE_CODE_AGENT_ID_HEADER, "").strip()
            else None
        )
        evidence = "x-claude-code-agent-id" if request_kind else "no-agent-id-header"

        if request_kind:
            request_class = {
                "collab_spawn": "subagent",
                "compaction": "compact",
            }.get(request_kind, "unknown")
            logger.info(
                "[anthropic_request] path=%s request_class=%s request_kind=%s evidence=%s",
                path,
                request_class,
                request_kind,
                evidence,
            )
        if logger.isEnabledFor(logging.DEBUG):
            shape = _request_shape(body)
            encoded_shape = json.dumps(shape, sort_keys=True, separators=(",", ":"))
            fingerprint = hashlib.sha256(encoded_shape.encode()).hexdigest()[:16]
            logger.debug(
                "[anthropic_request_diagnostics] path=%s fingerprint=%s markers=%s shape=%s",
                path,
                fingerprint,
                _request_marker_hits(body),
                encoded_shape,
            )
        return request_kind

    def is_compact(self, path: str) -> bool:
        """Whether the path is a compact endpoint."""
        return False

    # -- request: Anthropic wire -> internal messages/tools -----------------

    def parse_request(
        self, body: dict[str, Any], headers: Any = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
        """Parse an Anthropic Messages generation body."""
        if "messages" in body and "max_tokens" not in body and "max_new_tokens" not in body:
            raise ValueError("request is missing max_tokens")
        return self._parse_messages_body(body)

    def _parse_messages_body(
        self,
        body: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
        """Convert Anthropic messages and tools to the internal wire shape."""
        # Field renames for the middleware's generate-body builder (same
        # convention as the other adapters): Anthropic "max_tokens" /
        # "stop_sequences" become their OpenAI counterparts. Only set when
        # absent so a client speaking both spellings keeps priority.
        if "max_tokens" in body and "max_new_tokens" not in body:
            try:
                body["max_new_tokens"] = int(body["max_tokens"])
            except (TypeError, ValueError):
                pass
        if "stop_sequences" in body and "stop" not in body:
            body["stop"] = body["stop_sequences"]

        messages: list[dict[str, Any]] = []
        system = body.get("system")
        if system:
            messages.append({"role": "system", "content": flatten_content(system)})

        for message in body.get("messages") or []:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = message.get("content")
            if role == "user":
                messages.extend(self._convert_user_message(content))
            elif role == "assistant":
                messages.append(self._convert_assistant_message(content))
            elif role == "system":
                messages.append({"role": "system", "content": flatten_content(content)})

        messages = fold_mid_list_system_messages(
            messages,
            adjacent_tool_pattern=_TOTAL_TOKENS_REMINDER_RE,
        )
        tools = self._convert_tools(body.get("tools"))
        return messages, tools

    @staticmethod
    def _convert_user_message(content: Any) -> list[dict[str, Any]]:
        if isinstance(content, str):
            return [{"role": "user", "content": content}]
        if not isinstance(content, list):
            return [{"role": "user", "content": flatten_content(content)}]

        out: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                out.append({"role": "user", "content": block.get("text", "")})
            elif block_type == "tool_result":
                result_content = flatten_content(block.get("content"))
                if block.get("is_error"):
                    result_content = "[ERROR] " + result_content
                out.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id", ""),
                    "content": result_content,
                })
            elif block_type == "image":
                out.append({"role": "user", "content": "[image omitted]"})
            else:
                # Keep unsupported multimodal blocks visible as a placeholder
                # so an otherwise valid user turn is not silently dropped.
                out.append({"role": "user", "content": flatten_content(block)})
        return out

    @staticmethod
    def _convert_assistant_message(content: Any) -> dict[str, Any]:
        if isinstance(content, str):
            return {"role": "assistant", "content": content}

        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    text_parts.append(block.get("text", ""))
                elif block_type == "thinking":
                    # "signature" is intentionally dropped: see design doc §1.2/§4.3 —
                    # it does not participate in prefix comparison and coda's own
                    # /v1/messages endpoint performs no signature verification.
                    thinking_parts.append(block.get("thinking", ""))
                elif block_type == "tool_use":
                    tool_calls.append({
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input") or {}),
                        },
                    })

        # None vs "" distinction (design doc §4.4 constraint 2): content is None
        # when there was no text block at all, "" when there was an empty one.
        result: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(text_parts) if text_parts else None,
        }
        if tool_calls:
            result["tool_calls"] = tool_calls
        if thinking_parts:
            result["reasoning_content"] = "".join(thinking_parts)
        return result

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        converted = []
        for tool in tools:
            if not isinstance(tool, dict) or "name" not in tool:
                continue
            converted.append(_to_openai_tool(
                tool["name"],
                tool.get("description", ""),
                tool.get("input_schema") or {"type": "object", "properties": {}},
            ))
        return converted or None

    # -- response: internal assistant_message -> Anthropic wire -------------

    def build_response(
        self,
        body: dict[str, Any],
        assistant_message: dict[str, Any],
        finish_reason: str,
        usage: dict[str, Any],
        *,
        path: str | None = None,
        upstream_response: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build an Anthropic Messages wire response."""
        blocks = self._build_content_blocks(assistant_message)
        return {
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "model": body.get("model", "default"),
            "content": blocks,
            "stop_reason": self._stop_reason(finish_reason),
            "stop_sequence": None,
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            },
        }

    @staticmethod
    def _build_content_blocks(assistant_message: dict[str, Any]) -> list[dict[str, Any]]:
        # Exactly one block per populated field (design doc §4.4 constraint 1):
        # never split text/thinking across multiple blocks. Tool-call ids are
        # echoed back verbatim: Claude Code returns them in tool_use /
        # tool_result blocks on the next request, and the parser's prefix
        # comparison relies on the round-trip being identity.
        blocks: list[dict[str, Any]] = []
        reasoning_content = assistant_message.get("reasoning_content")
        if reasoning_content:
            blocks.append({"type": "thinking", "thinking": reasoning_content, "signature": ""})
        content = assistant_message.get("content")
        if content is not None:
            blocks.append({"type": "text", "text": content})
        for tool_call in assistant_message.get("tool_calls") or []:
            function = tool_call.get("function", {})
            try:
                tool_input = json.loads(function.get("arguments") or "{}")
            except (TypeError, ValueError):
                tool_input = {}
            blocks.append({
                "type": "tool_use",
                "id": tool_call.get("id", ""),
                "name": function.get("name", ""),
                "input": tool_input,
            })
        if not blocks:
            blocks.append({"type": "text", "text": ""})
        return blocks

    @staticmethod
    def _stop_reason(finish_reason: str) -> str:
        if finish_reason == "tool_calls":
            return "tool_use"
        if finish_reason == "length":
            return "max_tokens"
        return "end_turn"

    def build_streaming_response(self, wire_response: dict[str, Any]) -> StreamingResponse:
        """Build an Anthropic Messages SSE response."""
        return StreamingResponse(
            list(_anthropic_sse_events(wire_response)), media_type="text/event-stream"
        )

    def build_error_response(
        self,
        body: dict[str, Any],
        *,
        error_type: str,
        message: str,
        status_code: int,
    ) -> JSONResponse:
        """Build an Anthropic-compatible JSON error.

        Context-length exhaustion maps to ``invalid_request_error`` with a 4xx
        status so the Claude Code SDK fails fast instead of retrying (500-class
        errors make it retry the same un-shrinkable prompt with exponential
        backoff, wasting minutes per overflow).
        """
        anthropic_type = (
            "invalid_request_error"
            if error_type == "context_length_exceeded"
            else error_type
        )
        return JSONResponse(
            {
                "type": "error",
                "error": {"type": anthropic_type, "message": message},
            },
            status_code=status_code,
        )


def _anthropic_sse_events(wire_response: dict[str, Any]):
    """Yield the five-stage Anthropic SSE event sequence (design doc §1.4.4)."""
    usage = wire_response.get("usage") or {"input_tokens": 0, "output_tokens": 0}
    blocks = wire_response.get("content") or []

    message_start = {
        "type": "message_start",
        "message": {
            "id": wire_response.get("id", ""),
            "type": "message",
            "role": "assistant",
            "model": wire_response.get("model", "default"),
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": usage.get("input_tokens", 0), "output_tokens": 0},
        },
    }
    yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"

    for index, block in enumerate(blocks):
        block_type = block["type"]
        if block_type == "text":
            start_block = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": block.get("text", "")}
        elif block_type == "thinking":
            start_block = {"type": "thinking", "thinking": ""}
            delta = {"type": "thinking_delta", "thinking": block.get("thinking", "")}
        else:  # tool_use
            start_block = {"type": "tool_use", "id": block.get("id", ""), "name": block.get("name", ""), "input": {}}
            delta = {"type": "input_json_delta", "partial_json": json.dumps(block.get("input") or {})}

        yield (
            f"event: content_block_start\n"
            f"data: {json.dumps({'type': 'content_block_start', 'index': index, 'content_block': start_block})}\n\n"
        )
        yield (
            f"event: content_block_delta\n"
            f"data: {json.dumps({'type': 'content_block_delta', 'index': index, 'delta': delta})}\n\n"
        )
        yield (
            f"event: content_block_stop\n"
            f"data: {json.dumps({'type': 'content_block_stop', 'index': index})}\n\n"
        )

    message_delta = {
        "type": "message_delta",
        "delta": {"stop_reason": wire_response.get("stop_reason"), "stop_sequence": None},
        "usage": usage,
    }
    yield f"event: message_delta\ndata: {json.dumps(message_delta)}\n\n"
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
