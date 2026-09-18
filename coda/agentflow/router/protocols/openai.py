"""OpenAI Chat Completions and Responses protocol adapters."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from fastapi.responses import JSONResponse, StreamingResponse
from starlette.requests import Request

from .base import ProtocolAdapter, _route_re

#: Codex's legacy compaction endpoint. Current Codex builds signal compaction in
#: the request body instead, so this is only consulted to keep old clients working.
_LEGACY_COMPACT_ROUTE_RE = _route_re(r"v1/responses/compact")


def _is_legacy_compact_path(path: str) -> bool:
    return _LEGACY_COMPACT_ROUTE_RE.fullmatch(path) is not None


class OpenAIChatCompletionsAdapter(ProtocolAdapter):
    """OpenAI Chat Completions protocol adapter."""

    _ROUTE_RE = _route_re(r"v1/chat/completions")

    def parse_route(self, path: str) -> tuple[str, int] | None:
        """Match the /v1/chat/completions route."""
        match = self._ROUTE_RE.fullmatch(path)
        if match is None:
            return None
        return match.group("trajectory_id"), int(match.group("attempt_id"))

    def parse_request(self, body: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
        """Parse a chat/completions body into internal messages and tools."""
        messages = body["messages"]
        if not messages:
            raise ValueError("messages is empty")
        if body.get("n", 1) != 1:
            raise ValueError("n > 1 is not supported by Coda Router")
        max_new_tokens = body.get("max_completion_tokens")
        if max_new_tokens is None:
            max_new_tokens = body.get("max_tokens")
        if max_new_tokens is None:
            max_new_tokens = body.get("max_new_tokens")
        if max_new_tokens is not None:
            body["max_new_tokens"] = int(max_new_tokens)
        return messages, body.get("tools")

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
        """Build a chat.completion wire response."""
        if upstream_response and "choices" in upstream_response:
            return upstream_response
        meta = (upstream_response or {}).get("meta_info") or {}
        finish_reason_meta = meta.get("finish_reason") or {}
        matched_stop = (
            None
            if finish_reason == "tool_calls"
            else finish_reason_meta.get("matched")
            if isinstance(finish_reason_meta, dict)
            else None
        )
        metadata = dict(body.get("metadata") or {})
        if "weight_version" in meta:
            metadata["weight_version"] = meta["weight_version"]
        return {
            "id": meta.get("id", body.get("id", "")),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", "default"),
            "choices": [
                {
                    "index": 0,
                    "message": assistant_message,
                    "logprobs": None,
                    "finish_reason": finish_reason,
                    "matched_stop": matched_stop,
                }
            ],
            "usage": usage,
            "metadata": metadata,
        }

    def build_streaming_response(self, wire_response: dict[str, Any]) -> StreamingResponse:
        """Build a chat.completion.chunk SSE response."""
        choice = wire_response["choices"][0]
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
            "id": wire_response.get("id", ""),
            "object": "chat.completion.chunk",
            "created": wire_response.get("created", int(time.time())),
            "model": wire_response.get("model", "default"),
        }
        events = [
            {
                **chunk,
                "choices": [{"index": choice.get("index", 0), "delta": delta, "finish_reason": None}],
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
                "usage": wire_response.get("usage"),
            },
        ]
        body_lines = [f"data: {json.dumps(event)}\n\n" for event in events]
        body_lines.append("data: [DONE]\n\n")
        return StreamingResponse(body_lines, media_type="text/event-stream")

    def build_error_response(
        self,
        body: dict[str, Any],
        *,
        error_type: str,
        message: str,
        status_code: int,
    ) -> JSONResponse:
        """Build an OpenAI-compatible JSON error."""
        return JSONResponse(
            {"error": {"message": message, "type": error_type}},
            status_code=status_code,
        )


class OpenAIResponsesAdapter(ProtocolAdapter):
    """Adapt Codex Responses requests to Coda's internal message format."""

    _ROUTE_RE = _route_re(r"v1/responses(?:/compact)?")

    def parse_route(self, path: str) -> tuple[str, int] | None:
        match = self._ROUTE_RE.fullmatch(path)
        if match is None:
            return None
        return match.group("trajectory_id"), int(match.group("attempt_id"))

    def normalize_headers(self, request: Request) -> dict[str, str]:
        # Codex-specific compatibility: translate its private subagent and
        # compaction metadata into Coda's internal request kind. Revisit whether
        # this shim is still needed once subagent trajectories can be trained.
        normalized = super().normalize_headers(request)
        if normalized.get("x-openai-subagent", "").strip():
            normalized["request_kind"] = "collab_spawn"
            return normalized

        raw_metadata = normalized.get("x-codex-turn-metadata", "").strip()
        is_compaction_request = _is_legacy_compact_path(request.url.path)
        if raw_metadata:
            try:
                metadata = json.loads(raw_metadata)
            except (TypeError, ValueError):
                metadata = None
            is_compaction_request = is_compaction_request or (
                isinstance(metadata, dict)
                and metadata.get("request_kind") == "compaction"
            )
        if is_compaction_request:
            normalized["request_kind"] = "compaction"
        return normalized

    def parse_request(self, body: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
        max_new_tokens = body.get("max_output_tokens")
        if max_new_tokens is None:
            max_new_tokens = body.get("max_new_tokens")
        if max_new_tokens is not None:
            body["max_new_tokens"] = int(max_new_tokens)
        return _ResponsesRequestDecoder(body).decode()

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
        return _ResponsesResponseEncoder(
            body,
            assistant_message,
            finish_reason,
            usage,
            path,
        ).encode()

    def build_streaming_response(
        self,
        wire_response: dict[str, Any],
    ) -> StreamingResponse:
        return StreamingResponse(
            _ResponsesEventStream(wire_response),
            media_type="text/event-stream",
        )

    def build_error_response(
        self,
        body: dict[str, Any],
        *,
        error_type: str,
        message: str,
        status_code: int,
    ) -> JSONResponse | StreamingResponse:
        if body.get("stream") is not True:
            return JSONResponse(
                {"error": {"message": message, "type": error_type}},
                status_code=status_code,
            )

        response = {
            "id": f"resp_{uuid.uuid4().hex}",
            "object": "response",
            "created_at": int(time.time()),
            "model": body.get("model", "default"),
            "status": "failed",
            "error": {"code": error_type, "message": message},
            "output": [],
        }
        return StreamingResponse(
            _ResponsesEventStream(response),
            media_type="text/event-stream",
        )


@dataclass(frozen=True)
class _ToolBinding:
    internal_name: str
    name: str
    namespace: str | None
    description: str
    parameters: dict[str, Any]
    strict: bool | None


class _ToolBindings:
    """Translate Responses function and namespace tools in both directions."""

    def __init__(self, tools: list[dict[str, Any]] | None) -> None:
        self._by_internal_name: dict[str, _ToolBinding] = {}
        for tool in tools or []:
            if tool.get("type") == "namespace":
                namespace = str(tool.get("name") or "")
                children = tool.get("tools") or []
            else:
                namespace = None
                children = [tool]

            for child in children:
                if child.get("type") != "function":
                    continue
                name = str(child.get("name") or "")
                internal_name = f"{namespace}__{name}" if namespace else name
                self._by_internal_name[internal_name] = _ToolBinding(
                    internal_name=internal_name,
                    name=name,
                    namespace=namespace,
                    description=str(child.get("description") or ""),
                    parameters=child.get("parameters") or {},
                    strict=child.get("strict") if "strict" in child else None,
                )

    def to_chat_tools(self) -> list[dict[str, Any]] | None:
        """Convert the bound tools into OpenAI chat-format tool definitions."""
        tools: list[dict[str, Any]] = []
        for binding in self._by_internal_name.values():
            function: dict[str, Any] = {
                "name": binding.internal_name,
                "description": binding.description,
                "parameters": binding.parameters,
            }
            if binding.strict is not None:
                function["strict"] = binding.strict
            tools.append({"type": "function", "function": function})
        return tools or None

    def get(self, internal_name: str) -> _ToolBinding | None:
        """Return the tool binding for an internal name, or None."""
        return self._by_internal_name.get(internal_name)


class _ResponsesRequestDecoder:
    """Decode one Responses request into internal messages and tools."""

    def __init__(self, body: dict[str, Any]) -> None:
        self._body = body
        self._messages: list[dict[str, Any]] = []

    def decode(
        self,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
        """Decode the request body into (internal messages, chat tools)."""
        instructions = self._body.get("instructions")
        if instructions:
            self._messages.append({
                "role": "system",
                "content": self._flatten_content(instructions),
            })

        input_value = self._body.get("input")
        if isinstance(input_value, str):
            self._messages.append({"role": "user", "content": input_value})
        else:
            for item in input_value or []:
                self._decode_item(item)

        messages = self._fold_system_messages()
        tools = _ToolBindings(self._body.get("tools")).to_chat_tools()
        return messages, tools

    def _decode_item(self, item: Any) -> None:
        if not isinstance(item, dict):
            return

        item_type = item.get("type", "message")
        if item_type == "message":
            role = item.get("role")
            content = item.get("content")
            if role in ("system", "developer"):
                self._messages.append({
                    "role": "system",
                    "content": self._flatten_content(content),
                })
            elif role == "user":
                self._messages.append({
                    "role": "user",
                    "content": self._flatten_content(content),
                })
            elif role == "assistant":
                assistant_content = (
                    self._flatten_content(content)
                    if self._has_text_content(content)
                    else None
                )
                pending = self._pending_reasoning_message()
                if pending is not None:
                    pending["content"] = assistant_content
                else:
                    self._messages.append({
                        "role": "assistant",
                        "content": assistant_content,
                    })
            return

        if item_type == "function_call":
            name = str(item.get("name") or "")
            namespace = str(item.get("namespace") or "") or None
            function_name = f"{namespace}__{name}" if namespace else name
            tool_call = {
                "id": item.get("call_id", ""),
                "type": "function",
                "function": {
                    "name": function_name,
                    "arguments": item.get("arguments", ""),
                },
            }
            assistant = self._last_assistant_message()
            if assistant is None:
                assistant = {"role": "assistant", "content": None}
                self._messages.append(assistant)
            assistant.setdefault("tool_calls", []).append(tool_call)
            return

        if item_type == "function_call_output":
            output = item.get("output")
            output_text = (
                output if isinstance(output, str) else self._flatten_content(output)
            )
            self._messages.append({
                "role": "tool",
                "tool_call_id": item.get("call_id", ""),
                "content": output_text,
            })
            return

        if item_type == "reasoning":
            reasoning_text = "".join(
                block.get("text", "")
                for block in (item.get("content") or [])
                if isinstance(block, dict)
            ) or "".join(
                block.get("text", "")
                for block in (item.get("summary") or [])
                if isinstance(block, dict)
            )
            if not reasoning_text:
                return
            assistant = self._last_assistant_message()
            if assistant is None:
                assistant = {"role": "assistant", "content": None}
                self._messages.append(assistant)
            assistant["reasoning_content"] = (
                assistant.get("reasoning_content", "") + reasoning_text
            )
            return

        if item_type in {
            "compaction",
            "compaction_summary",
            "context_compaction",
        }:
            summary = item.get("encrypted_content")
            if summary is not None:
                self._messages.append({
                    "role": "assistant",
                    "content": str(summary),
                })

        # The remaining official Responses input item types are intentionally
        # omitted from Chat/Generate history:
        #
        # - compaction_trigger is a request-control marker. The response encoder
        #   reads it from the original body; it is not conversation content.
        # - custom_tool_call and custom_tool_call_output are absent because the
        #   current Codex and OpenCode profiles expose only function and namespace
        #   tools.
        # - local_shell_call, local_shell_call_output, shell_call,
        #   shell_call_output, apply_patch_call, and apply_patch_call_output are
        #   absent because shell access is exposed as ordinary function tools.
        # - file_search_call, web_search_call, computer_call,
        #   computer_call_output, image_generation_call, and code_interpreter_call
        #   require server-side tools that the current profiles do not advertise.
        # - mcp_list_tools, mcp_approval_request, mcp_approval_response, and
        #   mcp_call are absent because MCP is not enabled.
        # - tool_search_call, tool_search_output, additional_tools,
        #   configuration_update, and item_reference belong to dynamic tool loading
        #   or server-side item reuse, neither of which is enabled.
        #
        # These items should therefore never appear in current requests. If one
        # does appear, forwarding its wire JSON as model text would corrupt the
        # history, while rejecting it would turn a capability mismatch into a
        # framework failure. The adapter deliberately ignores it instead.

    @staticmethod
    def _flatten_content(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return str(content)

        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") in {"image", "input_image", "input_file", "document"}:
                    parts.append("[image omitted]")
                else:
                    parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "\n".join(parts)

    def _fold_system_messages(self) -> list[dict[str, Any]]:
        system_indices = [
            index
            for index, message in enumerate(self._messages)
            if index > 0 and message.get("role") == "system"
        ]
        if not system_indices:
            return self._messages

        messages = [dict(message) for message in self._messages]
        dropped: set[int] = set()
        for index in system_indices:
            target_index = next(
                (
                    candidate
                    for candidate in range(index - 1, -1, -1)
                    if candidate not in dropped and messages[candidate].get("role") == "user"
                ),
                None,
            )
            prepend = target_index is None
            if target_index is None:
                target_index = next(
                    (
                        candidate
                        for candidate in range(index + 1, len(messages))
                        if candidate not in dropped and messages[candidate].get("role") == "user"
                    ),
                    None,
                )

            system_text = self._flatten_content(messages[index].get("content"))
            reminder = f"<system-reminder>\n{system_text}\n</system-reminder>\n"
            if target_index is None:
                messages[index] = {"role": "user", "content": reminder}
                continue

            existing = self._flatten_content(messages[target_index].get("content"))
            messages[target_index] = {
                **messages[target_index],
                "content": f"{reminder}\n{existing}" if prepend else f"{existing}\n{reminder}",
            }
            dropped.add(index)

        return [message for index, message in enumerate(messages) if index not in dropped]

    @staticmethod
    def _has_text_content(content: Any) -> bool:
        return isinstance(content, str) or (
            isinstance(content, list)
            and any(
                isinstance(block, str)
                or (
                    isinstance(block, dict)
                    and block.get("type")
                    in {"input_text", "output_text", "text"}
                )
                for block in content
            )
        )

    def _last_assistant_message(self) -> dict[str, Any] | None:
        if self._messages and self._messages[-1].get("role") == "assistant":
            return self._messages[-1]
        return None

    def _pending_reasoning_message(self) -> dict[str, Any] | None:
        assistant = self._last_assistant_message()
        if (
            assistant is not None
            and assistant.get("reasoning_content") is not None
            and assistant.get("content") is None
            and not assistant.get("tool_calls")
        ):
            return assistant
        return None


class _ResponsesResponseEncoder:
    """Encode an internal assistant message as one Responses response."""

    def __init__(
        self,
        body: dict[str, Any],
        assistant_message: dict[str, Any],
        finish_reason: str,
        usage: dict[str, Any],
        path: str | None,
    ) -> None:
        self._body = body
        self._assistant_message = assistant_message
        self._finish_reason = finish_reason
        self._usage = usage
        self._path = path

    def encode(self) -> dict[str, Any]:
        """Encode the assistant message into a Responses wire response."""
        if self._is_legacy_compaction() or self._is_v2_compaction():
            return self._encode_compaction()
        return self._encode_response()

    def _encode_compaction(self) -> dict[str, Any]:
        is_v2_compaction = self._is_v2_compaction()
        summary = (
            self._assistant_message.get("content")
            or self._assistant_message.get("reasoning_content")
            or ""
        )
        input_tokens = self._usage.get("prompt_tokens", 0)
        output_tokens = self._usage.get("completion_tokens", 0)
        response = {
            "id": f"resp_{uuid.uuid4().hex}",
            "object": "response" if is_v2_compaction else "response.compaction",
            "created_at": int(time.time()),
            "output": self._compact_output_items(str(summary), is_v2_compaction),
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        }
        if is_v2_compaction:
            response.update({
                "model": self._body.get("model", "default"),
                "status": "completed",
                "incomplete_details": None,
            })
        return response

    def _encode_response(self) -> dict[str, Any]:
        input_tokens = self._usage.get("prompt_tokens", 0)
        output_tokens = self._usage.get("completion_tokens", 0)
        status = (
            "incomplete" if self._finish_reason == "length" else "completed"
        )
        return {
            "id": f"resp_{uuid.uuid4().hex}",
            "object": "response",
            "created_at": int(time.time()),
            "model": self._body.get("model", "default"),
            "status": status,
            "output": self._output_items(status),
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": input_tokens + output_tokens,
            },
            "incomplete_details": (
                {"reason": "max_output_tokens"}
                if self._finish_reason == "length"
                else None
            ),
        }

    def _compact_output_items(
        self,
        summary: str,
        is_v2_compaction: bool,
    ) -> list[dict[str, Any]]:
        if is_v2_compaction:
            return [{"type": "compaction", "encrypted_content": summary}]

        items: list[dict[str, Any]] = []
        input_value = self._body.get("input")
        if isinstance(input_value, str):
            items.append({
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": input_value}],
            })
        elif isinstance(input_value, list):
            for item in input_value:
                if not isinstance(item, dict):
                    continue
                if item.get("type", "message") != "message":
                    continue
                if item.get("role") not in {"user", "developer"}:
                    continue
                retained = dict(item)
                retained.setdefault("type", "message")
                items.append(retained)
        items.append({"type": "compaction", "encrypted_content": summary})
        return items

    def _output_items(self, status: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        bindings = _ToolBindings(self._body.get("tools"))
        reasoning_content = self._assistant_message.get("reasoning_content")
        if reasoning_content:
            items.append({
                "type": "reasoning",
                "id": f"reasoning_{uuid.uuid4().hex}",
                "status": status,
                "summary": [{
                    "type": "summary_text",
                    "text": reasoning_content,
                }],
            })

        content = self._assistant_message.get("content")
        if content is not None:
            items.append({
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex}",
                "role": "assistant",
                "status": status,
                "content": [{
                    "type": "output_text",
                    "text": content,
                    "annotations": [],
                    "logprobs": [],
                }],
            })

        for tool_call in self._assistant_message.get("tool_calls") or []:
            function = tool_call.get("function", {})
            internal_name = str(function.get("name") or "")
            binding = bindings.get(internal_name)
            item = {
                "id": f"fc_{uuid.uuid4().hex}",
                "call_id": tool_call.get("id", ""),
                "name": binding.name if binding else internal_name,
                "type": "function_call",
                "status": status,
                "arguments": function.get("arguments", ""),
            }
            if binding and binding.namespace:
                item["namespace"] = binding.namespace
            items.append(item)

        if not items:
            items.append({
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex}",
                "role": "assistant",
                "status": status,
                "content": [{
                    "type": "output_text",
                    "text": "",
                    "annotations": [],
                    "logprobs": [],
                }],
            })
        return items

    def _is_legacy_compaction(self) -> bool:
        return self._path is not None and _is_legacy_compact_path(self._path)

    def _is_v2_compaction(self) -> bool:
        input_value = self._body.get("input")
        return bool(
            isinstance(input_value, list)
            and input_value
            and isinstance(input_value[-1], dict)
            and input_value[-1].get("type") == "compaction_trigger"
        )


class _ResponsesEventStream:
    """Expand a buffered Response into its standard SSE lifecycle."""

    def __init__(self, response: dict[str, Any]) -> None:
        self._response = response

    def __iter__(self) -> Iterator[str]:
        for sequence_number, (event_type, fields) in enumerate(self._events()):
            event = {
                "type": event_type,
                "sequence_number": sequence_number,
                **fields,
            }
            yield f"event: {event_type}\ndata: {json.dumps(event)}\n\n"

    def _events(self) -> Iterator[tuple[str, dict[str, Any]]]:
        initial_response = {
            **self._response,
            "status": "in_progress",
            "output": [],
            "usage": None,
            "error": None,
            "incomplete_details": None,
        }
        yield "response.created", {"response": initial_response}
        yield "response.in_progress", {"response": initial_response}

        for output_index, item in enumerate(self._response.get("output") or []):
            yield from self._item_events(output_index, item)

        final_event_type = {
            "failed": "response.failed",
            "incomplete": "response.incomplete",
        }.get(self._response.get("status"), "response.completed")
        yield final_event_type, {"response": self._response}

    def _item_events(
        self,
        output_index: int,
        item: dict[str, Any],
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        yield "response.output_item.added", {
            "output_index": output_index,
            "item": self._in_progress_item(item),
        }

        position = {
            "item_id": item.get("id", ""),
            "output_index": output_index,
        }
        item_type = item.get("type")
        if item_type == "message":
            yield from self._message_events(item, position)
        elif item_type == "reasoning":
            yield from self._reasoning_events(item, position)
        elif item_type == "function_call":
            yield from self._function_call_events(item, position)

        yield "response.output_item.done", {
            "output_index": output_index,
            "item": item,
        }

    @staticmethod
    def _in_progress_item(item: dict[str, Any]) -> dict[str, Any]:
        empty_fields = {
            "message": {"content": []},
            "reasoning": {"summary": []},
            "function_call": {"arguments": ""},
        }.get(item.get("type"))
        if empty_fields is None:
            return item
        return {**item, "status": "in_progress", **empty_fields}

    @staticmethod
    def _message_events(
        item: dict[str, Any],
        position: dict[str, Any],
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        for content_index, part in enumerate(item.get("content") or []):
            if part.get("type") != "output_text":
                continue

            text = part.get("text", "")
            logprobs = part.get("logprobs", [])
            content_position = {**position, "content_index": content_index}
            yield "response.content_part.added", {
                **content_position,
                "part": {**part, "text": ""},
            }
            yield "response.output_text.delta", {
                **content_position,
                "delta": text,
                "logprobs": logprobs,
            }
            yield "response.output_text.done", {
                **content_position,
                "text": text,
                "logprobs": logprobs,
            }
            yield "response.content_part.done", {
                **content_position,
                "part": part,
            }

    @staticmethod
    def _reasoning_events(
        item: dict[str, Any],
        position: dict[str, Any],
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        reasoning_text = "".join(
            part.get("text", "") for part in item.get("summary") or []
        )
        summary_position = {**position, "summary_index": 0}
        summary_part = {"type": "summary_text", "text": reasoning_text}
        yield "response.reasoning_summary_part.added", {
            **summary_position,
            "part": {"type": "summary_text", "text": ""},
        }
        yield "response.reasoning_summary_text.delta", {
            **summary_position,
            "delta": reasoning_text,
        }
        yield "response.reasoning_summary_text.done", {
            **summary_position,
            "text": reasoning_text,
        }
        yield "response.reasoning_summary_part.done", {
            **summary_position,
            "part": summary_part,
        }

    @staticmethod
    def _function_call_events(
        item: dict[str, Any],
        position: dict[str, Any],
    ) -> Iterator[tuple[str, dict[str, Any]]]:
        arguments = item.get("arguments", "")
        yield "response.function_call_arguments.delta", {
            **position,
            "delta": arguments,
        }
        yield "response.function_call_arguments.done", {
            **position,
            "name": item.get("name", ""),
            "arguments": arguments,
        }
