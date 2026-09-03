"""Utilities for parsing RL trajectories."""

import base64
import copy
import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal

import torch
from pydantic import ValidationError
from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.entrypoints.openai.serving_chat import (
    normalize_assistant_tool_call_arguments,
)
from sglang.srt.environ import envs
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.parser.template_detection import (
    REASONING_PARSER_RULES,
    TOOL_CALL_PARSER_RULES,
    _architecture_auto_parsers,
    build_detection_context,
    detect_reasoning_pattern,
    match_rules,
)

from coda.agentflow.tokenizer_manager import DEFAULT_THINK_TAGS
from coda.agentflow.trajectory_store import (
    Segment,
    Trajectory,
    TrajectoryStatus,
    TrajectoryStore,
    Triplet,
)

logger = logging.getLogger(__name__)

_NORMALIZED_HISTORY_KEY = "normalized_history"

# Tool schemas from the first request of a trajectory. Cached because agents such as
# mini-swe-agent omit the "tools" field on follow-up turns while the LLM keeps emitting
# tool-call markup. Distinct from metadata["tools"], which the dataset layer owns.
_REQUEST_TOOLS_KEY = "request_tools"

# Standard OpenAI chat message fields retained during prefix-comparison normalisation.
_OPENAI_MESSAGE_FIELDS: frozenset[str] = frozenset({
    "role", "content", "name", "tool_call_id", "tool_calls",
})

# Roles that must always carry a "content" key (defaulting to "") after normalisation.
_OPENAI_MESSAGE_ROLES: frozenset[str] = frozenset({
    "system", "user", "assistant", "tool",
})

# Config values that disable a parser.
_PARSER_DISABLED_VALUES: frozenset[str] = frozenset({"", "none", "false", "null"})

# Placeholder passed to FunctionCallParser when the request carried no tool schemas:
# an empty list makes SGLang skip parsing entirely, so we keep one dummy entry and rely
# on SGLANG_FORWARD_UNKNOWN_TOOLS to forward calls whose name is not in the list.
_SENTINEL_TOOLS: list[Tool] = [Tool(type="function", function={"name": "_"})]

# Parser fields SGLang's architecture-based detection can fill in.
_PARSER_ATTRS = ("reasoning_parser", "tool_call_parser")


def _detect_parser_names_from_arch(model_path: str | None) -> tuple[str | None, str | None]:
    """Ask SGLang which parsers match the architecture in the model's ``config.json``.

    Reuses the fallback SGLang applies to models that ship no Jinja template, so parser
    keys stay owned by SGLang instead of being hardcoded here. The helper only reads the
    model-location fields off ``server_args``, so a plain namespace is enough.
    """
    if not model_path:
        logger.warning("No model path available for architecture-based parser detection")
        return None, None
    stub = SimpleNamespace(
        model_path=model_path,
        trust_remote_code=True,
        revision=None,
        model_config_parser="auto",
    )
    try:
        detected = _architecture_auto_parsers(stub, _PARSER_ATTRS)
    except Exception as exc:  # unreadable config.json, unknown model_type, ...
        logger.warning(
            "Architecture-based parser detection failed for %s: %s", model_path, exc
        )
        return None, None
    return detected.get("reasoning_parser"), detected.get("tool_call_parser")


def _needs_detection(config: str | bool | None) -> bool:
    """``None`` (unset) and ``True`` mean "auto-detect"."""
    return config is None or config is True


def _detect_parser_names(tokenizer: Any) -> tuple[str | None, str | None]:
    """Auto-detect ``(reasoning, tool_call)`` parser keys, mirroring SGLang's ``auto``.

    Same two stages as SGLang's ``resolve_auto_parsers``: match the Jinja chat template
    plus the tokenizer vocabulary first, then fall back to the model architecture for
    models that ship no template at all (DeepSeek-V4, whose wrapper replaces
    ``apply_chat_template`` with the official encoder).
    """
    template = getattr(tokenizer, "chat_template", None)
    force_reasoning, reasoning_config = detect_reasoning_pattern(template)
    ctx = build_detection_context(template, tokenizer, reasoning_config, force_reasoning)
    if ctx is None:
        return _detect_parser_names_from_arch(getattr(tokenizer, "name_or_path", None))
    return (
        match_rules(ctx, REASONING_PARSER_RULES, "reasoning parser"),
        match_rules(ctx, TOOL_CALL_PARSER_RULES, "tool-call parser"),
    )


def _resolve_parser_name(
    config: str | bool | None,
    detected: str | None,
    *,
    kind: str,
) -> str | None:
    """Validate a coda parser config value as a SGLang parser key.

    Config values are SGLang keys verbatim (``ReasoningParser.DetectorMap`` /
    ``FunctionCallParser.ToolCallParserEnum``). ``None`` / ``True`` fall back to
    *detected* (from template auto-detection); ``False`` and the values in
    ``_PARSER_DISABLED_VALUES`` disable parsing. Names SGLang does not know are
    logged and disabled rather than raising, so a new model name cannot block startup.
    """
    supported = (
        ReasoningParser.DetectorMap if kind == "reasoning"
        else FunctionCallParser.ToolCallParserEnum
    )

    if _needs_detection(config):
        name = detected
    elif config is False:
        return None
    else:
        name = str(config).strip().lower()
        if name in _PARSER_DISABLED_VALUES:
            return None
    if not name:
        logger.info("No %s parser configured or detected; %s parsing disabled", kind, kind)
        return None

    if name not in supported:
        logger.warning(
            "Unsupported %s parser %r; %s parsing disabled. Supported: %s",
            kind, name, kind, ", ".join(sorted(supported)),
        )
        return None
    return name


def _to_sglang_tools(tools: list[dict[str, Any]] | None) -> list[Tool]:
    """Convert OpenAI tool dicts from the request body into SGLang ``Tool`` models."""
    converted: list[Tool] = []
    for tool in tools or []:
        try:
            converted.append(Tool(**tool))
        except (TypeError, ValidationError) as exc:
            logger.warning("Skipping malformed tool schema %r: %s", tool, exc)
    return converted


@dataclass
class TurnInputContext:
    """Tokenized input for one agent turn, ready to send to the LLM worker."""

    delta_prompt_ids: list[int]
    input_ids: list[int]
    start_new_segment: bool
    raw_new_messages: list[dict[str, Any]]
    routed_experts_start_len: int = 0
    tools: list[dict[str, Any]] | None = None

    # --- Segment-tree addressing (new) ---

    # segment_id this turn should be appended to (existing segment for continuation,
    # or the newly-created segment's id when start_new_segment is True).
    target_segment_id: int = 0

    # Fields for the new segment when start_new_segment is True; ignored otherwise.
    new_segment_parent_id: int | None = None
    new_segment_origin: Literal["root", "compact", "subagent"] = "root"
    new_segment_depth: int = 0

    # True for a subagent-fork placeholder turn (case 3): the caller still forwards the
    # request to the LLM worker for a real response, but update_trajectory()/_append_triplet
    # must skip writing any token/loss/triplet data and must not move active_segment_id.
    is_subagent_placeholder: bool = False


@dataclass
class LLMGenerateResponse:
    """Decoded output from one LLM worker response."""

    generated_text: str
    response_ids: list[int]
    logprobs: list[float]
    weight_version: int
    routed_experts: torch.Tensor | None = None


def deserialize_routed_experts(b64_str: str, num_entries: int) -> torch.Tensor:
    """Decode SGLang base64+int32 bytes to a 2D torch.Tensor of shape [num_entries, block_size]."""
    raw = base64.b64decode(b64_str)
    flat = torch.frombuffer(bytearray(raw), dtype=torch.int32)
    tensor = flat.reshape(num_entries, -1)
    logger.debug(
        "r3_deserialize: entries=%d shape=%s trajectory=%s",
        num_entries, list(tensor.shape), tensor[0, :8].tolist() if len(tensor) > 0 else [],
    )
    return tensor


def _parse_weight_version(raw: Any) -> int:
    """Validate the worker-reported rollout weight version.

    -1 is reserved as the sentinel for prompt-token positions and for
    "not set yet" (Trajectory.start_rollout_weight_version), so a real rollout
    version must be a non-negative int.
    """
    try:
        version = int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"weight_version must be int or str, got {raw!r}") from None
    if version < 0:
        raise ValueError(f"weight_version must be >= 0, got {version}")
    return version


class TrajectoryParser:
    """Build and update RL trajectories from chat-completions traffic.

    Per-request lifecycle (called by ParserMiddleware):
        1. get_trajectory()         – fetch Trajectory from store
        2. build_turn_input()       – tokenize messages, return input_ids
        3. [LLM call via Router]
        4. build_assistant_message() – decode response, parse reasoning/tool_calls
        5. update_trajectory()      – append turn, persist to store
    """

    def __init__(
        self,
        trajectory_store: TrajectoryStore,
        tokenizer_manager: Any,
        accumulate_reasoning: bool = True,
        r3_enabled: bool = False,
        reasoning_parser: str | bool | dict[str, Any] | None = None,
        tool_call_parser: str | bool | dict[str, Any] | None = None,
    ):
        self.trajectory_store = trajectory_store
        self.tokenizer_manager = tokenizer_manager
        self.accumulate_reasoning = accumulate_reasoning
        self.r3_enabled = r3_enabled
        self.model_family = getattr(tokenizer_manager, "model_family", None)
        self.think_tags = getattr(tokenizer_manager, "think_tags", DEFAULT_THINK_TAGS)
        self.think_end_ids: list[int] = list(getattr(tokenizer_manager, "think_end_ids", []))
        self.think_start_ids: list[int] = list(getattr(tokenizer_manager, "think_start_ids", []))
        # Length of the template prelude for an empty system message, computed lazily
        # on the first continuation turn (not at tokenizer init). -1 caches "the chat
        # template rejects a system-only conversation" (e.g. Qwen3.5).
        self._system_prompt_len: int | None = None
        self._dummy_user_prefix_len: int | None = None
        gen_kwargs: dict = getattr(tokenizer_manager, "generation_prompt_kwargs", {}) or {}
        self.enable_thinking: bool = bool(gen_kwargs.get("enable_thinking", True))
        # Auto-detection reads the chat template + tokenizer vocab (or the model config
        # when there is no template), so only pay for it when a parser is left unset.
        if _needs_detection(reasoning_parser) or _needs_detection(tool_call_parser):
            detected_reasoning, detected_tool_call = _detect_parser_names(
                getattr(tokenizer_manager, "tokenizer", None)
            )
        else:
            detected_reasoning = detected_tool_call = None
        self._reasoning_parser_name = _resolve_parser_name(
            reasoning_parser, detected_reasoning, kind="reasoning"
        )
        self._tool_call_parser_name = _resolve_parser_name(
            tool_call_parser, detected_tool_call, kind="tool_call"
        )
        self._reasoning_parser: ReasoningParser | None = (
            ReasoningParser(model_type=self._reasoning_parser_name, stream_reasoning=False)
            if self._reasoning_parser_name else None
        )
        # Built on demand for generations that carry only the think end tag.
        self._forced_reasoning_parser: ReasoningParser | None = None

    async def _log_decode(self, ids: list[int], fmt: str, *args: Any) -> None:
        if logger.isEnabledFor(logging.DEBUG):
            text = await self.tokenizer_manager.decode(ids, skip_special_tokens=False)
            logger.debug(fmt, *args, text)

    # ── Public API (middleware call order) ──────────────────────────────
    def get_trajectory(self, trajectory_id: str, attempt_id: int) -> Trajectory | None:
        """Return the trajectory for the given id/attempt, or None if not found."""
        entries = self.trajectory_store.get([trajectory_id], attempt_id=attempt_id).get(trajectory_id)
        if not entries:
            logger.warning("[get_trajectory] not found: %s/%d", trajectory_id, attempt_id)
            return None
        return entries[-1]

    def _chat_segment(self, trajectory: Trajectory, segment_id: int) -> list[dict[str, Any]]:
        """Return the chat-completions slice for the given segment_id, or [] if absent."""
        return trajectory.chat_completions.get(segment_id, [])

    async def build_turn_input(
        self,
        trajectory: Trajectory,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        request_kind: str | None = None,
    ) -> TurnInputContext:
        """Compute tokenized input for the next LLM call on this trajectory.

        request_kind: value of the ``request_kind`` request header, used to disambiguate
        a prefix mismatch between context compaction (``None``/``"compaction"``/anything
        other than "collab_spawn") and a subagent fork (``"collab_spawn"``). 
        """
        normalized_messages = self._normalize_messages(messages)
        if tools:
            trajectory.metadata[_REQUEST_TOOLS_KEY] = copy.deepcopy(tools)
        turn_tools = trajectory.metadata.get(_REQUEST_TOOLS_KEY)

        # case 1: first turn — tokenize full messages as the trajectory's root segment.
        if not trajectory.tokens:
            input_ids = await self._tokenize_messages(normalized_messages, tools=tools)
            await self._log_decode(input_ids, "[build_input] %s first turn: %d tokens\n%s",
                                   trajectory.trajectory_id, len(input_ids))
            trajectory.metadata[_NORMALIZED_HISTORY_KEY] = normalized_messages
            return TurnInputContext(
                delta_prompt_ids=input_ids,
                input_ids=list(input_ids),
                start_new_segment=True,
                raw_new_messages=copy.deepcopy(messages),
                tools=turn_tools,
                target_segment_id=0,
                new_segment_parent_id=None,
                new_segment_origin="root",
                new_segment_depth=0,
            )

        # Detect how many of the incoming messages match the active segment's chat prefix.
        active_id = trajectory.active_segment_id
        normalized_history = trajectory.metadata.get(_NORMALIZED_HISTORY_KEY)
        if not isinstance(normalized_history, list):
            normalized_history = self._normalize_messages(self._chat_segment(trajectory, active_id))
            trajectory.metadata[_NORMALIZED_HISTORY_KEY] = normalized_history
        history_len = len(normalized_history)
        prefix_len = history_len if (
            history_len > 0
            and len(normalized_messages) >= history_len
            and normalized_messages[:history_len] == normalized_history
        ) else 0
        logger.debug("[build_input] %s: active_segment=%d history=%d messages=%d prefix=%d",
                     trajectory.trajectory_id, active_id, history_len, len(normalized_messages), prefix_len)

        current_chat_len = len(self._chat_segment(trajectory, active_id))
        raw_new_messages = copy.deepcopy(messages[current_chat_len:] if prefix_len else messages)

        # Case 2: prefix hits the active segment — continuation on the same segment.
        if prefix_len > 0:
            return await self._build_continuation_input(
                trajectory, normalized_messages, prefix_len, raw_new_messages, turn_tools,
                target_segment_id=active_id,
            )

        # Case 3: prefix miss + subagent fork — new placeholder segment. Not written to
        # trajectory.tokens (subagent turns are excluded from the training data), but the
        # subagent's own messages must still be tokenized so the LLM worker receives the
        # correct prompt. active_segment_id stays put. Think content is left untouched
        # (strip_think=False): subagent turns never re-enter the mainline context, so there
        # is no accumulate_reasoning bookkeeping to apply here.
        if request_kind == "collab_spawn":
            parent = trajectory.segments[active_id]
            logger.info(
                "[build_input] %s: subagent fork from segment %d (depth=%d)",
                trajectory.trajectory_id, active_id, parent.depth + 1,
            )
            subagent_messages = self._normalize_messages(messages, strip_think=False)
            input_ids = await self._tokenize_messages(subagent_messages, tools=tools)
            return TurnInputContext(
                delta_prompt_ids=[],
                input_ids=input_ids,
                start_new_segment=True,
                raw_new_messages=raw_new_messages,
                tools=turn_tools,
                target_segment_id=len(trajectory.segments),
                new_segment_parent_id=active_id,
                new_segment_origin="subagent",
                new_segment_depth=parent.depth + 1,
                is_subagent_placeholder=True,
            )

        # Case 4: prefix miss + not a subagent fork — context compaction. New mainline
        # segment, independently re-tokenized from this request's messages.
        logger.info(
            "[build_input] %s: prefix mismatch (active_segment=%d, request_kind=%r); "
            "treating as context compaction, opening new compact segment",
            trajectory.trajectory_id, active_id, request_kind,
        )
        input_ids = await self._tokenize_messages(normalized_messages, tools=tools)
        await self._log_decode(input_ids, "[build_input] %s compact segment: %d tokens\n%s",
                               trajectory.trajectory_id, len(input_ids))
        trajectory.metadata[_NORMALIZED_HISTORY_KEY] = normalized_messages
        return TurnInputContext(
            delta_prompt_ids=input_ids,
            input_ids=input_ids,
            start_new_segment=True,
            raw_new_messages=copy.deepcopy(messages),
            tools=turn_tools,
            target_segment_id=len(trajectory.segments),
            new_segment_parent_id=active_id,
            new_segment_origin="compact",
            new_segment_depth=0,
        )

    async def _build_continuation_input(
        self,
        trajectory: Trajectory,
        normalized_messages: list[dict[str, Any]],
        prefix_len: int,
        raw_new_messages: list[dict[str, Any]],
        turn_tools: list[dict[str, Any]] | None,
        *,
        target_segment_id: int,
    ) -> TurnInputContext:
        """Delta-tokenize a continuation turn appended to target_segment_id (cases 1/2)."""
        if self.enable_thinking and not self.accumulate_reasoning:
            await self._prune_last_response_think(trajectory)

        delta_prompt_ids = await self._tokenize_delta_messages(
            normalized_messages[prefix_len:]
        )
        await self._log_decode(delta_prompt_ids,
                               "[build_input] %s continuation: +%d tokens (total=%d)\n%s",
                               trajectory.trajectory_id, len(delta_prompt_ids),
                               len(trajectory.tokens) + len(delta_prompt_ids))
        routed_experts_start_len = 0
        if self.r3_enabled and trajectory.rollout_routed_experts is not None:
            segment_offset = trajectory.segments[target_segment_id].token_start
            # Subtract 1: _append_routed_experts strips the synthetic terminal entry from
            # the previous turn before extending, so SGLang must re-process that position.
            routed_experts_start_len = trajectory.rollout_routed_experts.shape[0] - 1 - segment_offset
        return TurnInputContext(
            delta_prompt_ids=delta_prompt_ids,
            input_ids=[*trajectory.tokens[trajectory.segments[target_segment_id].token_start:], *delta_prompt_ids],
            start_new_segment=False,
            raw_new_messages=raw_new_messages,
            routed_experts_start_len=routed_experts_start_len,
            tools=turn_tools,
            target_segment_id=target_segment_id,
        )

    def build_assistant_message(
        self,
        response_json: dict[str, Any],
        routed_experts_start_len: int = 0,
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[LLMGenerateResponse, dict[str, Any], str | None]:
        """Decode LLM worker response and build the assistant message.

        Extracts response_ids/logprobs, applies reasoning and tool_call parsing,
        and returns (payload, assistant_message, finish_reason_override).
        """
        meta = response_json.get("meta_info") or {}
        output_token_logprobs = meta.get("output_token_logprobs") or []
        if not output_token_logprobs:
            raise ValueError("output_token_logprobs missing from llm worker response")
        logprobs = [float(item[0]) for item in output_token_logprobs]
        response_ids = [int(item[1]) for item in output_token_logprobs]

        weight_version = _parse_weight_version(meta.get("weight_version"))
        generated_text: str = response_json.get("text") or ""
        routed_experts = None
        routed_experts_b64 = meta.get("routed_experts")
        if routed_experts_b64 and self.r3_enabled:
            prompt_tokens = meta.get("prompt_tokens", 0)
            completion_tokens = meta.get("completion_tokens", 0)
            # sglang captures routing for positions [start_len, seqlen - 1),
            # dropping the terminal position Re-append a synthetic terminal row 
            # so downstream replay (segment offset math, _append_routed_experts) 
            # sees the same shape it always has. The terminal value is irrelevant: 
            # _append_routed_experts overwrites it via merged[-1].copy_(merged[-2]).
            serialized_entries = (
                prompt_tokens + completion_tokens - 1 - routed_experts_start_len
            )
            logger.info(
                "[build_assistant_message] r3: prompt=%d completion=%d start=%d -> %d entries (+1 terminal)",
                prompt_tokens, completion_tokens, routed_experts_start_len, serialized_entries,
            )
            if serialized_entries > 0:
                routed_experts = deserialize_routed_experts(
                    routed_experts_b64, serialized_entries
                )
                routed_experts = torch.cat(
                    [routed_experts, routed_experts[-1:].clone()], dim=0
                )
            else:
                logger.warning(
                    "[build_assistant_message] r3: non-positive entry count (%d); skipping",
                    serialized_entries,
                )

        payload = LLMGenerateResponse(
            generated_text=generated_text,
            response_ids=response_ids,
            logprobs=logprobs,
            weight_version=weight_version,
            routed_experts=routed_experts,
        )

        reasoning_content, visible_text = self._parse_reasoning(generated_text)
        content, tool_calls = self._parse_tool_calls(visible_text, tools)
        if tool_calls and not content:
            content = None  # per OpenAI spec: content is null when tool_calls present
        assistant_message: dict[str, Any] = {"role": "assistant", "content": content}
        finish_reason_override = None
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
            finish_reason_override = "tool_calls"
        if reasoning_content:
            assistant_message["reasoning_content"] = reasoning_content

        return payload, assistant_message, finish_reason_override

    def _parse_reasoning(self, text: str) -> tuple[str | None, str]:
        """Split *text* into ``(reasoning_content, visible_text)`` via the SGLang detector."""
        parser = self._reasoning_parser
        if parser is None:
            return None, text

        detector = parser.detector
        # Some chat templates prefill the think start token, so the generation carries only
        # the end tag. Detectors that do not force reasoning would then treat the whole
        # string (end tag included) as content, so parse those with force_reasoning=True.
        if (
            not detector.force_reasoning
            and detector.think_end_token in text
            and detector.think_start_token not in text
        ):
            if self._forced_reasoning_parser is None:
                self._forced_reasoning_parser = ReasoningParser(
                    model_type=self._reasoning_parser_name,
                    stream_reasoning=False,
                    force_reasoning=True,
                )
            parser = self._forced_reasoning_parser

        reasoning, normal = parser.parse_non_stream(text)
        return (reasoning or None), (normal or "")

    def _parse_tool_calls(
        self,
        text: str,
        tools: list[dict[str, Any]] | None,
    ) -> tuple[str, list[dict[str, Any]] | None]:
        """Extract OpenAI-shaped tool calls from *text* via the SGLang detector."""
        if self._tool_call_parser_name is None:
            return text, None

        parser = FunctionCallParser(
            tools=_to_sglang_tools(tools) or _SENTINEL_TOOLS,
            tool_call_parser=self._tool_call_parser_name,
        )
        # Forward calls whose name is absent from the schema instead of dropping them:
        # agents may omit "tools" on follow-up turns while the LLM keeps emitting markup,
        # and a silently dropped call would leave raw markup in the assistant content.
        # parse_non_stream is synchronous, so the process-wide override cannot interleave.
        with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(True):
            normal_text, calls = parser.parse_non_stream(text)
        if not calls:
            return text, None

        tool_calls = [
            {
                "id": f"call_{index}",
                "type": "function",
                "function": {"name": call.name, "arguments": call.parameters},
            }
            for index, call in enumerate(calls)
        ]
        return normal_text or "", tool_calls

    def update_trajectory(
        self,
        trajectory: Trajectory,
        turn_ctx: TurnInputContext,
        payload: LLMGenerateResponse,
        assistant_message: dict[str, Any],
    ) -> None:
        """Append one completed turn to the trajectory and persist it to the store."""
        # Stale-response guard: only write to the current attempt in GENERATING state.
        stored = self.trajectory_store.get([trajectory.trajectory_id]).get(trajectory.trajectory_id)
        active = stored[-1] if stored else None
        stale = (active is None or active.attempt_id != trajectory.attempt_id
                 or active.status is not TrajectoryStatus.GENERATING)
        if stale:
            logger.warning("[update_trajectory] stale response dropped %s/%d (active=%s status=%s)",
                           trajectory.trajectory_id, trajectory.attempt_id,
                           active.attempt_id if active else None, active.status if active else None)
            return
        trajectory = active

        self._append_triplet(
            trajectory,
            delta_prompt_ids=turn_ctx.delta_prompt_ids,
            response_ids=payload.response_ids,
            logprobs=payload.logprobs,
            weight_version=payload.weight_version,
            turn_ctx=turn_ctx,
        )

        if turn_ctx.is_subagent_placeholder:
            # Subagent requests/responses never enter the training data pipeline: no
            # chat_completions entry, no weight-version bookkeeping, no history-cache update.
            logger.info(
                "[update_trajectory] %s#%d: subagent placeholder segment %d appended, "
                "active_segment_id unchanged (%d)",
                trajectory.trajectory_id, trajectory.attempt_id,
                turn_ctx.target_segment_id, trajectory.active_segment_id,
            )
            self.trajectory_store.update(trajectory.trajectory_id, trajectory)
            return

        if trajectory.start_rollout_weight_version == -1:
            trajectory.start_rollout_weight_version = payload.weight_version
        trajectory.end_rollout_weight_version = payload.weight_version

        if self.r3_enabled and payload.routed_experts is not None:
            self._append_routed_experts(
                trajectory, payload.routed_experts, turn_ctx.start_new_segment
            )
        segment_messages = copy.deepcopy(turn_ctx.raw_new_messages + [assistant_message])
        active_id = trajectory.active_segment_id
        if turn_ctx.start_new_segment or active_id not in trajectory.chat_completions:
            trajectory.chat_completions[active_id] = segment_messages
        else:
            trajectory.chat_completions[active_id].extend(segment_messages)
        trajectory.metadata[_NORMALIZED_HISTORY_KEY] = self._normalize_messages(
            trajectory.chat_completions[active_id]
        )

        logger.debug(
            "[update_trajectory] %s#%d: tokens=%d turns=%d history_cache_len=%d",
            trajectory.trajectory_id, trajectory.attempt_id,
            len(trajectory.tokens), trajectory.num_turns,
            len(trajectory.metadata[_NORMALIZED_HISTORY_KEY]),
        )
        self.trajectory_store.update(trajectory.trajectory_id, trajectory)

    async def _tokenize_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Tokenize *messages* as a full conversation."""
        # Chat templates (Qwen3-Coder's `arguments|items`, Hermes' `arguments|tojson`,
        # ...) expect assistant tool_call arguments as a mapping, but they are stored
        # as OpenAI-style JSON strings. Normalize a copy so the stored history keeps the
        # OpenAI string form while the template sees dicts. Mirrors SGLang's own
        # normalize_assistant_tool_call_arguments call before apply_chat_template.
        messages = copy.deepcopy(messages)
        for message in messages:
            normalize_assistant_tool_call_arguments(message)
        return await self.tokenizer_manager.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tools=tools,
        )

    async def _get_system_prompt_len(self) -> int | None:
        """Length of the template prelude rendered for an empty system message.

        Computed lazily on the first continuation turn instead of at tokenizer-manager
        init. Returns None when the chat template rejects a system-only conversation
        (e.g. Qwen3.5 requires at least one user message); either outcome is cached.
        """
        if self._system_prompt_len is None:
            try:
                sys_ids = await self.tokenizer_manager.apply_chat_template(
                    [{"role": "system", "content": ""}],
                    add_generation_prompt=False,
                    tools=None,
                )
                self._system_prompt_len = len(sys_ids)
            except Exception as exc:
                logger.warning(
                    "Failed to compute system prompt length (%s); "
                    "continuation turns will use the dummy-user fallback", exc
                )
                self._system_prompt_len = -1
        return None if self._system_prompt_len < 0 else self._system_prompt_len

    async def _tokenize_delta_messages(self, new_messages: list[dict[str, Any]]) -> list[int]:
        """Tokenize continuation messages, stripping the shared template prelude.

        Mirrors verl's chat-template helpers (verl/utils/tokenizer/chat_template.py):
        render the delta behind an empty system message and strip its rendered length;
        when the template rejects that (e.g. Qwen3.5), fall back to rendering behind a
        dummy user message and strip that prefix instead. Tools are omitted because
        they are already encoded in trajectory.tokens from the first turn.
        """
        system_prompt_len = await self._get_system_prompt_len()
        if system_prompt_len is not None:
            try:
                # Route through _tokenize_messages so assistant tool_call arguments
                # get normalized for the template, same as full-conversation renders.
                token_ids = await self._tokenize_messages(
                    [{"role": "system", "content": ""}, *new_messages],
                    tools=None,
                )
                if len(token_ids) >= system_prompt_len:
                    return token_ids[system_prompt_len:]
                return token_ids
            except Exception as exc:
                logger.warning(
                    "Empty-system delta tokenization failed (%s); "
                    "retrying with a dummy user prefix", exc
                )
        return await self._tokenize_delta_with_dummy_user(new_messages)

    async def _tokenize_delta_with_dummy_user(
        self, new_messages: list[dict[str, Any]]
    ) -> list[int]:
        """Qwen3.5-style fallback: prefix a dummy user turn, then strip its tokens."""
        dummy_user = [{"role": "user", "content": [{"type": "text", "text": ""}]}]
        if self._dummy_user_prefix_len is None:
            prefix_ids = await self.tokenizer_manager.apply_chat_template(
                dummy_user,
                add_generation_prompt=False,
                tools=None,
            )
            self._dummy_user_prefix_len = len(prefix_ids)
        token_ids = await self._tokenize_messages(
            [*dummy_user, *new_messages],
            tools=None,
        )
        return token_ids[self._dummy_user_prefix_len:]

    def _normalize_messages(
        self, messages: list[dict[str, Any]], *, strip_think: bool = True
    ) -> list[dict[str, Any]]:
        """Strip non-standard fields for prefix comparison.

        strip_think: whether to apply the accumulate_reasoning=False think-stripping to
        assistant content. Callers building input for a subagent fork (case 3) pass False,
        since subagent turns are excluded from training data and their think content
        should be left untouched.
        """
        normalized: list[dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            entry = {
                key: value
                for key, value in message.items()
                if key in _OPENAI_MESSAGE_FIELDS and value is not None
            }
            if role in _OPENAI_MESSAGE_ROLES and "content" not in entry:
                entry["content"] = ""
            if strip_think and not self.accumulate_reasoning and role == "assistant":
                entry["content"] = self._strip_think_from_text(entry.get("content") or "")
            # Canonicalize tool_call arguments to a dict for prefix comparison. We store the
            # SGLang-parsed JSON string (spaces after colons), but clients round-trip it
            # through JSON.stringify (no spaces), so a textual compare spuriously fails and
            # every follow-up turn is misread as context compaction. Compare structurally.
            if isinstance(entry.get("tool_calls"), list):
                entry["tool_calls"] = copy.deepcopy(entry["tool_calls"])
                normalize_assistant_tool_call_arguments(entry)
            normalized.append(entry)
        return normalized

    def _strip_think_from_ids(
        self,
        response_ids: list[int],
        logprobs: list[float],
    ) -> tuple[list[int], list[float], int, int]:
        """Strip think block and return answer tokens/logprobs plus their source slice.

        Cases: 1) <think>…</think>ans → ans  2) </think>ans → ans
               3) <think>… → ([], [], len(response_ids), len(response_ids))
               4) no tags → unchanged
        """
        if not response_ids or not self.think_end_ids:
            return response_ids, logprobs, 0, len(response_ids)

        def find_pattern(pattern: list[int]) -> int:
            """Return the index after the first occurrence of pattern, or -1."""
            pattern_len, first_token = len(pattern), pattern[0]
            search_from = 0
            while True:
                try:
                    i = response_ids.index(first_token, search_from)
                except ValueError:
                    return -1
                if response_ids[i : i + pattern_len] == pattern:
                    return i + pattern_len
                search_from = i + 1

        def rfind_pattern(pattern: list[int]) -> int:
            """Return the index after the LAST occurrence of pattern, or -1."""
            pattern_len, first_token = len(pattern), pattern[0]
            last_found = -1
            search_from = 0
            while True:
                try:
                    i = response_ids.index(first_token, search_from)
                except ValueError:
                    break
                if response_ids[i : i + pattern_len] == pattern:
                    last_found = i + pattern_len
                search_from = i + 1
            return last_found

        # Cases 1 & 2: found </think> — take content after the LAST occurrence
        answer_start = rfind_pattern(self.think_end_ids)
        if answer_start != -1:
            answer_end = len(response_ids)
            content_ids = response_ids[answer_start:]
            content_logprobs = logprobs[answer_start:] if logprobs else []
            return (
                (content_ids, content_logprobs, answer_start, answer_end)
                if content_ids else ([], [], answer_start, answer_start)
            )

        # Case 3: unclosed <think> block — strip everything
        if self.think_start_ids and find_pattern(self.think_start_ids) != -1:
            return [], [], len(response_ids), len(response_ids)

        # Case 4: no think markers
        return response_ids, logprobs, 0, len(response_ids)

    def _strip_think_from_text(self, text: str) -> str:
        """Remove think-block from text, returning only visible content (mirrors _strip_think_from_ids).

        Examples:
            # Full think-block (both open and close tags present)
            "<think>reasoning</think>Hello" -> "Hello"

            # Open tag only, no close tag (streaming: think-block not yet closed)
            "<think>reasoning..." -> ""

            # Close tag only (streaming: open tag was consumed in a previous chunk)
            "reasoning</think>Hello" -> "Hello"

            # Multiple close tags (LLM occasionally emits duplicates)
            "</think>\n\n</think>\n\nHello" -> "Hello"

            # No tags at all, return text as-is
            "Hello world" -> "Hello world"
        """
        think_start, think_end = self.think_tags
        has_start = think_start in text
        has_end = think_end in text
        if not has_start and not has_end:
            return text
        if has_start:
            _, _, after = text.partition(think_start)
            last_end = after.rfind(think_end)
            if last_end >= 0:
                return after[last_end + len(think_end):].strip("\n")
            return ""
        # No open tag, only close tag(s): take content after the LAST close tag
        last_end = text.rfind(think_end)
        return text[last_end + len(think_end):].strip("\n")

    async def _prune_last_response_think(self, trajectory: Trajectory) -> None:
        """Strip think tokens from the most recent response in-place (called on continuation turns)."""
        if not trajectory.segments:
            return
        last_segment = trajectory.segments[trajectory.active_segment_id]
        if not last_segment.triplets:
            return

        last_triplet = last_segment.triplets[-1]

        logprob_s = last_triplet.logprob_start
        logprob_e = last_triplet.logprob_end
        response_len = logprob_e - logprob_s
        if response_len == 0:
            return

        token_s = last_triplet.token_end - response_len
        token_e = last_triplet.token_end
        response_ids = trajectory.tokens[token_s:token_e]
        response_logprobs = trajectory.rollout_log_probs[logprob_s : logprob_e]
        response_loss_masks = trajectory.loss_masks[logprob_s : logprob_e]
        response_weight_versions = trajectory.rollout_weight_versions[logprob_s : logprob_e]

        await self._log_decode(response_ids, "[prune_think] %s: before pruning, response=%d tokens: %r",
                               trajectory.trajectory_id, len(response_ids))
        pruned_ids, pruned_logprobs, keep_start, keep_end = self._strip_think_from_ids(
            list(response_ids), list(response_logprobs)
        )

        if len(pruned_ids) == response_len:
            logger.debug("[prune_think] %s: no think block found (response=%d tokens)",
                         trajectory.trajectory_id, response_len)
            return

        pruned_len = len(pruned_ids)
        delta = response_len - pruned_len
        logger.info("[prune_think] %s: removed %d think tokens (%d -> %d)",
                    trajectory.trajectory_id, delta, response_len, pruned_len)

        # Replace the response slice in each array, keeping array lengths consistent.
        # Use del + insert to ensure each array is actually shortened by `delta` elements
        # rather than relying on Python's variable-length slice assignment.
        del trajectory.tokens[token_s:token_e]
        trajectory.tokens[token_s:token_s] = pruned_ids

        del trajectory.rollout_log_probs[logprob_s : logprob_e]
        trajectory.rollout_log_probs[logprob_s : logprob_s] = pruned_logprobs

        pruned_loss_masks = response_loss_masks[keep_start:keep_end] if pruned_len else []
        del trajectory.loss_masks[logprob_s : logprob_e]
        trajectory.loss_masks[logprob_s : logprob_s] = pruned_loss_masks

        pruned_weight_versions = response_weight_versions[keep_start:keep_end] if pruned_len else []
        del trajectory.rollout_weight_versions[logprob_s : logprob_e]
        trajectory.rollout_weight_versions[logprob_s : logprob_s] = pruned_weight_versions

        logger.info(
            "[prune_think] %s: mask/version slice %d -> %d, mask_sum %d -> %d, version_len %d -> %d",
            trajectory.trajectory_id,
            len(response_loss_masks),
            len(pruned_loss_masks),
            sum(response_loss_masks),
            sum(pruned_loss_masks),
            len(response_weight_versions),
            len(pruned_weight_versions),
        )
        assert len(trajectory.rollout_weight_versions) == len(trajectory.rollout_log_probs), (
            "rollout_weight_versions length "
            f"{len(trajectory.rollout_weight_versions)} != rollout_log_probs length "
            f"{len(trajectory.rollout_log_probs)}"
        )

        last_triplet.token_end -= delta
        last_triplet.logprob_end -= delta
        last_segment.token_end = last_triplet.token_end
        last_segment.logprob_end = last_triplet.logprob_end

        await self._log_decode(pruned_ids, "[prune_think] %s: after pruning, response=%d tokens: %r",
                               trajectory.trajectory_id, pruned_len)

        # Strip think content from the corresponding chat_completions entry
        active_id = trajectory.active_segment_id
        cc = trajectory.chat_completions.get(active_id, [])
        if cc and cc[-1].get("role") == "assistant":
            stripped = self._strip_think_from_text(cc[-1].get("content") or "")
            if stripped != cc[-1].get("content"):
                cc[-1] = {**cc[-1], "content": stripped}
                trajectory.metadata.pop(_NORMALIZED_HISTORY_KEY, None)
                logger.debug(
                    "[prune_think] %s: cache invalidated — cc[-1].content changed, stripped think block",
                    trajectory.trajectory_id,
                )

        # Sync R3 tensor: remove think-block routing entries from the response slice
        if self.r3_enabled and trajectory.rollout_routed_experts is not None:
            response_start = token_s
            experts_rows = trajectory.rollout_routed_experts.shape[0]
            if response_start < 0 or response_start + delta > experts_rows:
                logger.warning("[prune_think] %s: r3 shape mismatch (rows=%d start=%d delta=%d) — skip",
                               trajectory.trajectory_id, experts_rows, response_start, delta)
            else:
                trajectory.rollout_routed_experts = torch.cat([
                    trajectory.rollout_routed_experts[:response_start],
                    trajectory.rollout_routed_experts[response_start + delta:],
                ], dim=0)

    def _append_triplet(
        self,
        trajectory: Trajectory,
        *,
        delta_prompt_ids: list[int],
        response_ids: list[int],
        logprobs: list[float],
        weight_version: int,
        turn_ctx: TurnInputContext,
    ) -> None:
        """Append a (prompt, response, logprobs) triplet and update segment bookkeeping.

        Case 3 (subagent placeholder): appends an empty placeholder Segment carrying only
        tree metadata, and returns without touching tokens/loss_masks/rollout_log_probs/
        rollout_weight_versions or trajectory.active_segment_id.
        """
        if turn_ctx.is_subagent_placeholder:
            trajectory.segments.append(
                Segment(
                    triplets=[],
                    token_start=len(trajectory.tokens),
                    token_end=len(trajectory.tokens),
                    logprob_start=len(trajectory.rollout_log_probs),
                    logprob_end=len(trajectory.rollout_log_probs),
                    segment_id=turn_ctx.target_segment_id,
                    parent_segment_id=turn_ctx.new_segment_parent_id,
                    origin="subagent",
                    depth=turn_ctx.new_segment_depth,
                    trainable=False,
                )
            )
            return

        start_new_segment = turn_ctx.start_new_segment
        is_initial_turn = not trajectory.tokens
        # token_start: new segment starts here; continuation includes all prior segment tokens.
        if start_new_segment or not trajectory.segments:
            token_start = len(trajectory.tokens)
        else:
            token_start = trajectory.segments[trajectory.active_segment_id].token_start

        trajectory.tokens.extend(delta_prompt_ids)

        if not is_initial_turn and delta_prompt_ids:
            trajectory.loss_masks.extend([0] * len(delta_prompt_ids))
            trajectory.rollout_log_probs.extend([0.0] * len(delta_prompt_ids))
            trajectory.rollout_weight_versions.extend([-1] * len(delta_prompt_ids))

        assert len(logprobs) == len(response_ids), (
            f"logprobs length {len(logprobs)} != response_ids length {len(response_ids)}"
        )

        trajectory.tokens.extend(response_ids)
        trajectory.loss_masks.extend([1] * len(response_ids))
        response_logprob_start = len(trajectory.rollout_log_probs)
        trajectory.rollout_log_probs.extend(logprobs)
        trajectory.rollout_weight_versions.extend(
            [weight_version] * len(response_ids)
        )

        assert len(trajectory.rollout_weight_versions) == len(trajectory.rollout_log_probs), (
            "rollout_weight_versions length "
            f"{len(trajectory.rollout_weight_versions)} != rollout_log_probs length "
            f"{len(trajectory.rollout_log_probs)}"
        )

        triplet = Triplet(
            token_start=token_start,
            token_end=len(trajectory.tokens),
            logprob_start=response_logprob_start,
            logprob_end=len(trajectory.rollout_log_probs),
        )

        if start_new_segment or not trajectory.segments:
            trajectory.segments.append(
                Segment(
                    triplets=[triplet],
                    token_start=triplet.token_start,
                    token_end=triplet.token_end,
                    logprob_start=triplet.logprob_start,
                    logprob_end=triplet.logprob_end,
                    segment_id=turn_ctx.target_segment_id,
                    parent_segment_id=turn_ctx.new_segment_parent_id,
                    origin=turn_ctx.new_segment_origin,
                    depth=turn_ctx.new_segment_depth,
                    trainable=True,
                )
            )
            trajectory.active_segment_id = turn_ctx.target_segment_id
        else:
            segment = trajectory.segments[trajectory.active_segment_id]
            segment.triplets.append(triplet)
            segment.token_end = triplet.token_end
            segment.logprob_end = triplet.logprob_end

        trajectory.num_turns += 1

    def _append_routed_experts(
        self,
        trajectory: Trajectory,
        routed_experts_chunk: torch.Tensor,
        start_new_segment: bool = False,
    ) -> None:
        """Extend the rollout routing map with a new turn's chunk.

        Continuation strips the previous synthetic terminal (SGLang re-captures that
        position as a prompt token this turn). A new segment must NOT strip it: its
        prompt is re-tokenized fresh, so that placeholder is never re-captured — keep it
        so shape[0] stays aligned with len(tokens) per segment.
        """
        base = trajectory.rollout_routed_experts
        if base is None:
            merged = routed_experts_chunk.clone()
        elif start_new_segment:
            merged = torch.cat([base, routed_experts_chunk], dim=0)
        else:
            merged = torch.cat([base[:-1], routed_experts_chunk], dim=0)
        if merged.size(0) > 1:
            merged[-1].copy_(merged[-2])
        trajectory.rollout_routed_experts = merged
        assert merged.size(0) == len(trajectory.tokens), (
            f"r3 rows {merged.size(0)} != tokens {len(trajectory.tokens)} "
            f"({trajectory.trajectory_id}#{trajectory.attempt_id} turn {trajectory.num_turns}, "
            f"start_new_segment={start_new_segment})"
        )
        logger.debug("[append_r3] %s#%d: total=%s",
                     trajectory.trajectory_id, trajectory.attempt_id, list(merged.shape))
