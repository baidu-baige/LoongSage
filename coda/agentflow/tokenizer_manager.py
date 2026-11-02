"""Tokenizer execution manager for CPU-bound chat-template and encode calls.

The manager also probes the chat template's non-assistant / assistant prefixes, and
tokenizes SFT conversations with their assistant loss masks.
"""

from __future__ import annotations

import asyncio
import copy
import importlib
import json
import logging
import os
import types
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from pathlib import Path
from collections.abc import Mapping
from typing import Any

from coda.utils.path_utils import resolve_conf_path

logger = logging.getLogger(__name__)

_PROCESS_TOKENIZER: Any = None
DEFAULT_THINK_TAGS = ("<think>", "</think>")
DEEPSEEK_V4_MODEL_FAMILY = "deepseek_v4"
DEEPSEEK_V41_MODEL_FAMILY = "deepseek_v41"
# DeepSeek checkpoints ship no Jinja chat template; SGLang carries an official
# encoder per family instead. Keyed by the family the checkpoint declares.
DEEPSEEK_OFFICIAL_ENCODER_MODULES = {
    DEEPSEEK_V41_MODEL_FAMILY: "sglang.srt.entrypoints.openai.encoding_dsv41",
    DEEPSEEK_V4_MODEL_FAMILY: "sglang.srt.entrypoints.openai.encoding_dsv4",
}
# Probe messages: the prefixes are read off their renders.
_EMPTY_SYSTEM = {"role": "system", "content": ""}
_EMPTY_USER = {"role": "user", "content": ""}


def _deepseek_v4_thinking_mode(generation_prompt_kwargs: Mapping[str, Any]) -> str:
    """Resolve the DeepSeek-V4 prompt and completion parsing mode."""
    enabled = generation_prompt_kwargs.get("thinking", False) or generation_prompt_kwargs.get(
        "enable_thinking", False
    )
    return "thinking" if enabled else "chat"


def _deepseek_official_family(model_path: str) -> str | None:
    """DeepSeek family whose official SGLang encoder renders this checkpoint.

    V4.1 is matched before V4 because it reuses the ``DeepseekV4`` architecture
    substring (mirrors SGLang's ``is_deepseek_v41_config``).
    """
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        return None
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    arch = (config.get("architectures") or [""])[0]
    model_type = config.get("model_type") or ""
    if model_type == DEEPSEEK_V41_MODEL_FAMILY or "DeepseekV41" in arch:
        return DEEPSEEK_V41_MODEL_FAMILY
    if model_type == DEEPSEEK_V4_MODEL_FAMILY or "DeepseekV4" in arch:
        return DEEPSEEK_V4_MODEL_FAMILY
    return None


def _wrap_deepseek_v4_tokenizer(tokenizer: Any, family: str) -> Any:
    """Expose the official DeepSeek encoder through the HF tokenizer API."""
    # Only V4.1 renders an empty system message as a system token; V4 renders it
    # to nothing, so only tools justify inserting one there.
    v41 = family == DEEPSEEK_V41_MODEL_FAMILY
    encode_messages = importlib.import_module(
        DEEPSEEK_OFFICIAL_ENCODER_MODULES[family]
    ).encode_messages

    def apply_chat_template(
        self: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str | list[int]:
        conversation = copy.deepcopy(messages)
        if v41:
            # V4.1 consumes OpenAI content parts and requires a string content.
            for message in conversation:
                if message.get("content") is None:
                    message["content"] = ""
        if tools:
            if not conversation or conversation[0].get("role") != "system":
                conversation.insert(0, {"role": "system", "content": ""})
            conversation[0]["tools"] = tools

        thinking_mode = _deepseek_v4_thinking_mode(kwargs)
        prompt = encode_messages(conversation, thinking_mode=thinking_mode)
        if not kwargs.get("tokenize", True):
            return prompt
        return self.encode(prompt, add_special_tokens=False)

    tokenizer.apply_chat_template = types.MethodType(apply_chat_template, tokenizer)
    tokenizer.model_family = family
    return tokenizer


def _load_chat_template(template_path: str | None) -> str | None:
    """Read a Jinja2 chat template from a config-declared path.

    Args:
        template_path: Path to the template file, either absolute or relative to the project
                       `conf/` directory (e.g. `chat_template/bcp.jinja`).  Empty/`None` keeps
                       the tokenizer's built-in template.

    Returns:
        The template text, or `None` when no path is configured.

    Raises:
        FileNotFoundError: If the resolved path does not exist.
    """
    if not template_path:
        return None

    resolved = resolve_conf_path(template_path)
    logger.info("Loading custom chat template from %s", resolved)

    return Path(resolved).read_text(encoding="utf-8")


def load_tokenizer(model_path: str, custom_chat_template: str | None = None) -> Any:
    """Load a HuggingFace tokenizer and optionally override its chat template."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(model_path)),
        trust_remote_code=True,
    )
    if custom_chat_template:
        tokenizer.chat_template = custom_chat_template
    else:
        family = _deepseek_official_family(model_path)
        if family is not None:
            tokenizer = _wrap_deepseek_v4_tokenizer(tokenizer, family)
    return tokenizer


def _init_process_tokenizer(model_path: str, custom_chat_template: str | None) -> None:
    """Load one tokenizer per worker process."""
    global _PROCESS_TOKENIZER
    _PROCESS_TOKENIZER = load_tokenizer(model_path, custom_chat_template)


def _ensure_token_list(obj: Any) -> list[int]:
    """Normalize tokenizer outputs into a flat ``list[int]``.

    This handles Transformers 4/5 differences where ``apply_chat_template(tokenize=True)``
    may return either ``list[int]`` or a ``BatchEncoding``/mapping with ``input_ids``.
    """
    if hasattr(obj, "keys") and "input_ids" in obj:
        return _ensure_token_list(obj["input_ids"])
    if isinstance(obj, dict):
        raise TypeError(
            f"Tokenizer returned dict without 'input_ids': {list(obj.keys())}"
        )
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, list):
        return obj
    return list(obj)


def _normalize_think_tags(
    think_tags: Any | None,
) -> tuple[str, str]:
    """Normalize configurable reasoning delimiters to a `(start, end)` pair."""
    if think_tags is None:
        return DEFAULT_THINK_TAGS
    if not isinstance(think_tags, (list, tuple)) or len(think_tags) != 2:
        raise ValueError(
            "tokenizer.think_tags must be a list/tuple of two strings, "
            f"got {think_tags!r}"
        )
    start, end = think_tags
    if not isinstance(start, str) or not isinstance(end, str):
        raise ValueError(
            "tokenizer.think_tags entries must both be strings, "
            f"got {think_tags!r}"
        )
    if not start or not end:
        raise ValueError(
            "tokenizer.think_tags entries must both be non-empty strings, "
            f"got {think_tags!r}"
        )
    return start, end


# ---------------------------------------------------------------------------
# Process-mode worker entrypoints (module-level, run in child processes)
# ---------------------------------------------------------------------------

def _apply_chat_template_process(
    messages: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
    generation_prompt_kwargs: dict[str, Any],
    tools: list[dict[str, Any]] | None = None,
) -> list[int]:
    """Worker entrypoint for apply_chat_template in process mode."""
    if _PROCESS_TOKENIZER is None:
        raise RuntimeError("Process tokenizer not initialized")
    return _ensure_token_list(
        _PROCESS_TOKENIZER.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            **generation_prompt_kwargs,
        )
    )


def _encode_process(text: str) -> list[int]:
    """Worker entrypoint for encode in process mode."""
    if _PROCESS_TOKENIZER is None:
        raise RuntimeError("Process tokenizer not initialized")
    return _ensure_token_list(
        _PROCESS_TOKENIZER.encode(text, add_special_tokens=False)
    )


def _decode_process(token_ids: list[int], *, skip_special_tokens: bool) -> str:
    """Worker entrypoint for decode in process mode."""
    if _PROCESS_TOKENIZER is None:
        raise RuntimeError("Process tokenizer not initialized")
    return str(_PROCESS_TOKENIZER.decode(token_ids, skip_special_tokens=skip_special_tokens))


# ---------------------------------------------------------------------------
# Base class: abstract interface + common init
# ---------------------------------------------------------------------------

class BaseTokenizerManager(ABC):
    """Async tokenizer interface with common metadata.

    Attributes:
        non_assistant_prefix_message: Empty message a continuation message is rendered behind:
            an empty system message, or an empty user message when the chat template
            rejects a system-only conversation (Qwen3.5 needs a user message).
        non_assistant_prefix: Token ids ``non_assistant_prefix_message`` renders to, e.g. a lone BOS token
            (DeepSeek) or an empty system / user turn; cut off every render behind it.
        assistant_prefix: Token ids of an empty user message rendered with the
            generation prompt (the assistant header plus anything pre-filled after it);
            an assistant message is rendered behind the empty user message and this
            prefix is cut off, leaving what the model generates.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        mode: str,
        num_workers: int,
        generation_prompt_kwargs: dict[str, Any] | None = None,
        thought_words: tuple[str, str] | list[str] | None = None,
    ):
        self.tokenizer = tokenizer
        self.model_family = getattr(tokenizer, "model_family", None)
        self.mode = mode
        self.num_workers = num_workers
        self.generation_prompt_kwargs: dict[str, Any] = generation_prompt_kwargs or {}
        if self.model_family in DEEPSEEK_OFFICIAL_ENCODER_MODULES:
            self.thinking_mode = _deepseek_v4_thinking_mode(
                self.generation_prompt_kwargs
            )
        self.think_tags = _normalize_think_tags(thought_words)
        self.think_start_ids = _ensure_token_list(
            tokenizer.encode(self.think_tags[0], add_special_tokens=False)
        )
        self.think_end_ids = _ensure_token_list(
            tokenizer.encode(self.think_tags[1], add_special_tokens=False)
        )
        self.non_assistant_prefix_message, self.non_assistant_prefix = self._get_non_assistant_prefix()
        self.assistant_prefix = self._render([_EMPTY_USER], add_generation_prompt=True)
        logger.info(
            "probed from chat template: non_assistant_prefix=%r (empty %s message) assistant_prefix=%r",
            tokenizer.decode(self.non_assistant_prefix),
            self.non_assistant_prefix_message["role"],
            tokenizer.decode(self.assistant_prefix),
        )

    def _render(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        """Render ``messages`` to token ids in the calling thread, with ``generation_prompt_kwargs``."""
        return _ensure_token_list(
            self.tokenizer.apply_chat_template(
                messages,
                tools=tools,
                add_generation_prompt=add_generation_prompt,
                tokenize=True,
                **self.generation_prompt_kwargs,
            )
        )

    def _get_non_assistant_prefix(self) -> tuple[dict[str, Any], list[int]]:
        """Probe ``non_assistant_prefix_message`` and the ``non_assistant_prefix`` it renders to.

        empty system message, unless the chat template rejects a system-only conversation
        (Qwen3.5 requires at least one user message); then an empty user message.

        Returns:
            ``(non_assistant_prefix_message, non_assistant_prefix)``.

        Raises:
            Exception: The chat template rejects the empty user message too.
        """
        try:
            return _EMPTY_SYSTEM, self._render([_EMPTY_SYSTEM])
        except Exception as exc:
            logger.warning(
                "chat template rejects a system-only conversation (%s); "
                "continuation messages are rendered behind an empty user message", exc
            )
            return _EMPTY_USER, self._render([_EMPTY_USER])

    def tokenize(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[list[int], list[int]]:
        """Return the SFT ``(input_ids, loss_mask)`` of a conversation, in the calling thread.

        The conversation is cut into the pieces an RL rollout produces: each turn of
        consecutive non-assistant messages is a prompt, each assistant message a
        response.  The first turn is rendered at once with ``tools`` (an RL first turn);
        each later turn behind ``non_assistant_prefix_message`` with ``non_assistant_prefix`` cut off (an
        RL continuation turn, as the router's delta tokenization does).  A turn followed
        by an assistant message ends with the generation prompt, so it is context.
        An assistant message is rendered behind an empty user message, which makes it
        the turn after the last user query, and ``assistant_prefix`` is cut off: the
        rest is what the model generates after the generation prompt, and is trained.

        Templates that render a message depending on the messages around it make the
        concatenation differ from the whole conversation rendered at once: Qwen and
        DeepSeek drop the reasoning of history turns, while here every assistant
        message keeps its own.  The concatenation is trained on as is.

        Args:
            messages: OpenAI-style conversation; only ``role == "assistant"`` messages are trained.
                Not modified; assistant tool_call arguments given as JSON strings are parsed
                into mappings before rendering, as the router does.
            tools: Tool schemas the chat template renders with the prompt, if any.

        Returns:
            ``input_ids`` and a same-length 0/1 ``loss_mask``; both empty for no messages.

        Raises:
            ValueError: The conversation starts with an assistant message; a continuation
                turn does not render behind ``non_assistant_prefix`` so cutting it off would corrupt
                the context; or an assistant message does not render as a continuation of the
                generation prompt, e.g. it carries ``reasoning_content`` while the generation
                prompt pre-fills an empty think block (Qwen with ``enable_thinking: false``).
            Exception: The chat template rejects the prompt (Qwen3.5 given no user message
                before the first assistant message) or a turn rendered behind
                ``non_assistant_prefix_message`` (Qwen3.5 given a system message after the first).
        """
        if not messages:
            return [], []
        # Imported here, not at module level: importing sglang registers its own
        # DeepseekV41Config with transformers' AutoConfig, replacing Megatron-Bridge's,
        # in every process that imports this module (train workers via coda.custom).
        from sglang.srt.entrypoints.openai.serving_chat import (  # noqa: PLC0415
            normalize_assistant_tool_call_arguments,
        )

        # As the router's _tokenize_messages: chat templates expect assistant tool_call
        # arguments as a mapping (Qwen3.5 iterates them with ``|items``) while the data
        # stores OpenAI-style JSON strings, so a copy is normalized for rendering.
        messages = copy.deepcopy(messages)
        for message in messages:
            normalize_assistant_tool_call_arguments(message)
        # The prompt is the messages before the first assistant message; a conversation
        # must not start with an assistant message, so there is at least one.
        prompt_len = next(
            (index for index, message in enumerate(messages) if message.get("role") == "assistant"),
            len(messages),
        )
        if prompt_len == 0:
            raise ValueError("conversation must not start with an assistant message")
        turns: list[list[dict[str, Any]]] = [messages[:prompt_len]]
        for message in messages[prompt_len:]:
            if message.get("role") == "assistant" or turns[-1][-1].get("role") == "assistant":
                turns.append([message])
            else:
                turns[-1].append(message)

        input_ids: list[int] = []
        loss_mask: list[int] = []
        for index, turn in enumerate(turns):
            if turn[0].get("role") == "assistant":
                turn_ids = self._tokenize_assistant_message(turn[0])
                trained = 1
            else:
                # A non-assistant turn followed by an assistant message ends with the generation prompt
                # that opens that reply, as an RL turn does; the last turn of a conversation that ends
                # without a reply (or has no assistant message at all) does not.
                opens_reply = index + 1 < len(turns)
                if index == 0:
                    turn_ids = self._render(turn, tools, add_generation_prompt=opens_reply)
                else:
                    turn_ids = self._tokenize_non_assistant_messages(turn, add_generation_prompt=opens_reply)
                trained = 0
            input_ids += turn_ids
            loss_mask += [trained] * len(turn_ids)
        return input_ids, loss_mask

    def _tokenize_non_assistant_messages(
        self, messages: list[dict[str, Any]], *, add_generation_prompt: bool
    ) -> list[int]:
        """Tokenize non-assistant messages behind ``non_assistant_prefix_message``, as an RL continuation turn.

        Same rendering as the router's ``_tokenize_delta_messages`` (render behind
        ``non_assistant_prefix_message``, then cut ``non_assistant_prefix`` off); kept separate
        because the router runs async, and its delta can carry an assistant message.

        Raises:
            ValueError: The messages rendered behind ``non_assistant_prefix_message`` do not start
                with ``non_assistant_prefix``, so cutting it off would corrupt the context.
        """
        input_ids = self._render([self.non_assistant_prefix_message, *messages], add_generation_prompt=add_generation_prompt)
        if input_ids[: len(self.non_assistant_prefix)] != self.non_assistant_prefix:
            raise ValueError(
                f"messages {[m.get('role') for m in messages]} render as {self.tokenizer.decode(input_ids)!r}, "
                f"which does not start with the non-assistant prefix {self.tokenizer.decode(self.non_assistant_prefix)!r}"
            )
        return input_ids[len(self.non_assistant_prefix) :]

    def _tokenize_assistant_message(self, message: dict[str, Any]) -> list[int]:
        """Tokenize one assistant message as the model generates it after the generation prompt.

        Raises:
            ValueError: The message rendered behind an empty user message does not start
                with ``assistant_prefix``.
        """
        input_ids = self._render([_EMPTY_USER, message])
        if input_ids[: len(self.assistant_prefix)] != self.assistant_prefix:
            raise ValueError(
                f"assistant message renders as {self.tokenizer.decode(input_ids)!r}, which does not "
                f"continue the generation prompt {self.tokenizer.decode(self.assistant_prefix)!r}; "
                "reasoning_content must match the thinking mode in generation_prompt_kwargs: "
                "drop it when thinking is disabled, provide it when thinking is enabled"
            )
        return input_ids[len(self.assistant_prefix) :]

    @abstractmethod
    async def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool = True,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        """Apply chat template asynchronously."""

    @abstractmethod
    async def encode(self, text: str) -> list[int]:
        """Encode plain text asynchronously."""

    @abstractmethod
    async def decode(self, token_ids: list[int], *, skip_special_tokens: bool = False) -> str:
        """Decode token IDs to text asynchronously."""

    @abstractmethod
    def shutdown(self) -> None:
        """Release executor resources."""


# ---------------------------------------------------------------------------
# Thread-mode implementation
# ---------------------------------------------------------------------------

class ThreadedTokenizerManager(BaseTokenizerManager):
    """Tokenizer manager backed by a thread pool."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        num_workers: int,
        generation_prompt_kwargs: dict[str, Any] | None = None,
        thought_words: tuple[str, str] | list[str] | None = None,
    ):
        super().__init__(
            tokenizer,
            mode="thread",
            num_workers=num_workers,
            generation_prompt_kwargs=generation_prompt_kwargs,
            thought_words=thought_words,
        )
        self._executor = ThreadPoolExecutor(
            max_workers=num_workers,
            thread_name_prefix="tokenizer",
        )

    async def _run(self, fn, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            partial(fn, *args, **kwargs),
        )

    async def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool = True,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        return await self._run(
            lambda msgs, agp, tls: _ensure_token_list(
                self.tokenizer.apply_chat_template(
                    msgs, tokenize=True, add_generation_prompt=agp,
                    tools=tls,
                    **self.generation_prompt_kwargs,
                )
            ),
            messages,
            add_generation_prompt,
            tools,
        )

    async def encode(self, text: str) -> list[int]:
        return await self._run(
            lambda t: _ensure_token_list(
                self.tokenizer.encode(t, add_special_tokens=False)
            ),
            text,
        )

    async def decode(self, token_ids: list[int], *, skip_special_tokens: bool = False) -> str:
        return await self._run(
            lambda ids, sst: str(self.tokenizer.decode(ids, skip_special_tokens=sst)),
            token_ids,
            skip_special_tokens,
        )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)


# ---------------------------------------------------------------------------
# Process-mode implementation
# ---------------------------------------------------------------------------

class ProcessTokenizerManager(BaseTokenizerManager):
    """Tokenizer manager backed by worker processes.

    The main process keeps a tokenizer instance for init-time metadata;
    runtime calls run inside child processes with their own tokenizer.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        model_path: str,
        custom_chat_template: str | None,
        num_workers: int,
        generation_prompt_kwargs: dict[str, Any] | None = None,
        thought_words: tuple[str, str] | list[str] | None = None,
    ):
        super().__init__(
            tokenizer,
            mode="process",
            num_workers=num_workers,
            generation_prompt_kwargs=generation_prompt_kwargs,
            thought_words=thought_words,
        )
        self.model_path = model_path
        self.custom_chat_template = custom_chat_template
        self._executor = ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_process_tokenizer,
            initargs=(model_path, custom_chat_template),
        )

    async def _run(self, fn, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            partial(fn, *args, **kwargs),
        )

    async def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool = True,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int]:
        return await self._run(
            _apply_chat_template_process,
            messages,
            add_generation_prompt=add_generation_prompt,
            generation_prompt_kwargs=self.generation_prompt_kwargs,
            tools=tools,
        )

    async def encode(self, text: str) -> list[int]:
        return await self._run(_encode_process, text)

    async def decode(self, token_ids: list[int], *, skip_special_tokens: bool = False) -> str:
        return await self._run(_decode_process, token_ids, skip_special_tokens=skip_special_tokens)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_tokenizer_manager(tokenizer_config: Any, model_path: str) -> BaseTokenizerManager:
    """Build tokenizer manager from YAML config."""
    if not model_path:
        raise ValueError("hf_model_path is required in config")

    custom_chat_template = _load_chat_template(tokenizer_config.custom_chat_template_path)
    generation_prompt_kwargs = dict(tokenizer_config.generation_prompt_kwargs or {})
    think_tags = getattr(tokenizer_config, "think_tags", None) or getattr(tokenizer_config, "thought_words", None)
    manager_config = tokenizer_config.manager
    mode = str(manager_config.mode).strip().lower() if manager_config else "thread"
    default_workers = min(4, os.cpu_count() or 1)
    num_workers = int(manager_config.num_workers if manager_config and manager_config.num_workers else default_workers)
    if num_workers < 1:
        raise ValueError(f"tokenizer.manager.num_workers must be >= 1, got {num_workers}")

    logger.info(
        "Loading tokenizer manager: mode=%s workers=%d model=%s",
        mode,
        num_workers,
        model_path,
    )
    tokenizer = load_tokenizer(model_path, custom_chat_template)

    if mode == "thread":
        return ThreadedTokenizerManager(
            tokenizer,
            num_workers=num_workers,
            generation_prompt_kwargs=generation_prompt_kwargs,
            thought_words=think_tags,
        )
    if mode == "process":
        return ProcessTokenizerManager(
            tokenizer,
            model_path=str(Path(model_path)),
            custom_chat_template=custom_chat_template,
            num_workers=num_workers,
            generation_prompt_kwargs=generation_prompt_kwargs,
            thought_words=think_tags,
        )

    raise ValueError(
        f"Unsupported tokenizer.manager.mode: {mode!r} (expected 'thread' or 'process')"
    )
