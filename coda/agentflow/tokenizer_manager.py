"""Tokenizer execution manager for CPU-bound chat-template and encode calls."""

from __future__ import annotations

import asyncio
import copy
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


def _deepseek_v4_thinking_mode(generation_prompt_kwargs: Mapping[str, Any]) -> str:
    """Resolve the DeepSeek-V4 prompt and completion parsing mode."""
    enabled = generation_prompt_kwargs.get("thinking", False) or generation_prompt_kwargs.get(
        "enable_thinking", False
    )
    return "thinking" if enabled else "chat"


def _is_deepseek_v4(model_path: str) -> bool:
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        return False
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    return (
        config.get("model_type") == DEEPSEEK_V4_MODEL_FAMILY
        or "DeepseekV4ForCausalLM" in config.get("architectures", [])
    )


def _wrap_deepseek_v4_tokenizer(tokenizer: Any) -> Any:
    """Expose the official DeepSeek-V4 encoder through the HF tokenizer API."""
    from sglang.srt.entrypoints.openai.encoding_dsv4 import encode_messages

    def apply_chat_template(
        self: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str | list[int]:
        conversation = copy.deepcopy(messages)
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
    tokenizer.model_family = DEEPSEEK_V4_MODEL_FAMILY
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
    elif _is_deepseek_v4(model_path):
        tokenizer = _wrap_deepseek_v4_tokenizer(tokenizer)
    return tokenizer


def _init_process_tokenizer(model_path: str, custom_chat_template: str | None) -> None:
    """Load one tokenizer per worker process."""
    global _PROCESS_TOKENIZER
    _PROCESS_TOKENIZER = load_tokenizer(model_path, custom_chat_template)


def _ensure_token_list(obj: Any) -> list[int]:
    """Normalize tokenizer outputs to a plain list of ints."""
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
    """Async tokenizer interface with common metadata."""

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
        if self.model_family == DEEPSEEK_V4_MODEL_FAMILY:
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
