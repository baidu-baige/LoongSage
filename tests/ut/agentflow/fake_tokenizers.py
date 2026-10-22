"""Char-level fake tokenizers rendering Qwen / DeepSeek-V4 style chat templates.

One token per character keeps offsets trivial, so tests can compare decoded
masked spans against the exact expected text.  ``FakeMergingTokenizer`` instead
fuses chosen substrings into single multi-char tokens, to exercise loss-mask
projection when a token straddles a mask boundary (as a real BPE merge would).

The render methods mirror the behaviours that matter for the loss mask, pinned
against the real templates:
  * Qwen3 (``tests/.../qwen3.jinja``): a turn's ``reasoning_content`` is kept
    only after the last user turn; earlier reasoning is dropped.
  * DeepSeek-V4 (sglang ``encoding_dsv4``): thinking is dropped for turns at or
    before the last user turn unless tools are present, so history turns open
    with ``</think>`` and the final turn opens with ``<think>``.

The ``FakeQwen3`` / ``FakeDeepSeekV41`` / ``FakeLlama3`` / ``FakePlainText`` classes at the
end back ``test_sft_tokenizer_probe.py``; see each docstring for how it differs from the
similarly named class above.
"""

from __future__ import annotations

from typing import Any


class FakeCharTokenizer:
    """Minimal HF fast-tokenizer surface used by the SFT loss mask / sampler."""

    def __init__(self, model_family: str | None = None, with_offsets: bool = True):
        if model_family is not None:
            self.model_family = model_family
        self.with_offsets = with_offsets

    def __call__(self, text: str, add_special_tokens: bool = False, return_offsets_mapping: bool = False):
        out: dict[str, Any] = {"input_ids": self.encode(text)}
        if return_offsets_mapping and self.with_offsets:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(ch) for ch in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        return "".join(chr(t) for t in token_ids)

    def render(self, messages, tools, add_generation_prompt, **kwargs) -> str:
        raise NotImplementedError

    def apply_chat_template(
        self, messages, tools=None, tokenize=True, add_generation_prompt=False, **kwargs
    ):
        text = self.render(messages, tools, add_generation_prompt, **kwargs)
        return self.encode(text) if tokenize else text

    @staticmethod
    def _last_user_index(messages) -> int:
        """Index of the last ``user`` message (templates key history-drop off this)."""
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                return i
        return len(messages) - 1


class FakeQwenTokenizer(FakeCharTokenizer):
    """Qwen3.5-like: the generation prompt pre-fills ``gen_prompt_suffix``.

    Qwen3.5 pre-fills ``<think>\\n``; Qwen3 pre-fills ``""`` with thinking on, or
    the empty think block with ``enable_thinking: false``.  ``SFTTokenizer`` probes
    this suffix at runtime, so overriding it exercises the probe branches.
    """

    gen_prompt_suffix = "<think>\n"

    def render(self, messages, tools, add_generation_prompt, **kwargs) -> str:
        last_user = self._last_user_index(messages)
        parts = []
        if tools:
            parts.append(f"<|im_start|>system\n# Tools\n{tools}<|im_end|>\n")
        for index, msg in enumerate(messages):
            role, content = msg["role"], msg.get("content") or ""
            if role == "assistant":
                body = content
                # Qwen3 keeps a turn's reasoning only after the last user turn
                # (jinja ``loop.index0 > last_query_index``); earlier reasoning
                # is dropped from the rendered history.
                if msg.get("reasoning_content") and index > last_user:
                    body = f"<think>\n{msg['reasoning_content']}\n</think>\n\n{content}"
                for call in msg.get("tool_calls") or []:
                    body += f"\n<tool_call>{call['function']['name']}</tool_call>"
                parts.append(f"<|im_start|>assistant\n{body}<|im_end|>\n")
            elif role == "tool":
                parts.append(f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>\n")
            else:
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        if add_generation_prompt:
            parts.append(f"<|im_start|>assistant\n{self.gen_prompt_suffix}")
        return "".join(parts)


class FakeQwenThinkingOffTokenizer(FakeQwenTokenizer):
    """Qwen3 with ``enable_thinking: false``: the gen prompt pre-fills an empty think block."""

    gen_prompt_suffix = "<think>\n\n</think>\n\n"


class FakeQwenNoPrefillTokenizer(FakeQwenTokenizer):
    """Qwen3 with thinking on: the gen prompt pre-fills nothing (model emits ``<think>`` itself)."""

    gen_prompt_suffix = ""


class FakeDeepSeekV4Tokenizer(FakeCharTokenizer):
    """Mirrors sglang ``encoding_dsv4`` for the cases the loss mask cares about.

    ``<｜Assistant｜>`` plus a think token (``<think>``/``</think>``) follows each
    user/tool turn as the generation prompt.  In thinking mode the reasoning of
    turns at or before the last user turn is dropped (``drop_thinking``) unless
    any message carries tools, so those turns open with ``</think>`` and only the
    final turn opens with ``<think>`` + reasoning.  Pinned against
    ``encode_messages(..., thinking_mode=...)``.
    """

    BOS = "<｜begin▁of▁sentence｜>"

    def __init__(self, **kwargs):
        super().__init__(model_family="deepseek_v4", **kwargs)

    def render(self, messages, tools, add_generation_prompt, **kwargs) -> str:
        thinking = bool(kwargs.get("thinking", False) or kwargs.get("enable_thinking", False))
        # sglang forces drop_thinking off as soon as any message defines tools.
        drop = not (tools or any(msg.get("tools") for msg in messages))
        last_user = self._last_user_index(messages)
        out = self.BOS
        for index, msg in enumerate(messages):
            role, content = msg["role"], msg.get("content") or ""
            if role == "assistant":
                keep_reasoning = thinking and (not drop or index > last_user)
                reasoning = f"{msg.get('reasoning_content') or ''}</think>" if keep_reasoning else ""
                out += f"{reasoning}{content}<｜end▁of▁sentence｜>"
            elif role in ("user", "tool"):
                body = f"<tool_result>{content}</tool_result>" if role == "tool" else content
                out += f"<｜User｜>{body}"
                # The Assistant header + think token is the generation prompt into
                # the next assistant turn; sglang only emits it before an assistant.
                if index + 1 < len(messages) and messages[index + 1].get("role") == "assistant":
                    out += "<｜Assistant｜>" + self._think_token(index, last_user, thinking, drop)
            elif role == "system":
                out += content
        if add_generation_prompt:
            out += "<｜Assistant｜>" + self._think_token(len(messages) - 1, last_user, thinking, drop)
        return out

    @staticmethod
    def _think_token(index: int, last_user: int, thinking: bool, drop: bool) -> str:
        if thinking and (not drop or index >= last_user):
            return "<think>"
        return "</think>"


class FakeMergingTokenizer(FakeQwenTokenizer):
    """Qwen char tokenizer that fuses chosen substrings into single multi-char tokens.

    Lets a test place a token across a loss-mask seam (header/content or
    skip-prefix/reasoning) to pin the generator's any-overlap projection: a token
    is trainable iff it overlaps any trainable char.
    """

    def __init__(self, merges, **kwargs):
        super().__init__(**kwargs)
        self._merges = sorted((m for m in merges if m), key=len, reverse=True)
        self._str2id: dict[str, int] = {}
        self._id2str: dict[int, str] = {}
        self._next_id = 0x110000  # just past Unicode so ids never collide with ord()

    def _segment(self, text: str) -> list[tuple[str, int, int]]:
        pieces, i = [], 0
        while i < len(text):
            for merge in self._merges:
                if text.startswith(merge, i):
                    pieces.append((merge, i, i + len(merge)))
                    i += len(merge)
                    break
            else:
                pieces.append((text[i], i, i + 1))
                i += 1
        return pieces

    def _id_for(self, piece: str) -> int:
        if len(piece) == 1:
            return ord(piece)
        if piece not in self._str2id:
            self._str2id[piece] = self._next_id
            self._id2str[self._next_id] = piece
            self._next_id += 1
        return self._str2id[piece]

    def __call__(self, text: str, add_special_tokens: bool = False, return_offsets_mapping: bool = False):
        pieces = self._segment(text)
        out: dict[str, Any] = {"input_ids": [self._id_for(p) for p, _, _ in pieces]}
        if return_offsets_mapping and self.with_offsets:
            out["offset_mapping"] = [(start, end) for _, start, end in pieces]
        return out

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self._id_for(p) for p, _, _ in self._segment(text)]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        # Not dict.get(t, chr(t)): the chr() default would be evaluated for fused
        # ids too (which are out of chr()'s range) and raise.
        return "".join(self._id2str[t] if t in self._id2str else chr(t) for t in token_ids)


# ---------------------------------------------------------------------------
# Probe-test templates: renders pinned against the real Qwen3 / DeepSeek-V4.1 / Llama-3 layouts.
# ---------------------------------------------------------------------------

QWEN_IM_END = "<|im_end|>"
DS_BOS = "<｜begin▁of▁sentence｜>"
DS_EOS = "<｜end▁of▁sentence｜>"
DS_EFFORT = "Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n"



class FakeQwen3Tokenizer(FakeCharTokenizer):
    """Qwen3 (e.g. Qwen3-4B): turns after the last user turn always get a think block.

    History turns render without one; the final turn renders ``<think>\\n{reasoning}\\n</think>\\n\\n``
    even when the reasoning is empty.  ``enable_thinking=False`` only changes the
    generation prompt, which then pre-fills the empty think block.

    Unlike ``FakeQwenTokenizer`` (Qwen3.5-like: the generation prompt pre-fills
    ``gen_prompt_suffix`` and a think block is rendered only when there is reasoning),
    this pins the real Qwen3 renders used by ``test_sft_tokenizer_probe.py``.
    """

    def render(self, messages, tools, add_generation_prompt, **kwargs) -> str:
        last_user = self._last_user_index(messages)
        out = ""
        for index, msg in enumerate(messages):
            role, content = msg["role"], msg.get("content") or ""
            if role == "assistant" and index > last_user:
                reasoning = (msg.get("reasoning_content") or "").strip()
                content = f"<think>\n{reasoning}\n</think>\n\n{content}"
            out += f"<|im_start|>{role}\n{content}{QWEN_IM_END}\n"
        if add_generation_prompt:
            out += "<|im_start|>assistant\n"
            if kwargs.get("enable_thinking") is False:
                out += "<think>\n\n</think>\n\n"
        return out


class FakeDeepSeekV41Tokenizer(FakeCharTokenizer):
    """DeepSeek-V4.1 official encoder: no ChatML, special tokens back to back.

    The ``<｜Assistant｜>`` header plus a think token follows each user turn.  Chat mode
    always uses ``</think>``.  Thinking mode prepends a reasoning-effort system prefix,
    drops reasoning on turns before the last user turn (they open with ``</think>``)
    and opens the final turn with ``<think>{reasoning}</think>``.

    Unlike ``FakeDeepSeekV4Tokenizer`` (V4: no ``<｜System｜>`` token, no reasoning-effort
    prefix, tool-aware ``drop_thinking``), this pins the real V4.1 renders used by
    ``test_sft_tokenizer_probe.py`` and ignores tools.
    """

    def __init__(self, **kwargs):
        super().__init__(model_family="deepseek_v41", **kwargs)

    def render(self, messages, tools, add_generation_prompt, **kwargs) -> str:
        thinking = bool(kwargs.get("thinking"))
        last_user = self._last_user_index(messages)
        system = "".join(m.get("content") or "" for m in messages if m["role"] == "system")
        out = DS_BOS
        if thinking or system:
            out += "<｜System｜>" + (DS_EFFORT if thinking else "") + system
        for index, msg in enumerate(messages):
            role, content = msg["role"], msg.get("content") or ""
            if role == "user":
                out += f"<｜User｜>{content}"
                if index + 1 < len(messages) and messages[index + 1]["role"] == "assistant":
                    out += "<｜Assistant｜>" + ("<think>" if thinking and index + 1 > last_user else "</think>")
            elif role == "assistant":
                if thinking and index > last_user:
                    out += f"{msg.get('reasoning_content') or ''}</think>"
                out += f"{content}{DS_EOS}"
        if add_generation_prompt:
            out += "<｜Assistant｜>" + ("<think>" if thinking else "</think>")
        return out


class FakeLlama3Tokenizer(FakeCharTokenizer):
    """Llama-3 style headers: ``<|start_header_id|>{role}<|end_header_id|>\\n\\n ... <|eot_id|>``."""

    def render(self, messages, tools, add_generation_prompt, **kwargs) -> str:
        out = "<|begin_of_text|>"
        for msg in messages:
            out += f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n{msg.get('content') or ''}<|eot_id|>"
        if add_generation_prompt:
            out += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return out


class FakePlainTextTokenizer(FakeCharTokenizer):
    """``User: ... / Assistant: ...`` with no closing token: nothing marks where a reply ends."""

    def render(self, messages, tools, add_generation_prompt, **kwargs) -> str:
        out = "".join(f"{msg['role'].title()}: {msg.get('content') or ''}\n" for msg in messages)
        return out + ("Assistant: " if add_generation_prompt else "")
