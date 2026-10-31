"""Char-level fake tokenizers rendering Qwen / DeepSeek / Llama-3 style chat templates.

One token per character, so tests can compare decoded masked spans against the
exact expected text.  ``special_tokens`` each encode to one token, as a real
tokenizer's added tokens do.

The render methods mirror the behaviours that matter for the loss mask, pinned
against the real templates:
  * Qwen3.5 / Qwen3: a turn's ``reasoning_content`` is kept only after the last
    user turn; earlier reasoning is dropped.  Qwen3.5 rejects messages without a
    user message.
  * DeepSeek-V4 / V4.1 (sglang ``encoding_dsv4`` / ``encoding_dsv41``): see
    ``_FakeDeepSeekEncoder``.

The ``FakeQwen3`` / ``FakeLlama3`` classes at the end back
``test_sft_tokenizer_probe.py``; see each docstring for how it differs from the
similarly named class above.
"""

from __future__ import annotations

from jinja2.exceptions import TemplateError

QWEN_IM_END = "<|im_end|>"
DS_BOS = "<｜begin▁of▁sentence｜>"
DS_EOS = "<｜end▁of▁sentence｜>"
DS_EFFORT = "Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n"


class FakeCharTokenizer:
    """Minimal HF tokenizer surface used by the SFT loss mask / sampler.

    One token per character, except ``special_tokens``: like a real tokenizer's
    added tokens, each encodes to a single token.
    """

    special_tokens: tuple[str, ...] = ()

    def __init__(self, model_family: str | None = None):
        if model_family is not None:
            self.model_family = model_family
        self._merges = sorted(self.special_tokens, key=len, reverse=True)
        self._str2id: dict[str, int] = {}
        self._id2str: dict[int, str] = {}
        self._next_id = 0x110000  # just past Unicode so ids never collide with ord()

    def _segment(self, text: str) -> list[str]:
        pieces, i = [], 0
        while i < len(text):
            for merge in self._merges:
                if text.startswith(merge, i):
                    pieces.append(merge)
                    i += len(merge)
                    break
            else:
                pieces.append(text[i])
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

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [self._id_for(piece) for piece in self._segment(text)]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        # Not dict.get(t, chr(t)): the chr() default would be evaluated for fused
        # ids too (which are out of chr()'s range) and raise.
        return "".join(self._id2str[t] if t in self._id2str else chr(t) for t in token_ids)

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
    the empty think block with ``enable_thinking: false``.  ``BaseTokenizerManager`` probes
    this suffix at runtime, so overriding it exercises the probe branches.  Like
    Qwen3.5's template, it rejects messages without a user message and a system
    message anywhere but first.
    """

    gen_prompt_suffix = "<think>\n"

    def render(self, messages, tools, add_generation_prompt, **kwargs) -> str:
        if not any(msg["role"] == "user" for msg in messages):
            raise TemplateError("No user query found in messages.")
        if any(msg["role"] == "system" for msg in messages[1:]):
            raise TemplateError("System message must be at the beginning.")
        last_user = self._last_user_index(messages)
        parts = []
        if tools:
            parts.append(f"<|im_start|>system\n# Tools\n{tools}<|im_end|>\n")
        for index, msg in enumerate(messages):
            role, content = msg["role"], msg.get("content") or ""
            if role == "assistant":
                body = content
                # Qwen3.5 gives every turn after the last user turn a think block, empty
                # without reasoning (jinja ``loop.index0 > last_query_index``); earlier
                # turns render none, so their reasoning is dropped from the history.
                if index > last_user:
                    body = f"<think>\n{msg.get('reasoning_content') or ''}\n</think>\n\n{content}"
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
    """Qwen3.5 with ``enable_thinking: false``: the gen prompt pre-fills an empty think block."""

    gen_prompt_suffix = "<think>\n\n</think>\n\n"


class FakeQwenNoPrefillTokenizer(FakeQwenTokenizer):
    """Qwen3 with thinking on: the gen prompt pre-fills nothing (model emits ``<think>`` itself)."""

    gen_prompt_suffix = ""


def _merge_user_turns(messages) -> list[dict]:
    """Merge consecutive user / tool messages into one user turn, as sglang's DeepSeek encoders do.

    Contents are joined by ``"\\n\\n"``; a tool message renders as ``<tool_result>...</tool_result>``.
    ``has_text`` records whether a merged turn holds a real user message, not only tool results.
    """
    turns: list[dict] = []
    for msg in messages:
        role, content = msg["role"], msg.get("content") or ""
        if role not in ("user", "tool"):
            turns.append(msg)
            continue
        text = f"<tool_result>{content}</tool_result>" if role == "tool" else content
        if turns and turns[-1]["role"] == "user":
            turns[-1]["content"] += "\n\n" + text
            turns[-1]["has_text"] = turns[-1]["has_text"] or role == "user"
        else:
            turns.append({"role": "user", "content": text, "has_text": role == "user"})
    return turns


class _FakeDeepSeekEncoder(FakeCharTokenizer):
    """What sglang's ``encoding_dsv4`` / ``encoding_dsv41`` render for plain-text messages.

    Pinned against ``encode_messages(..., thinking_mode=...)``:

    * There is no ``add_generation_prompt`` (nor in LoongSage's wrapper): the
      ``<｜Assistant｜>`` header plus a think token follows every user turn that is
      last or followed by an assistant message.
    * Consecutive user / tool messages merge into one user turn (``_merge_user_turns``).
    * Thinking mode keeps an assistant turn's ``{reasoning}</think>`` only after the
      last user turn, unless any message carries tools (``drop_thinking`` off).  The
      think token after a user turn is then ``<think>`` from the last user turn on and
      ``</think>`` before it; chat mode always uses ``</think>``.
    """

    special_tokens = (DS_BOS, DS_EOS, "<｜System｜>", "<｜User｜>", "<｜Assistant｜>", "<think>", "</think>")
    # Whether a user turn holding only tool results counts as the last user turn.
    tool_results_count_as_user = True

    def render(self, messages, tools, add_generation_prompt, **kwargs) -> str:
        thinking = bool(kwargs.get("thinking") or kwargs.get("enable_thinking"))
        drop = not (tools or any(msg.get("tools") for msg in messages))
        turns = _merge_user_turns(messages)
        last_user = max(
            (
                index
                for index, turn in enumerate(turns)
                if turn["role"] == "user" and (self.tool_results_count_as_user or turn["has_text"])
            ),
            default=-1,
        )
        out = DS_BOS + self._preamble(thinking)
        for index, turn in enumerate(turns):
            role, content = turn["role"], turn.get("content") or ""
            if role == "system":
                out += self._system(content, index, thinking)
            elif role == "user":
                out += "<｜User｜>" + content
                if index + 1 == len(turns) or turns[index + 1]["role"] == "assistant":
                    opens_thinking = thinking and (not drop or index >= last_user)
                    out += "<｜Assistant｜>" + ("<think>" if opens_thinking else "</think>")
            elif role == "assistant":
                if thinking and (not drop or index > last_user):
                    out += f"{turn.get('reasoning_content') or ''}</think>"
                out += content + DS_EOS
        return out

    def _preamble(self, thinking: bool) -> str:
        return ""

    def _system(self, content: str, index: int, thinking: bool) -> str:
        return content


class FakeDeepSeekV4Tokenizer(_FakeDeepSeekEncoder):
    """DeepSeek-V4 (``encoding_dsv4``): system content renders as is, without a system token."""

    def __init__(self, **kwargs):
        super().__init__(model_family="deepseek_v4", **kwargs)


# ---------------------------------------------------------------------------
# Probe-test templates: renders pinned against the real Qwen3 / Llama-3 layouts.
# ---------------------------------------------------------------------------


class FakeQwen3Tokenizer(FakeCharTokenizer):
    """Qwen3 (e.g. Qwen3-4B): turns after the last user turn always get a think block.

    History turns render without one; the final turn renders ``<think>\\n{reasoning}\\n</think>\\n\\n``
    even when the reasoning is empty.  ``enable_thinking=False`` only changes the
    generation prompt, which then pre-fills the empty think block.

    Unlike ``FakeQwenTokenizer`` (Qwen3.5-like: the generation prompt pre-fills
    ``gen_prompt_suffix`` and a system-only conversation is rejected), this pins the
    real Qwen3 renders used by ``test_sft_tokenizer_probe.py``.
    """

    special_tokens = ("<|im_start|>", QWEN_IM_END, "<think>", "</think>")

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


class FakeDeepSeekV41Tokenizer(_FakeDeepSeekEncoder):
    """DeepSeek-V4.1 (``encoding_dsv41``): adds a ``<｜System｜>`` token and, in thinking mode,
    a reasoning-effort system prefix before the first message; a user turn of only tool
    results does not count as the last user turn."""

    tool_results_count_as_user = False

    def __init__(self, **kwargs):
        super().__init__(model_family="deepseek_v41", **kwargs)

    def _preamble(self, thinking: bool) -> str:
        return "<｜System｜>" + DS_EFFORT if thinking else ""

    def _system(self, content: str, index: int, thinking: bool) -> str:
        # In thinking mode the first message's system token is already in the preamble.
        return ("" if thinking and index == 0 else "<｜System｜>") + content


class FakeLlama3Tokenizer(FakeCharTokenizer):
    """Llama-3 style headers: ``<|start_header_id|>{role}<|end_header_id|>\\n\\n ... <|eot_id|>``."""

    special_tokens = ("<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>")

    def render(self, messages, tools, add_generation_prompt, **kwargs) -> str:
        out = "<|begin_of_text|>"
        for msg in messages:
            out += f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n{msg.get('content') or ''}<|eot_id|>"
        if add_generation_prompt:
            out += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return out
