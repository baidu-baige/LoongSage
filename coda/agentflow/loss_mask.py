"""Multi-turn SFT loss masks derived from the rendered chat template.

The whole conversation is rendered once with the tokenizer's chat template and
re-tokenized with ``return_offsets_mapping``.  Each assistant turn is located in
the rendered text by its ``assistant_header`` and closed by its ``end_marker``;
the characters in between (minus a leading generation-prompt prefix such as
``<think>\\n``) are marked trainable and projected onto tokens via the offset
mapping.  This mirrors slime's ``gen_multi_turn_loss_mask_qwen3_5`` with built-in
template presets for Qwen and DeepSeek-V4/V4.1.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from coda.agentflow.tokenizer_manager import _ensure_token_list

# ``skip_prefixes: None`` means "derive from the generation prompt" (see
# ``LossMaskGenerator._probe_generation_prefix``).
LOSS_MASK_PRESETS: dict[str, dict[str, Any]] = {
    # Qwen2.5/3/3.5 chat models that use the ChatML markers below.  The
    # generation-time thinking prefix is probed from the active chat template,
    # because it differs between model versions and thinking configurations.
    "qwen": {
        "assistant_header": "<|im_start|>assistant\n",
        "end_marker": "<|im_end|>",
        "skip_prefixes": None,
    },
    # DeepSeek-V4 and V4.1 use the same official assistant/eos markers.  The
    # official encoder emits ``<｜Assistant｜>`` plus ``<think>`` (thinking turn)
    # or ``</think>`` (chat / dropped-thinking turn) after the user message;
    # both are part of the generation prompt, not of the assistant's output.
    "deepseek_v4": {
        "assistant_header": "<｜Assistant｜>",
        "end_marker": "<｜end▁of▁sentence｜>",
        "skip_prefixes": ["<think>", "</think>"],
    },
}

_PROBE_MESSAGES = [{"role": "user", "content": "probe"}]


def resolve_loss_mask_config(mask_cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    """Resolve ``sft.loss_mask.preset`` to built-in template parameters.

    The preset must be set explicitly: SFT only supports the chat-template
    families in ``LOSS_MASK_PRESETS``.
    """
    preset = dict(mask_cfg or {}).get("preset")
    if preset not in LOSS_MASK_PRESETS:
        raise ValueError(
            f"sft.loss_mask.preset must be set to one of {list(LOSS_MASK_PRESETS)}, got {preset!r}; "
            "SFT supports only these chat-template families"
        )
    return dict(LOSS_MASK_PRESETS[preset], preset=preset)


class LossMaskGenerator:
    """Tokenize a conversation and mark the assistant tokens to train on."""

    def __init__(
        self,
        tokenizer: Any,
        mask_cfg: Mapping[str, Any] | None = None,
        generation_prompt_kwargs: Mapping[str, Any] | None = None,
    ):
        self.tokenizer = tokenizer
        self.template_kwargs = dict(generation_prompt_kwargs or {})
        resolved = resolve_loss_mask_config(mask_cfg)
        self.preset: str = resolved["preset"]
        self.assistant_header: str = resolved["assistant_header"]
        self.end_marker: str = resolved["end_marker"]
        if resolved["skip_prefixes"] is None:
            prefix = self._probe_generation_prefix()
            resolved["skip_prefixes"] = [prefix] if prefix else []
        self.skip_prefixes: list[str] = resolved["skip_prefixes"]

    def _render(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, **kwargs: Any) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, **kwargs, **self.template_kwargs
        )

    def _probe_generation_prefix(self) -> str:
        """Text the template appends after ``assistant_header`` in the generation prompt.

        Inference pre-fills it (e.g. Qwen3.5's ``<think>\\n``, or Qwen3's empty
        think block with ``enable_thinking: false``), so the model must not be
        trained to produce it.
        """
        text = self._render(_PROBE_MESSAGES, None, add_generation_prompt=True)
        pos = text.rfind(self.assistant_header)
        if pos < 0:
            raise ValueError(
                f"assistant_header {self.assistant_header!r} not found in the generation prompt "
                f"{text!r}; set sft.loss_mask.preset to match a supported chat template"
            )
        return text[pos + len(self.assistant_header):]

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[list[int], list[int]]:
        """Return ``(token_ids, loss_mask)`` over the full rendered conversation.

        Assistant messages with ``step_loss_mask: 0`` are kept as context but not
        trained on.

        Raises:
            ValueError: If the tokenizer gives no offsets, the re-tokenized text
                drifts from ``apply_chat_template(tokenize=True)``, or an
                assistant turn cannot be located in the rendered text.
        """
        rendered = self._render(messages, tools, add_generation_prompt=False)
        try:
            encoded = self.tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        except NotImplementedError as exc:  # slow (Python) tokenizers have no offsets
            raise ValueError("SFT loss masks need a fast tokenizer with return_offsets_mapping support") from exc
        token_ids = _ensure_token_list(encoded["input_ids"])
        offsets = encoded.get("offset_mapping")
        if offsets is None:
            raise ValueError("SFT loss masks need a fast tokenizer with return_offsets_mapping support")

        expected = _ensure_token_list(
            self.tokenizer.apply_chat_template(
                messages, tools=tools, tokenize=True, add_generation_prompt=False, **self.template_kwargs
            )
        )
        if token_ids != expected:
            raise ValueError(
                "Re-tokenized chat template text does not match apply_chat_template(tokenize=True); "
                "the char-offset loss mask would be misaligned"
            )

        char_mask = [0] * len(rendered)
        cursor = 0
        for index, message in enumerate(messages):
            if message.get("role") != "assistant":
                continue
            header_pos = rendered.find(self.assistant_header, cursor)
            if header_pos < 0:
                raise ValueError(
                    f"assistant message {index} has no {self.assistant_header!r} header in the rendered "
                    "template; check sft.loss_mask against the chat template"
                )
            start = header_pos + len(self.assistant_header)
            end_pos = rendered.find(self.end_marker, start)
            if end_pos < 0:
                raise ValueError(f"assistant message {index} is not closed by {self.end_marker!r}")
            end = end_pos + len(self.end_marker)
            cursor = end

            # ``== 0`` rather than ``!= 1``: parquet fills the field with None
            # on rows that omit it, which must still be trained.
            if message.get("step_loss_mask") == 0:
                continue
            for prefix in self.skip_prefixes:
                if rendered.startswith(prefix, start):
                    start += len(prefix)
                    break
            char_mask[start:end] = [1] * (end - start)

        prefix_sum = [0]
        for value in char_mask:
            prefix_sum.append(prefix_sum[-1] + value)
        loss_mask = [
            1 if end > start and prefix_sum[end] - prefix_sum[start] > 0 else 0
            for start, end in offsets
        ]
        return token_ids, loss_mask


def get_response_length(loss_mask: list[int]) -> int:
    """Length of the suffix that starts at the first trainable token (0 if none)."""
    for index, value in enumerate(loss_mask):
        if value:
            return len(loss_mask) - index
    return 0


def decode_masked_spans(tokenizer: Any, token_ids: list[int], loss_mask: list[int]) -> list[str]:
    """Decode each contiguous trainable run, to eyeball what the model learns."""
    spans: list[str] = []
    run: list[int] = []
    for token_id, value in zip(token_ids, loss_mask, strict=True):
        if value:
            run.append(token_id)
        elif run:
            spans.append(tokenizer.decode(run))
            run = []
    if run:
        spans.append(tokenizer.decode(run))
    return spans
