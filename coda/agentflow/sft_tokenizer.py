"""SFT tokenization: token ids plus assistant-turn loss masks from the rendered chat template.

The whole conversation is rendered once with the tokenizer's chat template and
re-tokenized with ``return_offsets_mapping``.  Each assistant turn is located in
the rendered text by its ``assistant_header`` and closed by its ``end_marker``;
the characters in between (minus a leading generation-prompt prefix such as
``<think>\\n``) are marked trainable and projected onto tokens via the offset
mapping.

The three markers are not configured per model: at construction a sentinel
conversation is rendered with the active chat template and the text the template
wraps around the sentinels is read back as ``assistant_header`` /
``skip_prefixes`` / ``end_marker``, then verified by masking that same
conversation.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from coda.agentflow.tokenizer_manager import _ensure_token_list

# One history and one final assistant turn.  Plain ASCII letters: no template
# strips, splits or escapes them.
_PROBE = [
    {"role": "user", "content": "LSUSERPROBEA"},
    {"role": "assistant", "content": "LSASSTPROBEA"},
    {"role": "user", "content": "LSUSERPROBEB"},
    {"role": "assistant", "content": "LSASSTPROBEB"},
]
_U1, _A1, _U2, _A2 = (message["content"] for message in _PROBE)


class SFTTokenizer:
    """Tokenize a conversation and mark the assistant tokens to train on."""

    def __init__(self, tokenizer: Any, generation_prompt_kwargs: Mapping[str, Any] | None = None):
        self.tokenizer = tokenizer
        self.template_kwargs = dict(generation_prompt_kwargs or {})
        self.assistant_header, self.end_marker, self.skip_prefixes = self._probe_markers()
        self._self_check()

    def _render(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, **kwargs: Any) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, **kwargs, **self.template_kwargs
        )

    def _probe_markers(self) -> tuple[str, str, list[str]]:
        """Derive ``(assistant_header, end_marker, skip_prefixes)`` from the sentinel renders.

        - Header: common prefix of a history turn's opening and the generation prompt,
          minus the user turn's own closing (only if that closing is a strict prefix).
        - Skip prefixes: what either opening adds beyond their common prefix (pre-filled
          at inference), longest first so that when one starts with another, ``tokenize``
          skips the longer one.
        - End marker: common prefix of the history and final turn's closing, right-stripped
          because what follows it (newline, next turn) differs between the two.
        """
        full = self._render(_PROBE, None, add_generation_prompt=False)
        gen = self._render(_PROBE[:1], None, add_generation_prompt=True)
        # Raw probe renders, kept for logging what the markers were parsed from.
        self.probe_full, self.probe_gen = full, gen
        open_hist = full.partition(_U1)[2].partition(_A1)[0]
        open_gen = gen.partition(_U1)[2]
        opening = os.path.commonprefix([open_hist, open_gen])
        skip_prefixes = sorted(
            {text[len(opening):] for text in (open_hist, open_gen)} - {""},
            key=lambda prefix: (-len(prefix), prefix),
        )

        # Drop the user turn's own closing (ChatML "<|im_end|>\n") so an assistant
        # turn need not follow a user turn.  Strict prefix only: DeepSeek's encoder
        # appends the generation prompt to a trailing user turn by itself.
        user_tail = self._render(_PROBE[:1], None, add_generation_prompt=False).partition(_U1)[2]
        header = opening
        if len(user_tail) < len(opening) and opening.startswith(user_tail):
            header = opening[len(user_tail):]

        close_hist = full.partition(_A1)[2].partition(_U2)[0]
        end_marker = os.path.commonprefix([close_hist, full.partition(_A2)[2]]).rstrip()
        return header, end_marker, skip_prefixes

    def _self_check(self) -> None:
        """Mask the sentinel conversation and verify the trained spans.

        Only the final turn's suffix is pinned: it may open with model output the
        template renders for an empty reasoning block (DeepSeek's ``</think>``).
        """
        spans = decode_trainable_spans(self.tokenizer, *self.tokenize(_PROBE))
        history, final = _A1 + self.end_marker, _A2 + self.end_marker
        if (
            len(spans) != 2
            or spans[0] != history
            or not spans[1].endswith(final)
            or any(user in span for span in spans for user in (_U1, _U2))
        ):
            raise ValueError(
                f"chat-template probe self-check failed: trained spans {spans!r}, expected "
                f"[{history!r}, ...{final!r}] (assistant_header={self.assistant_header!r}, "
                f"end_marker={self.end_marker!r}, skip_prefixes={self.skip_prefixes!r}); "
                "chat template unsupported"
            )

    def tokenize(
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
                    "template; the conversation does not follow the probed chat-template layout"
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


def trainable_suffix_length(loss_mask: list[int]) -> int:
    """Length of the suffix that starts at the first trainable token (0 if none)."""
    for index, value in enumerate(loss_mask):
        if value:
            return len(loss_mask) - index
    return 0


def decode_trainable_spans(tokenizer: Any, token_ids: list[int], loss_mask: list[int]) -> list[str]:
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
