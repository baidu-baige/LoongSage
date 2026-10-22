"""Unit tests for coda.agentflow.sft_tokenizer.

Run: python -m pytest tests/ut/agentflow/test_sft_tokenizer.py -v
"""

import os

import pytest

from coda.agentflow.sft_tokenizer import (
    SFTTokenizer,
    decode_trainable_spans,
    trainable_suffix_length,
)
from tests.ut.agentflow.fake_tokenizers import (
    FakeDeepSeekV4Tokenizer,
    FakeMergingTokenizer,
    FakeQwenNoPrefillTokenizer,
    FakeQwenThinkingOffTokenizer,
    FakeQwenTokenizer,
)

IM_END = "<|im_end|>"
EOS = "<｜end▁of▁sentence｜>"


def _spans(tokenizer, messages, tools=None, **template_kwargs):
    sft_tok = SFTTokenizer(tokenizer, template_kwargs)
    ids, mask = sft_tok.tokenize(messages, tools=tools)
    assert len(ids) == len(mask)
    return decode_trainable_spans(tokenizer, ids, mask)


# ---------------------------------------------------------------------------
# Marker probing
# ---------------------------------------------------------------------------

def test_qwen_skip_prefix_probed_from_generation_prompt():
    sft_tok = SFTTokenizer(FakeQwenTokenizer())
    assert sft_tok.skip_prefixes == ["<think>\n"]


def test_qwen_no_prefill_probes_no_skip_prefix():
    sft_tok = SFTTokenizer(FakeQwenNoPrefillTokenizer())
    assert sft_tok.skip_prefixes == []


def test_qwen_thinking_off_probes_empty_think_block():
    sft_tok = SFTTokenizer(FakeQwenThinkingOffTokenizer())
    assert sft_tok.skip_prefixes == ["<think>\n\n</think>\n\n"]


# ---------------------------------------------------------------------------
# Qwen-style template
# ---------------------------------------------------------------------------

def test_single_turn_masks_only_assistant_content_and_end_marker():
    messages = [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    assert _spans(FakeQwenTokenizer(), messages) == ["hello" + IM_END]


def test_multi_turn_masks_every_assistant_turn():
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    assert _spans(FakeQwenTokenizer(), messages) == ["a1" + IM_END, "a2" + IM_END]


def test_tool_turns_are_context_only():
    messages = [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "get_weather"}}]},
        {"role": "tool", "content": "sunny"},
        {"role": "assistant", "content": "It is sunny."},
    ]
    tools = [{"type": "function", "function": {"name": "get_weather"}}]
    assert _spans(FakeQwenTokenizer(), messages, tools=tools) == [
        "\n<tool_call>get_weather</tool_call>" + IM_END,
        "It is sunny." + IM_END,
    ]


def test_step_loss_mask_zero_keeps_turn_as_context():
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "bad", "step_loss_mask": 0},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "good"},
    ]
    assert _spans(FakeQwenTokenizer(), messages) == ["good" + IM_END]


def test_step_loss_mask_none_is_trained():
    """Parquet fills a missing step_loss_mask with None; that turn is still trained."""
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a", "step_loss_mask": None},
    ]
    assert _spans(FakeQwenTokenizer(), messages) == ["a" + IM_END]


def test_think_prefix_is_not_trained_but_reasoning_is():
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a", "reasoning_content": "r"},
    ]
    assert _spans(FakeQwenTokenizer(), messages) == ["r\n</think>\n\na" + IM_END]


def test_qwen_history_reasoning_dropped_last_turn_kept():
    """Qwen3 renders reasoning only after the last user turn, so only that reasoning is trained."""
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2", "reasoning_content": "r2"},
    ]
    assert _spans(FakeQwenTokenizer(), messages) == ["a1" + IM_END, "r2\n</think>\n\na2" + IM_END]


def test_qwen_no_prefill_trains_models_own_think_tag():
    """Without a pre-filled prefix the model emits ``<think>`` itself, so it is trained."""
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a", "reasoning_content": "r"},
    ]
    assert _spans(FakeQwenNoPrefillTokenizer(), messages) == ["<think>\nr\n</think>\n\na" + IM_END]


def test_end_marker_literal_in_content_closes_span_early():
    """Pins current behaviour: ``find(end_marker)`` stops at the content's own literal.

    The template's real ``<|im_end|>`` is then left untrained; content quoting
    the end marker is a known limitation of string-based span location.
    """
    messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "say " + IM_END}]
    assert _spans(FakeQwenTokenizer(), messages) == ["say " + IM_END]


def test_missing_offsets_raises():
    messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    with pytest.raises(ValueError, match="fast tokenizer"):
        _spans(FakeQwenTokenizer(with_offsets=False), messages)


def test_retokenization_drift_raises():
    class DriftingTokenizer(FakeQwenTokenizer):
        def apply_chat_template(self, messages, tools=None, tokenize=True, **kwargs):
            out = super().apply_chat_template(messages, tools=tools, tokenize=tokenize, **kwargs)
            return out + [0] if tokenize else out

    messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    with pytest.raises(ValueError, match="does not match"):
        _spans(DriftingTokenizer(), messages)


# ---------------------------------------------------------------------------
# Multi-char tokens straddling a mask boundary (any-overlap projection)
# ---------------------------------------------------------------------------

def test_token_straddling_header_seam_is_trained():
    """The header's trailing "\\n" fuses with content "x"; the fused token is trained whole."""
    messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "x"}]
    assert _spans(FakeMergingTokenizer(["\nx", IM_END]), messages) == ["\nx" + IM_END]


def test_token_straddling_skip_prefix_seam_is_trained():
    """The skipped "<think>\\n" fuses its "\\n" with reasoning "r"; the fused token is trained."""
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a", "reasoning_content": "r"},
    ]
    assert _spans(FakeMergingTokenizer(["\nr"]), messages) == ["\nr\n</think>\n\na" + IM_END]


# ---------------------------------------------------------------------------
# DeepSeek-V4 official encoder layout
# ---------------------------------------------------------------------------

def test_deepseek_chat_mode_skips_think_end_token():
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    assert _spans(FakeDeepSeekV4Tokenizer(), messages) == ["a1" + EOS, "a2" + EOS]


def test_deepseek_thinking_mode_trains_reasoning():
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a", "reasoning_content": "r"},
    ]
    assert _spans(FakeDeepSeekV4Tokenizer(), messages, thinking=True) == ["r</think>a" + EOS]


MULTI_TURN_REASONING = [
    {"role": "user", "content": "q1"},
    {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
    {"role": "user", "content": "q2"},
    {"role": "assistant", "content": "a2", "reasoning_content": "r2"},
]


def test_deepseek_drop_thinking_trains_only_last_turn_reasoning():
    """No tools: history turns open with "</think>" (reasoning dropped), the last with "<think>"."""
    spans = _spans(FakeDeepSeekV4Tokenizer(), MULTI_TURN_REASONING, thinking=True)
    assert spans == ["a1" + EOS, "r2</think>a2" + EOS]


def test_deepseek_tools_keep_reasoning_on_every_turn():
    tools = [{"type": "function", "function": {"name": "f"}}]
    spans = _spans(
        FakeDeepSeekV4Tokenizer(), MULTI_TURN_REASONING, tools=tools, thinking=True
    )
    assert spans == ["r1</think>a1" + EOS, "r2</think>a2" + EOS]


def test_deepseek_tool_result_turns_are_context_only():
    messages = [
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "call"},
        {"role": "tool", "content": "OUT"},
        {"role": "assistant", "content": "final"},
    ]
    assert _spans(FakeDeepSeekV4Tokenizer(), messages) == ["call" + EOS, "final" + EOS]


# ---------------------------------------------------------------------------
# Optional: real HF tokenizer (fake tokenizers cannot catch offset quirks)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("CODA_SFT_REAL_TOKENIZER"),
    reason="set CODA_SFT_REAL_TOKENIZER=<hf tokenizer dir> to run",
)
def test_real_tokenizer_trains_end_marker_but_not_prefill():
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        os.environ["CODA_SFT_REAL_TOKENIZER"], trust_remote_code=True
    )
    sft_tok = SFTTokenizer(tokenizer)
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "And 3+3?"},
        {"role": "assistant", "content": "6"},
    ]
    ids, mask = sft_tok.tokenize(messages)
    spans = decode_trainable_spans(tokenizer, ids, mask)

    # One span per assistant turn, each ending with the end marker so the model
    # learns to stop; neither the header nor a pre-filled prefix is trained.
    assert len(spans) == 2
    for span, answer in zip(spans, ("4", "6")):
        assert span.endswith(sft_tok.end_marker)
        assert answer in span
        assert sft_tok.assistant_header not in span
        assert not any(span.startswith(prefix) for prefix in sft_tok.skip_prefixes)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "mask, expected",
    [([0, 0, 1, 0, 1], 3), ([1, 1], 2), ([0, 0], 0), ([], 0)],
)
def test_trainable_suffix_length(mask, expected):
    assert trainable_suffix_length(mask) == expected
