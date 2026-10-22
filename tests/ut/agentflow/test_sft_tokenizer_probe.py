"""Chat-template probing: sentinel renders (full / gen) and the markers parsed from them.

Each fake (in ``fake_tokenizers.py``) reproduces one model family's chat-template layout
with a char-level tokenizer, so no weights or HF files are needed.  The Qwen3 and
DeepSeek-V4.1 expected renders are pinned against the real templates (Qwen3-4B tokenizer,
and the DeepSeek-V4.1-Flash tokenizer through sglang ``encoding_dsv41``), so a
template-layout change shows up here as a readable diff of the full/gen strings.

Run: python -m pytest tests/ut/agentflow/test_sft_tokenizer_probe.py -v
"""

import pytest

from coda.agentflow.sft_tokenizer import SFTTokenizer, decode_trainable_spans
from tests.ut.agentflow.fake_tokenizers import (
    DS_BOS,
    DS_EFFORT,
    DS_EOS,
    QWEN_IM_END as IM_END,
    FakeDeepSeekV41Tokenizer,
    FakeLlama3Tokenizer,
    FakePlainTextTokenizer,
    FakeQwen3Tokenizer,
)

# pytest.param(factory, kwargs, full, gen_prompt, header, end, skips, id=...)
PROBE_CASES = [
    pytest.param(
        FakeQwen3Tokenizer, {},
        "<|im_start|>user\nLSUSERPROBEA<|im_end|>\n<|im_start|>assistant\nLSASSTPROBEA<|im_end|>\n"
        "<|im_start|>user\nLSUSERPROBEB<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nLSASSTPROBEB<|im_end|>\n",
        "<|im_start|>user\nLSUSERPROBEA<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>assistant\n", IM_END, [],
        id="qwen3-thinking-default",
    ),
    pytest.param(
        FakeQwen3Tokenizer, {"enable_thinking": False},
        "<|im_start|>user\nLSUSERPROBEA<|im_end|>\n<|im_start|>assistant\nLSASSTPROBEA<|im_end|>\n"
        "<|im_start|>user\nLSUSERPROBEB<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nLSASSTPROBEB<|im_end|>\n",
        "<|im_start|>user\nLSUSERPROBEA<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "<|im_start|>assistant\n", IM_END, ["<think>\n\n</think>\n\n"],
        id="qwen3-thinking-off",
    ),
    pytest.param(
        FakeDeepSeekV41Tokenizer, {},
        f"{DS_BOS}<｜User｜>LSUSERPROBEA<｜Assistant｜></think>LSASSTPROBEA{DS_EOS}"
        f"<｜User｜>LSUSERPROBEB<｜Assistant｜></think>LSASSTPROBEB{DS_EOS}",
        f"{DS_BOS}<｜User｜>LSUSERPROBEA<｜Assistant｜></think>",
        "<｜Assistant｜></think>", DS_EOS, [],
        id="deepseek-v4.1-chat",
    ),
    pytest.param(
        # The common prefix of "</think>" (history) and "<think>" (gen prompt) ends inside
        # the think token, so the header carries a lone "<" and the skips are its two tails.
        FakeDeepSeekV41Tokenizer, {"thinking": True},
        f"{DS_BOS}<｜System｜>{DS_EFFORT}<｜User｜>LSUSERPROBEA<｜Assistant｜></think>LSASSTPROBEA{DS_EOS}"
        f"<｜User｜>LSUSERPROBEB<｜Assistant｜><think></think>LSASSTPROBEB{DS_EOS}",
        f"{DS_BOS}<｜System｜>{DS_EFFORT}<｜User｜>LSUSERPROBEA<｜Assistant｜><think>",
        "<｜Assistant｜><", DS_EOS, ["/think>", "think>"],
        id="deepseek-v4.1-thinking",
    ),
    pytest.param(
        FakeLlama3Tokenizer, {},
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nLSUSERPROBEA<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\nLSASSTPROBEA<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\nLSUSERPROBEB<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\nLSASSTPROBEB<|eot_id|>",
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nLSUSERPROBEA<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "<|start_header_id|>assistant<|end_header_id|>\n\n", "<|eot_id|>", [],
        id="llama3",
    ),
]


def _describe(sft_tok: SFTTokenizer) -> str:
    """Probe renders and parsed markers, printed on assertion failure."""
    return (
        f"\nfull={sft_tok.probe_full!r}\ngen={sft_tok.probe_gen!r}\nheader={sft_tok.assistant_header!r} "
        f"end={sft_tok.end_marker!r} skip_prefixes={sft_tok.skip_prefixes!r}"
    )


@pytest.mark.parametrize("factory, kwargs, full, gen_prompt, header, end, skips", PROBE_CASES)
def test_probe_renders_and_parsed_markers(factory, kwargs, full, gen_prompt, header, end, skips):
    sft_tok = SFTTokenizer(factory(), kwargs)
    info = _describe(sft_tok)
    assert sft_tok.probe_full == full, info
    assert sft_tok.probe_gen == gen_prompt, info
    assert sft_tok.assistant_header == header, info
    assert sft_tok.end_marker == end, info
    assert sft_tok.skip_prefixes == skips, info


# A 2-turn conversation with reasoning on both turns: Qwen3 and DeepSeek-V4.1 drop the
# history turn's reasoning and keep the final turn's; Llama-3 renders no reasoning at all.
CONVERSATION = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2=4.", "reasoning_content": "Simple arithmetic."},
    {"role": "user", "content": "And 3+3?"},
    {"role": "assistant", "content": "3+3=6.", "reasoning_content": "Add three and three."},
]


@pytest.mark.parametrize(
    "factory, kwargs, expected",
    [
        pytest.param(FakeQwen3Tokenizer, {},
                     ["2+2=4." + IM_END, "<think>\nAdd three and three.\n</think>\n\n3+3=6." + IM_END],
                     id="qwen3-thinking-default"),
        pytest.param(FakeQwen3Tokenizer, {"enable_thinking": False},
                     ["2+2=4." + IM_END, "<think>\nAdd three and three.\n</think>\n\n3+3=6." + IM_END],
                     id="qwen3-thinking-off"),
        pytest.param(FakeDeepSeekV41Tokenizer, {}, ["2+2=4." + DS_EOS, "3+3=6." + DS_EOS],
                     id="deepseek-v4.1-chat"),
        pytest.param(FakeDeepSeekV41Tokenizer, {"thinking": True},
                     ["2+2=4." + DS_EOS, "Add three and three.</think>3+3=6." + DS_EOS],
                     id="deepseek-v4.1-thinking"),
        pytest.param(FakeLlama3Tokenizer, {}, ["2+2=4.<|eot_id|>", "3+3=6.<|eot_id|>"], id="llama3"),
    ],
)
def test_parsed_markers_mask_a_real_conversation(factory, kwargs, expected):
    tokenizer = factory()
    sft_tok = SFTTokenizer(tokenizer, kwargs)
    ids, mask = sft_tok.tokenize(CONVERSATION)
    assert decode_trainable_spans(tokenizer, ids, mask) == expected, _describe(sft_tok)


def test_empty_qwen3_think_block_is_trained_only_with_thinking_on():
    """No reasoning in the data: the final turn's empty think block is model output when
    thinking is on (nothing pre-filled), and a skipped prefix when thinking is off."""
    messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    tokenizer = FakeQwen3Tokenizer()
    on = SFTTokenizer(tokenizer, {})
    off = SFTTokenizer(tokenizer, {"enable_thinking": False})
    assert decode_trainable_spans(tokenizer, *on.tokenize(messages)) == ["<think>\n\n</think>\n\na" + IM_END]
    assert decode_trainable_spans(tokenizer, *off.tokenize(messages)) == ["a" + IM_END]


def test_template_without_end_marker_fails_self_check():
    """Nothing closes a plain-text reply, so the probe cannot delimit assistant turns."""
    with pytest.raises(ValueError, match="self-check failed"):
        SFTTokenizer(FakePlainTextTokenizer())
