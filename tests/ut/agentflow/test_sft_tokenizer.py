"""Unit tests for SFT tokenization in coda.agentflow.tokenizer_manager.

Run: python -m pytest tests/ut/agentflow/test_sft_tokenizer.py -v
"""

import os

import pytest

from coda.agentflow.tokenizer_manager import ThreadedTokenizerManager
from coda.controller.sft_sampler import decode_trainable_tokens
from tests.ut.agentflow.fake_tokenizers import (
    FakeDeepSeekV4Tokenizer,
    FakeQwenNoPrefillTokenizer,
    FakeQwenThinkingOffTokenizer,
    FakeQwenTokenizer,
)

IM_END = "<|im_end|>"
# A ChatML assistant message's own render ends with "\n" after ``<|im_end|>``; as in verl it is trained.
TURN_END = IM_END + "\n"
# A Qwen3.5 turn without reasoning renders an empty think block; after the pre-filled
# "<think>\n" the model emits this.
EMPTY_REASONING = "\n</think>\n\n"
EOS = "<｜end▁of▁sentence｜>"


def _manager(tokenizer, generation_prompt_kwargs=None):
    """Tokenizer manager whose ``tokenize`` runs in the calling thread."""
    return ThreadedTokenizerManager(tokenizer, num_workers=1, generation_prompt_kwargs=generation_prompt_kwargs)


def _spans(tokenizer, messages, tools=None, **template_kwargs):
    manager = _manager(tokenizer, template_kwargs)
    ids, mask = manager.tokenize(messages, tools=tools)
    assert len(ids) == len(mask)
    return decode_trainable_tokens(tokenizer, ids, mask)


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "tokenizer, generation_prompt",
    [
        pytest.param(FakeQwenTokenizer(), "<|im_start|>assistant\n<think>\n", id="prefills-think"),
        pytest.param(FakeQwenNoPrefillTokenizer(), "<|im_start|>assistant\n", id="no-prefill"),
        pytest.param(FakeQwenThinkingOffTokenizer(), "<|im_start|>assistant\n<think>\n\n</think>\n\n", id="off"),
    ],
)
def test_qwen_probes(tokenizer, generation_prompt):
    """The template rejects a system-only conversation, so the empty user message is the prefix."""
    manager = _manager(tokenizer)
    assert manager.dummy_prefix_message == {"role": "user", "content": ""}
    assert tokenizer.decode(manager.dummy_prefix) == "<|im_start|>user\n<|im_end|>\n"
    assert tokenizer.decode(manager.generation_prompt) == generation_prompt


# ---------------------------------------------------------------------------
# Qwen3.5-style template
# ---------------------------------------------------------------------------

def test_single_turn_trains_only_assistant_content_and_closing_token():
    messages = [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    assert _spans(FakeQwenTokenizer(), messages) == [EMPTY_REASONING + "hello" + TURN_END]


def test_multi_turn_trains_every_assistant_turn():
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    assert _spans(FakeQwenTokenizer(), messages) == [
        EMPTY_REASONING + "a1" + TURN_END,
        EMPTY_REASONING + "a2" + TURN_END,
    ]


def test_concatenated_messages_equal_the_whole_render():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "get_weather"}}]},
        {"role": "tool", "content": "sunny"},
        {"role": "assistant", "content": "It is sunny."},
    ]
    tools = [{"type": "function", "function": {"name": "get_weather"}}]
    tokenizer = FakeQwenTokenizer()
    ids, _ = _manager(tokenizer).tokenize(messages, tools=tools)
    assert ids == tokenizer.apply_chat_template(messages, tools=tools)


def test_tool_turns_are_context_only():
    messages = [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "get_weather"}}]},
        {"role": "tool", "content": "sunny"},
        {"role": "assistant", "content": "It is sunny."},
    ]
    tools = [{"type": "function", "function": {"name": "get_weather"}}]
    assert _spans(FakeQwenTokenizer(), messages, tools=tools) == [
        EMPTY_REASONING + "\n<tool_call>get_weather</tool_call>" + TURN_END,
        EMPTY_REASONING + "It is sunny." + TURN_END,
    ]


def test_prefilled_think_tag_is_not_trained_but_reasoning_is():
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a", "reasoning_content": "r"},
    ]
    assert _spans(FakeQwenTokenizer(), messages) == ["r\n</think>\n\na" + TURN_END]


def test_no_prefill_trains_models_own_think_tag():
    """Without a pre-filled prefix the model emits ``<think>`` itself, so it is trained."""
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a", "reasoning_content": "r"},
    ]
    assert _spans(FakeQwenNoPrefillTokenizer(), messages) == ["<think>\nr\n</think>\n\na" + TURN_END]


def test_history_reasoning_is_trained():
    """A message rendered on its own keeps its reasoning, which the whole render drops on
    history turns; the per-message tokens are trained."""
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2", "reasoning_content": "r2"},
    ]
    tokenizer = FakeQwenTokenizer()
    ids, mask = _manager(tokenizer).tokenize(messages)
    assert decode_trainable_tokens(tokenizer, ids, mask) == [
        "r1\n</think>\n\na1" + TURN_END,
        "r2\n</think>\n\na2" + TURN_END,
    ]
    assert ids != tokenizer.apply_chat_template(messages)


def test_closing_token_quoted_in_content_is_trained():
    """Each message is masked as a whole, so content quoting ``<|im_end|>`` does not cut it short."""
    messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "say " + IM_END}]
    assert _spans(FakeQwenTokenizer(), messages) == [EMPTY_REASONING + "say " + IM_END + TURN_END]


def test_newline_after_closing_token_is_trained():
    """As in verl, an assistant message's whole render after the generation prompt is
    trained, including ChatML's "\\n" after ``<|im_end|>``."""
    tokenizer = FakeQwenTokenizer()
    messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    ids, mask = _manager(tokenizer).tokenize(messages)
    assert tokenizer.decode(ids).endswith(IM_END + "\n")
    assert mask[-1] == 1


# ---------------------------------------------------------------------------
# DeepSeek-V4 official encoder layout
# ---------------------------------------------------------------------------

def test_deepseek_probes_bos_and_no_generation_prompt():
    """The encoder ends every trailing user message with the assistant header itself."""
    manager = _manager(FakeDeepSeekV4Tokenizer())
    assert manager.tokenizer.decode(manager.dummy_prefix) == "<｜begin▁of▁sentence｜>"
    assert manager.generation_prompt == []


def test_deepseek_chat_mode_trains_content_and_eos():
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    assert _spans(FakeDeepSeekV4Tokenizer(), messages) == ["a1" + EOS, "a2" + EOS]


def test_deepseek_thinking_mode_trains_reasoning():
    """The user message's render ends with the pre-filled "<think>"; the reasoning after it is trained."""
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


def test_deepseek_thinking_keeps_every_turns_reasoning():
    """Rendered on its own, a history turn keeps the reasoning the whole render drops."""
    spans = _spans(FakeDeepSeekV4Tokenizer(), MULTI_TURN_REASONING, thinking=True)
    assert spans == ["r1</think>a1" + EOS, "r2</think>a2" + EOS]


def test_deepseek_tools_keep_reasoning_on_every_turn():
    tools = [{"type": "function", "function": {"name": "f"}}]
    spans = _spans(FakeDeepSeekV4Tokenizer(), MULTI_TURN_REASONING, tools=tools, thinking=True)
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
# Optional: real HF tokenizer (fake tokenizers cannot catch BPE quirks)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.environ.get("CODA_SFT_REAL_TOKENIZER"),
    reason="set CODA_SFT_REAL_TOKENIZER=<hf tokenizer dir> to run",
)
def test_real_tokenizer_trains_answers_but_not_generation_prompt():
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        os.environ["CODA_SFT_REAL_TOKENIZER"], trust_remote_code=True
    )
    manager = _manager(tokenizer)
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "And 3+3?"},
        {"role": "assistant", "content": "6"},
    ]
    ids, mask = manager.tokenize(messages)
    spans = decode_trainable_tokens(tokenizer, ids, mask)
    generation_prompt = tokenizer.decode(manager.generation_prompt)

    # One span per assistant turn, without the generation prompt.
    assert len(spans) == 2
    for span, answer in zip(spans, ("4", "6")):
        assert answer in span
        assert not (generation_prompt and generation_prompt in span)
