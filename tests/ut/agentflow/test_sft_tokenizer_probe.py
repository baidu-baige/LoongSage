"""Chat-template probing: ``non_assistant_prefix`` / ``assistant_prefix`` per model family.

Each fake (in ``fake_tokenizers.py``) reproduces one model family's chat-template layout
with a char-level tokenizer, so no weights or HF files are needed.  The Qwen3 layout is
pinned against the real Qwen3-4B template and the DeepSeek-V4.1 one against sglang's
``encoding_dsv41``, so a template-layout change shows up here as a readable diff.

Run: python -m pytest tests/ut/agentflow/test_sft_tokenizer_probe.py -v
"""

import pytest

from coda.agentflow.tokenizer_manager import ThreadedTokenizerManager
from coda.controller.sft_sampler import decode_trainable_tokens
from tests.ut.agentflow.fake_tokenizers import (
    DS_BOS,
    DS_EFFORT,
    DS_EOS,
    QWEN_IM_END as IM_END,
    FakeDeepSeekV41Tokenizer,
    FakeLlama3Tokenizer,
    FakeQwen3Tokenizer,
)

# A ChatML assistant message's own render ends with "\n" after ``<|im_end|>``; as in verl it is trained.
TURN_END = IM_END + "\n"

# pytest.param(factory, kwargs, non_assistant_prefix, assistant_prefix, id=...), both decoded;
# every template here accepts a system-only conversation, so ``non_assistant_prefix`` is the
# render of an empty system message.  ``assistant_prefix`` is an empty user message plus the
# generation prompt.
PROBE_CASES = [
    pytest.param(
        FakeQwen3Tokenizer, {}, "<|im_start|>system\n<|im_end|>\n",
        "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n",
        id="qwen3-thinking-default",
    ),
    pytest.param(
        FakeQwen3Tokenizer, {"enable_thinking": False}, "<|im_start|>system\n<|im_end|>\n",
        "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        id="qwen3-thinking-off",
    ),
    pytest.param(
        # The encoder ends a trailing user message with the assistant header itself;
        # there is no add_generation_prompt to append anything.
        FakeDeepSeekV41Tokenizer, {}, f"{DS_BOS}<｜System｜>", f"{DS_BOS}<｜User｜><｜Assistant｜></think>",
        id="deepseek-v4.1-chat",
    ),
    pytest.param(
        FakeDeepSeekV41Tokenizer, {"thinking": True}, f"{DS_BOS}<｜System｜>{DS_EFFORT}",
        f"{DS_BOS}<｜System｜>{DS_EFFORT}<｜User｜><｜Assistant｜><think>",
        id="deepseek-v4.1-thinking",
    ),
    pytest.param(
        FakeLlama3Tokenizer, {}, "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n<|eot_id|>",
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
        id="llama3",
    ),
]


def _manager(tokenizer, generation_prompt_kwargs=None):
    """Tokenizer manager whose ``tokenize`` runs in the calling thread."""
    return ThreadedTokenizerManager(tokenizer, num_workers=1, generation_prompt_kwargs=generation_prompt_kwargs)


def _probed(manager: ThreadedTokenizerManager) -> dict[str, str]:
    """Probed token ids, decoded."""
    decode = manager.tokenizer.decode
    return {
        "non_assistant_prefix": decode(manager.non_assistant_prefix),
        "assistant_prefix": decode(manager.assistant_prefix),
    }


@pytest.mark.parametrize("factory, kwargs, non_assistant_prefix, assistant_prefix", PROBE_CASES)
def test_probed_prompts(factory, kwargs, non_assistant_prefix, assistant_prefix):
    assert _probed(_manager(factory(), kwargs)) == {
        "non_assistant_prefix": non_assistant_prefix,
        "assistant_prefix": assistant_prefix,
    }


# A 2-turn conversation with reasoning on both turns.  Every assistant message is
# rendered as the turn after the last user query, so Qwen3 and DeepSeek-V4.1 in thinking
# mode keep each turn's reasoning; Llama-3 renders no reasoning at all.
CONVERSATION = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "2+2=4.", "reasoning_content": "Simple arithmetic."},
    {"role": "user", "content": "And 3+3?"},
    {"role": "assistant", "content": "3+3=6.", "reasoning_content": "Add three and three."},
]
CONVERSATION_NO_REASONING = [
    {key: value for key, value in message.items() if key != "reasoning_content"}
    for message in CONVERSATION
]


@pytest.mark.parametrize(
    "factory, kwargs, messages, expected",
    [
        pytest.param(FakeQwen3Tokenizer, {}, CONVERSATION,
                     ["<think>\nSimple arithmetic.\n</think>\n\n2+2=4." + TURN_END,
                      "<think>\nAdd three and three.\n</think>\n\n3+3=6." + TURN_END],
                     id="qwen3-thinking-default"),
        # The generation prompt pre-fills the empty think block, so only the content is trained.
        pytest.param(FakeQwen3Tokenizer, {"enable_thinking": False}, CONVERSATION_NO_REASONING,
                     ["2+2=4." + TURN_END, "3+3=6." + TURN_END],
                     id="qwen3-thinking-off"),
        pytest.param(FakeDeepSeekV41Tokenizer, {}, CONVERSATION, ["2+2=4." + DS_EOS, "3+3=6." + DS_EOS],
                     id="deepseek-v4.1-chat"),
        pytest.param(FakeDeepSeekV41Tokenizer, {"thinking": True}, CONVERSATION,
                     ["Simple arithmetic.</think>2+2=4." + DS_EOS, "Add three and three.</think>3+3=6." + DS_EOS],
                     id="deepseek-v4.1-thinking"),
        pytest.param(FakeLlama3Tokenizer, {}, CONVERSATION, ["2+2=4.<|eot_id|>", "3+3=6.<|eot_id|>"],
                     id="llama3"),
    ],
)
def test_probed_prompts_mask_a_conversation(factory, kwargs, messages, expected):
    tokenizer = factory()
    manager = _manager(tokenizer, kwargs)
    ids, mask = manager.tokenize(messages)
    assert decode_trainable_tokens(tokenizer, ids, mask) == expected, _probed(manager)


def test_qwen3_final_turn_matches_the_whole_render():
    """The final turn keeps the empty think block the whole render gives it."""
    messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    tokenizer = FakeQwen3Tokenizer()
    ids, mask = _manager(tokenizer).tokenize(messages)
    assert decode_trainable_tokens(tokenizer, ids, mask) == ["<think>\n\n</think>\n\na" + TURN_END]
    assert ids == tokenizer.apply_chat_template(messages)


def test_qwen3_thinking_off_rejects_reasoning():
    """With thinking off, reasoning in an assistant message cannot follow the pre-filled empty think block."""
    manager = _manager(FakeQwen3Tokenizer(), {"enable_thinking": False})
    with pytest.raises(ValueError, match="does not continue the generation prompt"):
        manager.tokenize(CONVERSATION)


class _GenerationRewritesTokenizer(FakeLlama3Tokenizer):
    """Changes the rendered messages themselves under ``add_generation_prompt``."""

    def render(self, messages, tools, add_generation_prompt, **kwargs) -> str:
        out = super().render(messages, tools, add_generation_prompt, **kwargs)
        return out.replace("<|begin_of_text|>", "<|begin_of_text|>GEN", 1) if add_generation_prompt else out


def test_generation_prompt_rewriting_the_messages_raises():
    """The assistant message no longer renders as a continuation of the generation prompt."""
    with pytest.raises(ValueError, match="does not continue the generation prompt"):
        _manager(_GenerationRewritesTokenizer()).tokenize(CONVERSATION)
