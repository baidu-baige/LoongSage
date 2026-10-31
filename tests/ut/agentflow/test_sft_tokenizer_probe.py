"""Chat-template probing: ``dummy_prefix`` / ``generation_prompt`` per model family.

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

# pytest.param(factory, kwargs, dummy_prefix, generation_prompt, id=...), both decoded;
# every template here accepts a system-only conversation, so ``dummy_prefix`` is the
# render of an empty system message.
PROBE_CASES = [
    pytest.param(
        FakeQwen3Tokenizer, {}, "<|im_start|>system\n<|im_end|>\n", "<|im_start|>assistant\n",
        id="qwen3-thinking-default",
    ),
    pytest.param(
        FakeQwen3Tokenizer, {"enable_thinking": False}, "<|im_start|>system\n<|im_end|>\n",
        "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        id="qwen3-thinking-off",
    ),
    pytest.param(
        # The encoder ends a trailing user message with the assistant header itself;
        # there is no add_generation_prompt to append anything.
        FakeDeepSeekV41Tokenizer, {}, f"{DS_BOS}<｜System｜>", "",
        id="deepseek-v4.1-chat",
    ),
    pytest.param(
        FakeDeepSeekV41Tokenizer, {"thinking": True}, f"{DS_BOS}<｜System｜>{DS_EFFORT}", "",
        id="deepseek-v4.1-thinking",
    ),
    pytest.param(
        FakeLlama3Tokenizer, {}, "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n<|eot_id|>",
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
        "dummy_prefix": decode(manager.dummy_prefix),
        "generation_prompt": decode(manager.generation_prompt),
    }


@pytest.mark.parametrize("factory, kwargs, dummy_prefix, generation_prompt", PROBE_CASES)
def test_probed_prompts(factory, kwargs, dummy_prefix, generation_prompt):
    assert _probed(_manager(factory(), kwargs)) == {
        "dummy_prefix": dummy_prefix,
        "generation_prompt": generation_prompt,
    }


# A 2-turn conversation with reasoning on both turns.  Rendered on its own, a Qwen3
# assistant message gets no think block, so its reasoning is dropped; a DeepSeek-V4.1
# one in thinking mode keeps it; Llama-3 renders no reasoning at all.
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
        pytest.param(FakeQwen3Tokenizer, {}, ["2+2=4." + TURN_END, "3+3=6." + TURN_END],
                     id="qwen3-thinking-default"),
        # Known limitation: rendered on its own, a Qwen3 assistant message lacks the empty
        # think block the generation prompt pre-fills, so cutting len(generation_prompt)
        # tokens also cuts its content.
        pytest.param(FakeQwen3Tokenizer, {"enable_thinking": False}, [TURN_END, TURN_END],
                     id="qwen3-thinking-off"),
        pytest.param(FakeDeepSeekV41Tokenizer, {}, ["2+2=4." + DS_EOS, "3+3=6." + DS_EOS],
                     id="deepseek-v4.1-chat"),
        pytest.param(FakeDeepSeekV41Tokenizer, {"thinking": True},
                     ["Simple arithmetic.</think>2+2=4." + DS_EOS, "Add three and three.</think>3+3=6." + DS_EOS],
                     id="deepseek-v4.1-thinking"),
        pytest.param(FakeLlama3Tokenizer, {}, ["2+2=4.<|eot_id|>", "3+3=6.<|eot_id|>"], id="llama3"),
    ],
)
def test_probed_prompts_mask_a_conversation(factory, kwargs, expected):
    tokenizer = factory()
    manager = _manager(tokenizer, kwargs)
    ids, mask = manager.tokenize(CONVERSATION)
    assert decode_trainable_tokens(tokenizer, ids, mask) == expected, _probed(manager)


def test_qwen3_final_turn_think_block_is_lost():
    """The whole render gives the final turn an empty think block; its own render does not."""
    messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    tokenizer = FakeQwen3Tokenizer()
    ids, mask = _manager(tokenizer).tokenize(messages)
    assert decode_trainable_tokens(tokenizer, ids, mask) == ["a" + TURN_END]
    assert "<think>" in tokenizer.decode(tokenizer.apply_chat_template(messages))
    assert "<think>" not in tokenizer.decode(ids)


class _GenerationRewritesTokenizer(FakeLlama3Tokenizer):
    """Changes the rendered messages themselves under ``add_generation_prompt``."""

    def render(self, messages, tools, add_generation_prompt, **kwargs) -> str:
        out = super().render(messages, tools, add_generation_prompt, **kwargs)
        return out.replace("<|begin_of_text|>", "<|begin_of_text|>GEN", 1) if add_generation_prompt else out


def test_unprobeable_generation_prompt_raises():
    with pytest.raises(ValueError, match="rewrites the rendered messages"):
        _manager(_GenerationRewritesTokenizer())
