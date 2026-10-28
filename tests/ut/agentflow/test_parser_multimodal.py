"""Multimodal (image) behaviour of ``TrajectoryParser`` with a fake processor and chat template."""

import asyncio
import base64
from types import SimpleNamespace

import pytest
import torch

# Same guard as test_parser.py: the parser needs the sglang build pinned by the Dockerfile.
pytest.importorskip(
    "sglang.srt.parser.template_detection",
    reason="sglang.srt.parser.template_detection is unavailable in this sglang build",
)

from coda.agentflow import multimodal
from coda.agentflow.multimodal import (
    MultimodalRequestError,
    ProcessedImage,
    find_pad_spans,
    image_sha256,
    parse_data_uri,
)
from coda.agentflow.router.parser import TrajectoryParser
from coda.agentflow.trajectory_store import Trajectory, TrajectoryStatus, TrajectoryStore

PAD, VS, VE, END = 9, 50, 51, 5
ROLE_IDS = {"system": 3, "user": 1, "assistant": 2, "tool": 4}


def uri(payload: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(payload).decode()


class FakeMM:
    """num_tokens = len(payload) + 1, so b'a' -> 2 and b'bb' -> 3."""

    image_pad_id = PAD

    async def prepare(self, refs, hashes=None):
        out = []
        for ref in refs:
            n = len(parse_data_uri(ref)) + 1
            tensors = {"pixel_values": torch.zeros(n * 4, 3), "image_grid_thw": torch.tensor([[1, 2, 2 * n]])}
            out.append(ProcessedImage(image_sha256(ref), "png:" + image_sha256(ref)[:8], tensors, n))
        return out


async def apply_chat_template(messages, add_generation_prompt=True, tools=None, **kwargs):
    ids = []
    for m in messages:
        ids.append(ROLE_IDS[m["role"]])
        content = m.get("content")
        if isinstance(content, str):
            ids.extend([100] if content else [])
        else:
            for part in content:
                ids.extend([VS, PAD, VE] if part["type"] == "image_url" else [100])
        ids.append(END)
    if add_generation_prompt:
        ids.append(ROLE_IDS["assistant"])
    return ids


def make_parser(multimodal=True) -> TrajectoryParser:
    manager = SimpleNamespace(
        apply_chat_template=apply_chat_template,
        multimodal_processor=FakeMM() if multimodal else None,
    )
    return TrajectoryParser(
        trajectory_store=TrajectoryStore(),
        tokenizer_manager=manager,
        reasoning_parser=False,
        tool_call_parser=False,
    )


def user_msg(*payloads: bytes) -> dict:
    parts = [{"type": "text", "text": "look"}]
    parts += [{"type": "image_url", "image_url": {"url": uri(p)}} for p in payloads]
    return {"role": "user", "content": parts}


def build(parser, trajectory, messages, **kw):
    return asyncio.run(parser.build_turn_input(trajectory, messages=messages, **kw))


def finish_turn(parser, trajectory, ctx):
    payload, assistant, _ = parser.build_assistant_message(
        {"text": "ok", "meta_info": {"weight_version": "1", "output_token_logprobs": [[-0.1, 7], [-0.2, 8]]}}
    )
    parser.update_trajectory(trajectory, ctx, payload, assistant)


def new_trajectory(parser, tid="t") -> Trajectory:
    traj = Trajectory(trajectory_id=tid, prompt_id="p", status=TrajectoryStatus.GENERATING)
    parser.trajectory_store.add(tid, traj)
    return traj


def test_first_turn_expands_and_folds():
    parser = make_parser()
    traj = new_trajectory(parser)
    ctx = build(parser, traj, [user_msg(b"a", b"bb")])

    assert [e - s for s, e in find_pad_spans(ctx.input_ids, PAD)] == [2, 3]
    assert ctx.send_input_ids == [1, 100, VS, PAD, VE, VS, PAD, VE, END, 2]
    assert len(ctx.image_data) == 2
    assert [(i["local_start"], i["local_end"]) for i in ctx.new_images] == find_pad_spans(ctx.input_ids, PAD)
    assert ctx.delta_prompt_ids == ctx.input_ids


def test_text_only_request_is_untouched():
    parser = make_parser(multimodal=False)
    ctx = build(parser, new_trajectory(parser), [{"role": "user", "content": "hi"}])
    assert ctx.send_input_ids is None and ctx.image_data == [] and ctx.new_images == []


def test_images_rejected_when_multimodal_disabled():
    parser = make_parser(multimodal=False)
    with pytest.raises(MultimodalRequestError):
        build(parser, new_trajectory(parser), [user_msg(b"a")])


def test_image_in_system_message_rejected():
    parser = make_parser()
    system = user_msg(b"a")
    system["role"] = "system"
    with pytest.raises(MultimodalRequestError):
        build(parser, new_trajectory(parser), [system])


def test_update_records_absolute_positions_and_history_holds_hashes():
    parser = make_parser()
    traj = new_trajectory(parser)
    ctx = build(parser, traj, [user_msg(b"a", b"bb")])
    finish_turn(parser, traj, ctx)

    stored = parser.trajectory_store.get(["t"])["t"][-1]
    items = stored.multimodal_train_inputs
    assert [(i["token_start"], i["token_end"]) for i in items] == find_pad_spans(stored.tokens, PAD)
    assert [i["image_hash"] for i in items] == [image_sha256(uri(b"a")), image_sha256(uri(b"bb"))]
    assert items[0]["tensors"]["image_grid_thw"].shape == (1, 3)
    assert "base64" not in repr(stored.metadata)


def test_continuation_only_processes_new_images():
    parser = make_parser()
    traj = new_trajectory(parser)
    first = [user_msg(b"a")]
    finish_turn(parser, traj, build(parser, traj, first))
    traj = parser.trajectory_store.get(["t"])["t"][-1]

    history = traj.chat_completions[0]
    ctx = build(parser, traj, [*history, user_msg(b"bb")])

    assert not ctx.start_new_segment
    assert len(ctx.image_data) == 2  # all images are sent to SGLang
    assert len(ctx.new_images) == 1  # but only the new one is recorded
    assert [e - s for s, e in find_pad_spans(ctx.input_ids, PAD)] == [2, 3]
    assert len(find_pad_spans(ctx.send_input_ids, PAD)) == 2
    finish_turn(parser, traj, ctx)
    stored = parser.trajectory_store.get(["t"])["t"][-1]
    assert [(i["token_start"], i["token_end"]) for i in stored.multimodal_train_inputs] == find_pad_spans(
        stored.tokens, PAD
    )


def test_compact_reprocesses_all_images_and_appends():
    parser = make_parser()
    traj = new_trajectory(parser)
    finish_turn(parser, traj, build(parser, traj, [user_msg(b"a")]))
    traj = parser.trajectory_store.get(["t"])["t"][-1]

    ctx = build(parser, traj, [user_msg(b"a", b"bb")])  # prefix mismatch -> compaction
    assert ctx.start_new_segment and ctx.new_segment_origin == "compact"
    assert len(ctx.new_images) == 2
    finish_turn(parser, traj, ctx)
    stored = parser.trajectory_store.get(["t"])["t"][-1]
    assert len(stored.multimodal_train_inputs) == 3
    # the same image shares its tensors by reference across segments
    assert stored.multimodal_train_inputs[0]["image_hash"] == stored.multimodal_train_inputs[1]["image_hash"]


def test_subagent_placeholder_records_nothing():
    parser = make_parser()
    traj = new_trajectory(parser)
    finish_turn(parser, traj, build(parser, traj, [user_msg(b"a")]))
    traj = parser.trajectory_store.get(["t"])["t"][-1]

    ctx = build(parser, traj, [user_msg(b"bb")], request_kind="collab_spawn")
    assert ctx.is_subagent_placeholder and ctx.new_images == []
    assert len(ctx.image_data) == 1 and ctx.send_input_ids is not None
    finish_turn(parser, traj, ctx)
    stored = parser.trajectory_store.get(["t"])["t"][-1]
    assert len(stored.multimodal_train_inputs) == 1


def test_continuation_hashes_each_history_image_once(monkeypatch):
    parser = make_parser()
    traj = new_trajectory(parser)
    finish_turn(parser, traj, build(parser, traj, [user_msg(b"a")]))
    traj = parser.trajectory_store.get(["t"])["t"][-1]
    history = traj.chat_completions[0]

    calls = []
    real = multimodal.image_sha256
    monkeypatch.setattr(multimodal, "image_sha256", lambda ref: calls.append(ref) or real(ref))

    ctx = build(parser, traj, [*history, user_msg(b"bb")])
    finish_turn(parser, traj, ctx)

    assert sorted(calls) == sorted({uri(b"a"), uri(b"bb")})  # one hash per distinct image, whole turn


def test_compact_hashes_each_image_once(monkeypatch):
    parser = make_parser()
    traj = new_trajectory(parser)
    finish_turn(parser, traj, build(parser, traj, [user_msg(b"a")]))
    traj = parser.trajectory_store.get(["t"])["t"][-1]

    calls = []
    real = multimodal.image_sha256
    monkeypatch.setattr(multimodal, "image_sha256", lambda ref: calls.append(ref) or real(ref))

    ctx = build(parser, traj, [user_msg(b"a", b"bb")])  # prefix mismatch -> compaction
    finish_turn(parser, traj, ctx)

    assert sorted(calls) == sorted({uri(b"a"), uri(b"bb")})


def _middleware(parser):
    from unittest.mock import MagicMock

    from coda.agentflow.router.parser_middleware import ParserMiddleware

    return ParserMiddleware(
        MagicMock(),
        config=SimpleNamespace(
            trainer=SimpleNamespace(temperature=1.0),
            rollout=SimpleNamespace(eval=SimpleNamespace(temperature=None)),
        ),
        trajectory_store=TrajectoryStore(),
        tokenizer_manager=parser.tokenizer_manager,
    )


def test_generate_body_carries_image_data_only_when_present():
    import json

    middleware = _middleware(make_parser())
    kwargs = dict(request_id="r", max_new_tokens=8)
    with_images = json.loads(middleware._build_generate_body([1, PAD, 2], image_data=["png:x"], **kwargs))
    assert with_images["image_data"] == ["png:x"] and with_images["input_ids"] == [1, PAD, 2]
    assert "image_data" not in json.loads(middleware._build_generate_body([1], **kwargs))
    assert "image_data" not in json.loads(middleware._build_generate_body([1], image_data=[], **kwargs))


def test_prompt_token_check():
    from coda.agentflow.router.parser_middleware import ParserMiddleware

    parser = make_parser()
    ctx = build(parser, new_trajectory(parser), [user_msg(b"a")])
    expected = len(ctx.input_ids)

    check = ParserMiddleware._check_prompt_tokens
    assert check(ctx, {"meta_info": {"prompt_tokens": expected}}) is None
    assert "mismatch" in check(ctx, {"meta_info": {"prompt_tokens": expected - 1}})
    assert "mismatch" in check(ctx, {"meta_info": {}})

    text_ctx = build(parser, new_trajectory(parser, "t2"), [{"role": "user", "content": "hi"}])
    assert check(text_ctx, {"meta_info": {"prompt_tokens": 12345}}) is None
