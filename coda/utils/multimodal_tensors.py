"""Torch-only helpers moving per-image tensors from a trajectory to the model forward.

`slice_segment_inputs` runs on the controller (`put_dp_shards_to_ray`); `merge_multimodal_batch` and
`move_multimodal_to_device` run on the Megatron workers. All are kept free of ray / megatron imports.

A *row* is one trainable Segment. Its multimodal payload is either `None` (no images) or a
`dict[str, Tensor]` whose tensors are the row's images concatenated along dim 0 in pad order
(e.g. `pixel_values` and `image_grid_thw`).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

MultimodalRow = dict[str, torch.Tensor] | None


def cat_by_key(dicts: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Concatenate `dicts` key by key along dim 0 (keys follow the first dict).

    A single dict is not concatenated: its tensors are reused as-is (only the dict is copied), so rows
    that share a cached image keep sharing one tensor object, which Ray then stores once per `ray.put`.
    """
    if len(dicts) == 1:
        return dict(dicts[0])
    return {key: torch.cat([d[key] for d in dicts], dim=0) for key in dicts[0]}


def check_segment_pads(
    selected: Sequence[dict[str, Any]],
    segment_tokens: Sequence[int],
    token_start: int,
    token_end: int,
    pad_id: int,
) -> None:
    """Assert the pad tokens of a Segment line up one-to-one with the selected images.

    Raises `ValueError` when a pad token has no image tensor (e.g. an image straddling the Segment boundary),
    when an image's token range is not entirely pad tokens, or when `pixel_values` does not hold
    `sum(t*h*w)` patches. Without this a bad row would silently train as text-only.
    """
    covered = 0
    for it in selected:
        lo, hi = it["token_start"] - token_start, it["token_end"] - token_start
        run = hi - lo
        if segment_tokens[lo:hi].count(pad_id) != run:
            raise ValueError(
                f"image {it['image_hash'][:12]} range [{it['token_start']}, {it['token_end']}) "
                f"is not entirely image pad tokens (pad_id={pad_id})"
            )
        tensors = it["tensors"]
        patches = int(tensors["image_grid_thw"].prod(dim=-1).sum())
        if tensors["pixel_values"].shape[0] != patches:
            raise ValueError(
                f"image {it['image_hash'][:12]}: pixel_values has {tensors['pixel_values'].shape[0]} patches "
                f"but image_grid_thw implies {patches}"
            )
        covered += run
    n_pad = segment_tokens.count(pad_id)
    if n_pad != covered:
        raise ValueError(
            f"Segment [{token_start}, {token_end}) has {n_pad} image pad tokens but its images cover {covered}"
        )


def slice_segment_inputs(
    items: Sequence[dict[str, Any]] | None,
    token_start: int,
    token_end: int,
    segment_tokens: list[int] | None = None,
) -> MultimodalRow:
    """Concatenate the tensors of the images whose pad run lies inside `[token_start, token_end)`.

    `items` is `Trajectory.multimodal_train_inputs` (absolute token positions, in pad order).
    Returns `None` when the span holds no image. When `segment_tokens` (the token ids of the span) is
    given, the pad tokens are checked against the selected images via `check_segment_pads`.
    """
    if not items:
        return None
    selected = [it for it in items if it["token_start"] >= token_start and it["token_end"] <= token_end]
    if segment_tokens is not None:
        check_segment_pads(selected, segment_tokens, token_start, token_end, items[0]["pad_id"])
    return cat_by_key([it["tensors"] for it in selected]) if selected else None


def merge_multimodal_batch(rows: Sequence[MultimodalRow] | None) -> MultimodalRow:
    """Merge the rows of one micro-batch into a single `forward` kwargs dict.

    `rows` follows the micro-batch row order, which is also the packed token order, so a plain
    per-key concat keeps images aligned with their pad runs. `None` rows are skipped; returns
    `None` when no row carries an image (the model then runs its text-only path).
    """
    if not rows:
        return None
    present = [row for row in rows if row]
    return cat_by_key(present) if present else None


def move_multimodal_to_device(rows: Sequence[MultimodalRow], device: torch.device | int | str) -> list[MultimodalRow]:
    """Move every tensor of every row to `device`, keeping `None` rows."""
    return [None if row is None else {k: v.to(device) for k, v in row.items()} for row in rows]
