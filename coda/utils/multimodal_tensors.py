"""Torch-only helpers moving per-image tensors from a trajectory to the model forward.

`slice_segment_inputs` runs on the controller (`put_dp_shards_to_ray`); `merge_multimodal_batch` and
`move_image_table_to_device` run on the Megatron workers. No ray / megatron imports.

A *row* is one trainable Segment. Its payload is `None` (no images) or the hashes of the images whose pad
runs lie in the Segment, in pad order. Each DP shard keeps one copy of every image's tensors in an
*image table* `{image_hash: {"pixel_values": ..., "image_grid_thw": ...}}`, shared by all rows.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

MultimodalRow = list[str] | None
ImageTable = dict[str, dict[str, torch.Tensor]]


def cat_by_key(dicts: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Concatenate `dicts` key by key along dim 0 (keys follow the first dict).

    A single dict is returned as a shallow copy, so its tensors are not duplicated.
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
    """Check that the pad tokens of a Segment line up one-to-one with the selected images.

    Raises `ValueError` when a pad token has no image tensor (e.g. an image straddling the Segment
    boundary), an image's token range is not all pad tokens, or `pixel_values` does not hold
    `sum(t*h*w)` patches. Otherwise a bad row would silently train as text-only.
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


def add_to_image_table(table: ImageTable, image_hash: str, tensors: dict[str, torch.Tensor]) -> None:
    """Register `tensors` under `image_hash`, keeping the first copy.

    Raises `ValueError` if the hash is already present with a different `pixel_values` shape or
    `image_grid_thw`, since one hash must map to one preprocessed image within a run.
    """
    known = table.get(image_hash)
    if known is None:
        table[image_hash] = tensors
        return
    if known["pixel_values"].shape != tensors["pixel_values"].shape or not torch.equal(
        known["image_grid_thw"], tensors["image_grid_thw"]
    ):
        raise ValueError(
            f"image {image_hash[:12]} appears with two different preprocessed tensors "
            f"(pixel_values {tuple(known['pixel_values'].shape)} vs {tuple(tensors['pixel_values'].shape)}, "
            f"image_grid_thw {known['image_grid_thw'].tolist()} vs {tensors['image_grid_thw'].tolist()})"
        )


def slice_segment_inputs(
    items: Sequence[dict[str, Any]] | None,
    token_start: int,
    token_end: int,
    table: ImageTable,
    segment_tokens: list[int] | None = None,
) -> MultimodalRow:
    """Return the hashes of the images whose pad run lies inside `[token_start, token_end)`, or `None` if none.

    `items` is `Trajectory.multimodal_train_inputs` (absolute token positions, in pad order). The selected
    images' tensors are added to `table`. If `segment_tokens` (the span's token ids) is given, the pad
    tokens are checked against the selected images via `check_segment_pads`.
    """
    if not items:
        return None
    selected = [it for it in items if it["token_start"] >= token_start and it["token_end"] <= token_end]
    if segment_tokens is not None:
        check_segment_pads(selected, segment_tokens, token_start, token_end, items[0]["pad_id"])
    if not selected:
        return None
    for it in selected:
        add_to_image_table(table, it["image_hash"], it["tensors"])
    return [it["image_hash"] for it in selected]


def merge_multimodal_batch(rows: Sequence[MultimodalRow] | None, table: ImageTable | None) -> dict[str, torch.Tensor] | None:
    """Merge the rows of one micro-batch into a single `forward` kwargs dict, or `None` if no row has an image.

    `rows` follows the micro-batch row order, which is also the packed token order, so a per-key concat
    of the table entries keeps images aligned with their pad runs. A hash that appears several times is
    concatenated several times, since each pad run needs its own patches.
    """
    if not rows:
        return None
    hashes = [h for row in rows if row for h in row]
    if not hashes:
        return None
    return cat_by_key([table[h] for h in hashes])


def move_image_table_to_device(table: ImageTable, device: torch.device | int | str) -> ImageTable:
    """Move every tensor of the image table to `device` (once per shard)."""
    return {h: {k: v.to(device) for k, v in tensors.items()} for h, tensors in table.items()}
