"""Torch-only helpers moving per-image tensors from a trajectory to the model forward.

`slice_segment_inputs` runs on the controller (`put_dp_shards_to_ray`); `merge_multimodal_batch` and
`move_image_table_to_device` run on the Megatron workers. No ray / megatron imports.

A *row* is one trainable Segment. Its payload is `None` (no images) or the hashes of the images whose pad
runs lie in the Segment, in pad order. Each DP shard keeps one copy of every image's tensors in an
*image table* `{image_hash: {"pixel_values": ..., "image_grid_thw" | "image_grid_hw": ...}}`, shared by all rows.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

MultimodalRow = list[str] | None
ImageTable = dict[str, dict[str, torch.Tensor]]

# Grid tensor key per model family: Qwen-VL stores `[t, h, w]` patches, DeepSeek-V4.1 stores `[h, w]`.
GRID_KEYS = ("image_grid_thw", "image_grid_hw")
# DeepSeek-V4.1 `vision_downsample_ratio`: ViT patches per LLM token along each axis.
DSV41_DOWNSAMPLE_RATIO = 3


def grid_key(tensors: dict[str, torch.Tensor]) -> str:
    """Return the name of the grid tensor in one image's `tensors`."""
    for key in GRID_KEYS:
        if key in tensors:
            return key
    raise ValueError(f"image tensors have none of {GRID_KEYS}: keys={sorted(tensors)}")


def dsv41_span_tokens(grid_hw: torch.Tensor, downsample_ratio: int) -> int:
    """Token count of the images in a DeepSeek-V4.1 `[n_img, 2]` grid: `h*(w+1)+2` per image on the LLM grid."""
    total = 0
    for n_vit_h, n_vit_w in grid_hw.tolist():
        n_llm_h = -(-n_vit_h // downsample_ratio)
        n_llm_w = -(-n_vit_w // downsample_ratio)
        total += n_llm_h * (n_llm_w + 1) + 2
    return total


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
    downsample_ratio: int = DSV41_DOWNSAMPLE_RATIO,
) -> None:
    """Check that the pad tokens of a Segment line up one-to-one with the selected images.

    Raises `ValueError` when a pad token has no image tensor (e.g. an image straddling the Segment
    boundary), an image's token range is not all pad tokens, or `pixel_values` does not hold
    `sum(t*h*w)` patches (`sum(h*w)` for `image_grid_hw`). For `image_grid_hw` the token range must also
    be `h_llm*(w_llm+1)+2` long, with `downsample_ratio` ViT patches per LLM token. Otherwise a bad row
    would silently train as text-only.
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
        key = grid_key(tensors)
        patches = int(tensors[key].prod(dim=-1).sum())
        if tensors["pixel_values"].shape[0] != patches:
            raise ValueError(
                f"image {it['image_hash'][:12]}: pixel_values has {tensors['pixel_values'].shape[0]} patches "
                f"but {key} implies {patches}"
            )
        if key == "image_grid_hw":
            expected = dsv41_span_tokens(tensors[key], downsample_ratio)
            if run != expected:
                raise ValueError(
                    f"image {it['image_hash'][:12]}: token range has {run} tokens but "
                    f"image_grid_hw {tensors[key].tolist()} needs {expected}"
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
    grid (`image_grid_thw` or `image_grid_hw`), since one hash must map to one preprocessed image within a run.
    """
    known = table.get(image_hash)
    if known is None:
        table[image_hash] = tensors
        return
    key = grid_key(tensors)
    if (
        key not in known
        or known["pixel_values"].shape != tensors["pixel_values"].shape
        or not torch.equal(known[key], tensors[key])
    ):
        known_grid = known[key].tolist() if key in known else "missing"
        raise ValueError(
            f"image {image_hash[:12]} appears with two different preprocessed tensors "
            f"(pixel_values {tuple(known['pixel_values'].shape)} vs {tuple(tensors['pixel_values'].shape)}, "
            f"{key} {known_grid} vs {tensors[key].tolist()})"
        )


def slice_segment_inputs(
    items: Sequence[dict[str, Any]] | None,
    token_start: int,
    token_end: int,
    table: ImageTable,
    segment_tokens: list[int] | None = None,
    downsample_ratio: int = DSV41_DOWNSAMPLE_RATIO,
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
        check_segment_pads(selected, segment_tokens, token_start, token_end, items[0]["pad_id"], downsample_ratio)
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
