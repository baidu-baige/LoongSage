"""Unit tests for coda.utils.multimodal_tensors (torch-only helpers)."""

import pytest
import torch

from coda.utils.multimodal_tensors import merge_multimodal_batch, move_multimodal_to_device, slice_segment_inputs

PAD = 9


def _item(token_start, token_end, n_patches, fill):
    return {
        "token_start": token_start,
        "token_end": token_end,
        "image_hash": f"h{fill}" * 4,
        "pad_id": PAD,
        "tensors": {
            "pixel_values": torch.full((n_patches, 4), float(fill)),
            "image_grid_thw": torch.tensor([[1, 2, n_patches // 2]]),
        },
    }


def _tokens(length, *runs):
    """`length` non-pad tokens with PAD written over each `[start, end)` run."""
    ids = [1] * length
    for start, end in runs:
        ids[start:end] = [PAD] * (end - start)
    return ids


class TestSegmentPadCheck:
    def test_consistent_segment_passes(self):
        items = [_item(2, 6, 4, 1), _item(10, 14, 4, 2)]
        row = slice_segment_inputs(items, 0, 20, _tokens(20, (2, 6), (10, 14)))
        assert row["pixel_values"].shape == (8, 4)

    def test_offset_segment_passes(self):
        row = slice_segment_inputs([_item(12, 16, 4, 1)], 10, 20, _tokens(10, (2, 6)))
        assert row["pixel_values"].shape == (4, 4)

    def test_text_only_segment_of_multimodal_trajectory_is_none(self):
        assert slice_segment_inputs([_item(2, 6, 4, 1)], 10, 20, _tokens(10)) is None

    def test_pad_without_image_raises(self):
        # a pad run in the segment that no item covers (the image was dropped)
        with pytest.raises(ValueError, match="pad tokens"):
            slice_segment_inputs([_item(2, 6, 4, 1)], 0, 20, _tokens(20, (2, 6), (10, 14)))

    def test_straddling_image_raises(self):
        # image [2, 6) straddles the segment starting at 4: its pads are in the segment but it is not selected
        with pytest.raises(ValueError, match="pad tokens"):
            slice_segment_inputs([_item(2, 6, 4, 1)], 4, 20, _tokens(16, (0, 2)))

    def test_item_range_not_all_pad_raises(self):
        with pytest.raises(ValueError, match="not entirely image pad"):
            slice_segment_inputs([_item(2, 6, 4, 1)], 0, 20, _tokens(20, (2, 5)))

    def test_patch_count_mismatch_raises(self):
        item = _item(2, 6, 4, 1)
        item["tensors"]["pixel_values"] = torch.zeros(3, 4)
        with pytest.raises(ValueError, match="pixel_values"):
            slice_segment_inputs([item], 0, 20, _tokens(20, (2, 6)))

    def test_check_is_skipped_without_tokens(self):
        assert slice_segment_inputs([_item(2, 6, 4, 1)], 0, 20) is not None


class TestSliceSegmentInputs:
    def test_none_or_empty_items(self):
        assert slice_segment_inputs(None, 0, 10) is None
        assert slice_segment_inputs([], 0, 10) is None

    def test_selects_images_inside_span_in_order(self):
        items = [_item(2, 6, 4, 1), _item(20, 26, 6, 2), _item(40, 44, 4, 3)]
        row = slice_segment_inputs(items, 0, 30)
        assert row["pixel_values"].shape == (10, 4)
        assert row["image_grid_thw"].shape == (2, 3)
        assert row["pixel_values"][0, 0] == 1.0
        assert row["pixel_values"][-1, 0] == 2.0

    def test_no_image_in_span_returns_none(self):
        assert slice_segment_inputs([_item(2, 6, 4, 1)], 10, 30) is None

    def test_image_straddling_boundary_is_not_selected(self):
        assert slice_segment_inputs([_item(2, 6, 4, 1)], 0, 5) is None
        assert slice_segment_inputs([_item(2, 6, 4, 1)], 3, 10) is None

    def test_span_boundaries_are_inclusive_of_whole_image(self):
        assert slice_segment_inputs([_item(2, 6, 4, 1)], 2, 6) is not None


class TestMergeMultimodalBatch:
    def test_single_image_row_shares_tensors_with_source(self):
        # Rows of the same prompt must keep pointing at the Router cache's tensors so Ray stores them once.
        item = _item(2, 6, 4, 1)
        row_a = slice_segment_inputs([item], 0, 20)
        row_b = slice_segment_inputs([item], 0, 20)
        assert row_a is not item["tensors"] and row_a["pixel_values"] is item["tensors"]["pixel_values"]
        assert row_a["pixel_values"] is row_b["pixel_values"]
        row_a["extra"] = torch.zeros(1)
        assert "extra" not in item["tensors"]

    def test_multi_image_row_is_a_fresh_concat(self):
        items = [_item(2, 6, 4, 1), _item(10, 14, 4, 2)]
        row = slice_segment_inputs(items, 0, 20)
        assert row["pixel_values"].data_ptr() != items[0]["tensors"]["pixel_values"].data_ptr()

    def test_none_or_empty_rows(self):
        assert merge_multimodal_batch(None) is None
        assert merge_multimodal_batch([]) is None

    def test_all_none_rows(self):
        assert merge_multimodal_batch([None, None]) is None

    def test_skips_none_rows_and_keeps_row_order(self):
        a = {"pixel_values": torch.full((2, 4), 1.0), "image_grid_thw": torch.tensor([[1, 1, 2]])}
        b = {"pixel_values": torch.full((3, 4), 2.0), "image_grid_thw": torch.tensor([[1, 1, 3]])}
        merged = merge_multimodal_batch([None, a, None, b])
        assert merged["pixel_values"].shape == (5, 4)
        assert merged["pixel_values"][0, 0] == 1.0 and merged["pixel_values"][-1, 0] == 2.0
        assert merged["image_grid_thw"].tolist() == [[1, 1, 2], [1, 1, 3]]

    def test_does_not_mutate_inputs(self):
        a = {"pixel_values": torch.zeros(2, 4)}
        merge_multimodal_batch([a, a])
        assert a["pixel_values"].shape == (2, 4)


class TestMoveMultimodalToDevice:
    def test_keeps_none_rows_and_tensor_values(self):
        row = {"pixel_values": torch.arange(8, dtype=torch.float32).reshape(2, 4)}
        out = move_multimodal_to_device([None, row], "cpu")
        assert out[0] is None
        assert torch.equal(out[1]["pixel_values"], row["pixel_values"])
        assert out[1]["pixel_values"].device.type == "cpu"
