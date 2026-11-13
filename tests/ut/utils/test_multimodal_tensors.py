"""Unit tests for coda.utils.multimodal_tensors (torch-only helpers)."""

import pytest
import torch

from coda.utils.multimodal_tensors import (
    add_to_image_table,
    merge_multimodal_batch,
    move_image_table_to_device,
    slice_segment_inputs,
)

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


def _slice(items, start, end, tokens=None):
    """Slice with a fresh table; returns (row, table)."""
    table = {}
    return slice_segment_inputs(items, start, end, table, tokens), table


class TestSegmentPadCheck:
    def test_consistent_segment_passes(self):
        items = [_item(2, 6, 4, 1), _item(10, 14, 4, 2)]
        row, table = _slice(items, 0, 20, _tokens(20, (2, 6), (10, 14)))
        assert row == [items[0]["image_hash"], items[1]["image_hash"]]
        assert set(table) == set(row)

    def test_offset_segment_passes(self):
        row, _ = _slice([_item(12, 16, 4, 1)], 10, 20, _tokens(10, (2, 6)))
        assert row == [_item(12, 16, 4, 1)["image_hash"]]

    def test_text_only_segment_of_multimodal_trajectory_is_none(self):
        row, table = _slice([_item(2, 6, 4, 1)], 10, 20, _tokens(10))
        assert row is None and table == {}

    def test_pad_without_image_raises(self):
        # a pad run in the segment that no item covers (the image was dropped)
        with pytest.raises(ValueError, match="pad tokens"):
            _slice([_item(2, 6, 4, 1)], 0, 20, _tokens(20, (2, 6), (10, 14)))

    def test_straddling_image_raises(self):
        # image [2, 6) straddles the segment starting at 4: its pads are in the segment but it is not selected
        with pytest.raises(ValueError, match="pad tokens"):
            _slice([_item(2, 6, 4, 1)], 4, 20, _tokens(16, (0, 2)))

    def test_item_range_not_all_pad_raises(self):
        with pytest.raises(ValueError, match="not entirely image pad"):
            _slice([_item(2, 6, 4, 1)], 0, 20, _tokens(20, (2, 5)))

    def test_patch_count_mismatch_raises(self):
        item = _item(2, 6, 4, 1)
        item["tensors"]["pixel_values"] = torch.zeros(3, 4)
        with pytest.raises(ValueError, match="pixel_values"):
            _slice([item], 0, 20, _tokens(20, (2, 6)))

    def test_check_is_skipped_without_tokens(self):
        assert _slice([_item(2, 6, 4, 1)], 0, 20)[0] is not None


class TestSliceSegmentInputs:
    def test_none_or_empty_items(self):
        assert _slice(None, 0, 10) == (None, {})
        assert _slice([], 0, 10) == (None, {})

    def test_selects_images_inside_span_in_order(self):
        items = [_item(2, 6, 4, 1), _item(20, 26, 6, 2), _item(40, 44, 4, 3)]
        row, table = _slice(items, 0, 30)
        assert row == [items[0]["image_hash"], items[1]["image_hash"]]
        assert set(table) == set(row)
        assert table[row[0]]["pixel_values"][0, 0] == 1.0
        assert table[row[1]]["pixel_values"].shape == (6, 4)

    def test_no_image_in_span_returns_none(self):
        assert _slice([_item(2, 6, 4, 1)], 10, 30) == (None, {})

    def test_image_straddling_boundary_is_not_selected(self):
        assert _slice([_item(2, 6, 4, 1)], 0, 5)[0] is None
        assert _slice([_item(2, 6, 4, 1)], 3, 10)[0] is None

    def test_span_boundaries_are_inclusive_of_whole_image(self):
        assert _slice([_item(2, 6, 4, 1)], 2, 6)[0] is not None

    def test_rows_sharing_an_image_share_one_table_entry(self):
        # Samples of the same prompt carry equal hashes; the table keeps one copy (the first one).
        a, b = _item(2, 6, 4, 1), _item(2, 6, 4, 1)
        table = {}
        row_a = slice_segment_inputs([a], 0, 20, table)
        row_b = slice_segment_inputs([b], 0, 20, table)
        assert row_a == row_b and len(table) == 1
        assert table[row_a[0]]["pixel_values"] is a["tensors"]["pixel_values"]

    def test_same_hash_with_different_tensors_raises(self):
        table = {}
        add_to_image_table(table, "hh", {"pixel_values": torch.zeros(4, 4), "image_grid_thw": torch.tensor([[1, 2, 2]])})
        with pytest.raises(ValueError, match="two different"):
            add_to_image_table(table, "hh", {"pixel_values": torch.zeros(6, 4), "image_grid_thw": torch.tensor([[1, 2, 3]])})
        with pytest.raises(ValueError, match="two different"):
            add_to_image_table(table, "hh", {"pixel_values": torch.zeros(4, 4), "image_grid_thw": torch.tensor([[1, 1, 4]])})

    def test_same_hash_with_equal_tensors_is_accepted(self):
        table = {}
        tensors = {"pixel_values": torch.zeros(4, 4), "image_grid_thw": torch.tensor([[1, 2, 2]])}
        add_to_image_table(table, "hh", tensors)
        add_to_image_table(table, "hh", {k: v.clone() for k, v in tensors.items()})
        assert table["hh"] is tensors

def _table(**fills):
    """`{hash: tensors}` where `name=fill` -> hash `name`, 2 patches filled with `fill`."""
    return {
        name: {
            "pixel_values": torch.full((2, 4), float(fill)),
            "image_grid_thw": torch.tensor([[1, 1, 2]]),
        }
        for name, fill in fills.items()
    }


class TestMergeMultimodalBatch:
    def test_none_or_empty_rows(self):
        table = _table(a=1)
        assert merge_multimodal_batch(None, table) is None
        assert merge_multimodal_batch([], table) is None

    def test_all_none_rows(self):
        assert merge_multimodal_batch([None, None], _table(a=1)) is None
        assert merge_multimodal_batch([None, None], None) is None

    def test_single_image_reuses_table_tensors(self):
        table = _table(a=1)
        merged = merge_multimodal_batch([["a"]], table)
        assert merged["pixel_values"] is table["a"]["pixel_values"]
        merged["extra"] = torch.zeros(1)
        assert "extra" not in table["a"]

    def test_skips_none_rows_and_keeps_row_order(self):
        table = _table(a=1, b=2)
        merged = merge_multimodal_batch([None, ["a"], None, ["b", "a"]], table)
        assert merged["pixel_values"].shape == (6, 4)
        assert [merged["pixel_values"][i, 0].item() for i in (0, 2, 4)] == [1.0, 2.0, 1.0]
        assert merged["image_grid_thw"].shape == (3, 3)

    def test_repeated_hash_is_concatenated_per_pad_run(self):
        merged = merge_multimodal_batch([["a"], ["a"]], _table(a=1))
        assert merged["pixel_values"].shape == (4, 4)

    def test_does_not_mutate_table(self):
        table = _table(a=1)
        merge_multimodal_batch([["a"], ["a"]], table)
        assert table["a"]["pixel_values"].shape == (2, 4)


class TestMoveImageTableToDevice:
    def test_keeps_hashes_and_tensor_values(self):
        table = _table(a=1, b=2)
        out = move_image_table_to_device(table, "cpu")
        assert set(out) == {"a", "b"}
        assert torch.equal(out["b"]["pixel_values"], table["b"]["pixel_values"])
        assert out["a"]["pixel_values"].device.type == "cpu"


def _dsv41_item(token_start, grid_hw, fill=1, n_tokens=None, n_patches=None):
    """A DeepSeek-V4.1 item: `image_grid_hw` grid, token range sized for downsample ratio 3 unless overridden."""
    h, w = grid_hw
    if n_tokens is None:
        n_tokens = -(-h // 3) * (-(-w // 3) + 1) + 2
    return {
        "token_start": token_start,
        "token_end": token_start + n_tokens,
        "image_hash": f"d{fill}" * 6,
        "pad_id": PAD,
        "tensors": {
            "pixel_values": torch.full((h * w if n_patches is None else n_patches, 4), float(fill)),
            "image_grid_hw": torch.tensor([grid_hw], dtype=torch.int64),
        },
    }


class TestDsv41ImageGrid:
    def test_consistent_segment_passes(self):
        item = _dsv41_item(2, (6, 9))  # 2 x 3 LLM grid -> 2 * (3 + 1) + 2 = 10 tokens
        assert item["token_end"] - item["token_start"] == 10
        row, table = _slice([item], 0, 20, _tokens(20, (2, 12)))
        assert row == [item["image_hash"]] and set(table) == set(row)

    def test_non_multiple_of_ratio_rounds_up(self):
        item = _dsv41_item(0, (7, 4))  # ceil(7/3) x ceil(4/3) = 3 x 2 -> 3 * 3 + 2 = 11 tokens
        assert item["token_end"] == 11
        row, _ = _slice([item], 0, 12, _tokens(12, (0, 11)))
        assert row == [item["image_hash"]]

    def test_wrong_token_count_raises(self):
        item = _dsv41_item(2, (6, 9), n_tokens=9)
        with pytest.raises(ValueError, match="needs 10"):
            _slice([item], 0, 20, _tokens(20, (2, 11)))

    def test_downsample_ratio_is_a_parameter(self):
        item = _dsv41_item(0, (6, 6), n_tokens=2 * 3 + 2)  # ratio 3: 2 x 2 LLM grid
        table = {}
        slice_segment_inputs([item], 0, 10, table, _tokens(10, (0, 8)))
        with pytest.raises(ValueError, match="needs"):
            slice_segment_inputs([item], 0, 10, {}, _tokens(10, (0, 8)), downsample_ratio=2)

    def test_patch_count_mismatch_raises(self):
        item = _dsv41_item(0, (6, 9), n_patches=50)
        with pytest.raises(ValueError, match="image_grid_hw implies 54"):
            _slice([item], 0, 12, _tokens(12, (0, 10)))

    def test_qwen_grid_does_not_get_span_check(self):
        # `image_grid_thw` items keep their previous behaviour: any all-pad run of the right patch count passes.
        row, _ = _slice([_item(2, 6, 4, 1)], 0, 10, _tokens(10, (2, 6)))
        assert row is not None

    def test_missing_grid_key_raises(self):
        item = _dsv41_item(0, (3, 3))
        del item["tensors"]["image_grid_hw"]
        with pytest.raises(ValueError, match="image_grid"):
            _slice([item], 0, 10, _tokens(10, (0, 5)))

    def test_same_hash_with_different_grid_raises(self):
        table = {}
        add_to_image_table(table, "hh", {"pixel_values": torch.zeros(6, 4), "image_grid_hw": torch.tensor([[2, 3]])})
        add_to_image_table(table, "hh", {"pixel_values": torch.zeros(6, 4), "image_grid_hw": torch.tensor([[2, 3]])})
        with pytest.raises(ValueError, match="two different"):
            add_to_image_table(table, "hh", {"pixel_values": torch.zeros(6, 4), "image_grid_hw": torch.tensor([[3, 2]])})

    def test_add_to_image_table_compares_pixel_shape_only(self):
        # Pins the current behaviour: equal shape and grid with different values count as the same image.
        table = {}
        a = {"pixel_values": torch.zeros(6, 4), "image_grid_hw": torch.tensor([[2, 3]])}
        b = {"pixel_values": torch.ones(6, 4), "image_grid_hw": torch.tensor([[2, 3]])}
        add_to_image_table(table, "hh", a)
        add_to_image_table(table, "hh", b)
        assert table["hh"] is a

    def test_merge_concatenates_hw_grids(self):
        table = {
            "a": {"pixel_values": torch.zeros(6, 4), "image_grid_hw": torch.tensor([[2, 3]])},
            "b": {"pixel_values": torch.ones(4, 4), "image_grid_hw": torch.tensor([[2, 2]])},
        }
        merged = merge_multimodal_batch([["a"], ["b", "a"]], table)
        assert merged["pixel_values"].shape == (16, 4)
        assert merged["image_grid_hw"].tolist() == [[2, 3], [2, 2], [2, 3]]
