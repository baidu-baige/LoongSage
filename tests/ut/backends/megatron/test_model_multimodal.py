"""Image-table plumbing in coda.backends.megatron.model (hash rows -> forward kwargs / MRoPE grids)."""

import pytest
import torch

pytest.importorskip("megatron", reason="Megatron-Core is not installed")

import coda.backends.megatron.model as model_mod  # noqa: E402


def _table():
    return {
        "a": {"pixel_values": torch.full((4, 8), 1.0), "image_grid_thw": torch.tensor([[1, 2, 2]])},
        "b": {"pixel_values": torch.full((2, 8), 2.0), "image_grid_thw": torch.tensor([[1, 1, 2]])},
    }


class TestMultimodalForwardKwargs:
    def test_hash_rows_are_resolved_from_table_in_row_order(self):
        kwargs = model_mod._multimodal_forward_kwargs({"multimodal_train_inputs": [["b"], None, ["a", "b"]]}, _table())
        assert kwargs["pixel_values"].shape == (8, 8)
        assert kwargs["image_grid_thw"].tolist() == [[1, 1, 2], [1, 2, 2], [1, 1, 2]]

    def test_text_only_batch_gives_empty_kwargs(self):
        assert model_mod._multimodal_forward_kwargs({"multimodal_train_inputs": None}, None) == {}
        assert model_mod._multimodal_forward_kwargs({"multimodal_train_inputs": [None, None]}, _table()) == {}
        assert model_mod._multimodal_forward_kwargs({}, None) == {}


class TestGetPositionIdsWithTable:
    def _run(self, monkeypatch, rows, table):
        captured = {}
        packed = torch.arange(6)

        def fake_prepare(tokens_list, cp_partition_mode, image_grid_thw_list, rope_index_fn):
            captured["grids"] = image_grid_thw_list
            captured["rope_index_fn"] = rope_index_fn
            return torch.zeros(3, packed.size(0), dtype=torch.long)

        monkeypatch.setattr(model_mod.mpu, "get_context_parallel_world_size", lambda: 2)
        monkeypatch.setattr(model_mod, "_uses_mrope", lambda _chunk: True)
        monkeypatch.setattr(model_mod, "_make_rope_index_fn", lambda _chunk: "rope_fn")
        monkeypatch.setattr(model_mod, "prepare_packed_mrope_position_ids", fake_prepare)
        model_mod._get_position_ids(
            object(), [packed], packed, "zigzag", multimodal_rows=rows, image_table=table
        )
        return captured

    def test_grids_are_concatenated_per_row_from_table(self, monkeypatch):
        captured = self._run(monkeypatch, [["a", "b"], None], _table())
        grids = captured["grids"]
        assert grids[0].tolist() == [[1, 2, 2], [1, 1, 2]]
        assert grids[1] is None
        assert captured["rope_index_fn"] == "rope_fn"

    def test_text_only_rows_pass_no_grids(self, monkeypatch):
        captured = self._run(monkeypatch, None, None)
        assert captured["grids"] is None
        assert captured["rope_index_fn"] is None
