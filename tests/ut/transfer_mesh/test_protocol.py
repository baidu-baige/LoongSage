"""Unit tests for transfer_mesh.protocol (TensorSpec, MetaFrame, dtype conversion)."""

import pytest
import torch

from coda.transfer_mesh.protocol import (
    TensorSpec,
    MetaFrame,
    str_to_dtype,
)


# ── TensorSpec ───────────────────────────────────────────────────────────────


class TestTensorSpec:

    def test_creation_and_fields(self):
        spec = TensorSpec(
            name="layer.weight", shape=(768, 3072), offset=0, dtype="torch.bfloat16"
        )
        assert spec.name == "layer.weight"
        assert spec.shape == (768, 3072)
        assert spec.offset == 0
        assert spec.dtype == "torch.bfloat16"

    def test_dtype_defaults_to_empty(self):
        assert TensorSpec(name="t", shape=(1,), offset=0).dtype == ""

    @pytest.mark.parametrize("shape, expected_numel", [
        ((2, 3, 4), 24),
        ((10,), 10),
        ((), 1),            # scalar
        ((0, 100), 0),      # zero dimension
        ((1,), 1),
        ((1, 1, 1, 1), 1),
    ])
    def test_numel(self, shape, expected_numel):
        spec = TensorSpec(name="t", shape=shape, offset=0)
        assert spec.numel() == expected_numel


# ── MetaFrame ────────────────────────────────────────────────────────────────


class TestMetaFrame:

    def test_default_values(self):
        mf = MetaFrame()
        assert mf.is_end is False
        assert mf.tensor_specs == []
        assert mf.payload_numel == 0

    def test_end_frame(self):
        mf = MetaFrame(is_end=True)
        assert mf.is_end is True
        assert mf.tensor_specs == []

    def test_tensor_specs_and_payload_numel(self):
        specs = [
            TensorSpec(name="a", shape=(10, 20), offset=0),
            TensorSpec(name="b", shape=(30,), offset=200),
            TensorSpec(name="c", shape=(2, 3, 4), offset=230),
        ]
        mf = MetaFrame(
            tensor_specs=specs,
            payload_numel=254,  # 200 + 30 + 24
        )

        assert len(mf.tensor_specs) == 3
        assert mf.tensor_specs[0].name == "a"
        assert mf.tensor_specs[1].offset == 200
        assert mf.tensor_specs[2].numel() == 24
        assert mf.payload_numel == 254

    def test_serialize_deserialize_roundtrip(self):
        specs = [
            TensorSpec(name="w1", shape=(100,), offset=0, dtype="torch.float32"),
            TensorSpec(name="w2", shape=(200,), offset=400, dtype="torch.bfloat16"),
        ]
        mf = MetaFrame(
            is_end=False,
            tensor_specs=specs,
            payload_numel=800,
        )

        data = mf.serialize()
        assert isinstance(data, bytes)

        restored = MetaFrame.deserialize(data)
        assert restored.is_end == mf.is_end
        assert restored.payload_numel == mf.payload_numel
        assert len(restored.tensor_specs) == 2
        assert [s.name for s in restored.tensor_specs] == ["w1", "w2"]
        # Per-spec dtype/offset must survive: the receiver reconstructs from them.
        assert [s.dtype for s in restored.tensor_specs] == [
            "torch.float32",
            "torch.bfloat16",
        ]
        assert [s.offset for s in restored.tensor_specs] == [0, 400]
        assert [s.shape for s in restored.tensor_specs] == [(100,), (200,)]

    def test_deserialize_wrong_type_raises_type_error(self):
        """Deserializing non-MetaFrame bytes must raise TypeError."""
        import pickle
        bad_data = pickle.dumps({"not": "a MetaFrame"})
        with pytest.raises(TypeError, match="MetaFrame"):
            MetaFrame.deserialize(bad_data)


# ── dtype conversion ─────────────────────────────────────────────────────────


class TestDtypeConversion:

    ALL_SUPPORTED_DTYPES = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.bool,
    ]

    @pytest.mark.parametrize("dtype", ALL_SUPPORTED_DTYPES)
    def test_roundtrip(self, dtype):
        restored = str_to_dtype(str(dtype))
        assert restored == dtype

    def test_str_to_dtype_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown dtype"):
            str_to_dtype("torch.not_a_real_dtype")

    def test_str_to_dtype_falls_back_to_getattr_for_unmapped_dtype(self):
        """A real torch dtype missing from dtype_map must resolve via getattr."""
        # complex64 is a genuine torch.dtype that dtype_map does not list.
        assert str_to_dtype("torch.complex64") == torch.complex64

    def test_str_to_dtype_rejects_non_dtype_torch_attribute(self):
        """getattr can find torch.nn, which is not a dtype — that must not pass."""
        with pytest.raises(ValueError, match="does not resolve to a torch.dtype"):
            str_to_dtype("torch.nn")
