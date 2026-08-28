"""TransferMesh protocol definitions.

Defines TensorSpec and MetaFrame for metadata transmission.
"""

import math
import pickle
from dataclasses import dataclass, field

import torch


@dataclass
class TensorSpec:
    """Describes single tensor location in physical buffer.

    Attributes:
        name: Tensor name (e.g., "model.layers.0.self_attn.q_proj.weight")
        shape: Original tensor shape
        offset: Start offset in flattened payload (in bytes)
        dtype: Original tensor dtype string (e.g., "torch.bfloat16")
    """
    name: str
    shape: tuple[int, ...]
    offset: int
    dtype: str = ""

    def numel(self) -> int:
        """Return the number of elements in this tensor."""
        return math.prod(self.shape)


@dataclass
class MetaFrame:
    """Control frame for bucket transmission.

    Attributes:
        is_end: Stream termination flag
        tensor_specs: List of tensor specifications in current bucket
        payload_numel: Total number of elements in the payload buffer
    """
    is_end: bool = False
    tensor_specs: list[TensorSpec] = field(default_factory=list)
    payload_numel: int = 0

    def serialize(self) -> bytes:
        """Serialize MetaFrame to bytes for transmission."""
        return pickle.dumps(self)

    @classmethod
    def deserialize(cls, data: bytes) -> "MetaFrame":
        """Deserialize bytes to MetaFrame."""
        result: "MetaFrame" = pickle.loads(data)
        if not isinstance(result, MetaFrame):
            raise TypeError(
                f"Deserialized object is not a MetaFrame: got {type(result)!r}"
            )
        return result


def str_to_dtype(dtype_str: str) -> torch.dtype:
    """Convert string representation to torch dtype."""
    dtype_map = {
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "torch.float32": torch.float32,
        "torch.float64": torch.float64,
        "torch.int8": torch.int8,
        "torch.int16": torch.int16,
        "torch.int32": torch.int32,
        "torch.int64": torch.int64,
        "torch.uint8": torch.uint8,
        "torch.bool": torch.bool,
    }
    try:
        return dtype_map[dtype_str]
    except KeyError:
        try:
            attr = getattr(torch, dtype_str.split(".")[-1])
        except AttributeError:
            raise ValueError(f"Unknown dtype string: {dtype_str}")
        if not isinstance(attr, torch.dtype):
            raise ValueError(f"'{dtype_str}' does not resolve to a torch.dtype")
        return attr
