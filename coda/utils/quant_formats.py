"""Block-scaled checkpoint quantization formats with power-of-two (ue8m0) scales (pure torch).

Two formats, described by a :class:`QuantScheme` (``block`` = elements covered by one scale):
  - ``fp8_block``: e4m3, one scale per ``(br, bc)`` tile -> (e4m3 [O, I], e8m0 [O/br, I/bc]).
    DeepSeek-V4.1 dense linears use 32x32 tiles, its engram table 1x32.
  - ``mxfp4``:     e2m1, one scale per ``bc`` elements along the input dim -> (int8 [O, I/2] packed,
    e8m0 [O, I/bc]); two codes per byte, LOW nibble = even column.

With ue8m0 scales ``dequant(quant(w))`` is exactly representable in bf16 and is a fixed point:
re-quantizing a snapped tensor reproduces the same values, so snapping the training weights (QAT)
and re-quantizing them at export time agree bit for bit.

:func:`read_ckpt_quant` recovers the per-tensor scheme of a safetensors checkpoint from its headers
alone (dtype + shape of every ``X.weight`` / ``X.scale`` pair), without reading tensor data.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass

import torch

FP8_MAX = 448.0
FP4_MAX = 6.0
# Scale floor for all-zero / tiny blocks (scale is irrelevant for their dequantized values).
AMAX_FLOOR = 1e-4

# e2m1 magnitude grid; code = index (sign in bit 3).
_FP4_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _ceil_pow2_exponent(x: torch.Tensor) -> torch.Tensor:
    """Biased fp32 exponent of 2^ceil(log2(x)) for positive normal fp32 ``x`` (int32)."""
    bits = x.float().contiguous().view(torch.int32)
    exp = (bits >> 23) & 0xFF
    return exp + ((bits & 0x7FFFFF) != 0).to(torch.int32)


def _exp_to_scale(exp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Biased exponent -> (fp32 scale, e8m0 scale)."""
    scale = (exp << 23).view(torch.float32)
    return scale, exp.to(torch.uint8).view(torch.float8_e8m0fnu)


def e8m0_to_float(scale: torch.Tensor) -> torch.Tensor:
    """e8m0 (or uint8 holding e8m0 bits) -> fp32 2^(e-127)."""
    exp = scale.view(torch.uint8).to(torch.int32)
    return (exp << 23).view(torch.float32)


# ---------------------------------------------------------------- FP8 e4m3, (br, bc) tiles

def quant_fp8_block(w: torch.Tensor, block: tuple[int, int] = (32, 32)) -> tuple[torch.Tensor, torch.Tensor]:
    """[O, I] float -> (e4m3 [O, I], e8m0 [O/br, I/bc]); O, I must be multiples of the tile."""
    (O, I), (br, bc) = w.shape, block
    assert O % br == 0 and I % bc == 0, f"fp8 {br}x{bc} blocks need aligned dims, got {tuple(w.shape)}"
    x = w.float().view(O // br, br, I // bc, bc)
    amax = x.abs().amax(dim=(1, 3)).clamp_min(AMAX_FLOOR)
    scale, scale_e8m0 = _exp_to_scale(_ceil_pow2_exponent(amax / FP8_MAX))
    q = (x / scale[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.view(O, I), scale_e8m0


def dequant_fp8_block(q: torch.Tensor, scale: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
    O, I = q.shape
    sO, sI = scale.shape
    s = e8m0_to_float(scale)
    x = q.float().view(sO, O // sO, sI, I // sI) * s[:, None, :, None]
    return x.view(O, I).to(dtype)


# ---------------------------------------------------------------- MXFP4 e2m1, groups of bc along I

def _round_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest-even onto the e2m1 grid; ``x`` already clamped to [-6, 6]."""
    mag = x.abs()
    step = torch.where(mag < 2.0, 0.5, torch.where(mag < 4.0, 1.0, 2.0))
    return torch.round(mag / step) * step * torch.sign(x)


def quant_mxfp4(w: torch.Tensor, group: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """[O, I] float -> (int8 [O, I/2] packed e2m1, e8m0 [O, I/group]); I must be a multiple of group."""
    O, I = w.shape
    assert I % group == 0, f"mxfp4 g{group} needs in-dim % {group} == 0, got {tuple(w.shape)}"
    x = w.float().view(O, I // group, group)
    amax = x.abs().amax(dim=-1).clamp_min(AMAX_FLOOR)
    scale, scale_e8m0 = _exp_to_scale(_ceil_pow2_exponent(amax / FP4_MAX))
    v = _round_e2m1((x / scale[..., None]).clamp(-FP4_MAX, FP4_MAX)).view(O, I)
    # magnitude -> code: grid is {0,.5,1,1.5,2,3,4,6}
    mag = v.abs()
    code = torch.where(mag <= 2.0, mag * 2, torch.where(mag <= 4.0, mag + 2, torch.full_like(mag, 7.0)))
    code = code.to(torch.uint8) | ((v < 0).to(torch.uint8) << 3)
    packed = code[:, 0::2] | (code[:, 1::2] << 4)
    return packed.view(torch.int8), scale_e8m0


def dequant_mxfp4(packed: torch.Tensor, scale: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
    x = packed.view(torch.uint8)
    O = x.shape[0]
    table = torch.tensor(_FP4_GRID + tuple(-g for g in _FP4_GRID), dtype=torch.float32, device=x.device)
    lo, hi = table[(x & 0x0F).long()], table[((x >> 4) & 0x0F).long()]
    v = torch.stack([lo, hi], dim=-1).view(O, -1)
    s = e8m0_to_float(scale)
    return (v.view(O, scale.shape[1], -1) * s[..., None]).view(O, -1).to(dtype)


# ---------------------------------------------------------------- scheme

@dataclass(frozen=True)
class QuantScheme:
    """``kind`` in {"fp8_block", "mxfp4"}; ``block`` = (rows, cols) covered by one ue8m0 scale."""

    kind: str
    block: tuple[int, int]

    def quant(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.kind == "fp8_block":
            return quant_fp8_block(w, self.block)
        return quant_mxfp4(w, self.block[1])

    def dequant(self, q: torch.Tensor, scale: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
        if self.kind == "fp8_block":
            return dequant_fp8_block(q, scale, dtype)
        return dequant_mxfp4(q, scale, dtype)

    @torch.no_grad()
    def snap_(self, w: torch.Tensor) -> torch.Tensor:
        """Overwrite ``w`` in place with ``dequant(quant(w))``."""
        w.copy_(self.dequant(*self.quant(w), dtype=w.dtype))
        return w


@dataclass(frozen=True)
class CkptQuant:
    """How one checkpoint ``*.weight`` is stored: scheme, companion scale name, logical [O, I] shape."""

    scheme: QuantScheme
    scale_name: str
    shape: tuple[int, int]


_SCALE_SUFFIX = ".scale"


def _safetensors_headers(path: str) -> dict[str, dict]:
    """Tensor name -> {"dtype", "shape"} for a safetensors checkpoint dir (headers only)."""
    index = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            files = sorted(set(json.load(f)["weight_map"].values()))
    else:
        files = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
    out: dict[str, dict] = {}
    for fname in files:
        with open(os.path.join(path, fname), "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(n))
        header.pop("__metadata__", None)
        out.update(header)
    return out


def read_ckpt_quant(path: str) -> dict[str, CkptQuant]:
    """Weight name -> :class:`CkptQuant` for every ``X.weight`` stored with an ``X.scale`` companion.

    The scheme is inferred from dtypes and the weight/scale geometry: e4m3 weight -> ``fp8_block``
    with tile ``(O/sO, I/sI)``; int8/uint8 (packed e2m1) weight with one scale row per weight row ->
    ``mxfp4`` groups of ``2*Ip/sI``. Only e8m0 scales are supported (raises otherwise), since only
    power-of-two scales make the snapped bf16 weights exact.
    """
    headers = _safetensors_headers(path)
    out: dict[str, CkptQuant] = {}
    for scale_name, sh in headers.items():
        if not scale_name.endswith(_SCALE_SUFFIX):
            continue
        name = scale_name[: -len(_SCALE_SUFFIX)] + ".weight"
        wh = headers.get(name)
        if wh is None:
            continue
        if sh["dtype"] != "F8_E8M0" or len(sh["shape"]) != 2 or len(wh["shape"]) != 2:
            raise NotImplementedError(f"{name}: unsupported quantized layout weight={wh} scale={sh}")
        (O, I), (sO, sI) = wh["shape"], sh["shape"]
        if wh["dtype"] == "F8_E4M3":
            kind, logical_i = "fp8_block", I
        elif wh["dtype"] in ("I8", "U8") and sO == O:
            kind, logical_i = "mxfp4", 2 * I
        else:
            raise NotImplementedError(f"{name}: unsupported quantized layout weight={wh} scale={sh}")
        if O % sO or logical_i % sI:
            raise NotImplementedError(f"{name}: scale {sh['shape']} does not tile weight [{O}, {logical_i}]")
        scheme = QuantScheme(kind, (O // sO, logical_i // sI))
        out[name] = CkptQuant(scheme, scale_name, (O, logical_i))
    return out
