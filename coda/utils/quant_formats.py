"""Block-scaled checkpoint quantization formats with power-of-two (ue8m0) scales (pure torch).

Two formats, described by a :class:`QuantScheme` (``block`` = elements covered by one scale):
  - ``fp8_block``: e4m3, one scale per ``(br, bc)`` tile -> (e4m3 [O, I], e8m0 [O/br, I/bc]).
    DeepSeek-V4.1 dense linears use 32x32 tiles, its engram table 1x32.
  - ``mxfp4``:     e2m1, one scale per ``bc`` elements along the input dim -> (int8 [O, I/2] packed,
    e8m0 [O, I/bc]); two codes per byte, LOW nibble = even column.

With ue8m0 scales ``dequant(quant(w))`` is exactly representable in bf16 and is a fixed point:
re-quantizing a snapped tensor with any quantizer on the same grid (same tiles, ceil-pow2 scale,
round-to-nearest) reproduces the same values, so what the bridge re-quantizes at export time
dequantizes to exactly the snapped training weights (QAT).

:func:`scan_ckpt_quant` recovers the per-tensor scheme of a safetensors checkpoint from its headers
alone (dtype + shape of every ``X.weight`` / ``X.scale`` pair), without reading tensor data.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Callable

import torch

from coda.utils.hf_ckpt import read_source_layout

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

def _round_e4m3_threshold(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Threshold rounding onto the e4m3 grid (fp32 out): between the two grid magnitudes around
    ``|y|`` take the upper one iff the fraction of the gap covered exceeds ``t``. ``y`` already
    clamped to [-448, 448]; e4m3 codes 0x00..0x7E are monotone in magnitude."""
    a = y.abs()
    c = a.to(torch.float8_e4m3fn).view(torch.uint8).to(torch.int32)  # RNE code of |y|
    lo = torch.where(_e4m3_code_value(c) > a, c - 1, c)
    hi = (lo + 1).clamp_max(0x7E)
    lo_v, hi_v = _e4m3_code_value(lo), _e4m3_code_value(hi)
    return torch.where(a - lo_v > t * (hi_v - lo_v), hi_v, lo_v) * torch.sign(y)


def _e4m3_code_value(code: torch.Tensor) -> torch.Tensor:
    return code.to(torch.uint8).view(torch.float8_e4m3fn).float()


def _round_fp8_block(
    w: torch.Tensor, block: tuple[int, int], t: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """[O, I] -> (e4m3 [O/br, br, I/bc, bc], fp32 scale broadcastable to it, e8m0 [O/br, I/bc]).
    With per-element thresholds ``t`` ([O, I] in (0, 1)) the values are threshold-rounded and
    returned as fp32 (exactly representable in e4m3) instead of round-to-nearest e4m3."""
    (O, I), (br, bc) = w.shape, block
    assert O % br == 0 and I % bc == 0, f"fp8 {br}x{bc} blocks need aligned dims, got {tuple(w.shape)}"
    x = w.float().view(O // br, br, I // bc, bc)
    amax = x.abs().amax(dim=(1, 3)).clamp_min(AMAX_FLOOR)
    scale, scale_e8m0 = _exp_to_scale(_ceil_pow2_exponent(amax / FP8_MAX))
    scale = scale[:, None, :, None]
    y = (x / scale).clamp(-FP8_MAX, FP8_MAX)
    q = y.to(torch.float8_e4m3fn) if t is None else _round_e4m3_threshold(y, t.view(y.shape))
    return q, scale, scale_e8m0


def quant_fp8_block(w: torch.Tensor, block: tuple[int, int] = (32, 32)) -> tuple[torch.Tensor, torch.Tensor]:
    """[O, I] float -> (e4m3 [O, I], e8m0 [O/br, I/bc]); O, I must be multiples of the tile."""
    q, _, scale_e8m0 = _round_fp8_block(w, block)
    return q.view(w.shape), scale_e8m0


def dequant_fp8_block(q: torch.Tensor, scale: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
    O, I = q.shape
    sO, sI = scale.shape
    s = e8m0_to_float(scale)
    x = q.float().view(sO, O // sO, sI, I // sI) * s[:, None, :, None]
    return x.view(O, I).to(dtype)


# ---------------------------------------------------------------- MXFP4 e2m1, groups of bc along I

def _round_e2m1(x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
    """Round onto the e2m1 grid; ``x`` already clamped to [-6, 6]. Round-to-nearest-even, or with
    per-element thresholds ``t`` in (0, 1): between the two grid magnitudes around ``|x|`` take the
    upper one iff the fraction of the gap covered exceeds ``t``."""
    mag = x.abs()
    step = torch.where(mag < 2.0, 0.5, torch.where(mag < 4.0, 1.0, 2.0))
    if t is None:
        return torch.round(mag / step) * step * torch.sign(x)
    lo = torch.floor(mag / step) * step
    return (lo + (mag - lo > t * step) * step).clamp_max(FP4_MAX) * torch.sign(x)


def _round_mxfp4(
    w: torch.Tensor, group: int, t: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """[O, I] -> (e2m1 grid values as fp32 [O, I/group, group], fp32 scale broadcastable to it,
    e8m0 [O, I/group]); ``t``: optional [O, I] rounding thresholds, see :func:`_round_e2m1`."""
    O, I = w.shape
    assert I % group == 0, f"mxfp4 g{group} needs in-dim % {group} == 0, got {tuple(w.shape)}"
    x = w.float().view(O, I // group, group)
    amax = x.abs().amax(dim=-1).clamp_min(AMAX_FLOOR)
    scale, scale_e8m0 = _exp_to_scale(_ceil_pow2_exponent(amax / FP4_MAX))
    scale = scale[..., None]
    t = None if t is None else t.view(x.shape)
    return _round_e2m1((x / scale).clamp(-FP4_MAX, FP4_MAX), t), scale, scale_e8m0


def quant_mxfp4(w: torch.Tensor, group: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """[O, I] float -> (int8 [O, I/2] packed e2m1, e8m0 [O, I/group]); I must be a multiple of group."""
    v, _, scale_e8m0 = _round_mxfp4(w, group)
    v = v.view(w.shape)
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
    def snap_(
        self, w: torch.Tensor, thresholds: Callable[[int, int, torch.device], torch.Tensor] | None = None
    ) -> torch.Tensor:
        """Overwrite ``w`` in place with ``dequant(quant(w))``, computed without packing to codes.

        ``thresholds(r0, r1, device)`` -> [r1 - r0, I] fp32 in (0, 1): per-element rounding
        thresholds for rows ``r0:r1`` (see :func:`element_thresholds`) instead of round-to-nearest.
        Rows are then processed in tile-aligned chunks to bound the temporaries."""
        if thresholds is None:
            v, scale, _ = self._round(w, None)
            w.copy_((v.float() * scale).view(w.shape))
            return w
        O, I = w.shape
        br = self.block[0]
        rows = max(br, _SNAP_CHUNK_ELEMS // max(I, 1) // br * br)
        for r0 in range(0, O, rows):
            r1 = min(O, r0 + rows)
            part = w[r0:r1]
            v, scale, _ = self._round(part, thresholds(r0, r1, w.device))
            part.copy_((v * scale).view(part.shape))
        return w

    def _round(self, w: torch.Tensor, t: torch.Tensor | None):
        if self.kind == "fp8_block":
            return _round_fp8_block(w, self.block, t)
        return _round_mxfp4(w, self.block[1], t)


# ---------------------------------------------------------------- per-element rounding thresholds
# Weight QAT snaps W to Q(W) after every step. With round-to-nearest, weights that start on the grid
# (any quantized checkpoint) only move once their fp32 master drifts half a grid step, which RL-size
# updates (lr ~1e-6) never reach on the coarse MXFP4 grid. A fixed, per-element threshold t ~ U(0, 1)
# instead of 1/2 keeps grid points fixed (initial weights are reproduced exactly) while the rounded
# value moves by the master's drift on average over elements; being fixed, it never flips back and
# forth while the master stands still. Thresholds hash the element's index in the full checkpoint
# tensor, so every DP replica and every TP/EP layout draws the same ones.

_M32 = 0xFFFFFFFF
_SNAP_CHUNK_ELEMS = 1 << 24


def _mul32(x: torch.Tensor, c: int) -> torch.Tensor:
    """(x * c) mod 2^32 for int64 ``x`` in [0, 2^32), without int64 overflow."""
    return (x * (c & 0xFFFF) + (((x * (c >> 16)) & 0xFFFF) << 16)) & _M32


def _hash32(x: torch.Tensor) -> torch.Tensor:
    """lowbias32 integer hash of int64 tensors holding uint32 values."""
    x = x ^ (x >> 16)
    x = _mul32(x, 0x7FEB352D)
    x = x ^ (x >> 15)
    x = _mul32(x, 0x846CA68B)
    return x ^ (x >> 16)


def tensor_seed(name: str) -> int:
    """Stable 32-bit seed of a checkpoint tensor name (not Python's per-process ``hash``)."""
    return zlib.crc32(name.encode())


def element_thresholds(seed: int, rows: torch.Tensor, cols: torch.Tensor, ncols: int) -> torch.Tensor:
    """Fixed rounding thresholds in (0, 1), fp32 [len(rows), len(cols)], for elements
    (rows[r], cols[c]) of the checkpoint tensor with ``seed`` and ``ncols`` columns."""
    idx = rows.long()[:, None] * ncols + cols.long()[None, :]
    h = _hash32(_hash32((idx >> 32) ^ (seed & _M32)) ^ (idx & _M32))
    return ((h >> 8).float() + 0.5) * 2.0**-24


@dataclass(frozen=True)
class CkptQuant:
    """How one checkpoint ``*.weight`` is stored: scheme, companion scale name, logical [O, I] shape."""

    scheme: QuantScheme
    scale_name: str
    shape: tuple[int, int]


@dataclass(frozen=True)
class UnsupportedQuant:
    """A ``*.weight`` stored with a ``scale_name`` companion in a layout :class:`QuantScheme` cannot
    express; ``dtype`` / ``shape`` are the stored (safetensors) dtype and shape of the weight."""

    reason: str
    scale_name: str
    dtype: str
    shape: tuple[int, ...]


_SCALE_SUFFIX = ".scale"


def scan_ckpt_quant(path: str) -> tuple[dict[str, CkptQuant], dict[str, UnsupportedQuant]]:
    """Every ``X.weight`` stored with an ``X.scale`` companion, split into
    (name -> :class:`CkptQuant` for supported layouts, name -> :class:`UnsupportedQuant`).

    The scheme is inferred from dtypes and the weight/scale geometry: e4m3 weight -> ``fp8_block``
    with tile ``(O/sO, I/sI)``; int8/uint8 (packed e2m1) weight with one scale row per weight row ->
    ``mxfp4`` groups of ``2*Ip/sI``. Only e8m0 scales are supported, since only power-of-two scales
    make the snapped bf16 weights exact. Whether an unsupported layout is an error depends on
    whether the trainer maps the tensor and on QAT (e.g. the BF16 DeepSeek-V4 conversion keeps the
    unused MTP e_proj/h_proj in FP8 with fp32 scales), so the caller decides.
    """
    layout = read_source_layout(path)
    out: dict[str, CkptQuant] = {}
    unsupported: dict[str, UnsupportedQuant] = {}
    for scale_name, sh in layout.items():
        if not scale_name.endswith(_SCALE_SUFFIX):
            continue
        name = scale_name[: -len(_SCALE_SUFFIX)] + ".weight"
        wh = layout.get(name)
        if wh is None:
            continue
        stored = scale_name, wh.dtype, tuple(wh.shape)
        if sh.dtype != "F8_E8M0" or len(sh.shape) != 2 or len(wh.shape) != 2:
            unsupported[name] = UnsupportedQuant(f"unsupported quantized layout weight={wh} scale={sh}", *stored)
            continue
        (O, I), (sO, sI) = wh.shape, sh.shape
        if wh.dtype == "F8_E4M3":
            kind, logical_i = "fp8_block", I
        elif wh.dtype in ("I8", "U8") and sO == O:
            kind, logical_i = "mxfp4", 2 * I
        else:
            unsupported[name] = UnsupportedQuant(f"unsupported quantized layout weight={wh} scale={sh}", *stored)
            continue
        if O % sO or logical_i % sI:
            unsupported[name] = UnsupportedQuant(f"scale {list(sh.shape)} does not tile weight [{O}, {logical_i}]", *stored)
            continue
        scheme = QuantScheme(kind, (O // sO, logical_i // sI))
        out[name] = CkptQuant(scheme, scale_name, (O, logical_i))
    return out, unsupported
