"""Save trained weights as a drop-in replacement of a source safetensors checkpoint.

:func:`save_like_source` writes every tensor of the source checkpoint into the same shard files
with the same names, dtypes and shapes: trained tensors come from the caller (already encoded
into the source format, e.g. FP8 weight + ``.scale``), every other tensor (vision tower, MTP,
untrained tables, ...) is byte-copied from the source. Non-weight files (config, tokenizer, code)
and the index are copied verbatim, so the result loads exactly like the source checkpoint.

Shards are streamed: a trained tensor is buffered (on CPU) only until its shard is complete, and
copied tensors are moved in fixed-size chunks without materializing them.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
from dataclasses import dataclass
from typing import Iterable

import torch

INDEX = "model.safetensors.index.json"
_COPY_CHUNK = 256 << 20

_DTYPE = {
    torch.float64: "F64", torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16",
    torch.int64: "I64", torch.int32: "I32", torch.int16: "I16", torch.int8: "I8", torch.uint8: "U8",
    torch.bool: "BOOL", torch.float8_e4m3fn: "F8_E4M3", torch.float8_e5m2: "F8_E5M2",
    torch.float8_e8m0fnu: "F8_E8M0",
}
_TORCH = {v: k for k, v in _DTYPE.items()}


@dataclass(frozen=True)
class _SrcTensor:
    file: str          # shard file name
    dtype: str
    shape: tuple
    begin: int         # absolute byte offsets in the shard file
    end: int


def read_source_layout(src_dir: str) -> dict[str, _SrcTensor]:
    """Tensor name -> where/how it is stored in the source checkpoint (headers only)."""
    index = os.path.join(src_dir, INDEX)
    if os.path.exists(index):
        with open(index) as f:
            files = sorted(set(json.load(f)["weight_map"].values()))
    else:
        files = sorted(f for f in os.listdir(src_dir) if f.endswith(".safetensors"))
    out: dict[str, _SrcTensor] = {}
    for fname in files:
        with open(os.path.join(src_dir, fname), "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(n))
        header.pop("__metadata__", None)
        for name, h in header.items():
            b, e = h["data_offsets"]
            out[name] = _SrcTensor(fname, h["dtype"], tuple(h["shape"]), 8 + n + b, 8 + n + e)
    return out


def _tensor_bytes(t: torch.Tensor) -> memoryview:
    return memoryview(t.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy())


def _write_shard(src_dir: str, out_dir: str, fname: str, keys: list[str], layout: dict[str, _SrcTensor],
                 trained: dict[str, torch.Tensor]) -> None:
    """Rewrite one source shard: same header bytes; trained tensors from ``trained``, rest copied."""
    keys = sorted(keys, key=lambda k: layout[k].begin)
    src_path, out_path = os.path.join(src_dir, fname), os.path.join(out_dir, fname)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(src_path, "rb") as src, open(out_path + ".tmp", "wb") as out:
        (n,) = struct.unpack("<Q", src.read(8))
        out.write(struct.pack("<Q", n) + src.read(n))
        pos = 8 + n
        for k in keys:
            info = layout[k]
            if info.begin != pos:
                raise RuntimeError(f"{fname}: non-contiguous tensor data at {k}")
            if k in trained:
                data = _tensor_bytes(trained.pop(k))
                assert len(data) == info.end - info.begin, k
                out.write(data)
            else:
                src.seek(info.begin)
                left = info.end - info.begin
                while left:
                    chunk = src.read(min(left, _COPY_CHUNK))
                    out.write(chunk)
                    left -= len(chunk)
            pos = info.end
    os.replace(out_path + ".tmp", out_path)


def _check_and_cast(name: str, t: torch.Tensor, info: _SrcTensor) -> torch.Tensor:
    if tuple(t.shape) != info.shape:
        raise ValueError(f"{name}: shape {tuple(t.shape)} != source {list(info.shape)}")
    want = _TORCH[info.dtype]
    if t.dtype != want:
        if not (t.is_floating_point() and want.is_floating_point and info.dtype in ("F32", "F16", "BF16", "F64")):
            raise ValueError(f"{name}: dtype {t.dtype} cannot be stored as source dtype {info.dtype}")
        t = t.to(want)
    return t


def _copy_metadata_files(src_dir: str, out_dir: str) -> None:
    """Everything except the weight shards (config, tokenizer, index, code, docs)."""
    for entry in os.listdir(src_dir):
        if entry.endswith(".safetensors"):
            continue
        s, d = os.path.join(src_dir, entry), os.path.join(out_dir, entry)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)


def save_like_source(named_tensors: Iterable[tuple[str, torch.Tensor]], src_dir: str, out_dir: str,
                     expected: set[str]) -> None:
    """Write ``out_dir`` as ``src_dir`` with the tensors in ``expected`` replaced.

    ``named_tensors`` must yield exactly the names in ``expected``, each in the source dtype/shape
    (float tensors are cast to the source float dtype; quantized ones must already be encoded).
    """
    layout = read_source_layout(src_dir)
    unknown = sorted(expected - layout.keys())
    if unknown:
        raise KeyError(f"{len(unknown)} trained tensors are not in {src_dir}, e.g. {unknown[:5]}")
    os.makedirs(out_dir, exist_ok=True)
    by_file: dict[str, list[str]] = {}
    for k, info in layout.items():
        by_file.setdefault(info.file, []).append(k)
    pending = {f: {k for k in ks if k in expected} for f, ks in by_file.items()}
    buffered: dict[str, torch.Tensor] = {}
    seen: set[str] = set()
    for name, t in named_tensors:
        if name not in expected or name in seen:
            raise KeyError(f"unexpected {'duplicate ' if name in seen else ''}tensor {name!r}")
        seen.add(name)
        info = layout[name]
        buffered[name] = _check_and_cast(name, t, info).detach().cpu()
        left = pending[info.file]
        left.discard(name)
        if not left:
            _write_shard(src_dir, out_dir, info.file, by_file[info.file], layout, buffered)
    missing = sorted(expected - seen)
    if missing:
        raise RuntimeError(f"{len(missing)} expected tensors were not produced, e.g. {missing[:5]}")
    for f, ks in by_file.items():
        if not any(k in expected for k in ks):  # nothing trained in this shard: pure copy
            _write_shard(src_dir, out_dir, f, ks, layout, {})
    _copy_metadata_files(src_dir, out_dir)
