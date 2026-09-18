#!/usr/bin/env python3
"""Cast the fp8/fp4 DeepSeek-V4.1-Flash HF checkpoint to a bf16 HF checkpoint.

  - dense fp8 (e4m3 + 32x32 ue8m0 block scale)      -> bf16
  - routed-expert fp4 (int8-packed e2m1 + block scale) -> bf16
  - the two engram tables (engram.embed.*)          -> kept as fp8 bits + e8m0 scales
  - MTP / DSpark / vision / aligner tensors          -> dropped
  - config.json flattened (text_config lifted, model_type=deepseek_v41)

The output keeps the source (DeepSeek-native flat) tensor names, which is what
the megatron.bridge DeepSeek-V4.1 weight mapping expects.
"""
import argparse
import json
import os
import re
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

# Standard e2m1 (fp4) value table, low nibble first (matches inference/convert.py).
FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)

DROP_PREFIXES = ("vision.", "aligner.", "mtp.")
DROP_EXACT = {"image_start", "image_end", "image_newline"}


def should_drop(name: str) -> bool:
    """Whether a source tensor is a vision/MTP/VL tensor that this cast discards."""
    if name.startswith(DROP_PREFIXES):
        return True
    if name in DROP_EXACT:
        return True
    if name.endswith(".bias_vl"):  # vision-language routing gate bias, unused for text
        return True
    return False


def dequant_fp8_dense(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """e4m3 weight [O, I] with e8m0 block scale [bO, bI] -> bf16 [O, I]."""
    w = weight.float()
    O, I = w.shape
    bO, bI = scale.shape
    obs, ibs = O // bO, I // bI
    w = w.view(bO, obs, bI, ibs) * scale.float().view(bO, 1, bI, 1)
    return w.view(O, I).bfloat16()


def dequant_fp4_experts(weight_i8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """int8-packed e2m1 weight [O, I/2] with e8m0 block scale [bO, bI] -> bf16 [O, I]."""
    x = weight_i8.view(torch.uint8)
    low = (x & 0x0F).long()
    high = ((x >> 4) & 0x0F).long()
    table = FP4_TABLE.to(low.device)
    unp = torch.stack([table[low], table[high]], dim=-1).flatten(1)  # [O, I]
    O, I = unp.shape
    bO, bI = scale.shape
    obs, ibs = O // bO, I // bI
    unp = unp.view(bO, obs, bI, ibs) * scale.float().view(bO, 1, bI, 1)
    return unp.view(O, I).bfloat16()


class ShardWriter:
    """Accumulates tensors and flushes size-bounded safetensors shards, tracking the index."""

    def __init__(self, dst: str, target_bytes: int):
        self.dst = dst
        self.target_bytes = target_bytes
        self.buf: dict[str, torch.Tensor] = {}
        self.buf_bytes = 0
        self.shard_idx = 0
        self.weight_map: dict[str, str] = {}
        self.total_bytes = 0

    def _flush(self):
        if not self.buf:
            return
        self.shard_idx += 1
        fname = f"model-{self.shard_idx:05d}.safetensors"
        save_file(self.buf, os.path.join(self.dst, fname), metadata={"format": "pt"})
        for name in self.buf:
            self.weight_map[name] = fname
        self.buf = {}
        self.buf_bytes = 0

    def add(self, name: str, tensor: torch.Tensor):
        """Buffer ``tensor`` under ``name``, flushing to a new shard when the size limit is hit."""
        tensor = tensor.contiguous()
        nbytes = tensor.numel() * tensor.element_size()
        self.total_bytes += nbytes
        # A single oversized tensor (engram table) gets its own shard.
        if nbytes >= self.target_bytes:
            self._flush()
            self.buf[name] = tensor
            self.buf_bytes = nbytes
            self._flush()
            return
        if self.buf_bytes + nbytes > self.target_bytes:
            self._flush()
        self.buf[name] = tensor
        self.buf_bytes += nbytes

    def finalize(self):
        """Flush remaining tensors, rename shards to the -of- convention and write the index."""
        self._flush()
        # Relabel shards to the conventional -of- naming and rewrite the index.
        n = self.shard_idx
        rename = {}
        for k in range(1, n + 1):
            old = f"model-{k:05d}.safetensors"
            new = f"model-{k:05d}-of-{n:05d}.safetensors"
            os.rename(os.path.join(self.dst, old), os.path.join(self.dst, new))
            rename[old] = new
        weight_map = {name: rename[fn] for name, fn in self.weight_map.items()}
        index = {"metadata": {"total_size": self.total_bytes}, "weight_map": weight_map}
        with open(os.path.join(self.dst, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f, indent=2)


def flatten_config(src_cfg: dict) -> dict:
    """Lift text_config to the top level, drop vision, set model_type=deepseek_v41.

    Also drops ``quantization_config`` (the dense/expert weights are now bf16; the
    engram tables stay fp8 but are loaded by the engram module, not an HF quantizer)
    and zeroes ``num_nextn_predict_layers`` (the MTP weights are dropped by this cast).
    """
    cfg = dict(src_cfg)
    text = cfg.pop("text_config", {}) or {}
    cfg.pop("vision_config", None)
    cfg.pop("quantization_config", None)
    text = dict(text)
    text.pop("model_type", None)
    out = {**text, **cfg}
    out["model_type"] = "deepseek_v41"
    out["architectures"] = ["DeepseekV41ForCausalLM"]
    # MTP layers are not exported by this cast (mtp.* tensors are dropped).
    out["num_nextn_predict_layers"] = 0
    return out


def main():
    """Cast every source shard into bf16 output shards, then write config and tokenizer."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--shard-gb", type=float, default=15.0)
    ap.add_argument("--limit-files", type=int, default=0, help="process only the first N input shards (validation)")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    index_path = os.path.join(args.src, "model.safetensors.index.json")
    weight_map = json.load(open(index_path))["weight_map"]
    scaled_stems = {k[:-6] for k in weight_map if k.endswith(".scale")}

    # group source tensor names by their shard file, in file order
    files: dict[str, list[str]] = {}
    for name, fn in weight_map.items():
        files.setdefault(fn, []).append(name)
    file_list = sorted(files)
    if args.limit_files:
        file_list = file_list[: args.limit_files]

    writer = ShardWriter(args.dst, int(args.shard_gb * 1024**3))
    kept = dropped = dequant_fp8 = dequant_fp4 = passthrough = 0

    for fn in tqdm(file_list, desc="input shards"):
        with safe_open(os.path.join(args.src, fn), framework="pt", device="cpu") as f:
            for name in files[fn]:
                # engram tables: keep fp8 bits + e8m0 scales verbatim (do not upcast).
                # Must be handled BEFORE the generic .scale skip so embed.scale is kept.
                if ".engram.embed." in name:
                    writer.add(name, f.get_tensor(name))
                    kept += 1
                    continue
                if name.endswith(".scale"):
                    continue  # consumed alongside its .weight (dequantized)
                if should_drop(name):
                    dropped += 1
                    continue
                stem = name[:-7] if name.endswith(".weight") else name
                has_scale = name.endswith(".weight") and stem in scaled_stems
                if has_scale:
                    w = f.get_tensor(name)
                    s = f.get_tensor(stem + ".scale")
                    if w.dtype == torch.int8:
                        writer.add(name, dequant_fp4_experts(w, s))
                        dequant_fp4 += 1
                    elif w.dtype == torch.float8_e4m3fn:
                        writer.add(name, dequant_fp8_dense(w, s))
                        dequant_fp8 += 1
                    else:
                        raise TypeError(f"{name}: unexpected scaled dtype {w.dtype}")
                    kept += 1
                else:
                    t = f.get_tensor(name)
                    assert t.dtype not in (torch.int8, torch.float8_e4m3fn, torch.float8_e8m0fnu), (
                        f"{name}: quantized dtype {t.dtype} without a scale"
                    )
                    writer.add(name, t)
                    passthrough += 1
                    kept += 1

    writer.finalize()

    # config + tokenizer (only on a full run)
    if not args.limit_files:
        src_cfg = json.load(open(os.path.join(args.src, "config.json")))
        with open(os.path.join(args.dst, "config.json"), "w") as f:
            json.dump(flatten_config(src_cfg), f, indent=2)
        for extra in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
            p = os.path.join(args.src, extra)
            if os.path.exists(p):
                shutil.copyfile(p, os.path.join(args.dst, extra))

    print(
        f"kept={kept} dropped={dropped} dequant_fp8={dequant_fp8} "
        f"dequant_fp4={dequant_fp4} passthrough={passthrough} "
        f"shards={writer.shard_idx} total_bytes={writer.total_bytes}"
    )


if __name__ == "__main__":
    main()
