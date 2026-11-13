#!/usr/bin/env python3
"""Build a pruned DeepSeek-V4.1-Flash checkpoint in the ORIGINAL storage format.

Copies every kept tensor verbatim (dense FP8 + FP4 experts stay as released), keeps only the
experts a routing mask selects (remapped to ids 0..K-1, gate rows sliced the same way), and
optionally drops the engram tables. Vision / MTP / VL routing bias are dropped.

The mask is a bool .npy [n_layers, n_experts] with the same number of True per layer, e.g. the
64 most-routed experts per layer measured on the training prompts.

  python coda/custom/dsv41_4090/tools/build_pruned.py --src /root/DeepSeek-V4.1-Flash \
      --dst /root/dsv41-k64 --mask mask_top64.npy --drop-engram-tables --expert-fp8
"""
import argparse
import json
import os
import shutil
import struct
from collections import defaultdict

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

FP4_TABLE = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                          -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def fp4_to_fp8_block32(w_int8: torch.Tensor, scale_e8m0: torch.Tensor):
    """[N, K/2] packed e2m1 + [N, K/32] e8m0  ->  fp8 [N, K] + f32 scale [N/32, K/32]."""
    t = FP4_TABLE.to(w_int8.device)
    x = w_int8.view(torch.uint8)
    vals = torch.stack([t[(x & 0x0F).long()], t[(x >> 4).long()]], dim=-1).flatten(1)
    N, K = vals.shape
    w = vals.view(N, K // 32, 32) * scale_e8m0.float().view(N, K // 32, 1)
    w = w.view(N // 32, 32, K // 32, 32)
    amax = w.abs().amax(dim=(1, 3)).clamp_min(1e-30)
    s = (amax / 448.0).float()
    q = (w / s.view(N // 32, 1, K // 32, 1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.reshape(N, K), s


DROP_PREFIX = ("vision.", "aligner.", "mtp.", "image_")


def read_files(src):
    """name -> (file, dtype, shape, start, end) from the safetensors headers."""
    meta = {}
    for fn in sorted(os.listdir(src)):
        if not fn.endswith(".safetensors"):
            continue
        path = os.path.join(src, fn)
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            hdr = json.loads(f.read(n))
        base = 8 + n
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            s, e = v["data_offsets"]
            meta[k] = (path, v["dtype"], tuple(v["shape"]), base + s, base + e)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--mask", required=True, help="bool [n_layers, n_experts]; True = keep")
    ap.add_argument("--no-engram", action="store_true")
    ap.add_argument("--drop-engram-tables", action="store_true",
                    help="keep engram wkv/q/k weights but drop the giant embed tables")
    ap.add_argument("--expert-fp8", action="store_true",
                    help="dequantize FP4 experts to 32x32 block FP8 (SM89 has no MXFP4 kernel)")
    ap.add_argument("--shard-gb", type=float, default=15.0)
    args = ap.parse_args()

    src_cfg = json.load(open(os.path.join(args.src, "config.json")))
    text = src_cfg["text_config"]
    n_layers, n_experts = text["num_hidden_layers"], text["n_routed_experts"]
    mask = np.load(args.mask).astype(bool)
    assert mask.shape == (n_layers, n_experts), (mask.shape, n_layers, n_experts)
    keep_ids = [np.flatnonzero(mask[l]).tolist() for l in range(n_layers)]
    n_keep = {len(x) for x in keep_ids}
    assert len(n_keep) == 1, f"mask keeps different expert counts per layer: {n_keep}"
    n_keep = n_keep.pop()
    remap = [{old: new for new, old in enumerate(ids)} for ids in keep_ids]

    meta = read_files(args.src)
    dropped = skipped = 0
    plan = []  # (src_name, dst_name)
    for name in meta:
        if name.startswith(DROP_PREFIX) or name.endswith((".bias_vl",)):
            dropped += 1
            continue
        if ".engram." in name and (
            args.no_engram or (args.drop_engram_tables and ".engram.embed." in name)
        ):
            dropped += 1
            continue
        dst = name
        parts = name.split(".")
        if parts[0] == "layers" and len(parts) > 4 and parts[2] == "ffn" and parts[3] == "experts":
            layer, old = int(parts[1]), int(parts[4])
            new = remap[layer].get(old)
            if new is None:
                skipped += 1
                continue
            parts[4] = str(new)
            dst = ".".join(parts)
        plan.append((name, dst))

    os.makedirs(args.dst, exist_ok=True)
    files = defaultdict(list)
    for name, dst in plan:
        files[meta[name][0]].append((name, dst))

    buf, buf_bytes, shard, total = {}, 0, 0, 0
    limit = int(args.shard_gb * 1024**3)
    index = {}
    shards = []

    def flush():
        nonlocal buf, buf_bytes, shard
        if not buf:
            return
        shard += 1
        fn = f"model-{shard:05d}.safetensors"
        save_file(buf, os.path.join(args.dst, fn), metadata={"format": "pt"})
        for k in buf:
            index[k] = fn
        shards.append(fn)
        buf, buf_bytes = {}, 0

    for src_file in sorted(files):
        with safe_open(src_file, framework="pt", device="cpu") as f:
            for name, dst in files[src_file]:
                if args.expert_fp8 and ".ffn.experts." in name and name.endswith(".weight"):
                    scale_name = name[: -len(".weight")] + ".scale"
                    q, s = fp4_to_fp8_block32(f.get_tensor(name).cuda(), f.get_tensor(scale_name).cuda())
                    for tname, t in ((dst, q), (dst[: -len(".weight")] + ".scale", s)):
                        t = t.contiguous().cpu()
                        nb = t.numel() * t.element_size()
                        if buf_bytes + nb > limit:
                            flush()
                        buf[tname] = t
                        buf_bytes += nb
                        total += nb
                    continue
                if args.expert_fp8 and ".ffn.experts." in name and name.endswith(".scale"):
                    continue
                if name.endswith(".ffn.gate.weight") or name.endswith(".ffn.gate.bias"):
                    layer = int(name.split(".")[1])
                    t = f.get_tensor(name)[keep_ids[layer]]
                else:
                    t = f.get_tensor(name)
                t = t.contiguous()
                nb = t.numel() * t.element_size()
                if buf_bytes + nb > limit:
                    flush()
                buf[dst] = t
                buf_bytes += nb
                total += nb
    flush()

    cfg = dict(src_cfg)
    t = dict(text)
    t.update(
        n_routed_experts=n_keep,
        num_experts_per_tok=min(text["num_experts_per_tok"], n_keep),
        num_nextn_predict_layers=0,
        dspark_block_size=0,
        dspark_target_layer_ids=[],
    )
    if args.no_engram:
        t.update(engram_layer_ids=[], engram_num_embeddings=[])
    cfg.pop("vision_config", None)
    cfg["text_config"] = t
    cfg["pruned_from"] = os.path.abspath(args.src)
    cfg["pruned_experts"] = n_keep
    with open(os.path.join(args.dst, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    with open(os.path.join(args.dst, "expert_map.json"), "w") as f:
        json.dump({"old_ids": keep_ids}, f)
    with open(os.path.join(args.dst, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": index}, f, indent=2)
    for extra in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        p = os.path.join(args.src, extra)
        if os.path.exists(p):
            shutil.copyfile(p, os.path.join(args.dst, extra))
    print(f"experts/layer={n_keep} shards={len(shards)} bytes={total / 1e9:.2f}GB "
          f"dropped={dropped} pruned_expert_tensors={skipped}")


if __name__ == "__main__":
    main()
