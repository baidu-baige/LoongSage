#!/usr/bin/env python3
"""Build the serving checkpoint for ``MegatronStreamActorWorker``.

Copies every tensor except the routed experts (dense, router, engram projections) of a
DeepSeek-V4.1-Flash checkpoint into one shard. SGLang serves from this directory (``hf_model_path``)
and reads the routed experts from the base checkpoint (``SGLANG_FILE_EXPERTS_CKPT``); the trainer
merges its LoRA adapters into the dense tail weights here in place.

  python coda/custom/dsv41_4090/tools/build_serve_ckpt.py --src /path/to/dsv41-k64 --dst /path/to/dsv41-k64-serve
"""
import argparse
import json
import os
import shutil

from safetensors import safe_open
from safetensors.torch import save_file

EXTRA_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json",
               "expert_map.json")


def build_serving_checkpoint(base_ckpt, dst):
    os.makedirs(dst, exist_ok=True)
    idx = json.load(open(os.path.join(base_ckpt, "model.safetensors.index.json")))["weight_map"]
    keep = {}
    for name, fn in idx.items():
        if ".ffn.experts." not in name:
            keep.setdefault(fn, []).append(name)
    out, total = {}, 0
    for fn, names in sorted(keep.items()):
        with safe_open(os.path.join(base_ckpt, fn), framework="pt") as f:
            for n in names:
                out[n] = f.get_tensor(n)
                total += out[n].numel() * out[n].element_size()
    shard = "model-00001-of-00001.safetensors"
    save_file(out, os.path.join(dst, shard), metadata={"format": "pt"})
    json.dump({"metadata": {"total_size": total}, "weight_map": {k: shard for k in out}},
              open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=2)
    for extra in EXTRA_FILES:
        p = os.path.join(base_ckpt, extra)
        if os.path.exists(p):
            shutil.copyfile(p, os.path.join(dst, extra))
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="base checkpoint (routed experts stay here)")
    ap.add_argument("--dst", required=True, help="serving checkpoint directory to create")
    a = ap.parse_args()
    print("bytes", build_serving_checkpoint(a.src, a.dst))


if __name__ == "__main__":
    main()
