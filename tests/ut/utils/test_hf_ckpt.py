import json
import os

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from coda.utils.hf_ckpt import save_like_source
from coda.utils.quant_formats import QuantScheme

FP8 = QuantScheme("fp8_block", (32, 32))


def _src(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    q, s = FP8.quant(torch.randn(64, 64).bfloat16())
    a = {"layers.0.attn.wq.weight": q, "layers.0.attn.wq.scale": s, "layers.0.norm.weight": torch.ones(64).bfloat16()}
    b = {"vision.w": torch.randn(8, 8).bfloat16(), "layers.0.engram.embed.weight": torch.randint(0, 9, (4, 32)).to(torch.uint8),
         "layers.0.hc": torch.randn(3)}
    save_file(a, str(src / "model-00001-of-00002.safetensors"), metadata={"format": "pt"})
    save_file(b, str(src / "model-00002-of-00002.safetensors"), metadata={"format": "pt"})
    weight_map = {**{k: "model-00001-of-00002.safetensors" for k in a}, **{k: "model-00002-of-00002.safetensors" for k in b}}
    (src / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": 1}, "weight_map": weight_map}))
    (src / "config.json").write_text('{"quantization_config": {"quant_method": "fp8"}, "text_config": {}}')
    (src / "encoding").mkdir()
    (src / "encoding" / "x.py").write_text("x = 1\n")
    return src, a, b


def _load(path):
    out = {}
    for f in sorted(os.listdir(path)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(path, f), "pt") as h:
                out.update({k: h.get_tensor(k) for k in h.keys()})
    return out


def test_save_like_source_roundtrip(tmp_path):
    src, a, b = _src(tmp_path)
    out = tmp_path / "out"
    new_w = torch.randn(64, 64).bfloat16()
    q, s = FP8.quant(new_w)
    trained = [("layers.0.norm.weight", torch.full((64,), 2.0)),  # fp32 -> cast to source bf16
               ("layers.0.attn.wq.weight", q), ("layers.0.attn.wq.scale", s), ("layers.0.hc", torch.zeros(3))]
    save_like_source(iter(trained), str(src), str(out), {n for n, _ in trained})
    got, ref = _load(out), _load(src)
    assert {k: (v.dtype, v.shape) for k, v in got.items()} == {k: (v.dtype, v.shape) for k, v in ref.items()}
    assert torch.equal(got["layers.0.attn.wq.weight"].view(torch.uint8), q.view(torch.uint8))
    assert torch.equal(got["layers.0.norm.weight"], torch.full((64,), 2.0).bfloat16())
    assert torch.equal(got["layers.0.hc"], torch.zeros(3))
    for k in b:  # untrained: byte copy
        if k != "layers.0.hc":
            assert torch.equal(got[k], ref[k])
    for f in ("config.json", "model.safetensors.index.json", "encoding/x.py"):
        assert (out / f).read_bytes() == (src / f).read_bytes()


def test_save_like_source_rejects_mismatch(tmp_path):
    src, _, _ = _src(tmp_path)
    with pytest.raises(KeyError, match="not in"):
        save_like_source(iter([]), str(src), str(tmp_path / "o1"), {"layers.9.x"})
    with pytest.raises(ValueError, match="shape"):
        save_like_source(iter([("layers.0.norm.weight", torch.ones(32))]), str(src), str(tmp_path / "o2"),
                         {"layers.0.norm.weight"})
    with pytest.raises(ValueError, match="dtype"):  # quantized tensor not encoded
        save_like_source(iter([("layers.0.attn.wq.weight", torch.ones(64, 64))]), str(src), str(tmp_path / "o3"),
                         {"layers.0.attn.wq.weight"})
    with pytest.raises(RuntimeError, match="not produced"):
        save_like_source(iter([]), str(src), str(tmp_path / "o4"), {"layers.0.norm.weight"})
