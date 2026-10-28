import json
import os

import pytest
import torch

from coda.utils.quant_formats import (
    QuantScheme,
    dequant_fp8_block,
    dequant_mxfp4,
    quant_fp8_block,
    quant_mxfp4,
    read_ckpt_quant,
)

FP8 = QuantScheme("fp8_block", (32, 32))
FP4 = QuantScheme("mxfp4", (1, 32))

ORIG_CKPT = "/mnt/cfs-ainf/workspace/coda-data/models/DeepSeek-V4.1-Flash-1017"
BF16_CKPT = ORIG_CKPT + "-bf16"


def _rand(o, i, seed=0):
    g = torch.Generator().manual_seed(seed)
    # Spread magnitudes over many binades so blocks get different scales.
    w = torch.randn(o, i, generator=g) * torch.exp2(torch.randint(-12, 2, (o, 1), generator=g).float())
    w[:32, :32] = 0.0  # an all-zero block
    return w.bfloat16()


@pytest.mark.parametrize("fmt", ["fp8", "fp4"])
def test_snap_is_fixed_point_and_bf16_exact(fmt):
    w = _rand(128, 256)
    scheme = {"fp8": FP8, "fp4": FP4}[fmt]
    quant, dequant, snap = scheme.quant, scheme.dequant, scheme.snap_
    snapped = snap(w.clone())
    # dequantized values are exactly representable in bf16
    torch.testing.assert_close(snapped.float(), dequant(*quant(w), dtype=torch.float32), rtol=0, atol=0)
    # snapping again is a no-op, and re-quantizing reproduces the same values
    torch.testing.assert_close(snap(snapped.clone()), snapped, rtol=0, atol=0)
    torch.testing.assert_close(dequant(*quant(snapped)), snapped, rtol=0, atol=0)
    assert not torch.equal(snapped, w)  # the quantization actually did something


def test_fp8_shapes_and_range():
    w = _rand(64, 96)
    q, s = quant_fp8_block(w)
    assert q.dtype == torch.float8_e4m3fn and q.shape == (64, 96)
    assert s.dtype == torch.float8_e8m0fnu and s.shape == (2, 3)
    assert q.float().abs().max() <= 448


def test_fp4_packing_low_nibble_first():
    # One row, one group: values chosen so that scale = 1 (amax = 6).
    vals = torch.tensor([[6.0, -0.5, 1.5, -3.0] + [0.0] * 28])
    q, s = quant_mxfp4(vals)
    assert q.shape == (1, 16) and s.shape == (1, 1)
    b = q.view(torch.uint8)
    assert b[0, 0].item() == (7 | (9 << 4))  # 6.0 -> 0x7 (low), -0.5 -> 0x9 (high)
    assert b[0, 1].item() == (3 | (13 << 4))  # 1.5 -> 0x3, -3.0 -> 0xD
    torch.testing.assert_close(dequant_mxfp4(q, s, torch.float32), vals, rtol=0, atol=0)


def test_fp4_round_half_even():
    # scale = 1 (amax 6); ties: 0.25->0, 0.75->1, 1.25->1, 1.75->2, 2.5->2, 3.5->4, 5->4
    x = torch.tensor([[6.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0] + [0.0] * 24])
    got = dequant_mxfp4(*quant_mxfp4(x), dtype=torch.float32)[0, :8]
    assert got.tolist() == [6.0, 0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0]


@pytest.mark.skipif(not (os.path.isdir(ORIG_CKPT) and os.path.isdir(BF16_CKPT)), reason="needs DSV4.1 ckpts")
@pytest.mark.parametrize(
    "name",
    ["layers.3.attn.wq_b", "layers.3.attn.wo_a", "layers.2.attn.indexer.wq_b",
     "layers.3.ffn.shared_experts.w1", "layers.3.ffn.experts.7.w1", "layers.3.ffn.experts.7.w2"],
)
def test_matches_original_checkpoint(name):
    """Quantizing the bf16 ckpt reproduces the ORIGINAL ckpt's dequantized values exactly."""
    from safetensors import safe_open

    def get(root, key):
        idx = json.load(open(f"{root}/model.safetensors.index.json"))["weight_map"]
        with safe_open(f"{root}/{idx[key]}", "pt") as f:
            return f.get_tensor(key)

    w_orig, s_orig = get(ORIG_CKPT, name + ".weight"), get(ORIG_CKPT, name + ".scale")
    w_bf16 = get(BF16_CKPT, name + ".weight")
    fp4 = w_orig.dtype == torch.int8
    scheme = read_ckpt_quant(ORIG_CKPT)[name + ".weight"].scheme
    assert scheme == (FP4 if fp4 else FP8)
    ref = scheme.dequant(w_orig, s_orig, torch.float32)
    torch.testing.assert_close(scheme.dequant(*scheme.quant(w_bf16), dtype=torch.float32), ref, rtol=0, atol=0)
    torch.testing.assert_close(w_bf16.float(), ref, rtol=0, atol=0)


@pytest.mark.parametrize("scheme", [QuantScheme("fp8_block", (1, 32)), QuantScheme("fp8_block", (16, 64)),
                                    QuantScheme("mxfp4", (1, 16))])
def test_other_tiles_fixed_point(scheme):
    w = _rand(64, 128)
    q, s = scheme.quant(w)
    br, bc = scheme.block
    assert s.shape == (64 // br, 128 // bc)
    snapped = scheme.snap_(w.clone())
    torch.testing.assert_close(scheme.snap_(snapped.clone()), snapped, rtol=0, atol=0)


def test_read_ckpt_quant(tmp_path):
    from safetensors.torch import save_file

    w8, s8 = quant_fp8_block(_rand(64, 96))
    w4, s4 = quant_mxfp4(_rand(32, 64))
    we, se = quant_fp8_block(_rand(8, 64), (1, 32))
    save_file({"a.weight": w8, "a.scale": s8, "e.weight": w4, "e.scale": s4, "t.weight": we, "t.scale": se,
               "n.weight": torch.zeros(4, 4).bfloat16()}, str(tmp_path / "model.safetensors"))
    got = read_ckpt_quant(str(tmp_path))
    assert set(got) == {"a.weight", "e.weight", "t.weight"}
    assert got["a.weight"].scheme == FP8 and got["a.weight"].shape == (64, 96) and got["a.weight"].scale_name == "a.scale"
    assert got["e.weight"].scheme == FP4 and got["e.weight"].shape == (32, 64)
    assert got["t.weight"].scheme == QuantScheme("fp8_block", (1, 32))
