import json
import os

import pytest
import torch

from coda.utils.quant_formats import (
    FP4_MAX,
    FP8_MAX,
    QuantScheme,
    dequant_fp8_block,
    dequant_mxfp4,
    element_thresholds,
    quant_fp8_block,
    quant_mxfp4,
    scan_ckpt_quant,
    tensor_seed,
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
    scheme = scan_ckpt_quant(ORIG_CKPT)[0][name + ".weight"].scheme
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


@pytest.mark.parametrize("scheme", [FP8, FP4, QuantScheme("fp8_block", (1, 32))])
def test_bridge_requant_of_snapped_is_exact(scheme):
    """The bridge's export quantizer (ties toward smaller magnitude, amax floor 2^-127 instead of
    AMAX_FLOOR) reproduces snapped weights exactly, so the rollout gets the training weights."""
    qu = pytest.importorskip("megatron.bridge.models.conversion.quantization_utils")
    w = _rand(128, 256)
    w[32:64, 64:128] *= 1e-6  # blocks below AMAX_FLOOR
    snapped = scheme.snap_(w.clone())
    _, s = scheme.quant(snapped)
    like = torch.empty(s.shape, dtype=s.dtype, device="meta")
    if scheme.kind == "mxfp4":
        bq, bs = qu.quantize_mxfp4_e2m1_like_scale(snapped, like, block_size=scheme.block[1])
    else:
        bq, bs = qu.quantize_fp8_e4m3fn_like_scale(snapped, like, block_size=scheme.block[0])
    torch.testing.assert_close(scheme.dequant(bq, bs, torch.float32), snapped.float(), rtol=0, atol=0)


# ---------------------------------------------------------------- threshold rounding

def _whole(w, seed=7):
    """thresholds(r0, r1, device) of a whole [O, I] tensor."""
    I = w.shape[1]
    return lambda r0, r1, device: element_thresholds(seed, torch.arange(r0, r1, device=device),
                                                     torch.arange(I, device=device), I)


SCHEMES = [FP8, FP4, QuantScheme("fp8_block", (1, 32)), QuantScheme("mxfp4", (1, 16))]


@pytest.mark.parametrize("scheme", SCHEMES)
def test_threshold_snap_keeps_grid_and_is_exportable(scheme):
    qu = pytest.importorskip("megatron.bridge.models.conversion.quantization_utils")
    w = _rand(128, 256)
    on_grid = scheme.snap_(w.clone())
    # initial (on-grid) weights are reproduced exactly
    torch.testing.assert_close(scheme.snap_(on_grid.clone(), _whole(w)), on_grid, rtol=0, atol=0)
    # off-grid input: output is on the grid (fixed point of round-to-nearest) and the bridge's
    # export quantizer reproduces it exactly
    snapped = scheme.snap_(w.clone(), _whole(w))
    torch.testing.assert_close(scheme.snap_(snapped.clone()), snapped, rtol=0, atol=0)
    _, s = scheme.quant(snapped)
    like = torch.empty(s.shape, dtype=s.dtype, device="meta")
    if scheme.kind == "mxfp4":
        bq, bs = qu.quantize_mxfp4_e2m1_like_scale(snapped, like, block_size=scheme.block[1])
    else:
        bq, bs = qu.quantize_fp8_e4m3fn_like_scale(snapped, like, block_size=scheme.block[0])
    torch.testing.assert_close(scheme.dequant(bq, bs, torch.float32), snapped.float(), rtol=0, atol=0)


def _on_grid_scale1(scheme, o=256, i=512, seed=1):
    """Random on-grid weights whose blocks all have scale 1 (one element per block at the format max)."""
    g = torch.Generator().manual_seed(seed)
    if scheme.kind == "mxfp4":
        grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
        w = grid[torch.randint(0, 8, (o, i), generator=g)]
    else:
        w = (torch.randn(o, i, generator=g) * 64).clamp(-440, 440).to(torch.float8_e4m3fn).float()
    w = w * torch.where(torch.rand(o, i, generator=g) < 0.5, -1.0, 1.0)
    br, bc = scheme.block
    w.view(o // br, br, i // bc, bc)[:, 0, :, 0] = FP4_MAX if scheme.kind == "mxfp4" else FP8_MAX
    return w


@pytest.mark.parametrize("scheme", [FP8, FP4])
def test_threshold_snap_follows_small_drift_on_average(scheme):
    """On-grid weights plus a drift far below half a grid step: round-to-nearest keeps every
    weight, threshold rounding moves them by the drift on average."""
    w0 = _on_grid_scale1(scheme)
    assert torch.equal(scheme.snap_(w0.clone()), w0)
    sign = torch.where(torch.arange(w0.shape[1]) % 2 == 0, 1.0, -1.0)
    # drift vs. half a grid step: e2m1 >= 0.25 near 0 and >= 1/8 of |w| above; e4m3 >= 1/32 of |w|
    rel, floor = (0.1, 0.5) if scheme.kind == "mxfp4" else (0.01, 0.0)
    delta = torch.where(w0.abs() == w0.abs().max(), 0.0, rel * w0.abs().clamp_min(floor) * sign)
    w = w0 + delta
    assert torch.equal(scheme.snap_(w.clone()), w0)
    moved = scheme.snap_(w.clone(), _whole(w)) - w0
    assert 0.01 < (moved != 0).float().mean() < 0.5
    ratio = (moved * delta.sign()).sum() / delta.abs().sum()
    assert abs(ratio - 1) < 0.05, ratio


def test_threshold_rounding_matches_nearest_at_one_half():
    from coda.utils.quant_formats import _round_e2m1, _round_e4m3_threshold

    y = torch.randn(1 << 16) * torch.exp2(torch.randint(-12, 9, (1 << 16,)).float())
    y = y.clamp(-448, 448)
    half = torch.full_like(y, 0.5)  # differs from round-to-nearest-even only on exact ties
    torch.testing.assert_close(_round_e4m3_threshold(y, half), y.to(torch.float8_e4m3fn).float(), rtol=0, atol=0)
    x = torch.rand(1 << 16) * 12 - 6
    torch.testing.assert_close(_round_e2m1(x, half), _round_e2m1(x), rtol=0, atol=0)
    # extremes: t -> 0 rounds every off-grid magnitude up, t -> 1 truncates toward zero
    assert torch.equal(_round_e2m1(torch.tensor([0.1, -2.2, 4.1]), torch.full((3,), 1e-6)), torch.tensor([0.5, -3.0, 6.0]))
    assert torch.equal(_round_e2m1(torch.tensor([0.4, -3.9, 5.9]), torch.full((3,), 1 - 1e-6)), torch.tensor([0.0, -3.0, 4.0]))


def test_element_thresholds():
    rows, cols = torch.arange(1000), torch.arange(300)
    t = element_thresholds(123, rows, cols, 300)
    assert t.dtype == torch.float32 and t.shape == (1000, 300)
    assert t.min() > 0 and t.max() < 1 and abs(t.mean() - 0.5) < 0.01
    assert abs((t < 0.1).float().mean() - 0.1) < 0.01
    # keyed by (seed, row * ncols + col): sub-blocks agree, other seeds differ
    torch.testing.assert_close(element_thresholds(123, rows[10:20], cols[100:], 300), t[10:20, 100:], rtol=0, atol=0)
    assert (element_thresholds(124, rows, cols, 300) != t).float().mean() > 0.99
    assert tensor_seed("layers.0.ffn.experts.0.w1.weight") == tensor_seed("layers.0.ffn.experts.0.w1.weight")
    if torch.cuda.is_available():
        got = element_thresholds(123, rows.cuda(), cols.cuda(), 300).cpu()
        torch.testing.assert_close(got, t, rtol=0, atol=0)


def test_scan_ckpt_quant(tmp_path):
    from safetensors.torch import save_file

    w8, s8 = quant_fp8_block(_rand(64, 96))
    w4, s4 = quant_mxfp4(_rand(32, 64))
    we, se = quant_fp8_block(_rand(8, 64), (1, 32))
    save_file({"a.weight": w8, "a.scale": s8, "e.weight": w4, "e.scale": s4, "t.weight": we, "t.scale": se,
               "n.weight": torch.zeros(4, 4).bfloat16()}, str(tmp_path / "model.safetensors"))
    got, unsupported = scan_ckpt_quant(str(tmp_path))
    assert set(got) == {"a.weight", "e.weight", "t.weight"} and not unsupported
    assert got["a.weight"].scheme == FP8 and got["a.weight"].shape == (64, 96) and got["a.weight"].scale_name == "a.scale"
    assert got["e.weight"].scheme == FP4 and got["e.weight"].shape == (32, 64)
    assert got["t.weight"].scheme == QuantScheme("fp8_block", (1, 32))


def test_scan_ckpt_quant_unsupported(tmp_path):
    from safetensors.torch import save_file

    w8, s8 = quant_fp8_block(_rand(64, 96))
    save_file({"a.weight": w8, "a.scale": s8,
               "m.weight": w8.clone(), "m.scale": torch.ones(1, 1)},  # fp32 scale (128x128-block FP8)
              str(tmp_path / "model.safetensors"))
    got, unsupported = scan_ckpt_quant(str(tmp_path))
    assert set(got) == {"a.weight"} and set(unsupported) == {"m.weight"}
    u = unsupported["m.weight"]
    assert (u.scale_name, u.dtype, u.shape) == ("m.scale", "F8_E4M3", (64, 96))
    assert u.reason.startswith("unsupported quantized layout")
