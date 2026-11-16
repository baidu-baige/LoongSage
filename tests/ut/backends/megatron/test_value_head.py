"""Unit tests for coda/backends/megatron/value_head.py (CPU, sequence parallel off)."""
import pytest
import torch

pytest.importorskip("megatron", reason="Megatron-Core is not installed")

from coda.backends.megatron.value_head import LinearValueHead


def make_head(hidden=64, std=0.02, seed=7, sp=False):
    return LinearValueHead(hidden, sequence_parallel=sp, init_std=std, seed=seed)


def test_forward_shape_dtype_and_signature():
    head = make_head().to(torch.bfloat16)
    hidden = torch.randn(5, 1, 64, dtype=torch.bfloat16)  # [s, b, H]
    values, bias = head(hidden, weight=torch.randn(100, 64), runtime_gather_output=True)
    assert values.shape == (5, 1, 1)
    assert values.dtype == torch.float32
    assert bias is None


def test_ignores_the_shared_embedding_weight():
    head = make_head()
    hidden = torch.randn(3, 1, 64)
    a, _ = head(hidden)
    b, _ = head(hidden, weight=torch.randn(10, 64))
    assert torch.equal(a, b)


def test_init_normal_with_zero_bias():
    head = make_head(hidden=4096, std=0.02)
    assert head.bias.item() == 0.0
    assert head.weight.std().item() == pytest.approx(0.02, rel=0.1)
    assert head.weight.mean().abs().item() < 0.002


def test_init_is_identical_for_the_same_seed():
    """Ranks holding a replica must start from the same head (DDP does not broadcast)."""
    torch.manual_seed(0)
    a = make_head(seed=3)
    torch.manual_seed(123)  # global RNG state must not matter
    b = make_head(seed=3)
    c = make_head(seed=4)
    assert torch.equal(a.weight, b.weight)
    assert not torch.equal(a.weight, c.weight)


def test_sequence_parallel_tags_params():
    head = make_head(sp=True)
    assert head.weight.sequence_parallel is True
    assert head.bias.sequence_parallel is True
    assert not hasattr(make_head().weight, "sequence_parallel")
