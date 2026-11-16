"""Tree-mode helpers of coda/backends/megatron/model.py (CPU, no process group)."""

import pytest

pytest.importorskip("megatron", reason="Megatron-Core is not installed")

from coda.backends.megatron.model import _TreeChunkCache  # noqa: E402


def test_tree_chunk_cache_builds_once_per_micro_batch():
    """VPP: every model chunk asks for the same micro-batch; the tree is built once, the
    first caller is flagged (stats), and the entry is dropped after the last chunk."""
    built = []
    cache = _TreeChunkCache(n_chunks=2)
    order = [1, 2, 1, 3, 2, 3]            # interleaved schedule: chunks reach mbs out of order
    out = [cache.get(k, lambda k=k: built.append(k) or f"tmb{k}") for k in order]
    assert built == [1, 2, 3]
    assert [f for _, f in out] == [True, True, False, True, False, False]
    assert [t for t, _ in out] == ["tmb1", "tmb2", "tmb1", "tmb3", "tmb2", "tmb3"]
    assert cache._entries == {}
    single = _TreeChunkCache(n_chunks=1)
    assert single.get(5, lambda: "a") == ("a", True) and single._entries == {}
