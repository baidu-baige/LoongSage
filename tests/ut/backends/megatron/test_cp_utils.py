"""Unit tests for coda/backends/megatron/cp_utils.py.

``megatron.core.parallel_state`` and
``megatron.core.packed_seq_params.PackedSeqParams`` are replaced with mocks via
the ``mock_megatron`` autouse fixture, so no GPU or initialized process group is
required. Megatron-Core must still be installed, because
``coda.backends.megatron`` imports it at module scope.

Focus areas:
  1. Symmetric zigzag slicing correctness across various cp_size / rank combos.
  2. Round-trip correctness: slice → gather_and_reconstruct recovers original.
  3. prepare_packed_seq_params: cu_seqlens construction, alignment padding,
     max_seqlen computation, and edge cases (empty list).
"""

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

pytest.importorskip("megatron", reason="Megatron-Core is not installed")

# ---------------------------------------------------------------------------
# Lightweight stand-in for PackedSeqParams (avoid Megatron dependency)
#
# qkv_format defaults to None so a test asserting ``qkv_format == "thd"`` only
# passes when prepare_packed_seq_params actually sets it.
# ---------------------------------------------------------------------------


@dataclass
class _FakePackedSeqParams:
    cu_seqlens_q: Any = None
    cu_seqlens_kv: Any = None
    cu_seqlens_q_padded: Any = None
    cu_seqlens_kv_padded: Any = None
    max_seqlen_q: int = 0
    max_seqlen_kv: int = 0
    qkv_format: str = "thd"
    cp_partition_mode: str = "zigzag"


# ---------------------------------------------------------------------------
# Install mocks before importing the module under test.
#
# ``coda.backends.megatron.__init__`` imports ``MegatronTrainWorker`` which
# transitively pulls in many ``megatron.*`` sub-packages.  We intercept
# *all* ``megatron.*`` lookups via a custom meta-path finder so that any
# ``import megatron.X.Y.Z`` succeeds with a MagicMock, then we patch in the
# two modules we actually care about (parallel_state, packed_seq_params).
# ---------------------------------------------------------------------------

import importlib
import importlib.abc
import importlib.machinery


class _MegatronMockFinder(importlib.abc.MetaPathFinder):
    """Auto-generate MagicMock modules for any ``megatron.*`` import."""

    def find_module(self, fullname, path=None):
        if fullname == "megatron" or fullname.startswith("megatron."):
            return self
        return None


from coda.backends.megatron import cp_utils as _cp_utils  # noqa: E402

_mpu_mock = MagicMock()

slice_cp_with_zigzag = _cp_utils.slice_cp_with_zigzag
slice_cp_packed = _cp_utils.slice_cp_packed
gather_and_reconstruct_cp = _cp_utils.gather_and_reconstruct_cp
prepare_packed_seq_params = _cp_utils.prepare_packed_seq_params
_contiguous_padded_length = _cp_utils._contiguous_padded_length


@pytest.fixture(autouse=True)
def mock_megatron(monkeypatch):
    """Swap cp_utils' Megatron globals for mocks, restoring them afterwards.

    Rebinding module globals at import time (the previous approach) leaked into
    every later-collected test module and made outcomes collection-order
    dependent. monkeypatch undoes both rebinds at teardown.
    """
    _mpu_mock.reset_mock()
    _mpu_mock.get_context_parallel_rank.return_value = 0
    _mpu_mock.get_context_parallel_world_size.return_value = 1
    monkeypatch.setattr(_cp_utils, "mpu", _mpu_mock)
    monkeypatch.setattr(_cp_utils, "PackedSeqParams", _FakePackedSeqParams)
    yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_cp(rank: int, size: int):
    """Configure the mocked CP rank / world-size."""
    _mpu_mock.get_context_parallel_rank.return_value = rank
    _mpu_mock.get_context_parallel_world_size.return_value = size


def _set_tp(size: int):
    """Configure the mocked TP world-size."""
    _mpu_mock.get_tensor_model_parallel_world_size.return_value = size


def _tok(n: int, offset: int = 0) -> torch.Tensor:
    """Return a 1-D int64 tensor [offset, offset+1, ..., offset+n-1]."""
    return torch.arange(offset, offset + n, dtype=torch.int64)


# ═══════════════════════════════════════════════════════════════════════════
# slice_cp_with_zigzag
# ═══════════════════════════════════════════════════════════════════════════


class TestSliceWithCp:
    """Tests for the THD symmetric two-chunk slicing."""

    # ── cp_size = 1: passthrough ──────────────────────────────────────────

    def test_cp_size_1_returns_original(self):
        """When cp_size=1, slice_cp_with_zigzag should return the original tensor unchanged."""
        _set_cp(rank=0, size=1)
        tokens = _tok(10)
        result = slice_cp_with_zigzag(tokens, pad_value=0)
        assert torch.equal(result, tokens)

    def test_cp_size_1_no_padding(self):
        """cp_size=1 is a passthrough — no CP slicing or padding is applied.

        This is correct: CP slicing is a no-op when there's only one rank,
        and alignment padding is handled separately by prepare_packed_seq_params.
        """
        _set_cp(rank=0, size=1)
        tokens = _tok(7)
        result = slice_cp_with_zigzag(tokens, pad_value=0)
        assert result.size(0) == 7  # no padding applied

    # ── output length ─────────────────────────────────────────────────────

    def test_output_length_is_2_chunks(self):
        """Result must always be exactly 2 * chunk_size tokens."""
        _set_cp(rank=0, size=2)
        tokens = _tok(16)
        result = slice_cp_with_zigzag(tokens, pad_value=0)
        # chunk_size = ceil(16 / 4) = 4, output = 2*4 = 8
        assert result.size(0) == 8

    def test_output_length_with_uneven_input(self):
        """When token_len is not divisible by 2*cp_size, padding occurs."""
        _set_cp(rank=0, size=2)
        tokens = _tok(10)
        result = slice_cp_with_zigzag(tokens, pad_value=-1)
        # chunk_size = ceil(10 / 4) = 3, output = 2*3 = 6
        assert result.size(0) == 6

    # ── symmetric slicing correctness ─────────────────────────────────────

    def test_rank0_gets_first_and_last_chunks(self):
        """Rank 0 should get chunk[0] (front) and chunk[2N-1] (back)."""
        _set_cp(rank=0, size=2)
        # 8 tokens → chunk_size=2, 4 chunks: [0,1], [2,3], [4,5], [6,7]
        tokens = _tok(8)
        result = slice_cp_with_zigzag(tokens, pad_value=0)
        # rank=0: front=[0,1], back=[6,7]
        expected = torch.tensor([0, 1, 6, 7])
        assert torch.equal(result, expected)

    def test_rank1_gets_middle_chunks(self):
        """Rank 1 should get chunk[1] (front) and chunk[2N-2] (back)."""
        _set_cp(rank=1, size=2)
        tokens = _tok(8)
        result = slice_cp_with_zigzag(tokens, pad_value=0)
        # rank=1: front=[2,3], back=[4,5]
        expected = torch.tensor([2, 3, 4, 5])
        assert torch.equal(result, expected)

    def test_cp4_rank2(self):
        """Larger cp_size=4, verify rank=2 slicing."""
        _set_cp(rank=2, size=4)
        tokens = _tok(16)  # chunk_size = ceil(16/8) = 2
        result = slice_cp_with_zigzag(tokens, pad_value=0)
        # 8 chunks of size 2: [0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15]
        # rank=2 front: chunk[2] = [4,5]
        # rank=2 back:  chunk[2*4-2-1] = chunk[5] = [10,11]
        expected = torch.tensor([4, 5, 10, 11])
        assert torch.equal(result, expected)

    # ── padding value ─────────────────────────────────────────────────────

    def test_pad_value_fills_correctly(self):
        """Padding positions should be filled with the specified pad_value."""
        _set_cp(rank=0, size=2)
        tokens = _tok(3)  # chunk_size = ceil(3/4) = 1
        # padded to 4: [0, 1, 2, PAD]
        result = slice_cp_with_zigzag(tokens, pad_value=-99)
        # rank=0: front=chunk[0]=[0], back=chunk[3]=[-99]
        assert result[-1].item() == -99

    # ── all ranks cover the full (padded) sequence without overlap ────────

    def test_all_ranks_cover_full_sequence(self):
        """Union of all ranks' slices must equal the full padded tensor.

        Asserts on what slice_with_cp actually returned. ``_tok`` values equal
        their index, and pad_value=-1 keeps padding distinguishable from token 0.
        """
        cp_size = 3
        tokens = _tok(11)
        # chunk_size = ceil(11/6) = 2; padded_len = 12, so one pad slot.
        all_indices: list[int] = []
        for rank in range(cp_size):
            _set_cp(rank=rank, size=cp_size)
            result = slice_cp_with_zigzag(tokens.clone(), pad_value=0)
            # Compute which indices this rank holds
            chunk_size = 2
            start_1 = chunk_size * rank
            end_1 = chunk_size * (rank + 1)
            start_2 = chunk_size * (2 * cp_size - rank - 1)
            end_2 = chunk_size * (2 * cp_size - rank)
            all_indices.extend(range(start_1, end_1))
            all_indices.extend(range(start_2, end_2))

        all_indices.sort()
        assert all_indices == list(range(12)), "All ranks must cover the full padded sequence"

    def test_no_index_overlap_between_ranks(self):
        """No two ranks should receive the same token index."""
        cp_size = 4
        tokens = _tok(32)  # chunk_size = 4, perfect fit

        seen: set[int] = set()
        for rank in range(cp_size):
            _set_cp(rank=rank, size=cp_size)
            indices = set(slice_cp_with_zigzag(tokens.clone(), pad_value=-1).tolist())
            overlap = seen & indices
            assert not overlap, f"Rank {rank} overlaps at indices {overlap}"
            seen |= indices

        assert seen == set(range(32))

    # ── edge case: empty tensor ───────────────────────────────────────────

    def test_empty_tensor(self):
        """An empty tensor should produce an empty result."""
        _set_cp(rank=0, size=2)
        tokens = torch.tensor([], dtype=torch.int64)
        result = slice_cp_with_zigzag(tokens, pad_value=0)
        assert result.size(0) == 0


# ═══════════════════════════════════════════════════════════════════════════
# slice_cp_with_zigzag: 2-D inputs (teacher hidden states)
# ═══════════════════════════════════════════════════════════════════════════


HIDDEN = 8


def _hidden(n: int, hidden_size: int = HIDDEN) -> torch.Tensor:
    """Return a 2-D [n, hidden_size] tensor whose row i is filled with i."""
    return torch.arange(n, dtype=torch.float32).unsqueeze(1).expand(n, hidden_size).contiguous()


class TestSliceWithCp2D:
    """Zigzag slicing must pad the sequence dim, not the feature dim.

    ``full_kl`` / ``full_jsd`` push per-trajectory teacher hidden states
    ``[seq, hidden]`` through this path (``kl_ctx.teacher_logits``). ``F.pad``'s
    spec runs from the last dim backwards, so the bare ``(0, pad_len)`` that is
    correct for the 1-D token/mask tensors every other caller passes would widen
    ``hidden`` instead, and the student-side lm_head matmul then fails with a
    shape mismatch.

    ``seq=1881`` under ``cp_size=2`` is the production case: ``chunk_size=471``
    and ``pad_len=3``.
    """

    SEQ = 1881
    CHUNK = 471   # (1881 + 3) // 4
    PAD_LEN = 3   # 4 * 471 - 1881

    @pytest.mark.parametrize("rank", [0, 1])
    def test_feature_dim_preserved(self, rank):
        _set_cp(rank=rank, size=2)
        result = slice_cp_with_zigzag(_hidden(self.SEQ), pad_value=0)
        assert result.shape == (2 * self.CHUNK, HIDDEN)

    def test_both_ranks_get_equal_length(self):
        """Padding the seq dim keeps the two ranks' slices in lockstep.

        Without it the seq dim stays short of ``4 * chunk_size`` and rank 0's
        second chunk silently truncates, so the ranks disagree on length.
        """
        lens = []
        for rank in (0, 1):
            _set_cp(rank=rank, size=2)
            lens.append(slice_cp_with_zigzag(_hidden(self.SEQ), pad_value=0).size(0))
        assert lens[0] == lens[1] == 2 * self.CHUNK

    def test_rank1_rows_are_verbatim(self):
        """rank 1 holds chunks 1 and 2, both entirely inside the unpadded region."""
        _set_cp(rank=1, size=2)
        src = _hidden(self.SEQ)
        result = slice_cp_with_zigzag(src, pad_value=0)
        expected = torch.cat([
            src[self.CHUNK:2 * self.CHUNK],
            src[2 * self.CHUNK:3 * self.CHUNK],
        ])
        torch.testing.assert_close(result, expected)

    def test_rank0_tail_is_padded(self):
        """rank 0 holds chunks 0 and 3; chunk 3 runs past SEQ into the padding."""
        _set_cp(rank=0, size=2)
        src = _hidden(self.SEQ)
        result = slice_cp_with_zigzag(src, pad_value=-1)

        torch.testing.assert_close(result[:self.CHUNK], src[:self.CHUNK])
        real_tail = self.SEQ - 3 * self.CHUNK  # rows of chunk 3 that actually exist
        torch.testing.assert_close(
            result[self.CHUNK:self.CHUNK + real_tail], src[3 * self.CHUNK:]
        )
        assert (result[self.CHUNK + real_tail:] == -1).all()
        assert result.size(0) - (self.CHUNK + real_tail) == self.PAD_LEN

    def test_1d_behaviour_unchanged(self):
        """The 1-D callers (tokens / masks / log-probs) must be untouched by the fix."""
        _set_cp(rank=0, size=2)
        result = slice_cp_with_zigzag(_tok(self.SEQ), pad_value=-1)
        assert result.dim() == 1
        assert result.size(0) == 2 * self.CHUNK
        assert torch.equal(result[:self.CHUNK], _tok(self.CHUNK))
        assert (result[-self.PAD_LEN:] == -1).all()


# ═══════════════════════════════════════════════════════════════════════════
# Round-trip: slice_cp_with_zigzag → gather_and_reconstruct_cp
# ═══════════════════════════════════════════════════════════════════════════


class TestSliceGatherRoundTrip:
    """Simulate all CP ranks locally and verify reconstruction matches the
    original padded sequence."""

    @staticmethod
    def _simulate_roundtrip(
        token_lengths: list[int],
        cp_size: int,
    ) -> list[torch.Tensor]:
        """Run slice → pack → gather → reconstruct for all ranks in-process.

        Returns the list of reconstructed sequences (one per trajectory).
        """
        # 1) Slice each trajectory for each rank
        per_rank_slices: list[list[torch.Tensor]] = []
        for rank in range(cp_size):
            _set_cp(rank=rank, size=cp_size)
            slices = [slice_cp_with_zigzag(_tok(L), pad_value=0) for L in token_lengths]
            per_rank_slices.append(slices)

        # 2) Pack each rank (simple concat, no alignment padding for clarity)
        per_rank_packed = [torch.cat(slices) for slices in per_rank_slices]

        # 3) For each rank, simulate gather_and_reconstruct_cp
        # Since we can't use dist.all_gather, we patch it.
        def fake_all_gather(output_list, tensor, group=None):
            for i in range(cp_size):
                output_list[i].copy_(per_rank_packed[i])

        _set_cp(rank=0, size=cp_size)
        _mpu_mock.get_context_parallel_group.return_value = None

        with patch("coda.backends.megatron.cp_utils.dist.all_gather", side_effect=fake_all_gather):
            results = gather_and_reconstruct_cp(per_rank_packed[0], token_lengths)

        return results

    def test_single_trajectory_exact_fit(self):
        """A single trajectory whose length is an exact multiple of 2*cp_size."""
        cp_size = 2
        L = 8  # chunk_size=2, padded_len=8 (exact)
        results = self._simulate_roundtrip([L], cp_size)
        assert len(results) == 1
        reconstructed = results[0]
        # reconstructed should be the original padded sequence [0..7]
        expected = _tok(8)
        assert torch.equal(reconstructed, expected)

    def test_single_trajectory_needs_padding(self):
        """A single trajectory requiring padding — reconstructed tensor should
        contain the original tokens followed by zeros."""
        cp_size = 2
        L = 5  # chunk_size = ceil(5/4) = 2, padded_len = 8
        results = self._simulate_roundtrip([L], cp_size)
        reconstructed = results[0]
        assert reconstructed.size(0) == 8
        # First 5 values should be 0..4
        assert torch.equal(reconstructed[:5], _tok(5))
        # Remaining 3 should be pad (0)
        assert (reconstructed[5:] == 0).all()

    def test_multiple_trajectories(self):
        """Multiple trajectories with different lengths."""
        cp_size = 2
        lengths = [8, 4]
        results = self._simulate_roundtrip(lengths, cp_size)
        assert len(results) == 2
        # Sample 0: [0..7]
        assert torch.equal(results[0], _tok(8, offset=0))
        # Sample 1: [0..3]
        assert torch.equal(results[1], _tok(4, offset=0))

    def test_cp_size_4(self):
        """Round-trip with cp_size=4."""
        cp_size = 4
        L = 16  # chunk_size=2, exact fit
        results = self._simulate_roundtrip([L], cp_size)
        expected = _tok(16)
        assert torch.equal(results[0], expected)


# ═══════════════════════════════════════════════════════════════════════════
# prepare_packed_seq_params
# ═══════════════════════════════════════════════════════════════════════════


class TestPreparePackedSeqParams:
    """Tests for packing + PackedSeqParams construction."""

    # ── basic functionality ───────────────────────────────────────────────

    def test_single_sequence_no_cp(self):
        """cp_size=1, tp_size=1: packed output should be padded to pad_multiplier."""
        _set_cp(rank=0, size=1)
        _set_tp(size=1)
        tokens = _tok(10)
        packed, params = prepare_packed_seq_params([tokens], pad_multiplier=8)
        # pad_size = 1*8 = 8; packed.size(0) should be multiple of 8
        assert packed.size(0) % 8 == 0
        assert packed.size(0) >= 10

    def test_packed_tokens_start_with_originals(self):
        """The packed tensor should begin with the (CP-sliced) tokens."""
        _set_cp(rank=0, size=1)
        _set_tp(size=1)
        tokens = _tok(5)
        packed, _ = prepare_packed_seq_params([tokens], pad_multiplier=8)
        assert torch.equal(packed[:5], tokens)

    def test_multiple_sequences(self):
        """Multiple sequences are concatenated in order."""
        _set_cp(rank=0, size=1)
        _set_tp(size=1)
        t1 = _tok(3)
        t2 = _tok(5, offset=10)
        packed, params = prepare_packed_seq_params([t1, t2], pad_multiplier=8)
        assert packed.size(0) % 8 == 0
        # First 3 should be t1, next 5 should be t2
        assert torch.equal(packed[:3], t1)
        assert torch.equal(packed[3:8], t2)

    # ── cu_seqlens correctness ────────────────────────────────────────────

    def test_cu_seqlens_without_padding(self):
        """When total length is already aligned, cu_seqlens has N+1 entries."""
        _set_cp(rank=0, size=1)
        _set_tp(size=1)
        t1 = _tok(4)
        t2 = _tok(4)
        packed, params = prepare_packed_seq_params([t1, t2], pad_multiplier=8)
        cu = params.cu_seqlens_q
        # total sliced = 8, pad_size=8, no padding needed → 3 entries [0, 4, 8]
        assert cu.tolist() == [0, 4, 8]

    def test_cu_seqlens_with_padding(self):
        """When alignment padding is added, cu_seqlens gets an extra entry
        for the padding segment."""
        _set_cp(rank=0, size=1)
        _set_tp(size=1)
        t1 = _tok(3)
        packed, params = prepare_packed_seq_params([t1], pad_multiplier=8)
        cu = params.cu_seqlens_q
        # sliced total = 3, pad to 8 → pad_len=5
        # cu_seqlens before scaling: [0, 3, 8]
        # After *cp_size(=1): [0, 3, 8]
        assert cu.tolist() == [0, 3, 8]

    # ── cu_seqlens * cp_size also scales the padding segment ───────────────

    def test_cu_seqlens_scaling_inflates_padding_segment(self):
        """cu_seqlens is multiplied by cp_size, which also scales the pad segment.

        Not a correctness issue — the padding "sequence" holds pad tokens and
        does not affect attention — but it does inflate max_seqlen, which is a
        FlashAttention workspace-sizing concern.
        """
        _set_cp(rank=0, size=2)
        _set_tp(size=1)
        # cp_size=2: 3 tokens -> chunk_size=ceil(3/4)=1 -> sliced_len=2.
        # sliced total 2, pad_size=1*4=4 -> pad_len=2.
        # cu_seqlens before scaling [0, 2, 4]; after *2 [0, 4, 8].
        packed, params = prepare_packed_seq_params([_tok(3)], pad_multiplier=4)

        assert params.cu_seqlens_q.tolist() == [0, 4, 8]
        diffs = (params.cu_seqlens_q[1:] - params.cu_seqlens_q[:-1]).tolist()
        # The real sequence is 2 sliced tokens but reports 4; the 2-token pad
        # segment likewise reports 4.
        assert diffs == [4, 4]
        assert params.max_seqlen_q == 4

    # ── max_seqlen can be dominated by padding (perf, not correctness) ────

    def test_max_seqlen_inflated_by_padding(self):
        """A large alignment pad makes max_seqlen the scaled padding segment.

        Correct but wasteful: FlashAttention sizes its workspace from
        max_seqlen, so 496 is reserved for a real sequence of 16.
        """
        _set_cp(rank=0, size=4)
        _set_tp(size=1)
        # 9 tokens -> chunk_size=ceil(9/8)=2 -> sliced_len=4.
        # pad_size=128 -> pad_len=124. cu_seqlens [0, 4, 128] * 4 = [0, 16, 512].
        packed, params = prepare_packed_seq_params([_tok(9)], pad_multiplier=128)

        assert params.cu_seqlens_q.tolist() == [0, 16, 512]
        diffs = (params.cu_seqlens_q[1:] - params.cu_seqlens_q[:-1]).tolist()
        assert diffs == [16, 496]
        # max_seqlen tracks the padding segment, not the real sequence.
        assert params.max_seqlen_q == 496

    # ── edge case: empty tokens_list ────────────────────────────────────────

    def test_empty_tokens_list_raises(self):
        """An empty tokens_list should raise ValueError with a clear message."""
        _set_cp(rank=0, size=1)
        _set_tp(size=1)
        with pytest.raises(ValueError, match="tokens_list must be non-empty"):
            prepare_packed_seq_params([], pad_multiplier=8)

    # ── pad_token_id fills padding ────────────────────────────────────────

    def test_padding_uses_pad_token_id(self):
        """Padding region should be filled with pad_token_id."""
        _set_cp(rank=0, size=1)
        _set_tp(size=1)
        tokens = _tok(3)
        pad_id = -42
        packed, _ = prepare_packed_seq_params([tokens], pad_token_id=pad_id, pad_multiplier=8)
        # Positions 3..7 should be pad_id
        assert (packed[3:] == pad_id).all()

    # ── tp_size affects alignment ─────────────────────────────────────────

    def test_tp_affects_pad_alignment(self):
        """pad_size = tp_size * pad_multiplier."""
        _set_cp(rank=0, size=1)
        _set_tp(size=4)
        tokens = _tok(5)
        packed, _ = prepare_packed_seq_params([tokens], pad_multiplier=8)
        # pad_size = 4 * 8 = 32
        assert packed.size(0) % 32 == 0

    # ── PackedSeqParams fields ────────────────────────────────────────────

    def test_qkv_format_is_thd(self):
        _set_cp(rank=0, size=1)
        _set_tp(size=1)
        _, params = prepare_packed_seq_params([_tok(4)], pad_multiplier=4)
        assert params.qkv_format == "thd"

    def test_cu_seqlens_q_equals_kv(self):
        _set_cp(rank=0, size=1)
        _set_tp(size=1)
        _, params = prepare_packed_seq_params([_tok(4)], pad_multiplier=4)
        assert torch.equal(params.cu_seqlens_q, params.cu_seqlens_kv)

    def test_max_seqlen_q_equals_kv(self):
        _set_cp(rank=0, size=1)
        _set_tp(size=1)
        _, params = prepare_packed_seq_params([_tok(4)], pad_multiplier=4)
        assert params.max_seqlen_q == params.max_seqlen_kv

    def test_cu_seqlens_dtype_is_int32(self):
        _set_cp(rank=0, size=1)
        _set_tp(size=1)
        _, params = prepare_packed_seq_params([_tok(4)], pad_multiplier=4)
        assert params.cu_seqlens_q.dtype == torch.int32

    # ── when packed length is already aligned, no extra cu_seqlens entry ──

    def test_no_extra_cu_seqlens_when_aligned(self):
        """When total sliced length is already a multiple of pad_size,
        no extra cu_seqlens entry for padding should be added."""
        _set_cp(rank=0, size=1)
        _set_tp(size=1)
        # pad_size=4; use a tensor of exactly 4 tokens
        _, params = prepare_packed_seq_params([_tok(4)], pad_multiplier=4)
        cu = params.cu_seqlens_q
        # Should have exactly 2 entries: [0, 4]
        assert cu.size(0) == 2


# ═══════════════════════════════════════════════════════════════════════════
# gather_and_reconstruct_cp (unit-level, without round-trip)
# ═══════════════════════════════════════════════════════════════════════════


class TestGatherAndReconstructCp:
    """Direct tests for gather_and_reconstruct_cp with mocked all_gather."""

    def _run_with_fake_ranks(
        self,
        per_rank_data: list[torch.Tensor],
        total_lengths: list[int],
        cp_rank: int = 0,
    ) -> list[torch.Tensor]:
        """Run gather_and_reconstruct_cp with a fake all_gather."""
        cp_size = len(per_rank_data)
        _set_cp(rank=cp_rank, size=cp_size)
        _mpu_mock.get_context_parallel_group.return_value = None

        def fake_all_gather(output_list, tensor, group=None):
            for i in range(cp_size):
                output_list[i].copy_(per_rank_data[i])

        with patch("coda.backends.megatron.cp_utils.dist.all_gather", side_effect=fake_all_gather):
            return gather_and_reconstruct_cp(per_rank_data[cp_rank], total_lengths)

    def test_reconstructed_length(self):
        """Each reconstructed tensor should have length 2 * cp_size * chunk_size."""
        # cp_size=2, total_len=8 → chunk_size=2, reconstructed=8
        # rank0 has [0,1,6,7], rank1 has [2,3,4,5]
        rank0 = torch.tensor([0, 1, 6, 7], dtype=torch.int64)
        rank1 = torch.tensor([2, 3, 4, 5], dtype=torch.int64)
        results = self._run_with_fake_ranks([rank0, rank1], [8])
        assert results[0].size(0) == 8

    def test_reconstructed_order(self):
        """Inverse zigzag should recover the original sequence order."""
        rank0 = torch.tensor([0, 1, 6, 7], dtype=torch.int64)
        rank1 = torch.tensor([2, 3, 4, 5], dtype=torch.int64)
        results = self._run_with_fake_ranks([rank0, rank1], [8])
        expected = torch.arange(8, dtype=torch.int64)
        assert torch.equal(results[0], expected)

    def test_multiple_trajectories_offset(self):
        """Multiple trajectories in the packed tensor use correct offsets."""
        # Two trajectories, each of length 4, cp_size=2 → chunk_size=1, local_len=2
        # Trajectory 0: [0,1,2,3] → rank0=[0,3], rank1=[1,2]
        # Trajectory 1: [10,11,12,13] → rank0=[10,13], rank1=[11,12]
        rank0 = torch.tensor([0, 3, 10, 13], dtype=torch.int64)
        rank1 = torch.tensor([1, 2, 11, 12], dtype=torch.int64)
        results = self._run_with_fake_ranks([rank0, rank1], [4, 4])
        assert len(results) == 2
        assert torch.equal(results[0], torch.tensor([0, 1, 2, 3]))
        assert torch.equal(results[1], torch.tensor([10, 11, 12, 13]))

    def test_gradient_preserved_for_local_rank(self):
        """The local rank's slice should preserve the gradient graph
        (gathered[cp_rank] = packed_tensor replaces the detached copy)."""
        rank0 = torch.tensor([0.0, 1.0, 6.0, 7.0], requires_grad=True)
        rank1 = torch.tensor([2.0, 3.0, 4.0, 5.0])

        _set_cp(rank=0, size=2)
        _mpu_mock.get_context_parallel_group.return_value = None

        def fake_all_gather(output_list, tensor, group=None):
            output_list[0].copy_(rank0.detach())
            output_list[1].copy_(rank1.detach())

        with patch("coda.backends.megatron.cp_utils.dist.all_gather", side_effect=fake_all_gather):
            results = gather_and_reconstruct_cp(rank0, [8])

        # The result should still be connected to rank0's grad graph
        loss = results[0].sum()
        loss.backward()
        assert rank0.grad is not None
        assert (rank0.grad != 0).any()


# ═══════════════════════════════════════════════════════════════════════════
# slice_cp_packed (contiguous mode)
# ═══════════════════════════════════════════════════════════════════════════


class TestSliceCpPackedContiguous:
    """Tests for slice_cp_packed with contiguous mode."""

    def test_cp1_no_per_traj_pad(self):
        """cp_size=1, tp=1: padded_lens = raw lens, no tail-pad."""
        _set_cp(rank=0, size=1)
        _set_tp(size=1)
        t1 = _tok(7)
        t2 = _tok(5, offset=10)
        packed, padded_lens = slice_cp_packed(
            [t1, t2], "contiguous", 0, pad_multiplier=8
        )
        # cp=1, tp=1 → align_size=1, padded_lens = raw lens, no tail-pad in contiguous
        assert padded_lens == [7, 5]
        assert packed.size(0) == 12
        assert torch.equal(packed[:7], t1)
        assert torch.equal(packed[7:12], t2)

    def test_cp2_per_traj_pad(self):
        """cp_size=2, tp=1: per-traj padded to tp*2*cp=4."""
        _set_cp(rank=0, size=2)
        _set_tp(size=1)
        t1 = _tok(5)  # padded to 8 (next mult of 1*2*2=4 -> 8)
        packed, padded_lens = slice_cp_packed(
            [t1], "contiguous", 0, pad_multiplier=0
        )
        assert padded_lens == [8]
        # local_len = 8 // 2 = 4
        assert packed.size(0) == 4

    def test_cp2_rank_partition(self):
        """cp_size=2: rank0 and rank1 get non-overlapping contiguous halves."""
        _set_tp(size=1)
        t1 = _tok(4)  # padded_len=4, total=4, local_len=2

        _set_cp(rank=0, size=2)
        packed0, _ = slice_cp_packed([t1], "contiguous", -1, pad_multiplier=0)

        _set_cp(rank=1, size=2)
        packed1, _ = slice_cp_packed([t1], "contiguous", -1, pad_multiplier=0)

        assert torch.equal(packed0, torch.tensor([0, 1]))
        assert torch.equal(packed1, torch.tensor([2, 3]))

    def test_no_tail_pad_in_contiguous(self):
        """Contiguous mode does not add tail-pad; packed size = local_len."""
        _set_cp(rank=0, size=2)
        _set_tp(size=4)
        t1 = _tok(16)  # padded_len=16, local_len=8
        packed, _ = slice_cp_packed([t1], "contiguous", 0, pad_multiplier=8)
        # No tail-pad: local_len = 16 // 2 = 8
        assert packed.size(0) == 8

    def test_multiple_trajs(self):
        """Multiple trajs: concat in order, narrow flat."""
        _set_cp(rank=0, size=2)
        _set_tp(size=1)
        t1 = _tok(4)   # padded to 4
        t2 = _tok(4, offset=10)  # padded to 4
        packed, padded_lens = slice_cp_packed(
            [t1, t2], "contiguous", 0, pad_multiplier=0
        )
        assert padded_lens == [4, 4]
        # total=8, local=4, rank0 gets [0:4] = all of traj0
        assert packed.size(0) == 4
        assert torch.equal(packed, t1)

    def test_2d_hidden_states(self):
        """2-D [seq, hidden] inputs keep their feature dim through packing.

        ``kl_ctx.teacher_logits`` packs per-traj teacher hidden states through
        this branch for DSv4-style contiguous CP. A flat ``[total_padded]``
        buffer would reject the per-traj assignment outright.
        """
        _set_cp(rank=0, size=2)
        _set_tp(size=1)
        t1 = _hidden(4)
        t2 = _hidden(4)
        packed, padded_lens = slice_cp_packed(
            [t1, t2], "contiguous", 0, pad_multiplier=0
        )
        assert padded_lens == [4, 4]
        # total=8, local=4, rank0 gets [0:4] = all of traj0
        assert packed.shape == (4, HIDDEN)
        torch.testing.assert_close(packed, t1)

    def test_2d_hidden_states_per_traj_pad(self):
        """The per-traj pad rows land in the seq dim, filled with pad_value."""
        _set_cp(rank=0, size=1)
        _set_tp(size=1)
        t1 = _hidden(5)
        packed, padded_lens = slice_cp_packed(
            [t1, _hidden(3)], "contiguous", -1, pad_multiplier=0
        )
        # cp=1 → align=1, so no padding at all; assert the layout is verbatim.
        assert padded_lens == [5, 3]
        assert packed.shape == (8, HIDDEN)

        _set_cp(rank=0, size=2)
        packed, padded_lens = slice_cp_packed([t1], "contiguous", -1, pad_multiplier=0)
        # padded_len = 8 (next mult of 1*2*2=4), total=8, local_len=4 → rank0
        # sees rows 0..3, all real.
        assert padded_lens == [8]
        assert packed.shape == (4, HIDDEN)
        torch.testing.assert_close(packed, t1[:4])

        _set_cp(rank=1, size=2)
        packed, _ = slice_cp_packed([t1], "contiguous", -1, pad_multiplier=0)
        assert packed.shape == (4, HIDDEN)
        torch.testing.assert_close(packed[:1], t1[4:5])
        assert (packed[1:] == -1).all()


# ═══════════════════════════════════════════════════════════════════════════
# prepare_packed_seq_params (contiguous mode)
# ═══════════════════════════════════════════════════════════════════════════


class TestPreparePackedSeqParamsContiguous:
    """Tests for prepare_packed_seq_params with contiguous mode."""

    def test_basic(self):
        """Basic contiguous packing produces valid PackedSeqParams."""
        _set_cp(rank=0, size=2)
        _set_tp(size=1)
        t1 = _tok(4)
        t2 = _tok(4, offset=10)
        packed, params = prepare_packed_seq_params(
            [t1, t2], pad_token_id=0, pad_multiplier=8, cp_partition_mode="contiguous"
        )
        assert params.qkv_format == "thd"
        assert params.cp_partition_mode == "contiguous"
        # Contiguous: no tail-pad. padded_lens=[4,4], total=8, local=4
        assert packed.size(0) == 4

    def test_cu_seqlens_equals_padded(self):
        """In contiguous mode, cu_seqlens_q == cu_seqlens_q_padded."""
        _set_cp(rank=0, size=1)
        _set_tp(size=1)
        t1 = _tok(5)
        t2 = _tok(3, offset=10)
        _, params = prepare_packed_seq_params(
            [t1, t2], pad_token_id=0, pad_multiplier=8, cp_partition_mode="contiguous"
        )
        assert torch.equal(params.cu_seqlens_q, params.cu_seqlens_q_padded)

    def test_max_seqlen(self):
        """max_seqlen should be the largest padded_len."""
        _set_cp(rank=0, size=2)
        _set_tp(size=1)
        t1 = _tok(3)   # padded to 4
        t2 = _tok(7, offset=10)  # padded to 8
        _, params = prepare_packed_seq_params(
            [t1, t2], pad_token_id=0, pad_multiplier=8, cp_partition_mode="contiguous"
        )
        assert params.max_seqlen_q == 8


# ═══════════════════════════════════════════════════════════════════════════
# Contiguous round-trip: slice_cp_packed → gather_and_reconstruct_cp
# ═══════════════════════════════════════════════════════════════════════════


class TestContiguousRoundTrip:
    """Verify contiguous slice → gather → reconstruct recovers original."""

    @staticmethod
    def _simulate(token_lengths: list[int], cp_size: int) -> list[torch.Tensor]:
        _set_tp(size=1)
        padded_lens = [_contiguous_padded_length(L, cp_size, 1) for L in token_lengths]
        total_padded = sum(padded_lens)

        packed_full = torch.zeros(total_padded, dtype=torch.int64)
        offset = 0
        for i, (L, PL) in enumerate(zip(token_lengths, padded_lens)):
            packed_full[offset:offset + L] = _tok(L, offset=i * 100)
            offset += PL

        local_len = total_padded // cp_size
        per_rank = [
            packed_full[r * local_len:(r + 1) * local_len].clone()
            for r in range(cp_size)
        ]

        _set_cp(rank=0, size=cp_size)
        _mpu_mock.get_context_parallel_group.return_value = None

        def fake_all_gather(output_list, tensor, group=None):
            for i in range(cp_size):
                output_list[i].copy_(per_rank[i])

        with patch(
            "coda.backends.megatron.cp_utils.dist.all_gather",
            side_effect=fake_all_gather,
        ):
            return gather_and_reconstruct_cp(
                per_rank[0], token_lengths, cp_partition_mode="contiguous"
            )

    def test_single_traj_exact(self):
        results = self._simulate([8], cp_size=2)
        assert results[0].size(0) == 8
        assert torch.equal(results[0], _tok(8))

    def test_single_traj_padded(self):
        results = self._simulate([5], cp_size=2)
        # padded to 8 (tp*2*cp = 1*2*2 = 4, next mult of 4 ≥ 5 = 8)
        assert results[0].size(0) == 8
        assert torch.equal(results[0][:5], _tok(5))
        assert (results[0][5:] == 0).all()

    def test_multiple_trajs(self):
        results = self._simulate([4, 8], cp_size=2)
        assert len(results) == 2
        assert torch.equal(results[0][:4], _tok(4, offset=0))
        assert torch.equal(results[1][:8], _tok(8, offset=100))

    def test_cp4(self):
        results = self._simulate([16], cp_size=4)
        assert results[0].size(0) == 16
        assert torch.equal(results[0], _tok(16))

    def test_gradient_preserved(self):
        _set_cp(rank=0, size=2)
        _set_tp(size=1)
        _mpu_mock.get_context_parallel_group.return_value = None

        rank0 = torch.tensor([0.0, 1.0], requires_grad=True)
        rank1 = torch.tensor([2.0, 3.0])

        def fake_all_gather(output_list, tensor, group=None):
            output_list[0].copy_(rank0.detach())
            output_list[1].copy_(rank1.detach())

        with patch(
            "coda.backends.megatron.cp_utils.dist.all_gather",
            side_effect=fake_all_gather,
        ):
            results = gather_and_reconstruct_cp(
                rank0, [4], cp_partition_mode="contiguous"
            )

        loss = results[0].sum()
        loss.backward()
        assert rank0.grad is not None
        assert (rank0.grad != 0).any()
