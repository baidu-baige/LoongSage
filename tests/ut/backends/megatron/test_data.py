"""Unit tests for coda/backends/megatron/data.py.

Covers:
  * DataIterator — static and dynamic modes, exhaustion, edge cases.
  * _get_min_num_microbatches — first-fit bin-packing logic.
  * get_rollout_data — Ray resolve + device transfer.
  * get_data_iterator — static / dynamic / VPP factory paths.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

pytest.importorskip("megatron", reason="Megatron-Core is not installed")

from coda.backends.megatron.data import (
    DataIterator,
    _get_min_num_microbatches,
    get_data_iterator,
    get_rollout_data,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rollout(
    n: int,
    extra_fields: dict | None = None,
    lengths: list[int] | None = None,
) -> dict:
    """Per-Segment rollout: *n* rows, each its own single-Segment trajectory."""
    lengths = lengths or [5 + i for i in range(n)]
    data: dict = {
        "tokens": [torch.arange(length) for length in lengths],
        "loss_masks": [torch.ones(length) for length in lengths],
        "total_lengths": lengths,
        "response_lengths": lengths,
        "trajectory_id": list(range(n)),
    }
    if extra_fields:
        data.update(extra_fields)
    return data


def _make_segment_rollout(traj_seg_lengths: list[list[int]]) -> dict:
    """Per-Segment rollout from per-trajectory Segment lengths.

    ``traj_seg_lengths[t]`` lists the token lengths of trajectory ``t``'s Segment
    rows; those rows share ``trajectory_id == t`` and are emitted contiguously.
    """
    lengths: list[int] = []
    trajectory_id: list[int] = []
    for tid, seg_lengths in enumerate(traj_seg_lengths):
        for length in seg_lengths:
            lengths.append(length)
            trajectory_id.append(tid)
    return {
        "tokens": [torch.arange(length) for length in lengths],
        "loss_masks": [torch.ones(length) for length in lengths],
        "total_lengths": lengths,
        "response_lengths": lengths,
        "trajectory_id": trajectory_id,
    }



import coda.backends.megatron.data as _data_mod

_mpu_mock = MagicMock()


@pytest.fixture(autouse=True)
def mock_mpu(monkeypatch):
    """Swap data.mpu for a controllable mock, restoring it afterwards.

    Rebinding at import time leaked the mock into every later-collected module
    and made results collection-order dependent.
    """
    _mpu_mock.reset_mock()
    monkeypatch.setattr(_data_mod, "mpu", _mpu_mock)
    yield


def _set_parallel(
    dp_rank: int = 0,
    dp_size: int = 1,
    cp_size: int = 1,
    vpp_size: int | None = None,
    tp_rank: int = 0,
    pp_rank: int = 0,
    cp_rank: int = 0,
):
    """Configure the mocked mpu for get_data_iterator / get_rollout_data."""
    _mpu_mock.get_data_parallel_rank.return_value = dp_rank
    _mpu_mock.get_data_parallel_world_size.return_value = dp_size
    _mpu_mock.get_context_parallel_world_size.return_value = cp_size
    _mpu_mock.get_virtual_pipeline_model_parallel_world_size.return_value = vpp_size
    _mpu_mock.get_tensor_model_parallel_rank.return_value = tp_rank
    _mpu_mock.get_pipeline_model_parallel_rank.return_value = pp_rank
    _mpu_mock.get_context_parallel_rank.return_value = cp_rank
    _mpu_mock.get_data_parallel_group.return_value = None


def _iter(cfg, model, data, **kwargs):
    """Build the data iterator from already-flattened per-Segment rows."""
    return get_data_iterator(cfg, model, data, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# DataIterator — static mode
# ═══════════════════════════════════════════════════════════════════════════


class TestDataIteratorStatic:
    """Static micro-batch mode (micro_batch_size is set)."""

    def test_basic_iteration(self):
        """Returns the correct slice for each step."""
        data = _make_rollout(6)
        it = DataIterator(data, micro_batch_size=2)

        b0 = it.get_next(["tokens"])
        assert len(b0["tokens"]) == 2
        assert torch.equal(b0["tokens"][0], data["tokens"][0])
        assert torch.equal(b0["tokens"][1], data["tokens"][1])

        b1 = it.get_next(["tokens"])
        assert torch.equal(b1["tokens"][0], data["tokens"][2])

    def test_multiple_keys(self):
        """Multiple keys are returned correctly."""
        data = _make_rollout(4)
        it = DataIterator(data, micro_batch_size=2)
        batch = it.get_next(["tokens", "loss_masks"])
        assert "tokens" in batch
        assert "loss_masks" in batch
        assert len(batch["tokens"]) == 2
        assert len(batch["loss_masks"]) == 2

    def test_missing_key_returns_none(self):
        """A key not present in rollout_data returns None."""
        data = _make_rollout(4)
        it = DataIterator(data, micro_batch_size=2)
        batch = it.get_next(["tokens", "nonexistent_field"])
        assert batch["nonexistent_field"] is None

    def test_exhaustion_raises_index_error(self):
        """After all micro-batches are consumed, IndexError is raised."""
        data = _make_rollout(4)
        it = DataIterator(data, micro_batch_size=2)
        it.get_next(["tokens"])  # offset 0
        it.get_next(["tokens"])  # offset 1
        with pytest.raises(IndexError, match="exhausted"):
            it.get_next(["tokens"])  # offset 2 → start=4 >= 4

    def test_micro_batch_size_equals_total(self):
        """Single micro-batch covering all trajectories."""
        data = _make_rollout(3)
        it = DataIterator(data, micro_batch_size=3)
        batch = it.get_next(["tokens"])
        assert len(batch["tokens"]) == 3

    def test_not_divisible_causes_index_error(self):
        data = _make_rollout(5)
        it = DataIterator(data, micro_batch_size=3)
        it.get_next(["tokens"])
        with pytest.raises(IndexError):
            it.get_next(["tokens"])

    def test_micro_batch_size_larger_than_total(self):
        data = _make_rollout(2)
        it = DataIterator(data, micro_batch_size=5)
        with pytest.raises(IndexError):
            it.get_next(["tokens"])


# ═══════════════════════════════════════════════════════════════════════════
# DataIterator — dynamic mode
# ═══════════════════════════════════════════════════════════════════════════


class TestDataIteratorDynamic:
    """Dynamic micro-batch mode (micro_batch_indices is set)."""

    def test_basic_iteration(self):
        data = _make_rollout(6)
        indices = [[0, 2], [1, 3, 5], [4]]
        it = DataIterator(data, micro_batch_indices=indices)

        b0 = it.get_next(["tokens"])
        assert len(b0["tokens"]) == 2
        assert torch.equal(b0["tokens"][0], data["tokens"][0])
        assert torch.equal(b0["tokens"][1], data["tokens"][2])

        b1 = it.get_next(["tokens"])
        assert len(b1["tokens"]) == 3

        b2 = it.get_next(["tokens"])
        assert len(b2["tokens"]) == 1
        assert torch.equal(b2["tokens"][0], data["tokens"][4])

    def test_exhaustion_raises_index_error(self):
        data = _make_rollout(4)
        indices = [[0, 1], [2, 3]]
        it = DataIterator(data, micro_batch_indices=indices)
        it.get_next(["tokens"])
        it.get_next(["tokens"])
        with pytest.raises(IndexError, match="exhausted"):
            it.get_next(["tokens"])

    def test_missing_key_returns_none(self):
        data = _make_rollout(2)
        it = DataIterator(data, micro_batch_indices=[[0, 1]])
        batch = it.get_next(["tokens", "no_such_key"])
        assert batch["no_such_key"] is None

    def test_non_contiguous_indices(self):
        """Dynamic mode supports arbitrary index orderings."""
        data = _make_rollout(5)
        indices = [[4, 0, 2]]
        it = DataIterator(data, micro_batch_indices=indices)
        batch = it.get_next(["tokens"])
        assert torch.equal(batch["tokens"][0], data["tokens"][4])
        assert torch.equal(batch["tokens"][2], data["tokens"][2])

    def test_padding_index_zeros_masks(self):
        """A -1 index pads: masks are zeroed and row 0's data is reused."""
        data = _make_rollout(3)
        data["raw_loss_masks"] = [m.clone() for m in data["loss_masks"]]
        it = DataIterator(data, micro_batch_indices=[[1, -1]])
        assert it.padding_stats == (1, 2)
        batch = it.get_next(["tokens", "loss_masks", "raw_loss_masks"])
        assert torch.equal(batch["tokens"][1], data["tokens"][0])
        assert not batch["loss_masks"][1].any()
        assert not batch["raw_loss_masks"][1].any()


class TestDataIteratorInit:
    def test_both_args_set_raises(self):
        data = _make_rollout(4)
        with pytest.raises(AssertionError, match="Exactly one"):
            DataIterator(data, micro_batch_size=2, micro_batch_indices=[[0, 1]])

    def test_neither_arg_set_raises(self):
        data = _make_rollout(4)
        with pytest.raises(AssertionError, match="Exactly one"):
            DataIterator(data)


# ═══════════════════════════════════════════════════════════════════════════
# _get_min_num_microbatches
# ═══════════════════════════════════════════════════════════════════════════


class TestGetMinNumMicrobatches:
    """First-fit bin-packing for micro-batch count estimation."""

    def test_all_fit_in_one_bin(self):
        assert _get_min_num_microbatches([10, 20, 30], 100) == 1

    def test_exact_fit_boundary(self):
        """Items that sum exactly to the budget fit in one bin."""
        assert _get_min_num_microbatches([50, 50], 100) == 1

    def test_one_item_per_bin(self):
        """Each item exceeds half the budget → each gets its own bin."""
        assert _get_min_num_microbatches([60, 70, 80], 100) == 3

    def test_first_fit_packing(self):
        # [100, 50, 80, 20] with budget 120:
        #   bin0: 100; bin1: 50 → 50+80>120 → bin2: 80; try 20: bin0→100+20=120 ✓
        #   → 3 bins? Let's trace: 100→bin0; 50: bin0→150>120, new bin1(50);
        #     80: bin0→180>120, bin1→130>120, new bin2(80);
        #     20: bin0→120<=120 ✓ → 3 bins
        assert _get_min_num_microbatches([100, 50, 80, 20], 120) == 3

    def test_single_item(self):
        assert _get_min_num_microbatches([42], 100) == 1

    def test_empty_list(self):
        assert _get_min_num_microbatches([], 100) == 0

    def test_item_exceeds_budget(self):
        """A single item larger than the budget still gets one bin."""
        assert _get_min_num_microbatches([200], 100) == 1

    def test_multiple_large_items(self):
        """Each item exceeds the budget → one bin each."""
        assert _get_min_num_microbatches([200, 300], 100) == 2

    def test_first_fit_matches_optimal_for_this_input(self):
        """[7,7,6,6,5,5] budget=13.

        Optimal: {7+6, 7+6, 5+5} = 3 bins.
        First-fit: 7→b0; 7→b1; 6→b0(13); 6→b1(13); 5→b2; 5→b2(10) → 3 bins too.
        """
        assert _get_min_num_microbatches([7, 7, 6, 6, 5, 5], 13) == 3


# ═══════════════════════════════════════════════════════════════════════════
# get_rollout_data
# ═══════════════════════════════════════════════════════════════════════════


class TestGetRolloutData:
    """Test the Ray-resolve + device-move helper."""

    def test_moves_tensor_fields_to_device(self):
        _set_parallel(dp_rank=0)
        cpu_tokens = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
        cpu_masks = [torch.tensor([1.0, 1.0, 1.0]), torch.tensor([1.0, 1.0])]
        raw = {
            "tokens": cpu_tokens,
            "loss_masks": cpu_masks,
            "rewards": [1.0, 2.0],
        }

        with (
            patch.object(_data_mod, "ray") as ray_mock,
            patch("torch.cuda.current_device", return_value="cpu"),
        ):
            ray_mock.get.return_value = raw
            result = get_rollout_data([MagicMock(ref="ref_placeholder", teacher_worker_ref=[])])

        # Every tensor field lands on the device returned by current_device().
        for field in ("tokens", "loss_masks", "raw_loss_masks"):
            assert all(t.device.type == "cpu" for t in result[field])
        # rewards is not in the tensor fields list, should be unchanged
        assert result["rewards"] == [1.0, 2.0]

    def test_raw_loss_masks_is_an_independent_snapshot(self):
        """M2PO overwrites loss_masks, so raw_loss_masks must be a separate clone."""
        _set_parallel(dp_rank=0)
        raw = {
            "tokens": [torch.tensor([1, 2, 3])],
            "loss_masks": [torch.tensor([1.0, 1.0, 0.0])],
        }

        with (
            patch.object(_data_mod, "ray") as ray_mock,
            patch("torch.cuda.current_device", return_value="cpu"),
        ):
            ray_mock.get.return_value = raw
            result = get_rollout_data([MagicMock(ref="ref", teacher_worker_ref=[])])

        assert torch.equal(result["raw_loss_masks"][0], torch.tensor([1.0, 1.0, 0.0]))
        assert result["raw_loss_masks"][0] is not result["loss_masks"][0]

        result["loss_masks"][0][0] = 0.0
        assert result["raw_loss_masks"][0][0] == 1.0

    def test_none_field_is_skipped(self):
        _set_parallel(dp_rank=0)
        raw = {
            "tokens": [torch.tensor([1])],
            "loss_masks": [torch.tensor([1.0])],
            "rollout_log_probs": None,
            "rollout_routed_experts": None,
        }

        with (
            patch.object(_data_mod, "ray") as ray_mock,
            patch("torch.cuda.current_device", return_value="cpu"),
        ):
            ray_mock.get.return_value = raw
            result = get_rollout_data([MagicMock(ref="ref", teacher_worker_ref=[])])

        assert result["rollout_log_probs"] is None
        assert result["rollout_routed_experts"] is None


# ═══════════════════════════════════════════════════════════════════════════
# get_data_iterator — static batching
# ═══════════════════════════════════════════════════════════════════════════


class TestGetDataIteratorStatic:
    """Factory function with use_dynamic_batch_size=False."""

    def _make_config(self, **overrides):
        from unittest.mock import MagicMock as M
        cfg = M()
        cfg.use_dynamic_batch_size = False
        cfg.mini_batch_size = overrides.get("mini_batch_size", 4)
        cfg.micro_batch_size = overrides.get("micro_batch_size", 2)
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    def test_basic_static(self):
        _set_parallel(dp_size=1, cp_size=1, vpp_size=None)
        data = _make_rollout(8)
        cfg = self._make_config(mini_batch_size=4, micro_batch_size=2)

        iterators, num_mbs_list = _iter(cfg, MagicMock(), data)

        # vpp_size=None → 1 iterator
        assert len(iterators) == 1
        assert iterators[0].micro_batch_indices is not None
        # num_mini_batch = 8 // 4 = 2, each with 4//2 = 2 micro-batches
        assert num_mbs_list == [2, 2]

    def test_vpp_replication(self):
        """With VPP, the factory creates vpp_size identical iterators."""
        _set_parallel(dp_size=1, cp_size=1, vpp_size=3)
        data = _make_rollout(4)
        cfg = self._make_config(mini_batch_size=4, micro_batch_size=2)

        mock_cfg = MagicMock()
        mock_cfg.microbatch_group_size_per_vp_stage = 1
        with patch("coda.backends.megatron.data.get_model_config", return_value=mock_cfg):
            iterators, _ = _iter(cfg, MagicMock(), data)
        assert len(iterators) == 3

    def test_static_vpp_ceils_num_microbatches(self):
        """Static + VPP ceils num_microbatches UP to a multiple of
        microbatch_group_size_per_vp_stage, adding padding-only micro-batches
        without dropping any real Segment rows."""
        _set_parallel(dp_size=1, cp_size=1, vpp_size=2)
        # One trajectory with 5 trainable Segment rows.
        data = _make_segment_rollout([[2, 2, 2, 2, 2]])
        cfg = self._make_config(mini_batch_size=1, micro_batch_size=2)

        mock_cfg = MagicMock()
        mock_cfg.microbatch_group_size_per_vp_stage = 2
        with patch("coda.backends.megatron.data.get_model_config", return_value=mock_cfg):
            iterators, num_mbs_list = _iter(cfg, MagicMock(), data)

        # ceil(5 rows / mbs 2) = 3 → ceil up to multiple of 2 → 4
        assert num_mbs_list == [4]
        indices = iterators[0].micro_batch_indices
        assert len(indices) == 4
        assert sorted(i for b in indices for i in b if i >= 0) == [0, 1, 2, 3, 4]
        assert sum(i == -1 for b in indices for i in b) == 3

    def test_dp_size_divides_mini_batch(self):
        """mini_batch_size_per_dp = mini_batch_size // dp_size."""
        _set_parallel(dp_size=2, cp_size=1, vpp_size=None)
        data = _make_rollout(4)
        cfg = self._make_config(mini_batch_size=8, micro_batch_size=2)
        # mini_batch_size_per_dp = 8 // 2 = 4
        # num_mini_batch = 4 // 4 = 1, num_microbatches = 4 // 2 = 2
        # all_reduce(MAX) is a no-op here (single-segment rows, equal counts).
        with patch("coda.backends.megatron.data.dist.all_reduce"):
            _, num_mbs_list = _iter(cfg, MagicMock(), data)
        assert num_mbs_list == [2]

    def test_single_mini_batch(self):
        """use_single_mini_batch=True → one giant mini-batch covering all trajectories."""
        _set_parallel(dp_size=1, cp_size=1, vpp_size=None)
        data = _make_rollout(6)
        cfg = self._make_config(mini_batch_size=3, micro_batch_size=2)

        iterators, num_mbs_list = _iter(cfg, MagicMock(), data, use_single_mini_batch=True)
        # mini_batch_size_per_dp = 6 (all trajectories)
        # num_mini_batch = 1, num_microbatches = 6 // 2 = 3
        assert num_mbs_list == [3]

    def test_static_dp_sync_adds_only_noop_rows(self):
        _set_parallel(dp_size=2, cp_size=1, vpp_size=None)
        # traj0: 2 Segment rows; traj1: 1 row → 3 rows, trajectory_id=[0,0,1].
        data = _make_segment_rollout([[2, 3], [5]])
        cfg = self._make_config(mini_batch_size=4, micro_batch_size=2)

        def sync_to_three(tensor, **_kwargs):
            tensor.fill_(3)

        with patch(
            "coda.backends.megatron.data.dist.all_reduce",
            side_effect=sync_to_three,
        ):
            iterators, num_mbs_list = _iter(cfg, MagicMock(), data)

        assert num_mbs_list == [3]
        indices = iterators[0].micro_batch_indices
        assert sorted(i for batch in indices for i in batch if i >= 0) == [0, 1, 2]
        assert sum(i == -1 for batch in indices for i in batch) == 3

    def test_mini_batch_keeps_trajectory_segments_together(self):
        _set_parallel(dp_size=1, cp_size=1, vpp_size=None)
        # traj0: 2 rows, traj1: 2 rows, traj2/traj3: 1 row → trajectory_id=[0,0,1,1,2,3].
        data = _make_segment_rollout([[2, 3], [3, 3], [7], [8]])
        cfg = self._make_config(mini_batch_size=2, micro_batch_size=3)

        iterators, num_mbs_list = _iter(cfg, MagicMock(), data)

        assert num_mbs_list == [2, 1]
        assert iterators[0].micro_batch_indices == [
            [0, 1, 2],
            [3, -1, -1],
            [4, 5, -1],
        ]


# ═══════════════════════════════════════════════════════════════════════════
# get_data_iterator — dynamic batching
# ═══════════════════════════════════════════════════════════════════════════


class TestGetDataIteratorDynamic:
    """Factory function with use_dynamic_batch_size=True."""

    def _make_config(self, **overrides):
        cfg = MagicMock()
        cfg.use_dynamic_batch_size = True
        cfg.mini_batch_size = overrides.get("mini_batch_size", 4)
        cfg.max_tokens_per_gpu = overrides.get("max_tokens_per_gpu", 1000)
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    def test_basic_dynamic(self):
        """Dynamic mode creates DataIterator with micro_batch_indices."""
        _set_parallel(dp_size=1, cp_size=1, vpp_size=None)
        data = _make_rollout(4, lengths=[100, 200, 150, 250])
        cfg = self._make_config(mini_batch_size=4, max_tokens_per_gpu=1000)

        with patch("coda.backends.megatron.data.dist.all_reduce"):
            iterators, num_mbs_list = _iter(cfg, MagicMock(), data)

        assert len(iterators) == 1
        assert len(num_mbs_list) == 1
        # All 4 trajectories should fit in one micro-batch (budget=1000)
        assert num_mbs_list == [1]

    def test_dynamic_splits_when_budget_tight(self):
        """When budget is tight, more micro-batches are needed."""
        _set_parallel(dp_size=1, cp_size=1, vpp_size=None)
        data = _make_rollout(4, lengths=[100, 200, 150, 250])
        cfg = self._make_config(mini_batch_size=4, max_tokens_per_gpu=300)

        with patch("coda.backends.megatron.data.dist.all_reduce"):
            _, num_mbs_list = _iter(cfg, MagicMock(), data)

        # Budget 300: first-fit on [100,200,150,250]:
        #   100→b0; 200→b0(300); 150→b1; 250→b2 → 3 micro-batches
        # all_reduce is mocked as a no-op, so tensor values are unchanged.
        assert num_mbs_list == [3]

    def test_synced_dynamic_k_uses_padding_when_segments_are_insufficient(self):
        _set_parallel(dp_size=2, cp_size=1, vpp_size=None)
        data = _make_rollout(2)
        cfg = self._make_config(mini_batch_size=4, max_tokens_per_gpu=1000)

        def sync_to_three(tensor, **_kwargs):
            tensor.fill_(3)

        with patch("coda.backends.megatron.data.dist.all_reduce", side_effect=sync_to_three):
            iterators, num_mbs_list = _iter(cfg, MagicMock(), data)

        assert num_mbs_list == [3]
        assert sum(batch == [-1] for batch in iterators[0].micro_batch_indices) == 1

    # ── precondition: caller guarantees divisibility ───────────────────

    def test_tail_drop_assertion_failure(self):
        """When sequence_num_per_dp is not divisible by mini_batch_size_per_dp,
        the integer division at line 171 drops tail trajectories, and the
        assertion at line 243 fires. This is expected — the caller is
        required to guarantee divisibility.

        Documents the behaviour so future refactors remain aware of the
        precondition.
        """
        _set_parallel(dp_size=1, cp_size=1, vpp_size=None)
        # 5 trajectories, mini_batch_size=4 → mini_batch_size_per_dp=4
        # num_mini_batch = 5 // 4 = 1, covers indices 0-3, drops index 4
        data = _make_rollout(5)
        data["total_lengths"] = [100, 200, 150, 250, 300]
        cfg = self._make_config(mini_batch_size=4, max_tokens_per_gpu=1000)

        with patch("coda.backends.megatron.data.dist.all_reduce"):
            with pytest.raises(AssertionError):
                _iter(cfg, MagicMock(), data)
