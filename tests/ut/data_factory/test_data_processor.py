"""Unit tests for coda/data_factory/data_processor.py."""

from __future__ import annotations

import types
import unittest
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Patch targets – used by tests that want to intercept torch calls made by
# coda.data_factory.data_processor without touching real torch globally.
# ---------------------------------------------------------------------------

class _StubScalar:
    def __init__(self, v):
        self._v = v

    def item(self):
        return self._v


class _StubTensor(list):
    def numel(self):
        return len(self)

    def sum(self):
        total = 0
        for v in self:
            total += v
        return _StubScalar(total)


_torch_mod = types.ModuleType("torch")
_torch_mod.long = "long"  # type: ignore[attr-defined]
_torch_mod.int = "int"  # type: ignore[attr-defined]
_torch_mod.int32 = "int32"  # type: ignore[attr-defined]
_torch_mod.tensor = lambda data, dtype=None, device=None: _StubTensor(data)  # type: ignore[attr-defined]
_torch_mod.Tensor = _StubTensor  # type: ignore[attr-defined]
_torch_cuda = types.ModuleType("torch.cuda")
_torch_cuda.current_device = lambda: 0  # type: ignore[attr-defined]
_torch_mod.cuda = _torch_cuda  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Actual imports
# ---------------------------------------------------------------------------
from coda.agentflow.trajectory_store import (  # noqa: E402
    Segment,
    Trajectory,
    TrajectoryGroup,
    Triplet,
)
from coda.data_factory.data_processor import (  # noqa: E402
    Box,
    get_dp_shard_from_ray,
    put_dp_shards_to_ray,
    split_traj_group_by_dp,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# put_dp_shards_to_ray flattens each trajectory into per-trainable-Segment rows.
_EXPECTED_ROLLOUT_KEYS = {
    "prompt_id",
    "trajectory_id",
    "tokens",
    "loss_masks",
    "rollout_log_probs",
    "rollout_weight_versions",
    "rewards",
    "token_rewards",
    "response_lengths",
    "total_lengths",
    "metadata",
    "ds_indices",
}


def _make_triplet(ts: int = 0, te: int = 2, ls: int = 0, le: int = 2) -> Triplet:
    return Triplet(token_start=ts, token_end=te, logprob_start=ls, logprob_end=le)


def _make_trajectory(
    n_tokens: int,
    reward: float | None = 1.0,
    tid: str = "t0",
    pid: str = "p0",
    with_segments: bool = False,
    rollout_routed_experts = None,
) -> Trajectory:
    """Create a minimal Trajectory with one full-span trainable Segment.

    Real trajectories always have at least a root Segment; the source flatten
    (``put_dp_shards_to_ray``) only emits rows for trainable Segments.
    """
    trip = _make_triplet(0, n_tokens, 0, n_tokens)
    segs = [Segment(
        token_start=0, token_end=n_tokens,
        logprob_start=0, logprob_end=n_tokens, triplets=[trip],
    )]
    return Trajectory(
        trajectory_id=tid,
        prompt_id=pid,
        tokens=list(range(n_tokens)),
        loss_masks=[1] * n_tokens,
        rollout_log_probs=[-0.5] * n_tokens,
        rollout_weight_versions=[0] * n_tokens,
        reward=reward,
        token_rewards=[0.0] * (n_tokens - 1) + [reward or 0.0],
        rollout_routed_experts=rollout_routed_experts,
        segments=segs,
    )


def _make_group(prompt_id: str, token_counts: list[int], with_segments: bool = False) -> TrajectoryGroup:
    """Create a TrajectoryGroup whose trajectories have the given token counts."""
    trajs = [
        _make_trajectory(n, tid=f"{prompt_id}_t{i}", pid=prompt_id, with_segments=with_segments)
        for i, n in enumerate(token_counts)
    ]
    return TrajectoryGroup(prompt_id=prompt_id, trajectories=trajs)


def _fake_ray_put():
    """Return a (mock_ray, captured) pair; captured collects every dict passed to ray.put."""
    captured: list[dict] = []

    class _MockRay:
        put = staticmethod(lambda x: (captured.append(x), id(x))[1])

    return _MockRay(), captured


# ---------------------------------------------------------------------------
# Tests: put_dp_shards_to_ray
# ---------------------------------------------------------------------------

class TestPutDpShardsToRay(unittest.TestCase):

    def _shards(self, token_sums: list[int], dp_size: int) -> list[list[TrajectoryGroup]]:
        groups = [_make_group(f"p{i}", [n]) for i, n in enumerate(token_sums)]
        return split_traj_group_by_dp(groups, dp_size=dp_size, num_mini_batch=1)

    def test_dp_size_mismatch_raises(self):
        shards = self._shards([10, 20, 30, 40], dp_size=2)
        mock_ray, _ = _fake_ray_put()
        with patch("coda.data_factory.data_processor.ray", mock_ray):
            with self.assertRaises(AssertionError):
                put_dp_shards_to_ray(shards, dp_size=3)  # 2 shards vs dp_size=3

    # --- lengths ---

    def test_response_lengths_equals_loss_masks_len(self):
        group = _make_group("p0", [7, 4])
        mock_ray, captured = _fake_ray_put()
        with patch("coda.data_factory.data_processor.torch", _torch_mod):
            with patch("coda.data_factory.data_processor.ray", mock_ray):
                put_dp_shards_to_ray([[group]], dp_size=1)
        d = captured[0]
        for rl, lm in zip(d["response_lengths"], d["loss_masks"]):
            self.assertEqual(rl, len(lm))

    def test_total_lengths_equals_tokens_len(self):
        group = _make_group("p0", [7, 4])
        mock_ray, captured = _fake_ray_put()
        with patch("coda.data_factory.data_processor.torch", _torch_mod):
            with patch("coda.data_factory.data_processor.ray", mock_ray):
                put_dp_shards_to_ray([[group]], dp_size=1)
        d = captured[0]
        for tl, tok in zip(d["total_lengths"], d["tokens"]):
            self.assertEqual(tl, len(tok))

    def test_token_rewards_preserved(self):
        traj = _make_trajectory(n_tokens=5, reward=0.42, pid="p0")
        traj.token_rewards = [0.0, 0.0, 0.0, 0.0, 0.42]
        group = TrajectoryGroup(prompt_id="p0", trajectories=[traj])
        mock_ray, captured = _fake_ray_put()
        with patch("coda.data_factory.data_processor.torch", _torch_mod):
            with patch("coda.data_factory.data_processor.ray", mock_ray):
                put_dp_shards_to_ray([[group]], dp_size=1)
        self.assertEqual(captured[0]["token_rewards"][0], [0.0, 0.0, 0.0, 0.0, 0.42])

    # --- per-Segment flatten + trajectory_id ---

    def test_trajectory_id_and_segment_flatten(self):
        # traj0: 2 trainable Segments; traj1: 1 Segment → 3 rows.
        traj0 = _make_trajectory(n_tokens=10, pid="p0", tid="p0_t0")
        traj0.segments = [
            Segment(token_start=0, token_end=4, logprob_start=0, logprob_end=4, trainable=True),
            Segment(token_start=4, token_end=10, logprob_start=4, logprob_end=10, trainable=True),
        ]
        traj1 = _make_trajectory(n_tokens=5, pid="p0", tid="p0_t1")
        group = TrajectoryGroup(prompt_id="p0", trajectories=[traj0, traj1])
        mock_ray, captured = _fake_ray_put()
        with patch("coda.data_factory.data_processor.torch", _torch_mod):
            with patch("coda.data_factory.data_processor.ray", mock_ray):
                put_dp_shards_to_ray([[group]], dp_size=1)
        d = captured[0]
        self.assertEqual(d["trajectory_id"], [0, 0, 1])
        self.assertEqual(d["total_lengths"], [4, 6, 5])
        self.assertEqual(d["response_lengths"], [4, 6, 5])
        self.assertEqual(len(d["tokens"]), 3)
        # scalars are replicated across a trajectory's Segment rows
        self.assertEqual(d["prompt_id"], ["p0", "p0", "p0"])

    def test_non_trainable_segments_skipped(self):
        traj = _make_trajectory(n_tokens=10, pid="p0")
        traj.segments = [
            Segment(token_start=0, token_end=4, logprob_start=0, logprob_end=4, trainable=True),
            Segment(token_start=4, token_end=10, logprob_start=4, logprob_end=10, trainable=False),
        ]
        group = TrajectoryGroup(prompt_id="p0", trajectories=[traj])
        mock_ray, captured = _fake_ray_put()
        with patch("coda.data_factory.data_processor.torch", _torch_mod):
            with patch("coda.data_factory.data_processor.ray", mock_ray):
                put_dp_shards_to_ray([[group]], dp_size=1)
        d = captured[0]
        self.assertEqual(d["trajectory_id"], [0])
        self.assertEqual(d["total_lengths"], [4])
        self.assertEqual(set(d.keys()), _EXPECTED_ROLLOUT_KEYS)


# ---------------------------------------------------------------------------
# Tests: get_dp_shard_from_ray
# ---------------------------------------------------------------------------

class TestGetDpShardFromRay(unittest.TestCase):

    def _make_rollout_data(self) -> dict:
        """Create mock rollout data with per-trajectory segments/triplets.

        Structure:
        - 1 trajectory, 17 tokens total, 14 response positions
        - 2 segments, segment 0 has 2 triplets, segment 1 has 1 triplet
        """
        return {
            "prompt_id": ["p0"],
            "tokens": [list(range(17))],
            "loss_masks": [[1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1]],
            "rollout_log_probs": [[-0.5] * 14],
            "rollout_weight_versions": [[0] * 14],
            "token_rewards": [[0.0] * 13 + [1.0]],
            "total_lengths": [17],
            "response_lengths": [14],
            "rewards": [1.0],
            "rollout_routed_experts": [None],
            "segments": [[
                {"token_start": 0, "token_end": 10,
                 "logprob_start": 0, "logprob_end": 7, "reward": 0.5},
                {"token_start": 10, "token_end": 17,
                 "logprob_start": 12, "logprob_end": 14, "reward": 0.5},
            ]],
            "triplets": [[
                [
                    {"token_start": 0, "token_end": 5,
                     "logprob_start": 0, "logprob_end": 2,
                     "reward": 0.3, "metadata": {"tag": "a"}},
                    {"token_start": 5, "token_end": 10,
                     "logprob_start": 5, "logprob_end": 7,
                     "reward": 0.2, "metadata": {"tag": "b"}},
                ],
                [
                    {"token_start": 10, "token_end": 17,
                     "logprob_start": 12, "logprob_end": 14,
                     "reward": 0.5, "metadata": {"tag": "c"}},
                ],
            ]],
        }
    def _get_result(self, data: dict) -> dict:
        with patch("coda.data_factory.data_processor.ray") as mock_ray:
            mock_ray.get.return_value = data
            return get_dp_shard_from_ray(Box(ref=object()))

    # --- basic passthrough fields ---

    def test_passthrough_fields(self):
        data = self._make_rollout_data()
        result = self._get_result(data)

        self.assertIs(result["tokens"], data["tokens"])
        self.assertIs(result["loss_masks"], data["loss_masks"])
        self.assertIs(result["rollout_log_probs"], data["rollout_log_probs"])
        self.assertIs(result["rollout_weight_versions"], data["rollout_weight_versions"])
        self.assertEqual(result["total_lengths"], [17])
        self.assertEqual(result["response_lengths"], [14])
        self.assertEqual(result["rewards"], [1.0])
        self.assertEqual(len(result["token_rewards"]), 1)

    def test_all_expected_keys_present(self):
        result = self._get_result(self._make_rollout_data())
        expected_keys = {
            "prompt_id",
            "tokens", "loss_masks", "rollout_log_probs", "rollout_weight_versions",
            "total_lengths", "response_lengths",
            "token_rewards", "rewards",
            "rollout_routed_experts",
            "segment_tokens", "segment_loss_masks",
            "segment_log_probs", "segment_rewards",
            "triplet_tokens", "triplet_loss_masks",
            "triplet_log_probs", "triplet_rewards",
            "triplet_metadata",
            "metadata",
            "ds_indices",
        }
        self.assertEqual(set(result.keys()), expected_keys)

    def test_rollout_routed_experts_passthrough(self):
        # rollout_routed_experts should be passed through unchanged
        mock_tensor = object()
        data = self._make_rollout_data()
        data["rollout_routed_experts"] = [mock_tensor, None, mock_tensor]
        result = self._get_result(data)
        self.assertIs(result["rollout_routed_experts"], data["rollout_routed_experts"])

    def test_missing_rollout_routed_experts_omits_key(self):
        data = self._make_rollout_data()
        del data["rollout_routed_experts"]
        result = self._get_result(data)
        self.assertNotIn("rollout_routed_experts", result)

    def test_missing_rollout_weight_versions_raises(self):
        data = self._make_rollout_data()
        del data["rollout_weight_versions"]

        with self.assertRaises(KeyError):
            self._get_result(data)
    # --- segment slicing ---

    def test_segment_slicing_shapes(self):
        result = self._get_result(self._make_rollout_data())

        # 1 trajectory, 2 segments
        self.assertEqual(len(result["segment_tokens"]), 1)
        self.assertEqual(len(result["segment_tokens"][0]), 2)

    def test_segment_0_lengths(self):
        result = self._get_result(self._make_rollout_data())

        seg0_tok = result["segment_tokens"][0][0]
        seg0_mask = result["segment_loss_masks"][0][0]
        seg0_logp = result["segment_log_probs"][0][0]

        self.assertEqual(len(seg0_tok), 10)   # tokens[0:10]
        self.assertEqual(len(seg0_mask), 7)   # logprob[0:7]
        self.assertEqual(len(seg0_logp), 7)

    def test_segment_1_lengths(self):
        result = self._get_result(self._make_rollout_data())

        seg1_tok = result["segment_tokens"][0][1]
        seg1_mask = result["segment_loss_masks"][0][1]
        seg1_logp = result["segment_log_probs"][0][1]

        self.assertEqual(len(seg1_tok), 7)    # tokens[10:17]
        self.assertEqual(len(seg1_mask), 2)   # logprob[12:14]
        self.assertEqual(len(seg1_logp), 2)

    def test_segment_rewards(self):
        result = self._get_result(self._make_rollout_data())
        self.assertEqual(result["segment_rewards"][0], [0.5, 0.5])
    # --- triplet slicing ---

    def test_triplet_slicing_shapes(self):
        result = self._get_result(self._make_rollout_data())

        # [traj][seg][trip]: 1 traj, 2 segs, seg0 has 2 trips, seg1 has 1
        self.assertEqual(len(result["triplet_tokens"]), 1)
        self.assertEqual(len(result["triplet_tokens"][0]), 2)
        self.assertEqual(len(result["triplet_tokens"][0][0]), 2)
        self.assertEqual(len(result["triplet_tokens"][0][1]), 1)

    def test_triplet_00_lengths(self):
        result = self._get_result(self._make_rollout_data())

        trip00_tok = result["triplet_tokens"][0][0][0]
        trip00_mask = result["triplet_loss_masks"][0][0][0]
        trip00_logp = result["triplet_log_probs"][0][0][0]

        self.assertEqual(len(trip00_tok), 5)   # tokens[0:5]
        self.assertEqual(len(trip00_mask), 2)  # logprob[0:2]
        self.assertEqual(len(trip00_logp), 2)

    def test_triplet_01_lengths(self):
        result = self._get_result(self._make_rollout_data())

        trip01_tok = result["triplet_tokens"][0][0][1]
        trip01_mask = result["triplet_loss_masks"][0][0][1]

        self.assertEqual(len(trip01_tok), 5)   # tokens[5:10]
        self.assertEqual(len(trip01_mask), 2)  # logprob[5:7]

    def test_triplet_10_lengths(self):
        result = self._get_result(self._make_rollout_data())

        trip10_tok = result["triplet_tokens"][0][1][0]
        trip10_mask = result["triplet_loss_masks"][0][1][0]

        self.assertEqual(len(trip10_tok), 7)   # tokens[10:17]
        self.assertEqual(len(trip10_mask), 2)  # logprob[12:14]

    def test_triplet_rewards(self):
        result = self._get_result(self._make_rollout_data())
        self.assertEqual(
            result["triplet_rewards"][0],
            [[0.3, 0.2], [0.5]],
        )

    def test_triplet_metadata(self):
        result = self._get_result(self._make_rollout_data())
        self.assertEqual(
            result["triplet_metadata"][0],
            [[{"tag": "a"}, {"tag": "b"}], [{"tag": "c"}]],
        )
    # --- empty segments / triplets ---

    def test_empty_segments_and_triplets(self):
        data = self._make_rollout_data()
        data["segments"] = [[]]
        data["triplets"] = [[]]
        result = self._get_result(data)

        self.assertEqual(result["segment_tokens"], [[]])
        self.assertEqual(result["segment_rewards"], [[]])
        self.assertEqual(result["triplet_tokens"], [[]])
        self.assertEqual(result["triplet_rewards"], [[]])
        self.assertEqual(result["triplet_metadata"], [[]])

    # --- multiple trajectories ---

    def test_multiple_trajectories(self):
        data = {
            "prompt_id": ["p0", "p1"],
            "tokens": [[1, 2, 3], [4, 5, 6, 7]],
            "loss_masks": [[1, 1], [1, 0, 0]],
            "rollout_log_probs": [[-0.5, -0.3], [-0.2, 0.0, 0.0]],
            "rollout_weight_versions": [[0, 0], [0, 0, 0]],
            "token_rewards": [[0.0, 0.5], [0.0, 0.0, 0.8]],
            "total_lengths": [3, 4],
            "response_lengths": [2, 3],
            "rewards": [0.5, 0.8],
            "rollout_routed_experts": [None, None],
            "segments": [[], []],
            "triplets": [[], []],
        }
        result = self._get_result(data)

        self.assertEqual(len(result["tokens"]), 2)
        self.assertEqual(result["total_lengths"], [3, 4])
        self.assertEqual(result["rewards"], [0.5, 0.8])
        # empty segments → empty sliced lists per trajectory
        self.assertEqual(result["segment_tokens"], [[], []])
        self.assertEqual(result["triplet_tokens"], [[], []])

    # --- missing required field ---

    def test_missing_required_field_raises(self):
        data = self._make_rollout_data()
        del data["tokens"]
        with self.assertRaises(KeyError):
            self._get_result(data)


if __name__ == "__main__":
    unittest.main()
