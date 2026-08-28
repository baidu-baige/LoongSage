"""Unit tests for coda/data_factory/data_filter.py."""

import unittest

from coda.agentflow.trajectory_store import Trajectory, TrajectoryGroup
from coda.data_factory import DATA_FILTER_REGISTRY
from coda.data_factory.data_filter import (
    DataFilter,
    _BoundFilter,
    _filter_by_reward,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _traj(reward=None, tid="t1", pid="p1"):
    """Build a minimal Trajectory with the given reward."""
    return Trajectory(trajectory_id=tid, prompt_id=pid, reward=reward)


def _group(*rewards, prompt_id="p1"):
    """Build a TrajectoryGroup from a sequence of reward values.

    Example:
        _group(0, 0, 1)  →  TrajectoryGroup with 3 trajectories
    """
    trajs = [_traj(reward=r, tid=f"t{i}", pid=prompt_id) for i, r in enumerate(rewards)]
    return TrajectoryGroup(prompt_id=prompt_id, trajectories=trajs)


# ===========================================================================
# _filter_by_reward  (built-in, module-level function)
# ===========================================================================

class TestFilterByReward(unittest.TestCase):

    # --- groups that should be KEPT ---

    def test_keeps_group_with_mixed_zero_and_one(self):
        """Some 0s and some 1s → not trivial → keep."""
        result = _filter_by_reward(_group(0, 1, 0, 1))
        self.assertIsNotNone(result)
        self.assertEqual(len(result.trajectories), 4)

    def test_keeps_group_with_non_zero_non_one_rewards(self):
        """Float rewards that are neither 0 nor 1 → keep."""
        result = _filter_by_reward(_group(0.3, 0.7, 0.5))
        self.assertIsNotNone(result)
        self.assertEqual(len(result.trajectories), 3)

    def test_keeps_group_with_partial_zero_rewards(self):
        """Mix of 0 and non-zero/non-one → keep."""
        result = _filter_by_reward(_group(0, 0, 0.5))
        self.assertIsNotNone(result)
        self.assertEqual(len(result.trajectories), 3)

    def test_keeps_group_with_partial_one_rewards(self):
        """Mix of 1 and non-zero/non-one → keep."""
        result = _filter_by_reward(_group(1, 1, 0.5))
        self.assertIsNotNone(result)
        self.assertEqual(len(result.trajectories), 3)

    # --- groups that should be DROPPED ---

    def test_drops_group_all_zero(self):
        """All rewards are 0 → trivial → drop."""
        self.assertIsNone(_filter_by_reward(_group(0, 0, 0)))

    def test_drops_group_all_one(self):
        """All rewards are 1 → trivial → drop."""
        self.assertIsNone(_filter_by_reward(_group(1, 1, 1)))

    def test_drops_single_trajectory_with_zero(self):
        """Single trajectory with reward 0 → all-zero → drop."""
        self.assertIsNone(_filter_by_reward(_group(0)))

    def test_drops_single_trajectory_with_one(self):
        """Single trajectory with reward 1 → all-one → drop."""
        self.assertIsNone(_filter_by_reward(_group(1)))

    def test_drops_empty_group(self):
        """Empty trajectories → drop (nothing to evaluate)."""
        self.assertIsNone(_filter_by_reward(_group()))

    # --- return-value properties ---

    def test_returns_same_group_object_when_kept(self):
        """Kept group must be the same TrajectoryGroup object (no copy)."""
        group = _group(0, 1)
        result = _filter_by_reward(group)
        self.assertIs(result, group)

    def test_preserves_trajectory_identity(self):
        """Trajectories inside the returned group must be the same objects."""
        t0 = _traj(reward=0, tid="a")
        t1 = _traj(reward=1, tid="b")
        group = TrajectoryGroup(prompt_id="p1", trajectories=[t0, t1])
        result = _filter_by_reward(group)
        self.assertIs(result.trajectories[0], t0)
        self.assertIs(result.trajectories[1], t1)


# ===========================================================================
# DATA_FILTER_REGISTRY
# ===========================================================================

class TestDataFilterRegistry(unittest.TestCase):

    def test_reward_registered(self):
        self.assertIn("reward", DATA_FILTER_REGISTRY)

    def test_reward_resolves_to_correct_function(self):
        self.assertIs(DATA_FILTER_REGISTRY.get("reward"), _filter_by_reward)

    def test_unknown_names_not_registered(self):
        self.assertNotIn("reward_threshold", DATA_FILTER_REGISTRY)
        self.assertNotIn("non_trivial_reward", DATA_FILTER_REGISTRY)

    def test_unknown_name_raises_key_error(self):
        with self.assertRaises(KeyError):
            DATA_FILTER_REGISTRY.get("no_such_filter_xyz")

    def test_register_custom_filter(self):
        def _my_custom(traj_group, **_kwargs):
            return traj_group

        DATA_FILTER_REGISTRY.register("_test_custom_filter")(_my_custom)
        try:
            self.assertIs(DATA_FILTER_REGISTRY.get("_test_custom_filter"), _my_custom)
        finally:
            del DATA_FILTER_REGISTRY._registry["_test_custom_filter"]

    def test_duplicate_registration_raises_value_error(self):
        def _dummy(traj_group):
            return traj_group

        DATA_FILTER_REGISTRY.register("_test_dup")(_dummy)
        try:
            with self.assertRaises(ValueError):
                DATA_FILTER_REGISTRY.register("_test_dup")(_dummy)
        finally:
            del DATA_FILTER_REGISTRY._registry["_test_dup"]


# ===========================================================================
# DataFilter.__init__
# ===========================================================================

class TestDataFilterInit(unittest.TestCase):

    def test_none_config_builds_empty_chain(self):
        """DataFilter(None) is valid and produces an empty chain."""
        df = DataFilter(None)
        self.assertEqual(len(df.chain), 0)

    def test_empty_list_config_builds_empty_chain(self):
        df = DataFilter({})
        self.assertEqual(len(df.chain), 0)

    def test_single_filter_builds_chain_of_one(self):
        """A YAML key with no value parses to None; that must build one filter."""
        df = DataFilter({"reward": None})
        self.assertEqual(len(df.chain), 1)

    def test_two_filters_build_chain_of_two(self):
        def _pass_through(traj_group, **_kwargs):
            return traj_group

        DATA_FILTER_REGISTRY.register("_test_pass")(_pass_through)
        try:
            df = DataFilter({"reward": None, "_test_pass": None})
            self.assertEqual(len(df.chain), 2)
        finally:
            del DATA_FILTER_REGISTRY._registry["_test_pass"]

    def test_unknown_filter_name_raises_key_error(self):
        with self.assertRaises(KeyError):
            DataFilter({"does_not_exist": None})

    def test_custom_registered_filter_resolves_in_init(self):
        def _custom_filter(traj_group, **_kwargs):
            return traj_group

        DATA_FILTER_REGISTRY.register("_test_init_custom")(_custom_filter)
        try:
            df = DataFilter({"_test_init_custom": None})
            self.assertEqual(len(df.chain), 1)
        finally:
            del DATA_FILTER_REGISTRY._registry["_test_init_custom"]

    def test_chain_entries_are_bound_filters(self):
        df = DataFilter({"reward": None})
        self.assertIsInstance(df.chain[0], _BoundFilter)

    def test_bound_filter_name_matches_registry_key(self):
        def _custom(traj_group, **_kwargs):
            return traj_group

        DATA_FILTER_REGISTRY.register("_test_bound_name")(_custom)
        try:
            df = DataFilter({"reward": None, "_test_bound_name": None})
            self.assertEqual(df.chain[0].name, "reward")
            self.assertEqual(df.chain[1].name, "_test_bound_name")
        finally:
            del DATA_FILTER_REGISTRY._registry["_test_bound_name"]

    def test_invalid_param_raises_type_error_at_init(self):
        """Unknown params must raise TypeError at construction, not at apply()."""
        def _strict(traj_group, _required_param):
            return traj_group

        DATA_FILTER_REGISTRY.register("_test_strict")(_strict)
        try:
            with self.assertRaises(TypeError):
                DataFilter({"_test_strict": {"bad_param": 1}})
        finally:
            del DATA_FILTER_REGISTRY._registry["_test_strict"]

    def test_false_value_disables_filter(self):
        """Setting a filter to False skips it entirely."""
        df = DataFilter({"reward": False})
        self.assertEqual(len(df.chain), 0)

    def test_false_among_enabled_filters(self):
        """Only the False entry is skipped; others are active."""
        df = DataFilter({"status": None, "reward": False})
        self.assertEqual(len(df.chain), 1)
        self.assertEqual(df.chain[0].name, "status")


# ===========================================================================
# DataFilter.apply  (pipeline execution)
# ===========================================================================

class TestDataFilterPipeline(unittest.TestCase):

    def test_empty_chain_returns_same_input_object(self):
        """Empty chain passes the TrajectoryGroup through unchanged (same reference)."""
        df = DataFilter({})
        group = _group(0, 1)
        result = df.apply(group)
        self.assertIs(result, group)

    def test_reward_keeps_mixed_group(self):
        df = DataFilter({"reward": None})
        group = _group(0, 1, 0, 1)
        result = df.apply(group)
        self.assertIsNotNone(result)
        self.assertEqual(len(result.trajectories), 4)

    def test_reward_drops_all_zero_group(self):
        df = DataFilter({"reward": None})
        self.assertIsNone(df.apply(_group(0, 0, 0)))

    def test_reward_drops_all_one_group(self):
        df = DataFilter({"reward": None})
        self.assertIsNone(df.apply(_group(1, 1, 1)))

    def test_chain_applies_filters_in_order(self):
        """reward filter followed by a keep-all filter."""
        def _pass_through(traj_group, **_kwargs):
            return traj_group

        DATA_FILTER_REGISTRY.register("_test_chain_pass")(_pass_through)
        try:
            df = DataFilter({"reward": None, "_test_chain_pass": None})
            result_kept = df.apply(_group(0, 1))
            self.assertIsNotNone(result_kept)
            self.assertEqual(len(result_kept.trajectories), 2)
            self.assertIsNone(df.apply(_group(0, 0)))
        finally:
            del DATA_FILTER_REGISTRY._registry["_test_chain_pass"]

    def test_first_dropping_filter_short_circuits_chain(self):
        """If first filter drops, the second filter must not be called."""
        call_log = []

        def _drop_all(_traj_group, **_kwargs):
            call_log.append("drop")
            return None

        def _should_not_run(traj_group, **_kwargs):
            call_log.append("second")
            return traj_group

        DATA_FILTER_REGISTRY.register("_test_drop_first")(_drop_all)
        DATA_FILTER_REGISTRY.register("_test_second")(_should_not_run)
        try:
            df = DataFilter({"_test_drop_first": None, "_test_second": None})
            result = df.apply(_group(0, 1))
            self.assertIsNone(result)
            self.assertNotIn("second", call_log)
        finally:
            del DATA_FILTER_REGISTRY._registry["_test_drop_first"]
            del DATA_FILTER_REGISTRY._registry["_test_second"]

    def test_filter_drops_group_returns_none(self):
        df = DataFilter({"reward": None})
        self.assertIsNone(df.apply(_group(1, 1)))

    def test_filter_keeps_group_returns_traj_group(self):
        df = DataFilter({"reward": None})
        result = df.apply(_group(0, 1, 0.5))
        self.assertIsNotNone(result)
        self.assertIsInstance(result, TrajectoryGroup)
        self.assertEqual(len(result.trajectories), 3)

    def test_empty_trajectories_group_is_dropped(self):
        """TrajectoryGroup with no trajectories → drop."""
        df = DataFilter({"reward": None})
        empty_group = TrajectoryGroup(prompt_id="p", trajectories=[])
        self.assertIsNone(df.apply(empty_group))

    def test_custom_filter_returning_none_drops_group(self):
        def _drop_all(_traj_group, **_kwargs):
            return None

        DATA_FILTER_REGISTRY.register("_test_drop_all")(_drop_all)
        try:
            df = DataFilter({"_test_drop_all": None})
            self.assertIsNone(df.apply(_group(0, 1)))
        finally:
            del DATA_FILTER_REGISTRY._registry["_test_drop_all"]

    def test_multiple_calls_are_independent(self):
        df = DataFilter({"reward": None})
        self.assertEqual(len(df.apply(_group(0, 1)).trajectories), 2)
        self.assertIsNone(df.apply(_group(0, 0)))
        self.assertEqual(len(df.apply(_group(0, 1)).trajectories), 2)

    # --- isolation ---

    def test_apply_does_not_mutate_input_group(self):
        """apply() must not modify the trajectories list of the input group."""
        df = DataFilter({"reward": None})
        group = _group(0, 1, 0)
        original_len = len(group.trajectories)
        df.apply(group)
        self.assertEqual(len(group.trajectories), original_len)

    def test_mutating_config_dict_after_init_does_not_affect_filter(self):
        """_BoundFilter copies params at construction; later mutation is safe."""
        cfg = {"reward": {}}
        df = DataFilter(cfg)
        cfg["reward"]["injected"] = 999
        result = df.apply(_group(0, 1))
        self.assertIsNotNone(result)
        self.assertEqual(len(result.trajectories), 2)

    # --- logging ---

    def test_apply_emits_info_log_when_group_is_dropped(self):
        """Dropped group must say which filter dropped it and how many trajectories.

        Matching loosely on "reward" would pass on the logger name alone
        (coda.data_factory.data_filter), so assert on the message body.
        """
        df = DataFilter({"reward": None})
        with self.assertLogs("coda.data_factory.data_filter", level="INFO") as log:
            df.apply(_group(0, 0, 0))
        messages = [r.getMessage() for r in log.records]
        self.assertTrue(
            any("dropped all 3 trajectories" in m for m in messages),
            messages,
        )

    def test_apply_no_warning_when_group_is_kept(self):
        """Kept group must not emit any WARNING-level messages."""
        import logging as _log_mod

        df = DataFilter({"reward": None})
        warning_records: list = []

        class _WarningCapture(_log_mod.Handler):
            def emit(self, record):
                warning_records.append(record)

        log = _log_mod.getLogger("coda.data_factory.data_filter")
        cap = _WarningCapture(level=_log_mod.WARNING)
        log.addHandler(cap)
        try:
            df.apply(_group(0, 1))
        finally:
            log.removeHandler(cap)
        self.assertEqual(warning_records, [])


if __name__ == "__main__":
    unittest.main()
