"""Unit tests for coda/data_factory/data_source.py.
"""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

# ---------------------------------------------------------------------------
# Actual imports
# ---------------------------------------------------------------------------
import torch  # noqa: E402
from coda.agentflow.trajectory_store import Trajectory, TrajectoryGroup  # noqa: E402
from coda.data_factory import BUFFER_REPLAY_STRATEGY_REGISTRY  # noqa: E402
from coda.data_factory.data_source import (  # noqa: E402
    RolloutDataSource,
    RolloutDataSourceWithBuffer,
    _ID_SEP,
    fifo,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_config(
    *,
    prompt_data_path="",
    max_prompt_len=None,
    input_key="text",
    label_key=None,
    metadata_key="metadata",
    data_pre_processor=None,
    seed=42,
    shuffle=False,
    num_trajectories_per_prompt=2,
    checkpoint_path="/tmp/ckpt",
    load=None,
    buffer_replay_strategy=None,
):
    """Build a nested SimpleNamespace matching the fields accessed by RolloutDataSource.

    Returns a single object usable as both ds_config and global_config:
    - ds_config.dataset.* for dataset fields
    - ds_config.num_trajectories_per_prompt at top level
    - global_config.seed, global_config.checkpoint_path for global fields
    """
    dataset_cfg = SimpleNamespace(
        prompt_data_path=prompt_data_path,
        max_prompt_len=max_prompt_len,
        input_key=input_key,
        label_key=label_key,
        metadata_key=metadata_key,
        data_pre_processor=data_pre_processor,
        shuffle=shuffle,
        load=load,
        buffer_replay_strategy=buffer_replay_strategy,
    )
    return SimpleNamespace(
        dataset=dataset_cfg,
        num_trajectories_per_prompt=num_trajectories_per_prompt,
        seed=seed,
        checkpoint_path=checkpoint_path,
    )


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _traj_group(n, prompt="p"):
    trajs = [Trajectory(trajectory_id=f"t{i}", prompt_id="p0", prompt=prompt) for i in range(n)]
    return TrajectoryGroup(prompt_id="p0", trajectories=trajs)


# ===========================================================================
# fifo  (defined in data_source, registered in BUFFER_REPLAY_STRATEGY_REGISTRY)
# ===========================================================================

class TestFifo(unittest.TestCase):

    def test_pops_exact_count(self):
        buf = [_traj_group(1), _traj_group(1), _traj_group(1)]
        result = fifo(None, None, buf, 2)
        self.assertEqual(len(result), 2)
        self.assertEqual(len(buf), 1)

    def test_pops_all_when_fewer_than_requested(self):
        buf = [_traj_group(1), _traj_group(1)]
        result = fifo(None, None, buf, 10)
        self.assertEqual(len(result), 2)
        self.assertEqual(len(buf), 0)

    def test_empty_buffer_returns_empty(self):
        buf = []
        self.assertEqual(fifo(None, None, buf, 5), [])

    def test_zero_num_returns_empty_and_leaves_buffer(self):
        buf = [_traj_group(1)]
        result = fifo(None, None, buf, 0)
        self.assertEqual(result, [])
        self.assertEqual(len(buf), 1)

    def test_mutates_buffer_in_place(self):
        group = _traj_group(1)
        buf = [group]
        fifo(None, None, buf, 1)
        self.assertEqual(len(buf), 0)

    def test_returned_groups_are_original_objects(self):
        """fifo must not copy; it returns the original group references."""
        group = _traj_group(1)
        buf = [group]
        result = fifo(None, None, buf, 1)
        self.assertIs(result[0], group)

    def test_registered_in_buffer_replay_strategies_registry(self):
        """fifo must be reachable via the registry under its canonical name."""
        self.assertIs(BUFFER_REPLAY_STRATEGY_REGISTRY.get("fifo"), fifo)


# ===========================================================================
# RolloutDataSource.__init__
# ===========================================================================

class TestRolloutDataSourceInit(unittest.TestCase):

    def setUp(self):
        self.tmpdir = self.enterContext(tempfile.TemporaryDirectory())

    def _jsonl(self, records):
        p = os.path.join(self.tmpdir, "ds.jsonl")
        _write_jsonl(p, records)
        return p

    def test_counters_initialised_to_zero(self):
        path = self._jsonl([{"text": "x"}])
        cfg = _make_config(prompt_data_path=path)
        ds = RolloutDataSource(cfg, cfg)
        self.assertEqual(ds.epoch_id, 0)
        self.assertEqual(ds.step, 0)
        self.assertEqual(ds.prompt_offset, 0)
        self.assertEqual(ds.trajectory_count, 0)
        self.assertEqual(ds.prompt_index, 0)
        self.assertEqual(ds.trajectory_index, 0)

    def test_dataset_loaded(self):
        path = self._jsonl([{"text": "hello"}, {"text": "world"}])
        cfg = _make_config(prompt_data_path=path)
        ds = RolloutDataSource(cfg, cfg)
        self.assertEqual(len(ds.dataset), 2)

    def test_dataset_forwarded_keys(self):
        """input_key / label_key are passed through to the Dataset."""
        path = self._jsonl([{"prompt_col": "q", "lbl": "a"}])
        cfg = _make_config(
            prompt_data_path=path,
            input_key="prompt_col",
            label_key="lbl",
        )
        ds = RolloutDataSource(cfg, cfg)
        self.assertEqual(ds.dataset[0].prompt, [{"role": "user", "content": "q"}])
        self.assertEqual(ds.dataset[0].label, {"value": "a"})

    def test_shuffle_applied_on_init(self):
        path = self._jsonl([{"text": str(i)} for i in range(10)])
        cfg = _make_config(prompt_data_path=path, shuffle=True, seed=0)
        ds = RolloutDataSource(cfg, cfg)
        self.assertEqual(ds.dataset.epoch_id, 0)  # shuffle(0) was called

    def test_no_shuffle_on_init_when_disabled(self):
        path = self._jsonl([{"text": "x"}])
        cfg = _make_config(prompt_data_path=path, shuffle=False)
        ds = RolloutDataSource(cfg, cfg)
        self.assertEqual(ds.dataset.epoch_id, -1)  # shuffle never called


# ===========================================================================
# RolloutDataSource._fetch_prompts
# ===========================================================================

class TestFetchPrompts(unittest.TestCase):

    def setUp(self):
        self.tmpdir = self.enterContext(tempfile.TemporaryDirectory())

    def _ds(self, n_rows, shuffle=False, n_trajs=1):
        path = os.path.join(self.tmpdir, "ds.jsonl")
        _write_jsonl(path, [{"text": f"p{i}"} for i in range(n_rows)])
        cfg = _make_config(
            
            prompt_data_path=path,
            num_trajectories_per_prompt=n_trajs,
            shuffle=shuffle,
        )
        return RolloutDataSource(cfg, cfg)

    def test_advances_offset_within_epoch(self):
        ds = self._ds(5)
        ds._fetch_prompts(3)
        self.assertEqual(ds.prompt_offset, 3)

    def test_returns_requested_count_within_epoch(self):
        ds = self._ds(5)
        result = ds._fetch_prompts(3)
        self.assertEqual(len(result), 3)

    def test_epoch_wrap_increments_epoch_id(self):
        ds = self._ds(3)
        ds._fetch_prompts(3)       # exhaust epoch 0
        ds._fetch_prompts(2)       # crosses into epoch 1
        self.assertEqual(ds.epoch_id, 1)

    def test_epoch_wrap_offset_set_to_remainder(self):
        ds = self._ds(3)
        ds._fetch_prompts(3)       # offset = 3 (= dataset size)
        ds._fetch_prompts(2)       # 2 from epoch 1 → offset = 2
        self.assertEqual(ds.prompt_offset, 2)

    def test_epoch_wrap_returns_correct_total(self):
        ds = self._ds(3)
        ds._fetch_prompts(2)       # consume 2; 1 remains in epoch 0
        result = ds._fetch_prompts(4)  # 1 tail + 3 head
        self.assertEqual(len(result), 4)

    def test_tail_prefix_is_old_epoch(self):
        ds = self._ds(3)
        ds._fetch_prompts(2)       # consume 2; 1 left in epoch 0
        result = ds._fetch_prompts(3)  # 1 tail(epoch0) + 2 head(epoch1)
        self.assertIn("epoch0", result[0].prompt_id)

    def test_head_prefix_is_new_epoch(self):
        ds = self._ds(3)
        ds._fetch_prompts(2)
        result = ds._fetch_prompts(3)
        self.assertIn("epoch1", result[1].prompt_id)
        self.assertIn("epoch1", result[2].prompt_id)

    def test_shuffle_called_on_epoch_wrap(self):
        ds = self._ds(3, shuffle=True)
        ds._fetch_prompts(3)
        with patch.object(ds.dataset, "shuffle") as mock_shuffle:
            ds._fetch_prompts(1)
        mock_shuffle.assert_called_once_with(1)

    def test_no_shuffle_on_wrap_when_disabled(self):
        ds = self._ds(3, shuffle=False)
        ds._fetch_prompts(3)
        with patch.object(ds.dataset, "shuffle") as mock_shuffle:
            ds._fetch_prompts(1)
        mock_shuffle.assert_not_called()

    def test_prompt_index_reset_on_epoch_wrap(self):
        """When epoch wraps, prompt_index should be reset to 0."""
        ds = self._ds(3, n_trajs=2)
        # First, build some groups to increment prompt_index
        prompts = ds._fetch_prompts(2)
        ds._build_trajectory_groups(prompts)  # prompt_index becomes 2
        self.assertEqual(ds.prompt_index, 2)
        # Now trigger epoch wrap
        prompts = ds._fetch_prompts(3)  # epoch wrap, resets prompt_index to 0
        ds._build_trajectory_groups(prompts)  # prompt_index becomes 3
        self.assertEqual(ds.prompt_index, 3)


# ===========================================================================
# RolloutDataSource._assign_prompt_and_trajectory_id_prefix
# ===========================================================================

class TestAssignPromptAndTrajectoryIdPrefix(unittest.TestCase):

    def setUp(self):
        self.tmpdir = self.enterContext(tempfile.TemporaryDirectory())

    def _ds(self):
        path = os.path.join(self.tmpdir, "ds.jsonl")
        _write_jsonl(path, [{"text": "x"}])
        cfg = _make_config(prompt_data_path=path)
        return RolloutDataSource(cfg, cfg)

    def test_prompt_id_encodes_epoch_and_step(self):
        ds = self._ds()
        ds.epoch_id = 2
        ds.step = 5
        t = Trajectory(trajectory_id="t0", prompt_id="p0")
        ds._assign_prompt_and_trajectory_id_prefix([t])
        self.assertEqual(t.prompt_id, f"epoch2{_ID_SEP}step5{_ID_SEP}ds0{_ID_SEP}")

    def test_trajectory_id_same_as_prompt_id(self):
        ds = self._ds()
        ds.epoch_id = 1
        ds.step = 3
        t = Trajectory(trajectory_id="t0", prompt_id="p0")
        ds._assign_prompt_and_trajectory_id_prefix([t])
        self.assertEqual(t.trajectory_id, t.prompt_id)

    def test_all_prompts_get_same_prefix(self):
        ds = self._ds()
        ds.epoch_id = 0
        ds.step = 7
        prompts = [Trajectory(trajectory_id=f"t{i}", prompt_id="p0") for i in range(3)]
        ds._assign_prompt_and_trajectory_id_prefix(prompts)
        expected = f"epoch0{_ID_SEP}step7{_ID_SEP}ds0{_ID_SEP}"
        for p in prompts:
            self.assertEqual(p.prompt_id, expected)


# ===========================================================================
# RolloutDataSource._build_trajectory_groups
# ===========================================================================

class TestBuildTrajectoryGroups(unittest.TestCase):

    def setUp(self):
        self.tmpdir = self.enterContext(tempfile.TemporaryDirectory())

    def _ds(self, num_trajectories_per_prompt=3):
        path = os.path.join(self.tmpdir, "ds.jsonl")
        _write_jsonl(path, [{"text": "x"}])
        cfg = _make_config(prompt_data_path=path, num_trajectories_per_prompt=num_trajectories_per_prompt)
        return RolloutDataSource(cfg, cfg)

    def _stamped_trajectory(self, ds, prompt_id_val=None):
        """Return a Trajectory with prompt_id/trajectory_id prefix already stamped."""
        t = Trajectory(trajectory_id="t0", prompt_id="p0")
        ds._assign_prompt_and_trajectory_id_prefix([t])
        return t

    def test_returns_one_group_per_prompt(self):
        ds = self._ds()
        prompts = [self._stamped_trajectory(ds) for _ in range(4)]
        groups = ds._build_trajectory_groups(prompts)
        self.assertEqual(len(groups), 4)

    def test_returns_trajectory_group_instances(self):
        ds = self._ds(num_trajectories_per_prompt=2)
        prompts = [self._stamped_trajectory(ds)]
        groups = ds._build_trajectory_groups(prompts)
        self.assertIsInstance(groups[0], TrajectoryGroup)

    def test_each_group_has_num_trajectories_per_prompt(self):
        ds = self._ds(num_trajectories_per_prompt=3)
        prompts = [self._stamped_trajectory(ds) for _ in range(2)]
        groups = ds._build_trajectory_groups(prompts)
        for g in groups:
            self.assertEqual(len(g.trajectories), 3)

    def test_group_prompt_id_matches_trajectory_prompt_id(self):
        """TrajectoryGroup.prompt_id must equal the shared prompt_id of its trajectories."""
        ds = self._ds(num_trajectories_per_prompt=2)
        prompts = [self._stamped_trajectory(ds)]
        groups = ds._build_trajectory_groups(prompts)
        g = groups[0]
        self.assertEqual(g.prompt_id, g.trajectories[0].prompt_id)

    def test_trajectories_are_deep_copies(self):
        ds = self._ds(num_trajectories_per_prompt=2)
        original = Trajectory(trajectory_id="t0", prompt_id="p0", prompt="orig")
        ds._assign_prompt_and_trajectory_id_prefix([original])
        groups = ds._build_trajectory_groups([original])
        # Mutating one copy must not affect the other
        groups[0].trajectories[0].prompt = "modified"
        self.assertEqual(groups[0].trajectories[1].prompt, "orig")

    def test_prompt_id_increments_per_group(self):
        """Each successive prompt group gets a higher prompt_index in the suffix."""
        ds = self._ds(num_trajectories_per_prompt=1)
        prompts = [self._stamped_trajectory(ds) for _ in range(3)]
        initial_pid = ds.prompt_index
        groups = ds._build_trajectory_groups(prompts)
        for i, group in enumerate(groups):
            self.assertIn(f"prompt{initial_pid + i}", group.trajectories[0].prompt_id)

    def test_trajectory_count_increments_globally(self):
        """trajectory_count counter increments monotonically across all groups."""
        ds = self._ds(num_trajectories_per_prompt=2)
        prompts = [self._stamped_trajectory(ds) for _ in range(2)]
        initial_count = ds.trajectory_count
        groups = ds._build_trajectory_groups(prompts)
        all_tids = [s.trajectory_id for g in groups for s in g.trajectories]
        # All 4 trajectory IDs should be unique
        self.assertEqual(len(all_tids), len(set(all_tids)))
        # The global counter should have advanced by n_groups * num_trajectories_per_prompt
        self.assertEqual(ds.trajectory_count, initial_count + 4)

    def test_trajectory_index_resets_per_group(self):
        """trajectory_index resets to 0 at the start of each prompt group."""
        ds = self._ds(num_trajectories_per_prompt=3)
        prompts = [self._stamped_trajectory(ds) for _ in range(3)]
        groups = ds._build_trajectory_groups(prompts)
        # Check that trajectory_index resets for each group
        # First group: trajectory0, trajectory1, trajectory2
        self.assertIn("trajectory0", groups[0].trajectories[0].trajectory_id)
        self.assertIn("trajectory1", groups[0].trajectories[1].trajectory_id)
        self.assertIn("trajectory2", groups[0].trajectories[2].trajectory_id)
        # Second group: trajectory0, trajectory1, trajectory2 (index resets)
        self.assertIn("trajectory0", groups[1].trajectories[0].trajectory_id)
        self.assertIn("trajectory1", groups[1].trajectories[1].trajectory_id)
        self.assertIn("trajectory2", groups[1].trajectories[2].trajectory_id)
        # Third group: same pattern
        self.assertIn("trajectory0", groups[2].trajectories[0].trajectory_id)
        self.assertIn("trajectory1", groups[2].trajectories[1].trajectory_id)
        self.assertIn("trajectory2", groups[2].trajectories[2].trajectory_id)
        # Verify all trajectory_ids are still unique (due to count)
        all_tids = [s.trajectory_id for g in groups for s in g.trajectories]
        self.assertEqual(len(all_tids), len(set(all_tids)))

    def test_prompt_id_counter_advances(self):
        ds = self._ds(num_trajectories_per_prompt=1)
        prompts = [self._stamped_trajectory(ds) for _ in range(3)]
        ds._build_trajectory_groups(prompts)
        self.assertEqual(ds.prompt_index, 3)

    def test_trajectory_ids_are_unique_across_groups(self):
        ds = self._ds(num_trajectories_per_prompt=2)
        prompts = [self._stamped_trajectory(ds) for _ in range(3)]
        groups = ds._build_trajectory_groups(prompts)
        all_tids = [s.trajectory_id for g in groups for s in g.trajectories]
        self.assertEqual(len(all_tids), len(set(all_tids)))


# ===========================================================================
# RolloutDataSource.get (integration of fetch + build)
# ===========================================================================

class TestGet(unittest.TestCase):

    def setUp(self):
        self.tmpdir = self.enterContext(tempfile.TemporaryDirectory())

    def _ds(self, n_rows=5, n_trajs=2, shuffle=False):
        path = os.path.join(self.tmpdir, "ds.jsonl")
        _write_jsonl(path, [{"text": f"p{i}"} for i in range(n_rows)])
        cfg = _make_config(
            
            prompt_data_path=path,
            num_trajectories_per_prompt=n_trajs,
            shuffle=shuffle,
        )
        return RolloutDataSource(cfg, cfg)

    def test_correct_number_of_groups(self):
        groups = self._ds().get(3)
        self.assertEqual(len(groups), 3)

    def test_returns_list_of_trajectory_groups(self):
        groups = self._ds().get(2)
        for g in groups:
            self.assertIsInstance(g, TrajectoryGroup)

    def test_group_size_matches_num_trajectories_per_prompt(self):
        groups = self._ds(n_trajs=4).get(2)
        for g in groups:
            self.assertEqual(len(g.trajectories), 4)

    def test_prompt_id_contains_step(self):
        ds = self._ds()
        ds.step = 9
        groups = ds.get(1)
        self.assertIn("step9", groups[0].trajectories[0].prompt_id)

    def test_consecutive_calls_advance_offset(self):
        ds = self._ds(n_rows=6, n_trajs=1)
        ds.get(2)
        ds.get(2)
        self.assertEqual(ds.prompt_offset, 4)

    def test_prompt_id_counter_advances_across_calls(self):
        ds = self._ds(n_rows=6, n_trajs=1)
        ds.get(2)
        self.assertEqual(ds.prompt_index, 2)
        ds.get(3)
        self.assertEqual(ds.prompt_index, 5)

    def test_trajectory_count_counter_advances_across_calls(self):
        ds = self._ds(n_rows=6, n_trajs=2)
        ds.get(2)   # 2 groups × 2 trajectories = 4 trajectories
        self.assertEqual(ds.trajectory_count, 4)


# ===========================================================================
# RolloutDataSource.add
# ===========================================================================

class TestAddReadOnly(unittest.TestCase):

    def setUp(self):
        self.tmpdir = self.enterContext(tempfile.TemporaryDirectory())

    def _cfg(self):
        path = os.path.join(self.tmpdir, "ds.jsonl")
        _write_jsonl(path, [{"text": "x"}])
        return _make_config(prompt_data_path=path)

    def test_raises_runtime_error(self):
        cfg = self._cfg()
        ds = RolloutDataSource(cfg, cfg)
        with self.assertRaises(RuntimeError):
            ds.add([TrajectoryGroup(prompt_id="p0", trajectories=[Trajectory(trajectory_id="t0", prompt_id="p0")])])

    def test_error_message_contains_class_name(self):
        cfg = self._cfg()
        ds = RolloutDataSource(cfg, cfg)
        with self.assertRaises(RuntimeError) as ctx:
            ds.add([TrajectoryGroup(prompt_id="p0", trajectories=[Trajectory(trajectory_id="t0", prompt_id="p0")])])
        self.assertIn("RolloutDataSource", str(ctx.exception))


# ===========================================================================
# RolloutDataSource.save
# ===========================================================================

class TestSave(unittest.TestCase):

    def setUp(self):
        self.tmpdir = self.enterContext(tempfile.TemporaryDirectory())
        self._save_patcher = patch("torch.save")
        self._load_patcher = patch("torch.load")
        self.mock_save = self._save_patcher.start()
        self.mock_load = self._load_patcher.start()

    def tearDown(self):
        self._save_patcher.stop()
        self._load_patcher.stop()

    def _ds(self, checkpoint_path=None, ds_index=0):
        path = os.path.join(self.tmpdir, "ds.jsonl")
        _write_jsonl(path, [{"text": "x"}])
        cfg = _make_config(
            prompt_data_path=path,
            checkpoint_path=checkpoint_path or self.tmpdir,
        )
        return RolloutDataSource(cfg, cfg, ds_index=ds_index)

    def test_saves_all_state_fields(self):
        ds = self._ds()
        ds.epoch_id = 2
        ds.step = 5
        ds.prompt_offset = 1
        ds.prompt_index = 10
        ds.trajectory_count = 20
        self.mock_save.reset_mock()
        with patch("os.makedirs"):
            ds.save("r1")
        state = self.mock_save.call_args[0][0]
        self.assertEqual(state["prompt_data_path"], ds.dataset_config.prompt_data_path)
        self.assertEqual(state["epoch_id"], 2)
        self.assertEqual(state["step"], 5)
        self.assertEqual(state["prompt_offset"], 1)
        self.assertEqual(state["prompt_index"], 10)
        self.assertEqual(state["trajectory_count"], 20)

    def test_save_path_contains_step(self):
        ds = self._ds(checkpoint_path="/mydir")
        self.mock_save.reset_mock()
        with patch("os.makedirs"):
            ds.save(10)
        saved_path = self.mock_save.call_args[0][1]
        self.assertIn("10", saved_path)

    def test_save_path_uses_ds_index_not_prompt_data_path(self):
        ds = self._ds(checkpoint_path="/mydir", ds_index=7)
        self.mock_save.reset_mock()
        with patch("os.makedirs"):
            ds.save(3)
        saved_path = self.mock_save.call_args[0][1]
        self.assertEqual(
            saved_path,
            "/mydir/train_step_3/data_source/global_dataset_state_dict_ds7.pt",
        )
        self.assertNotIn(ds.dataset_config.prompt_data_path, saved_path)

    def test_save_creates_parent_dirs(self):
        ds = self._ds()
        self.mock_save.reset_mock()
        with patch("os.makedirs") as mock_makedirs:
            ds.save("r0")
        mock_makedirs.assert_called_once()


# ===========================================================================
# RolloutDataSource.load
# ===========================================================================

class TestLoad(unittest.TestCase):

    def setUp(self):
        self.tmpdir = self.enterContext(tempfile.TemporaryDirectory())
        self._save_patcher = patch("torch.save")
        self._load_patcher = patch("torch.load")
        self.mock_save = self._save_patcher.start()
        self.mock_load = self._load_patcher.start()

    def tearDown(self):
        self._save_patcher.stop()
        self._load_patcher.stop()

    def _ds(self, load_dir=None, shuffle=False):
        path = os.path.join(self.tmpdir, "ds.jsonl")
        _write_jsonl(path, [{"text": f"p{i}"} for i in range(5)])
        cfg = _make_config(
            prompt_data_path=path,
            load=load_dir,
            shuffle=shuffle,
        )
        return RolloutDataSource(cfg, cfg)

    def test_skipped_when_no_candidate_files(self):
        ds = self._ds(load_dir=None)
        self.mock_load.reset_mock()
        with patch("coda.data_factory.data_source.glob.glob", return_value=[]):
            ds.load("r0")
        self.mock_load.assert_not_called()

    def test_skipped_when_no_checkpoint_matches_dataset_path(self):
        ds = self._ds(load_dir="/no/such/path")
        self.mock_load.reset_mock()
        self.mock_load.return_value = {"prompt_data_path": "/other/dataset.jsonl"}
        with patch(
            "coda.data_factory.data_source.glob.glob",
            return_value=["/tmp/ckpt/train_step_r0/data_source/global_dataset_state_dict_ds0.pt"],
        ):
            result = ds.load("r0")
        self.assertIsNone(result)
        self.mock_load.assert_called_once()

    def test_restores_all_state_fields(self):
        ds = self._ds(load_dir="/some/path")
        state = {
            "prompt_data_path": ds.dataset_config.prompt_data_path,
            "epoch_id": 3, "step": 10,
            "prompt_offset": 2, "prompt_index": 15, "trajectory_count": 30,
        }
        self.mock_load.reset_mock()
        self.mock_load.return_value = state
        with patch(
            "coda.data_factory.data_source.glob.glob",
            return_value=["/tmp/ckpt/train_step_r0/data_source/global_dataset_state_dict_ds0.pt"],
        ):
            ds.load("r0")
        self.assertEqual(ds.epoch_id, 3)
        self.assertEqual(ds.step, 10)
        self.assertEqual(ds.prompt_offset, 2)
        self.assertEqual(ds.prompt_index, 15)
        self.assertEqual(ds.trajectory_count, 30)

    def test_shuffle_called_after_restore_when_enabled(self):
        ds = self._ds(load_dir="/some/path", shuffle=True)
        state = {
            "prompt_data_path": ds.dataset_config.prompt_data_path,
            "epoch_id": 3, "step": 0, "prompt_offset": 0,
            "prompt_index": 0, "trajectory_count": 0,
        }
        self.mock_load.reset_mock()
        self.mock_load.return_value = state
        with patch(
            "coda.data_factory.data_source.glob.glob",
            return_value=["/tmp/ckpt/train_step_r0/data_source/global_dataset_state_dict_ds0.pt"],
        ):
            with patch.object(ds.dataset, "shuffle") as mock_shuffle:
                ds.load("r0")
        mock_shuffle.assert_called_once_with(3)

    def test_shuffle_not_called_after_restore_when_disabled(self):
        ds = self._ds(load_dir="/some/path", shuffle=False)
        state = {
            "prompt_data_path": ds.dataset_config.prompt_data_path,
            "epoch_id": 1, "step": 0, "prompt_offset": 0,
            "prompt_index": 0, "trajectory_count": 0,
        }
        self.mock_load.reset_mock()
        self.mock_load.return_value = state
        with patch(
            "coda.data_factory.data_source.glob.glob",
            return_value=["/tmp/ckpt/train_step_r0/data_source/global_dataset_state_dict_ds0.pt"],
        ):
            with patch.object(ds.dataset, "shuffle") as mock_shuffle:
                ds.load("r0")
        mock_shuffle.assert_not_called()

    def test_state_defaults_to_zero_for_missing_keys(self):
        ds = self._ds(load_dir="/some/path")
        ds.epoch_id = ds.step = ds.prompt_offset = ds.prompt_index = ds.trajectory_count = 1
        self.mock_load.reset_mock()
        self.mock_load.return_value = {
            "prompt_data_path": ds.dataset_config.prompt_data_path,
        }
        with patch(
            "coda.data_factory.data_source.glob.glob",
            return_value=["/tmp/ckpt/train_step_r0/data_source/global_dataset_state_dict_ds0.pt"],
        ):
            ds.load("r0")
        self.assertEqual(ds.epoch_id, 0)
        self.assertEqual(ds.step, 0)
        self.assertEqual(ds.prompt_offset, 0)
        self.assertEqual(ds.prompt_index, 0)
        self.assertEqual(ds.trajectory_count, 0)

    def test_traverses_candidates_until_dataset_path_matches(self):
        ds = self._ds(load_dir="/some/path")
        paths = [
            "/tmp/ckpt/train_step_5/data_source/global_dataset_state_dict_ds0.pt",
            "/tmp/ckpt/train_step_5/data_source/global_dataset_state_dict_ds1.pt",
        ]
        self.mock_load.reset_mock()
        self.mock_load.side_effect = [
            {"prompt_data_path": "/other/dataset.jsonl", "epoch_id": 99},
            {"prompt_data_path": ds.dataset_config.prompt_data_path, "epoch_id": 4},
        ]

        with patch(
            "coda.data_factory.data_source.glob.glob",
            return_value=list(reversed(paths)),
        ):
            state = ds.load(5)

        self.assertEqual(state["prompt_data_path"], ds.dataset_config.prompt_data_path)
        self.assertEqual(ds.epoch_id, 4)
        self.assertEqual(
            self.mock_load.call_args_list,
            [call(paths[0]), call(paths[1])],
        )


# ===========================================================================
# RolloutDataSource save/load matching across datasource reordering
# ===========================================================================

class TestSaveLoadDatasetMatching(unittest.TestCase):

    def setUp(self):
        self.tmpdir = self.enterContext(tempfile.TemporaryDirectory())

    def _ds(self, filename, ds_index):
        path = os.path.join(self.tmpdir, filename)
        _write_jsonl(path, [{"text": filename}])
        cfg = _make_config(
            prompt_data_path=path,
            checkpoint_path=self.tmpdir,
        )
        return RolloutDataSource(cfg, cfg, ds_index=ds_index)

    def test_load_matches_dataset_path_after_indices_change(self):
        original_a = self._ds("a.jsonl", ds_index=0)
        original_b = self._ds("b.jsonl", ds_index=1)
        original_a.epoch_id = 3
        original_b.epoch_id = 8
        original_a.save(5)
        original_b.save(5)

        reordered_a = self._ds("a.jsonl", ds_index=1)
        reordered_b = self._ds("b.jsonl", ds_index=0)
        reordered_a.load(5)
        reordered_b.load(5)

        self.assertEqual(reordered_a.epoch_id, 3)
        self.assertEqual(reordered_b.epoch_id, 8)


# ===========================================================================
# RolloutDataSource.__len__
# ===========================================================================

class TestLen(unittest.TestCase):

    def setUp(self):
        self.tmpdir = self.enterContext(tempfile.TemporaryDirectory())

    def test_returns_dataset_length(self):
        path = os.path.join(self.tmpdir, "ds.jsonl")
        _write_jsonl(path, [{"text": f"p{i}"} for i in range(7)])
        cfg = _make_config(prompt_data_path=path)
        self.assertEqual(len(RolloutDataSource(cfg, cfg)), 7)


# ===========================================================================
# RolloutDataSourceWithBuffer.__init__
# ===========================================================================

class TestBufferInit(unittest.TestCase):

    def setUp(self):
        self.tmpdir = self.enterContext(tempfile.TemporaryDirectory())

    def _ds(self, buffer_replay_strategy=None):
        path = os.path.join(self.tmpdir, "ds.jsonl")
        _write_jsonl(path, [{"text": "x"}])
        cfg = _make_config(prompt_data_path=path,
                           buffer_replay_strategy=buffer_replay_strategy)
        return RolloutDataSourceWithBuffer(cfg, cfg)

    def test_buffer_empty_on_init(self):
        self.assertEqual(self._ds().buffer, [])

    def test_default_filter_is_fifo(self):
        """When buffer_replay_strategy is None/empty the built-in fifo is used."""
        self.assertIs(self._ds().buffer_replay_strategy, fifo)

    def test_custom_filter_loaded_from_registry(self):
        """A registered name is resolved through BUFFER_REPLAY_STRATEGY_REGISTRY."""
        custom_fn = MagicMock()
        BUFFER_REPLAY_STRATEGY_REGISTRY._registry["_test_custom"] = custom_fn
        try:
            ds = self._ds(buffer_replay_strategy="_test_custom")
            self.assertIs(ds.buffer_replay_strategy, custom_fn)
        finally:
            del BUFFER_REPLAY_STRATEGY_REGISTRY._registry["_test_custom"]

    def test_unknown_filter_name_raises_key_error(self):
        with self.assertRaises(KeyError):
            self._ds(buffer_replay_strategy="no_such_filter_xyz")


# ===========================================================================
# RolloutDataSourceWithBuffer._get_from_buffer
# ===========================================================================

class TestGetFromBuffer(unittest.TestCase):

    def setUp(self):
        self.tmpdir = self.enterContext(tempfile.TemporaryDirectory())

    def _ds(self, n_trajs=2):
        path = os.path.join(self.tmpdir, "ds.jsonl")
        _write_jsonl(path, [{"text": f"p{i}"} for i in range(5)])
        cfg = _make_config(prompt_data_path=path, num_trajectories_per_prompt=n_trajs)
        return RolloutDataSourceWithBuffer(cfg, cfg)

    def test_empty_buffer_returns_empty(self):
        ds = self._ds()
        self.assertEqual(ds._get_from_buffer(5), [])

    def test_zero_num_returns_empty(self):
        ds = self._ds()
        ds.buffer.append(_traj_group(2))
        self.assertEqual(ds._get_from_buffer(0), [])

    def test_calls_buffer_replay_strategy(self):
        ds = self._ds()
        ds.buffer.append(_traj_group(2))
        mock_filter = MagicMock(return_value=[ds.buffer[0]])
        ds.buffer_replay_strategy = mock_filter
        ds._get_from_buffer(1)
        mock_filter.assert_called_once_with(ds.config, ds.step, ds.buffer, 1)

    def test_returns_filter_result(self):
        ds = self._ds()
        expected = [_traj_group(2)]
        ds.buffer_replay_strategy = MagicMock(return_value=expected)
        ds.buffer.append(_traj_group(2))
        result = ds._get_from_buffer(1)
        self.assertIs(result, expected)


# ===========================================================================
# RolloutDataSourceWithBuffer.add
# ===========================================================================

class TestAddBuffer(unittest.TestCase):

    def setUp(self):
        self.tmpdir = self.enterContext(tempfile.TemporaryDirectory())

    def _ds(self, n_trajs=2):
        path = os.path.join(self.tmpdir, "ds.jsonl")
        _write_jsonl(path, [{"text": "x"}])
        cfg = _make_config(prompt_data_path=path, num_trajectories_per_prompt=n_trajs)
        return RolloutDataSourceWithBuffer(cfg, cfg)

    def test_empty_input_is_noop(self):
        ds = self._ds()
        ds.add([])
        self.assertEqual(len(ds.buffer), 0)

    def test_appends_single_group(self):
        ds = self._ds(n_trajs=2)
        ds.add([_traj_group(2)])
        self.assertEqual(len(ds.buffer), 1)

    def test_appends_multiple_groups(self):
        ds = self._ds(n_trajs=2)
        ds.add([_traj_group(2), _traj_group(2), _traj_group(2)])
        self.assertEqual(len(ds.buffer), 3)

    def test_wrong_outer_type_raises_assertion(self):
        ds = self._ds()
        with self.assertRaises(AssertionError):
            ds.add(Trajectory(trajectory_id="t0", prompt_id="p0"))  # not a list

    def test_wrong_inner_type_raises_assertion(self):
        """Inner elements must be TrajectoryGroup, not bare Trajectory."""
        ds = self._ds()
        with self.assertRaises(AssertionError):
            ds.add([Trajectory(trajectory_id="t0", prompt_id="p0")])  # not TrajectoryGroup

    def test_wrong_group_size_raises_assertion(self):
        ds = self._ds(n_trajs=2)
        with self.assertRaises(AssertionError):
            ds.add([_traj_group(3)])  # expected 2, got 3

    def test_groups_stored_by_reference(self):
        ds = self._ds(n_trajs=2)
        group = _traj_group(2)
        ds.add([group])
        self.assertIs(ds.buffer[0], group)

    def test_buffer_contains_trajectory_group_instances(self):
        ds = self._ds(n_trajs=2)
        ds.add([_traj_group(2)])
        self.assertIsInstance(ds.buffer[0], TrajectoryGroup)


# ===========================================================================
# RolloutDataSourceWithBuffer.get
# ===========================================================================

class TestBufferGet(unittest.TestCase):

    def setUp(self):
        self.tmpdir = self.enterContext(tempfile.TemporaryDirectory())

    def _ds(self, n_rows=8, n_trajs=2):
        path = os.path.join(self.tmpdir, "ds.jsonl")
        _write_jsonl(path, [{"text": f"p{i}"} for i in range(n_rows)])
        cfg = _make_config(prompt_data_path=path, num_trajectories_per_prompt=n_trajs)
        return RolloutDataSourceWithBuffer(cfg, cfg)

    def test_records_step(self):
        ds = self._ds()
        ds.get(1, step=7)
        self.assertEqual(ds.step, 7)

    def test_fully_from_buffer(self):
        ds = self._ds(n_trajs=2)
        for _ in range(3):
            ds.add([_traj_group(2)])
        result = ds.get(3, step=0)
        self.assertEqual(len(result), 3)
        self.assertEqual(ds.get_buffer_length(), 0)

    def test_fully_from_dataset_when_buffer_empty(self):
        ds = self._ds(n_rows=5, n_trajs=1)
        result = ds.get(3, step=0)
        self.assertEqual(len(result), 3)

    def test_mixed_buffer_and_dataset(self):
        ds = self._ds(n_rows=5, n_trajs=2)
        ds.add([_traj_group(2)])   # 1 in buffer
        result = ds.get(3, step=0)
        self.assertEqual(len(result), 3)
        self.assertEqual(ds.get_buffer_length(), 0)

    def test_buffer_groups_appear_first(self):
        ds = self._ds(n_rows=5, n_trajs=1)
        sentinel = Trajectory(trajectory_id="t0", prompt_id="p0", prompt="sentinel")
        ds.add([TrajectoryGroup(prompt_id="p0", trajectories=[sentinel])])
        result = ds.get(2, step=0)
        self.assertIs(result[0].trajectories[0], sentinel)

    def test_returns_early_when_buffer_satisfies_request(self):
        """When buffer has enough, parent get must NOT be called."""
        ds = self._ds(n_trajs=1)
        for _ in range(3):
            ds.add([_traj_group(1)])
        with patch.object(RolloutDataSource, "get") as mock_parent:
            ds.get(3, step=0)
        mock_parent.assert_not_called()

    def test_result_contains_trajectory_group_instances(self):
        ds = self._ds(n_rows=5, n_trajs=2)
        result = ds.get(2, step=0)
        for g in result:
            self.assertIsInstance(g, TrajectoryGroup)


# ===========================================================================
# RolloutDataSourceWithBuffer.get_buffer_length
# ===========================================================================

class TestGetBufferLength(unittest.TestCase):

    def setUp(self):
        self.tmpdir = self.enterContext(tempfile.TemporaryDirectory())

    def _ds(self):
        path = os.path.join(self.tmpdir, "ds.jsonl")
        _write_jsonl(path, [{"text": "x"}])
        cfg = _make_config(prompt_data_path=path, num_trajectories_per_prompt=1)
        return RolloutDataSourceWithBuffer(cfg, cfg)

    def test_zero_on_init(self):
        self.assertEqual(self._ds().get_buffer_length(), 0)

    def test_increases_after_add(self):
        ds = self._ds()
        ds.add([_traj_group(1)])
        ds.add([_traj_group(1)])
        self.assertEqual(ds.get_buffer_length(), 2)

    def test_decreases_after_get(self):
        ds = self._ds()
        ds.add([_traj_group(1), _traj_group(1)])
        ds.get(1, step=0)
        self.assertEqual(ds.get_buffer_length(), 1)


# ===========================================================================
# RolloutDataSourceWithBuffer.save / load — buffered prompt persistence
# ===========================================================================

class TestBufferedSaveLoad(unittest.TestCase):
    """Cover the extra state_dict payload added by RolloutDataSourceWithBuffer.

    The parent 5-counter payload is already covered by TestSave / TestLoad; this
    test class focuses on the ``unused_prompt_list`` round-trip plus the
    trajectory_count increment on restore.
    """

    def setUp(self):
        self.tmpdir = self.enterContext(tempfile.TemporaryDirectory())
        self._save_patcher = patch("torch.save")
        self._load_patcher = patch("torch.load")
        self.mock_save = self._save_patcher.start()
        self.mock_load = self._load_patcher.start()

    def tearDown(self):
        self._save_patcher.stop()
        self._load_patcher.stop()

    def _ds(self, num_traj=3):
        path = os.path.join(self.tmpdir, "ds.jsonl")
        _write_jsonl(path, [{"text": f"p{i}"} for i in range(3)])
        cfg = _make_config(
            prompt_data_path=path,
            num_trajectories_per_prompt=num_traj,
        )
        return RolloutDataSourceWithBuffer(cfg, cfg)

    @staticmethod
    def _group(prompt_id, prompt, num_traj):
        trajs = [
            Trajectory(
                trajectory_id=f"{prompt_id}_traj{i}",
                prompt_id=prompt_id,
                prompt=prompt,
                label={"gold": "a"},
                metadata={"src": "test"},
            )
            for i in range(num_traj)
        ]
        return TrajectoryGroup(prompt_id=prompt_id, trajectories=trajs)

    def test_save_captures_unused_prompt_list(self):
        ds = self._ds(num_traj=3)
        ds.buffer = [
            self._group("pid_A", "prompt-A", 3),
            self._group("pid_B", "prompt-B", 3),
        ]
        self.mock_save.reset_mock()
        with patch("os.makedirs"):
            ds.save(7)
        state = self.mock_save.call_args[0][0]
        self.assertEqual(state["prompt_data_path"], ds.dataset_config.prompt_data_path)
        self.assertEqual(len(state["unused_prompt_list"]), 2)
        self.assertEqual(state["unused_prompt_list"][0]["prompt_id"], "pid_A")
        self.assertEqual(state["unused_prompt_list"][0]["prompt"], "prompt-A")
        self.assertEqual(state["unused_prompt_list"][0]["label"], {"gold": "a"})
        self.assertEqual(state["unused_prompt_list"][0]["metadata"], {"src": "test"})
        self.assertEqual(state["unused_prompt_list"][1]["prompt_id"], "pid_B")

    def test_save_empty_buffer_yields_empty_unused_list(self):
        ds = self._ds()
        self.mock_save.reset_mock()
        with patch("os.makedirs"):
            ds.save(1)
        state = self.mock_save.call_args[0][0]
        self.assertEqual(state["unused_prompt_list"], [])

    def test_save_includes_additional_groups_without_mutating_buffer(self):
        ds = self._ds(num_traj=3)
        buffered_group = self._group("pid_buffer", "prompt-buffer", 3)
        pipeline_group = self._group("pid_pipeline", "prompt-pipeline", 3)
        ds.buffer = [buffered_group]
        self.mock_save.reset_mock()

        with patch("os.makedirs"):
            ds.save(7, additional_groups=[pipeline_group])

        state = self.mock_save.call_args[0][0]
        self.assertEqual(
            [item["prompt_id"] for item in state["unused_prompt_list"]],
            ["pid_buffer", "pid_pipeline"],
        )
        self.assertEqual(ds.buffer, [buffered_group])

    def test_load_reconstructs_buffer_and_increments_trajectory_count(self):
        ds = self._ds(num_traj=3)
        state = {
            "prompt_data_path": ds.dataset_config.prompt_data_path,
            "epoch_id": 0, "step": 5,
            "prompt_offset": 4, "prompt_index": 2, "trajectory_count": 100,
            "unused_prompt_list": [
                {"prompt_id": "pid_A", "prompt": "prompt-A",
                 "label": {"gold": "a"}, "metadata": {"src": "A"}},
                {"prompt_id": "pid_B", "prompt": "prompt-B",
                 "label": {"gold": "b"}, "metadata": {"src": "B"}},
            ],
        }
        self.mock_load.reset_mock()
        self.mock_load.return_value = state
        with patch(
            "coda.data_factory.data_source.glob.glob",
            return_value=["/tmp/ckpt/train_step_5/data_source/global_dataset_state_dict_ds0.pt"],
        ):
            ds.load(5)

        # 2 groups × 3 trajectories/group = 6 trajectories restored.
        self.assertEqual(len(ds.buffer), 2)
        for g in ds.buffer:
            self.assertEqual(len(g.trajectories), 3)
        # trajectory_count moves from 100 by exactly 6.
        self.assertEqual(ds.trajectory_count, 106)

        # trajectory_id count suffix strictly increases from the base counter.
        counts = [
            int(t.trajectory_id.rsplit("count", 1)[1])
            for g in ds.buffer for t in g.trajectories
        ]
        self.assertEqual(counts, [100, 101, 102, 103, 104, 105])

        # prompt_id is preserved verbatim on both group and inner trajectories.
        self.assertEqual(ds.buffer[0].prompt_id, "pid_A")
        for t in ds.buffer[0].trajectories:
            self.assertEqual(t.prompt_id, "pid_A")
            self.assertEqual(t.prompt, "prompt-A")
            self.assertEqual(t.label, {"gold": "a"})
            self.assertEqual(t.metadata, {"src": "A"})

    def test_load_missing_ckpt_returns_none_and_leaves_buffer_untouched(self):
        ds = self._ds()
        ds.buffer = [self._group("pre-existing", "x", 3)]
        self.mock_load.reset_mock()
        with patch("coda.data_factory.data_source.glob.glob", return_value=[]):
            result = ds.load(9)
        self.assertIsNone(result)
        self.mock_load.assert_not_called()
        # Pre-existing buffer entries are not clobbered.
        self.assertEqual(len(ds.buffer), 1)
        self.assertEqual(ds.buffer[0].prompt_id, "pre-existing")

    def test_load_without_unused_prompt_list_leaves_buffer_empty(self):
        # Backwards-compat: legacy state_dicts had no unused_prompt_list.
        ds = self._ds()
        state = {
            "prompt_data_path": ds.dataset_config.prompt_data_path,
            "epoch_id": 0, "step": 0,
            "prompt_offset": 0, "prompt_index": 0, "trajectory_count": 50,
        }
        self.mock_load.reset_mock()
        self.mock_load.return_value = state
        with patch(
            "coda.data_factory.data_source.glob.glob",
            return_value=["/tmp/ckpt/train_step_0/data_source/global_dataset_state_dict_ds0.pt"],
        ):
            ds.load(0)
        self.assertEqual(len(ds.buffer), 0)
        self.assertEqual(ds.trajectory_count, 50)


if __name__ == "__main__":
    unittest.main()
