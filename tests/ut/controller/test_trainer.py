"""Unit tests for Trainer trajectory processing methods.

Run: python -m pytest tests/ut/controller/test_trainer.py -v
"""
import sys
from unittest.mock import patch, MagicMock, Mock

import pytest
from omegaconf import OmegaConf

# coda.controller.trainer pulls in coda.controller.teacher_manager, which imports
# Megatron-Core. Skipping here (before setup_module installs the sys.modules
# mocks) keeps a failed import from leaving those mocks behind for other files.
pytest.importorskip("megatron", reason="Megatron-Core is not installed")


_MOCKED_MODULE_KEYS = [
    "coda.resource_scheduler",
    "coda.controller.rollout_sampler",
    "coda.data_factory.data_source",
    "coda.agentflow.agent_flow",
    "coda.controller.train_manager",
    "coda.controller.rollout_manager",
    "coda.transfer_mesh.channel",
    "coda.data_factory.data_processor",
    "coda.utils.tracking",
    "ray",
    "coda.controller.trainer",
]
_ORIGINAL_SYS_MODULES = {}

# Populated lazily by setup_module so that mocks only apply while these tests run.
Trainer = None
Mode = None
Trajectory = None
TrajectoryGroup = None


def _install_mocks():
    sys.modules["coda.resource_scheduler"] = Mock()
    sys.modules["coda.resource_scheduler"].ResourceScheduler = type(
        "ResourceScheduler", (), {
            "__init__": lambda self, x: None,
            "get_gloo_master_address": lambda self: ("127.0.0.1", 8080),
        }
    )
    sys.modules["coda.controller.rollout_sampler"] = Mock()
    sys.modules["coda.controller.rollout_sampler"].RolloutSampler = type(
        "RolloutSampler", (), {"__init__": lambda self, *args: None}
    )
    sys.modules["coda.data_factory.data_source"] = Mock()
    sys.modules["coda.data_factory.data_source"].RolloutDataSourceWithBuffer = type(
        "RolloutDataSourceWithBuffer", (), {"__init__": lambda self, *args, **kwargs: None}
    )
    sys.modules["coda.agentflow.agent_flow"] = Mock()
    sys.modules["coda.agentflow.agent_flow"].AgentFlow = type(
        "AgentFlow", (), {"__init__": lambda self, *args: None}
    )
    sys.modules["coda.controller.train_manager"] = Mock()
    sys.modules["coda.controller.train_manager"].TrainManager = type(
        "TrainManager", (), {
            "__init__": lambda self, *args, **kwargs: None,
            "async_init": lambda self: None,
            "async_offload": lambda self: None,
            "async_onload": lambda self: None,
        }
    )
    sys.modules["coda.controller.rollout_manager"] = Mock()
    sys.modules["coda.controller.rollout_manager"].RolloutManager = type(
        "RolloutManager", (), {
            "__init__": lambda self, *args, **kwargs: None,
            "offload": lambda self: None,
            "onload": lambda self: None,
        }
    )
    sys.modules["coda.transfer_mesh.channel"] = Mock()
    # coda.utils.checkpoint_utils is deliberately left real: it is a pure os.path
    # leaf module, and _validate_config's ref/teacher path checks go through it.
    sys.modules["coda.data_factory.data_processor"] = Mock()
    sys.modules["coda.data_factory.data_processor"].split_traj_group_by_dp = (
        lambda x, y, z=1: [[g] for g in x]
    )
    sys.modules["coda.data_factory.data_processor"].put_dp_shards_to_ray = (
        lambda x, y=1: [f"ray_ref_{i}" for i in range(len(x))]
    )
    mock_tracking = Mock()
    mock_tracking.configure_tracking = lambda x: "test_run_id"
    mock_tracking.time_marker = MagicMock()
    mock_tracking.TimeMarkerAcc = MagicMock()
    sys.modules["coda.utils.tracking"] = mock_tracking

    sys.modules["ray"] = Mock()
    sys.modules["ray"].is_initialized = lambda: True
    sys.modules["ray"].get = lambda x: x


def setup_module(module):
    """Install sys.modules mocks and import Trainer at test-run time.

    Doing this here (not at import time) prevents the mocks from leaking to
    sibling test files that pytest collects before running us. If the import
    fails, the mocks are rolled back before propagating, otherwise every later
    test that touches ray or the mocked coda modules would see a Mock.
    """
    global Trainer, Mode, Trajectory, TrajectoryGroup

    for k in _MOCKED_MODULE_KEYS:
        _ORIGINAL_SYS_MODULES[k] = sys.modules.get(k)
    # Force fresh import of coda.controller.trainer against the mocked deps.
    sys.modules.pop("coda.controller.trainer", None)

    _install_mocks()

    try:
        from coda.agentflow.trajectory_store import Trajectory as _T, TrajectoryGroup as _TG
        from coda.controller.trainer import Trainer as _Trainer, Mode as _Mode
    except BaseException:
        teardown_module(module)
        raise

    Trajectory = _T
    TrajectoryGroup = _TG
    Trainer = _Trainer
    Mode = _Mode


def teardown_module(module):
    """Restore sys.modules entries so other test files see real modules."""
    for k, v in _ORIGINAL_SYS_MODULES.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


def make_minimal_config(**overrides):
    """Build a minimal config for Trainer testing."""
    base = {
        "colocate": False,
        "checkpoint_path": "/tmp/fake_ckpt",
        "run_mode": "default",
        "fully_async": {
            "enable": False,
            "sliding_window": "no-window",
            "stale_steps": 0,
        },
        "trainer": {
            "num_nodes": 1,
            "num_gpus_per_node": 1,
            "save_freq": 1,
            "project_name": "test_project",
            "experiment_name": "test_experiment",
            "tracking_backend": "console",
            # 4 prompts x 2 trajectories = 8 trajectories per step. _validate_config
            # requires 8 % mini_batch_size == 0 and, with num_mini_batch = 8/4 = 2,
            # also num_prompts_per_step % (dp_size * num_mini_batch) == 0.
            "mini_batch_size": 4,
        },
        "rollout": {
            "colocate": False,
            "num_gpus": 1,
            "total_steps": 10,
            "partial": False,
            "mask_offpolicy_in_partial_rollout": False,
            "data_source": {
                "num_samples_per_prompt": 2,
            },
            "sampler": {
                "num_prompt_per_step": 4,
                "num_oversample": 0,
            },
            "eval": {
                "interval": 0,
            },
            "use_fault_tolerance": False,
        },
        "data_sources": [
            {
                "dataset": {"eval_prompt_data_path": None},
                "num_trajectories_per_prompt": 2,
                "num_prompts_per_step": 4,
            }
        ],
        "rollout_sampler": {
            "timeout": 60,
        },
        "algorithm": {
            "m2po": {
                "enable": False,
            },
            "ref_kl": {
                "enable": False,
            },
        },
        "ref_dist_ckpt_path": None,
        "ref_hf_model_path": None,
        "opd": {
            "enable": False,
            "teachers": [],
        },
        "megatron": {
            "model": {
                "tensor_model_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
                "context_parallel_size": 1,
            },
        },
        "tracking": {
            "run_id": None,
        },
    }
    cfg = OmegaConf.create(base)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))
    return cfg


def create_test_trajectory_group(
    prompt_id: str = "test_prompt_0",
    num_trajectories: int = 2,
):
    """Create a test TrajectoryGroup for testing."""
    trajectories = []
    for i in range(num_trajectories):
        traj = Trajectory(
            trajectory_id=f"{prompt_id}_traj_{i}",
            prompt_id=prompt_id,
            prompt="Test prompt",
            tokens=[1, 2, 3, 4, 5],
            loss_masks=[0, 1, 1, 1, 0],
            rollout_log_probs=[0.0, -0.5, -0.3, -0.2, 0.0],
            reward=0.5,
            token_rewards=[0.0, 0.0, 0.0, 0.0, 0.5],
            status="completed",
        )
        trajectories.append(traj)
    return TrajectoryGroup(prompt_id=prompt_id, trajectories=trajectories)


class TestTrainerValidateConfig:
    """Tests for Trainer._validate_config."""

    @staticmethod
    def _trainer(config):
        """A bare Trainer carrying only what _validate_config reads."""
        trainer = object.__new__(Trainer)
        trainer.config = config
        trainer.dp_size = 1
        return trainer

    def test_mask_offpolicy_requires_partial_rollout(self):
        config = make_minimal_config(
            rollout={
                "partial": False,
                "mask_offpolicy_in_partial_rollout": True,
            }
        )

        with pytest.raises(ValueError, match="requires rollout.partial=true"):
            self._trainer(config)._validate_config()

    def test_mask_offpolicy_allowed_with_partial_rollout(self):
        config = make_minimal_config(
            rollout={
                "partial": True,
                "mask_offpolicy_in_partial_rollout": True,
            }
        )
        trainer = self._trainer(config)

        trainer._validate_config()

        assert trainer.config.rollout.partial is True
        assert trainer.config.rollout.mask_offpolicy_in_partial_rollout is True

    def test_total_trajectories_must_divide_mini_batch_size(self):
        """8 trajectories with mini_batch_size=3 leaves a partial mini-batch."""
        config = make_minimal_config(trainer={"mini_batch_size": 3})

        with pytest.raises(
            ValueError, match=r"Total trajectories \(8\) is not divisible by"
        ):
            self._trainer(config)._validate_config()

    def test_prompts_per_step_must_divide_across_dp_and_mini_batches(self):
        """num_prompts_per_step must split evenly over dp_size * num_mini_batch."""
        config = make_minimal_config()
        trainer = self._trainer(config)
        trainer.dp_size = 3  # 4 prompts cannot split over 3 * 2 = 6

        with pytest.raises(
            ValueError, match=r"num_prompts_per_step \(4\) is not divisible by"
        ):
            trainer._validate_config()


class TestTrainerValidateReadOnlyCkptPaths:
    """ref / teacher dist checkpoint dirs are checked before GPUs are reserved.

    _validate_config runs on the driver ahead of ResourceScheduler, so a typo
    costs seconds rather than failing once every train worker has built its model.
    """

    @staticmethod
    def _validate(config):
        trainer = object.__new__(Trainer)
        trainer.config = config
        trainer.dp_size = 1
        trainer._validate_config()

    @staticmethod
    def _dist_ckpt(tmp_path):
        d = tmp_path / "train_step_100" / "dist_ckpt"
        d.mkdir(parents=True)
        (d / "metadata.json").write_text('{"sharded_backend": "torch_dist"}')
        return str(d)

    def test_bad_ref_path_raises(self, tmp_path):
        config = make_minimal_config(
            algorithm={"ref_kl": {"enable": True}},
            ref_dist_ckpt_path=str(tmp_path / "nope"),
        )
        with pytest.raises(ValueError, match="ref_dist_ckpt_path"):
            self._validate(config)

    def test_good_ref_path_passes(self, tmp_path):
        config = make_minimal_config(
            algorithm={"ref_kl": {"enable": True}},
            ref_dist_ckpt_path=self._dist_ckpt(tmp_path),
        )
        self._validate(config)

    def test_ref_path_ignored_when_ref_kl_disabled(self, tmp_path):
        """A stale path costs nothing while the feature reading it is off."""
        config = make_minimal_config(ref_dist_ckpt_path=str(tmp_path / "nope"))
        self._validate(config)

    def test_bad_teacher_path_raises(self, tmp_path):
        config = make_minimal_config(opd={
            "enable": True,
            "teachers": [
                {"name": "a", "hf_path": "/hf/a"},
                {"name": "b", "hf_path": "/hf/b", "dist_ckpt_path": str(tmp_path / "nope")},
            ],
        })
        with pytest.raises(ValueError, match=r"opd\.teachers\[1\]\.dist_ckpt_path"):
            self._validate(config)

    def test_good_teacher_path_passes(self, tmp_path):
        config = make_minimal_config(opd={
            "enable": True,
            "teachers": [
                {"name": "a", "hf_path": "/hf/a", "dist_ckpt_path": self._dist_ckpt(tmp_path)},
            ],
        })
        self._validate(config)

    def test_teacher_path_ignored_when_opd_disabled(self, tmp_path):
        config = make_minimal_config(opd={
            "enable": False,
            "teachers": [
                {"name": "a", "hf_path": "/hf/a", "dist_ckpt_path": str(tmp_path / "nope")},
            ],
        })
        self._validate(config)


class TestWeightVersionHelpers:
    """Tests for trainer-side rollout weight version helpers."""

    def test_post_train_weight_version_advances_from_current_version(self):
        trainer = object.__new__(Trainer)

        assert trainer._post_train_weight_version(7) == 8


class TestBatchDiskRoundTrip:
    """_save_batch_to_disk / _load_batch_from_disk must round-trip a batch.

    Both only read ``config.rollout_data_path``, so a bare Trainer is enough —
    no need to drive the full constructor.
    """

    @staticmethod
    def _trainer(tmp_path):
        trainer = object.__new__(Trainer)
        trainer.config = make_minimal_config()
        trainer.config.rollout_data_path = str(tmp_path / "rollout_data")
        return trainer

    def test_round_trip_preserves_shape_and_content(self, tmp_path):
        trainer = self._trainer(tmp_path)

        batch = [
            [create_test_trajectory_group("prompt_0")],
            [create_test_trajectory_group("prompt_1", num_trajectories=3)],
        ]
        trainer._save_batch_to_disk(batch, step=20)

        assert (tmp_path / "rollout_data" / "step_20.pt").exists()

        loaded = trainer._load_batch_from_disk(step=20)

        assert [len(shard) for shard in loaded] == [1, 1]
        assert [shard[0].prompt_id for shard in loaded] == ["prompt_0", "prompt_1"]
        assert isinstance(loaded[0][0], TrajectoryGroup)
        assert len(loaded[1][0].trajectories) == 3
        # Trajectory payload must survive serialization, not just the group shell.
        assert loaded[0][0].trajectories[0].tokens == [1, 2, 3, 4, 5]
        assert loaded[0][0].trajectories[0].reward == 0.5

    def test_save_is_idempotent_when_directory_already_exists(self, tmp_path):
        trainer = self._trainer(tmp_path)
        (tmp_path / "rollout_data").mkdir()

        batch = [[create_test_trajectory_group("prompt_0")]]
        trainer._save_batch_to_disk(batch, step=10)

        assert (tmp_path / "rollout_data" / "step_10.pt").exists()

    def test_load_missing_step_raises_with_actionable_message(self, tmp_path):
        trainer = self._trainer(tmp_path)

        with pytest.raises(FileNotFoundError, match="Ensure ROLLOUT_ONLY mode was executed"):
            trainer._load_batch_from_disk(step=7)
