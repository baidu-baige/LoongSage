"""training entry """
import asyncio
import threading
import ray
import hydra
import os
import logging
import torch
from enum import Enum
from omegaconf import DictConfig, OmegaConf
from typing import Optional

from coda.controller.rollout_sampler import RolloutSampler
from coda.resource_scheduler import ResourceScheduler
from coda.data_factory.data_source import RolloutDataSourceWithBuffer
from coda.agentflow.agent_flow import AgentFlow
from coda.agentflow.trajectory_store import TrajectoryGroup
from coda.data_factory.data_processor import (
    split_traj_group_by_dp,
    put_dp_shards_to_ray,
)
from coda.controller.train_manager import TrainManager
from coda.controller.rollout_manager import RolloutManager
from coda.controller.teacher_manager import TeacherManager
from coda.utils.channel_helper import ChannelMeta
from coda.utils.checkpoint_utils import get_tracker_file, resolve_dist_ckpt_dir
from coda.utils.tracking import configure_tracking, time_marker, TimeMarkerAcc, track
from coda.utils import logging_utils

logger = logging.getLogger(__name__)


def normalize_data_sources(config: DictConfig) -> DictConfig:
    """Normalize config to ensure data_sources is a properly merged list.

    Each entry in data_sources inherits missing keys from data_source.
    """
    defaults = config.data_source

    merged = []
    for ds in config.data_sources:
        merged_ds = OmegaConf.merge(defaults, ds)
        merged.append(merged_ds)

    config.data_sources = merged
    logger.info("Normalized %d data source(s)", len(merged))
    return config


class Mode(str, Enum):
    """Training mode enum."""
    DEFAULT = "default"
    TRAIN_ONLY = "train-only"
    ROLLOUT_ONLY = "rollout-only"


class Trainer:
    """Main training orchestrator with multi-data-source support."""

    def __init__(self, config: DictConfig):
        self.config = config
        train_world_size = self.config.trainer.num_gpus_per_node * self.config.trainer.num_nodes
        # num_gpu_per_dp = tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
        # https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py
        num_gpu_per_dp = self.config.megatron.model.tensor_model_parallel_size * \
            self.config.megatron.model.pipeline_model_parallel_size * \
            self.config.megatron.model.context_parallel_size
        if num_gpu_per_dp == 0:
            raise ValueError(
                f"Invalid model parallelism config: product of "
                f"tensor_parallel({self.config.megatron.model.tensor_model_parallel_size}), "
                f"pipeline_parallel({self.config.megatron.model.pipeline_model_parallel_size}), "
                f"context_parallel({self.config.megatron.model.context_parallel_size}) must be > 0"
            )
        if train_world_size % num_gpu_per_dp != 0:
            raise ValueError(
                f"trainer GPUs ({train_world_size}) must be divisible by "
                f"TP*PP*CP ({num_gpu_per_dp})"
            )
        self.dp_size = train_world_size // num_gpu_per_dp
        self._validate_config()
        self.is_colocated = self.config.colocate
        self.is_fully_async = self.config.fully_async.enable
        self.num_mini_batches = 0
        if self.is_fully_async:
            ds = self.config.data_sources[0]
            self.num_mini_batches = (
                int(ds.num_prompts_per_step) * int(ds.num_trajectories_per_prompt)
                // self.config.trainer.mini_batch_size
            )
        self._init_tracking()

        self.scheduler = ResourceScheduler(self.config)

        # Create one RolloutDataSourceWithBuffer per data source unit
        self.datasources: list[RolloutDataSourceWithBuffer] = []
        for ds_index, ds_cfg in enumerate(self.config.data_sources):
            ds = RolloutDataSourceWithBuffer(ds_cfg, self.config, ds_index=ds_index)
            self.datasources.append(ds)

        agent_flow = AgentFlow(self.config)

        self.train_manager = TrainManager(self.config, self.scheduler)
        ray.get(self.train_manager.async_init())

        if self.is_colocated:
            logger.info("[init] colocated mode: offloading train workers ...")
            self.train_manager.offload()
            logger.info("[init] train workers offloaded")

        self.teacher_manager = None
        if self.config.opd.enable:
            self.teacher_manager = TeacherManager(self.config, self.scheduler)
            ray.get(self.teacher_manager.async_init())
            if self.is_colocated:
                logger.info("[init] colocated mode: offloading teacher workers ...")
                self.teacher_manager.offload()
                logger.info("[init] teacher workers offloaded")

        # train-only mode replays rollout data from disk and never runs
        # inference, so skip creating the rollout engines (which would launch
        # the whole sglang cluster and load weights).
        self.rollout_manager = None
        self.rollout_sampler = None
        if Mode(self.config.run_mode) != Mode.TRAIN_ONLY:
            self.rollout_manager = RolloutManager(self.config, self.scheduler)
            self.rollout_sampler = RolloutSampler(self.config, self.datasources, agent_flow)
            self.channel_master_addr, self.channel_gloo_port = self.scheduler.get_gloo_master_address()
            self.rollout_manager.clear_num_new_engines()

    def _validate_config(self) -> None:
        if self.config.fully_async.enable:
            if self.config.colocate:
                raise ValueError("fully_async and colocate cannot be enabled at the same time")
            if Mode(self.config.run_mode) in (Mode.ROLLOUT_ONLY, Mode.TRAIN_ONLY):
                raise ValueError("fully_async cannot be used with rollout_only or train_only mode")
            if self.config.rollout.sampler.num_oversample != 0:
                raise ValueError("in fully async mode, rollout.sampler.num_oversample must be 0")
            if self.config.fully_async.stale_steps < 0:
                raise ValueError(f"fully_async.stale_steps ({self.config.fully_async.stale_steps}) must be >= 0")

            # Fully-async mode currently supports only a single data source.
            if len(self.config.data_sources) != 1:
                raise ValueError(f"fully_async supports only one data source, got {len(self.config.data_sources)}")

        partial_rollout = self.config.rollout.partial
        mask_offpolicy = self.config.rollout.mask_offpolicy_in_partial_rollout
        if mask_offpolicy and not partial_rollout:
            raise ValueError(
                "rollout.mask_offpolicy_in_partial_rollout requires rollout.partial=true"
            )

        # Validate multi-datasource batch-size divisibility constraint.
        data_sources = self.config.data_sources
        mini_batch_size = self.config.trainer.mini_batch_size
        total_trajs = sum(
            int(ds.num_prompts_per_step) * int(ds.num_trajectories_per_prompt)
            for ds in data_sources
        )
        
        if total_trajs <= 0:
            raise ValueError(
                f"Total trajectories ({total_trajs}) must be > 0. "
            )
        if mini_batch_size <= 0:
            raise ValueError(
                f"trainer.mini_batch_size ({mini_batch_size}) must be > 0."
            )
        if total_trajs % mini_batch_size != 0:
            raise ValueError(
                f"Total trajectories ({total_trajs}) is not divisible by "
                f"trainer.mini_batch_size ({mini_batch_size}). "
            )
        
        num_mini_batch = total_trajs // mini_batch_size
        for idx, ds in enumerate(data_sources):
            n_prompts = int(ds.num_prompts_per_step)
            if n_prompts % (self.dp_size * num_mini_batch) != 0:
                raise ValueError(
                    f"data_sources[{idx}] num_prompts_per_step ({n_prompts}) is not divisible by "
                    f"(dp_size * num_mini_batch) = ({self.dp_size} * {num_mini_batch}) = "
                    f"{self.dp_size * num_mini_batch}. "
                )

        if int(self.config.rollout.eval.interval) > 0 and not any(
            ds.dataset.eval_prompt_data_path for ds in self.config.data_sources
        ):
            raise ValueError(
                "rollout.eval.interval > 0 requires at least one dataset with eval_prompt_data_path"
            )

        self._validate_read_only_ckpt_paths()

    def _validate_read_only_ckpt_paths(self) -> None:
        """Fail fast on unusable ref / teacher dist checkpoint dirs.

        The workers check these too, but only after the placement group is
        reserved and every model is built. Checking on the driver first turns a
        typo into a seconds-long failure.
        """
        if self.config.algorithm.ref_kl.enable:
            resolve_dist_ckpt_dir(self.config.ref_dist_ckpt_path, "ref_dist_ckpt_path")
        if self.config.opd.enable:
            for idx, teacher in enumerate(self.config.opd.teachers):
                resolve_dist_ckpt_dir(
                    teacher.get("dist_ckpt_path"), f"opd.teachers[{idx}].dist_ckpt_path"
                )

    def _init_tracking(self) -> None:
        """Initialize tracking system.

        ``configure_tracking`` writes the per-backend run ids back into
        ``self.config.tracking``, so the workers that later receive this config
        attach to these runs instead of creating their own.
        """
        configure_tracking(self.config)

    def _init_channel_meta(self) -> ChannelMeta:
        """Initialize channel metadata for weight synchronization.

        Returns:
            The ChannelMeta for this weight-sync round.
        """
        # this world_size is for gloo ring; it spans the trainer and rollout processes
        train_world_size = self.config.trainer.num_gpus_per_node * self.config.trainer.num_nodes
        rollout_world_size = sum(
            replica.num_nodes * self.config.rollout.num_gpus_per_node
            for _, replica in self.config.rollout.sglang_replicas.items()
        )
        world_size = train_world_size + rollout_world_size
        logger.info("[channel_meta] train_world_size=%d rollout_world_size=%d world_size=%d",
                    train_world_size, rollout_world_size, world_size)

        if self.config.rollout.use_fault_tolerance:
            logger.info("[channel_meta] recovering faulty engines ...")
            self.rollout_manager.recover_faulty_engines()

        recreate = self.rollout_manager.num_new_engines > 0
        if recreate:
            logger.info("[channel_meta] new engines detected, refreshing gloo master address ...")
            self.channel_master_addr, self.channel_gloo_port \
            = self.scheduler.get_gloo_master_address()
            self.rollout_manager.clear_num_new_engines()

        meta = ChannelMeta(
                world_size=world_size,
                master_addr=self.channel_master_addr,
                gloo_port=self.channel_gloo_port,
                train_world_size=train_world_size,
                recreate=recreate,
            )
        logger.info("[channel_meta] ChannelMeta: %s", meta)
        return meta


    def _update_weights(self, meta: ChannelMeta, weight_version: int) -> None:
        """Update weights for both train and rollout managers concurrently.

        This method fires both update requests and waits for all of them to complete.

        Args:
            meta: ChannelMeta for 'train' and 'rollout' manager
            weight_version: Rollout-visible model version label.
        """
        if not isinstance(weight_version, int) or weight_version < 0:
            raise ValueError("weight_version must be a non-negative int")

        train_update_ref = self.train_manager.async_update_weights(meta)
        rollout_update_ref = self.rollout_manager.async_update_weights(meta, weight_version=weight_version)

        all_refs = train_update_ref + rollout_update_ref
        ray.get(all_refs)


    async def train_loop(self) -> None:
        """training loop: shared setup, then dispatch to the mode-specific loop."""
        # resume_datasource returns the last completed step (0 if fresh);
        # +1 because we are starting a new training round.
        start_step = current_weight_version = self.resume_datasource() + 1

        if start_step > self.config.total_steps:
            logger.info(
                "Training already complete: start_step=%d > total_steps=%d.",
                start_step, self.config.total_steps,
            )
            return

        self.rollout_manager._health_monitoring_pause()
        channel_meta = self._init_channel_meta()
        self._update_weights(channel_meta, weight_version=current_weight_version)
        self.rollout_manager.health_monitoring_resume()

        if self.is_fully_async:
            await self._fully_async_train_loop(start_step, current_weight_version)
        else:
            await self._sync_train_loop(start_step, current_weight_version)

        self._finalize_training()

    async def _fully_async_train_loop(self, start_step: int, current_weight_version: int) -> None:
        """Fully-async loop: a background rollout thread feeds mini-batches to the trainer.

        fully_async and colocate are mutually exclusive (see _validate_config),
        so there is no offload/onload handling here.
        """
        self.rollout_sampler.step = start_step
        rollout_thread = threading.Thread(
            target=lambda: asyncio.run(self.rollout_sampler.rollout_loop()), daemon=True)
        rollout_thread.start()

        for step in range(start_step, self.config.total_steps + 1):
            with time_marker("step", step=step):
                with TimeMarkerAcc(step=step) as timers:
                    for _ in range(self.num_mini_batches):
                        with timers("rollout"):
                            groups = await self.rollout_sampler(step)

                        with timers("process_traj"):
                            refs = self._process_traj_for_train(
                                groups, Mode.DEFAULT, step,
                                current_weight_version=current_weight_version,
                            )

                        if self.teacher_manager is not None:
                            with timers("teacher"):
                                refs = self.teacher_manager.compute_teacher(refs)

                        with timers.inverse_timer("wait"), timers("train"):
                            ray.get(self.train_manager.async_train(step, refs))

                    # collect train metrics
                    ray.get(self.train_manager.async_flush_train_metrics())

                    with timers("pause_delay"):
                        self.rollout_sampler.pause()

                # Fraction of the step the trainer spent NOT training (waiting
                # for / preparing rollout data, flushing, pausing) vs training.
                wait = timers.elapsed("wait")
                train = timers.elapsed("train")
                if wait + train > 0:
                    track({"perf/wait_ratio": wait / (wait + train)}, step=step)

                with time_marker("save_ckpt", step=step):
                    self.save_ckpt(step)

                if step == self.config.total_steps:
                    break

                current_weight_version = self._sync_weights_after_train(step, current_weight_version)
                self.rollout_sampler.resume()

        self.rollout_sampler.stop()
        rollout_thread.join(timeout=30)

    async def _sync_train_loop(self, start_step: int, current_weight_version: int) -> None:
        """Synchronous loop: rollout the whole step, then train on it."""
        for step in range(start_step, self.config.total_steps + 1):
            with time_marker("step", step=step):
                with time_marker("rollout", step=step):
                    accepted_traj_group_list = await self.rollout_sampler(step)

                with time_marker("process_traj", step=step):
                    splited_batch_refs = self._process_traj_for_train(
                        accepted_traj_group_list, Mode.DEFAULT, step,
                        current_weight_version=current_weight_version,
                    )

                if self.is_colocated:
                    with time_marker("offload_rollout", step=step):
                        self.rollout_manager._health_monitoring_pause()
                        self.rollout_manager.offload()

                splited_batch_refs = self._compute_teacher(splited_batch_refs, step)

                if self.is_colocated:
                    with time_marker("onload_train", step=step):
                        self.train_manager.onload()

                with time_marker("train", step=step):
                    ray.get(self.train_manager.async_train(step, splited_batch_refs))

                with time_marker("save_ckpt", step=step):
                    self.save_ckpt(step)

                if step == self.config.total_steps:
                    break

                if self.is_colocated:
                    with time_marker("offload_train", step=step):
                        self.train_manager.offload()
                    with time_marker("onload_rollout_weights", step=step):
                        self.rollout_manager.onload_weights()

                current_weight_version = self._sync_weights_after_train(step, current_weight_version)

                if self.is_colocated:
                    with time_marker("onload_rollout_kv", step=step):
                        self.rollout_manager.onload_kv()
                        self.rollout_manager.health_monitoring_resume()

    def _sync_weights_after_train(self, step: int, current_weight_version: int) -> int:
        """Bump the weight version and push the new weights to the rollout engines."""
        with time_marker("update_weights", step=step):
            channel_meta = self._init_channel_meta()
            current_weight_version = self._post_train_weight_version(current_weight_version)
            self._update_weights(channel_meta, weight_version=current_weight_version)
        return current_weight_version

    def _compute_teacher(self, splited_batch_refs: list, step: int) -> list:
        """Run teacher inference, onloading/offloading the teacher when colocated."""
        if self.teacher_manager is None:
            return splited_batch_refs

        if self.is_colocated:
            with time_marker("onload_teacher", step=step):
                self.teacher_manager.onload()
        with time_marker("teacher", step=step):
            splited_batch_refs = self.teacher_manager.compute_teacher(splited_batch_refs)
        if self.is_colocated:
            with time_marker("offload_teacher", step=step):
                self.teacher_manager.offload()
        return splited_batch_refs

    async def rollout_only_loop(self) -> None:
        """rollout only loop."""
        start_step = current_weight_version = self.resume_datasource() + 1

        if start_step > self.config.total_steps:
            logger.info(
                "Rollout only already complete: start_step=%d > total_steps=%d.",
                start_step, self.config.total_steps,
            )
            return

        channel_meta = self._init_channel_meta()
        self._update_weights(channel_meta, weight_version=current_weight_version)

        for step in range(start_step, self.config.total_steps + 1):
            with time_marker("step", step=step):
                # ROLLOUT_ONLY mode: run rollouts and persist the split batch
                # to disk so a downstream TRAIN_ONLY worker can consume it.
                with time_marker("rollout", step=step):
                    accepted_traj_group_list = await self.rollout_sampler(step)

                with time_marker("process_traj", step=step):
                    self._process_traj_for_train(
                        accepted_traj_group_list, Mode.ROLLOUT_ONLY, step,
                        current_weight_version=current_weight_version,
                    )

    async def train_only_loop(self) -> None:
        """Train-only loop: consumes rollout batches pre-generated by ROLLOUT_ONLY."""
        start_step = current_weight_version = self.resume_datasource() + 1

        if start_step > self.config.total_steps:
            logger.info(
                "Train only already complete: start_step=%d > total_steps=%d.",
                start_step, self.config.total_steps,
            )
            return

        for step in range(start_step, self.config.total_steps + 1):
            with time_marker("step", step=step):
                # Batches are loaded from disk inside _process_traj_for_train (TRAIN_ONLY branch).
                with time_marker("process_traj", step=step):
                    splited_batch_refs = self._process_traj_for_train(
                        [], Mode.TRAIN_ONLY, step, current_weight_version)

                # Run teacher inference before training.
                splited_batch_refs = self._compute_teacher(splited_batch_refs, step)

                if self.is_colocated:
                    with time_marker("onload_train", step=step):
                        self.train_manager.onload()

                with time_marker("train", step=step):
                    ray.get(self.train_manager.async_train(step, splited_batch_refs))
                                      
                current_weight_version = self._post_train_weight_version(current_weight_version)

                with time_marker("save_ckpt", step=step):
                    self.save_ckpt(step)

                if self.is_colocated:
                    with time_marker("offload_train", step=step):
                        self.train_manager.offload()

        self._finalize_training()

    def _process_traj_for_train(
        self,
        batch: list[TrajectoryGroup],
        mode: Mode,
        step: int,
        current_weight_version: int,
    ) -> list:
        """Split batch by data parallelism."""

        if mode == Mode.TRAIN_ONLY:
            splited_batch = self._load_batch_from_disk(step)
            self._set_current_weight_version(splited_batch, current_weight_version)              
            splited_batch_refs = put_dp_shards_to_ray(splited_batch, self.dp_size)
            return splited_batch_refs

        total_trajs = sum(len(g.trajectories) for g in batch)
        num_mini_batch = total_trajs // self.config.trainer.mini_batch_size
        self._set_current_weight_version(batch, current_weight_version)
        splited_batch = split_traj_group_by_dp(batch, self.dp_size, num_mini_batch)

        if mode == Mode.ROLLOUT_ONLY:
            self._save_batch_to_disk(splited_batch, step)
            return []

        splited_batch_refs = put_dp_shards_to_ray(splited_batch, self.dp_size)
        return splited_batch_refs

    def _set_current_weight_version(
        self,
        batch: list[TrajectoryGroup] | list[list[TrajectoryGroup]],
        current_weight_version: int,
    ) -> None:
        """Annotate trajectories with the trainer-side current model version."""
        groups = (
            [group for shard in batch for group in shard]
            if batch and isinstance(batch[0], list) else batch
        )

        traj_count = 0
        for group in groups:
            for traj in group.trajectories:
                traj.metadata["current_weight_version"] = current_weight_version
                traj_count += 1
        logger.info(
            "[_set_current_weight_version] annotate train batch: current_weight_version=%s groups=%d trajectories=%d",
            current_weight_version,
            len(groups),
            traj_count,
        )

    def _save_batch_to_disk(self, batch: list[list[TrajectoryGroup]], step: int):
        dest_dir = self.config.rollout_data_path
        logger.info("[step %s] saving rollout data to disk at %s", step, dest_dir)
        os.makedirs(dest_dir, exist_ok=True)

        torch.save(batch, os.path.join(dest_dir, f"step_{step}.pt"))

    def _load_batch_from_disk(self, step: int) -> list[list[TrajectoryGroup]]:
        src_dir = self.config.rollout_data_path
        logger.info("[step %s] loading rollout data from disk at %s", step, src_dir)
        src_file = os.path.join(src_dir, f"step_{step}.pt")

        if not os.path.isfile(src_file):
            raise FileNotFoundError(
                f"Rollout data file not found: {src_file}. "
                f"Ensure ROLLOUT_ONLY mode was executed for step {step} before TRAIN_ONLY."
            )
        try:
            batch = torch.load(src_file, map_location='cpu', weights_only=False)
            return batch
        except Exception as e:
            raise ValueError(f"Failed to load TrajectoryGroup from {src_file}: {e}") from e

    def resume_datasource(self) -> int:
        """Return the last completed step, or 0 if no checkpoint exists.

        Callers add 1 to get the next step to run. On resume, each
        datasource's cursor and unused-prompt buffer are also rehydrated.
        """
        latest_step_file = get_tracker_file(self.config.checkpoint_path)
        if not os.path.isfile(latest_step_file):
            return 0
        with open(latest_step_file, "r") as fp:
            saved_step = int(fp.readline().strip())
        for ds in self.datasources:
            ds.load(saved_step)
        return saved_step

    def _post_train_weight_version(self, current_weight_version: int) -> int:
        """Return the rollout-visible model version after one training update."""
        return current_weight_version + 1

    def save_ckpt(self, step: int) -> None:
        """Save on cadence or at the final step, so training always ends with a save."""
        save_freq = self.config.trainer.save_freq
        if save_freq <= 0:
            return
        if step % save_freq != 0 and step != self.config.total_steps:
            return
        save_checkpoint = self.config.trainer.save_checkpoint
        save_hf = self.config.trainer.save_hf
        if not save_checkpoint and not save_hf:
            logger.info("save_checkpoint and save_hf are both disabled, not saving anything.")
            return
        pipeline_groups = []
        if self.is_fully_async:
            # fully_async supports only one datasource for now
            pipeline_groups = self.rollout_sampler.snapshot_pipeline_buf()
        ray.get(self.train_manager.async_save_model(step, checkpoint=save_checkpoint, hf=save_hf))
        for ds in self.datasources:
            ds.save(step, additional_groups=pipeline_groups)

    def _finalize_training(self) -> None:
        """Drain queued async saves so the last save's tracker write lands on disk.

        The loop itself already issues the final save; this only flushes.
        """
        if self.config.trainer.save_freq <= 0:
            return
        ray.get(self.train_manager.async_save_model(None))

@hydra.main(config_path="../../conf", config_name=None, version_base=None)
def main(config: Optional[DictConfig] = None) -> None:
    """main entry point."""
    assert len(config) > 0, "config is none, spicify the config file with --config-name=xxx"
    OmegaConf.set_struct(config, False)
    OmegaConf.resolve(config)
    normalize_data_sources(config)
    logging_utils.configure_logger(level=config.log_level)

    # Load user extensions from coda/custom/ so their @register_* decorators run
    # before any registry lookup. Imported here rather than at module scope so a
    # custom module is free to import any coda module.
    import coda.custom  # noqa: F401,PLC0415

    # Hydra is done parsing; keep a credential passed as an override out of anything
    # that copies sys.argv (wandb records it in the run's config and metadata).
    logging_utils.redact_argv()

    logger.info("main config: %s", logging_utils.redacted_config_yaml(config))

    if not ray.is_initialized():
        ray.init(log_to_driver=True)

    trainer = Trainer(config)

    mode_str = config.run_mode
    mode = Mode(mode_str)
    if mode == Mode.TRAIN_ONLY:
        asyncio.run(trainer.train_only_loop())
    elif mode == Mode.ROLLOUT_ONLY:
        asyncio.run(trainer.rollout_only_loop())
    else:
        asyncio.run(trainer.train_loop())

if __name__ == "__main__":
    main()
