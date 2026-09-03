""" Rollout sampler with multi-data-source support. """
import abc
import asyncio
import logging
import threading
from typing import Optional
from collections import defaultdict
from omegaconf import DictConfig

from coda.utils.pipeline_queue import PipelineBuffer
from coda.agentflow.trajectory_store import Trajectory, TrajectoryGroup, TrajectoryStatus
from coda.data_factory.data_source import RolloutDataSource, RolloutDataSourceWithBuffer
from coda.agentflow.agent_flow import AgentFlow
from coda.data_factory.data_filter import DataFilter
from coda.controller import (
    SLIDING_WINDOW_STRATEGY_REGISTRY,
    get_sliding_window_strategy,
    list_sliding_window_strategies,
    register_sliding_window_strategy,
)
from coda.utils import eval_utils
from coda.utils.tracking import track

logger = logging.getLogger(__name__)

# Key prefixes forwarded from RolloutSampler.metrics to the tracking backend.
_TRACKED_PREFIXES = ("rollout/", "rollout_per_ds/")


def _ds_bucket() -> dict:
    """Raw per-step metric accumulators for a single data source."""
    return {
        "rewards": [],
        "prompt_length": [],
        "response_len": [],
        "num_turns": [],
        "compactions_per_trajectory": [],
        "response_clipped": 0,
        "turn_clipped": 0,
        # Group-level correctness counters.
        "group_total": 0,
        "group_all_correct": 0,
        "group_all_wrong": 0,
    }


def _accumulate_ds_metrics(bucket: dict, traj_group: TrajectoryGroup, data_sources: list) -> None:
    """Fold one prompt group's raw values and clip counters into ``bucket``.

    Group-level correctness is counted before data_filter.apply() so it reflects the raw
    difficulty distribution the rollout produced.
    """
    correctness = [t.is_correct for t in traj_group.trajectories]
    if correctness:
        bucket["group_total"] += 1
        if all(correctness):
            bucket["group_all_correct"] += 1
        elif not any(correctness):
            bucket["group_all_wrong"] += 1

    for t in traj_group.trajectories:
        ds_cfg = data_sources[t.ds_index]
        response_len = len(t.rollout_log_probs)
        prompt_len = len(t.tokens) - response_len

        bucket["rewards"].append(t.reward)
        bucket["prompt_length"].append(prompt_len)
        bucket["response_len"].append(response_len)
        bucket["num_turns"].append(t.num_turns)
        max_response = ds_cfg.max_response_len_per_trajectory
        if max_response > 0 and response_len >= max_response:
            bucket["response_clipped"] += 1

        max_turns = int(ds_cfg.get("agent", {}).get("max_turns") or 0)
        if max_turns > 0 and t.num_turns >= max_turns:
            bucket["turn_clipped"] += 1

        # A compaction round produces two "compact" segments: one for the summarization
        # request and one for the agent's follow-up request built from the summary.
        num_compact_segments = sum(1 for seg in t.segments if seg.origin == "compact")
        compaction_events = (num_compact_segments + 1) // 2
        bucket["compactions_per_trajectory"].append(compaction_events)


def _summary_metrics(bucket: dict, key_prefix: str) -> dict[str, float]:
    """Summarize one bucket into ``{key_prefix}<metric>`` reward / turn / response stats.

    Every statistic is scoped to the given bucket, so the caller controls the scope by
    passing either the whole-batch accumulator or one data source's bucket.
    The ``_mean`` / ``_max`` / ``_min`` suffixes are load-bearing: TrainMetricsAggregator
    picks its cross-mini-batch reduction from them.

    ``prompt_length`` and ``completed_count`` are deliberately absent; they are reported
    globally only, by the caller.
    """
    n = len(bucket["rewards"])
    summary: dict[str, float] = {}
    if n > 0:
        for name, field in (("reward", "rewards"),
                            ("response_length", "response_len"),
                            ("num_turns", "num_turns"),
                            ("compactions", "compactions_per_trajectory")):
            values = bucket[field]
            summary[f"{key_prefix}{name}_mean"] = sum(values) / n
            summary[f"{key_prefix}{name}_max"] = max(values)
            summary[f"{key_prefix}{name}_min"] = min(values)
        summary[f"{key_prefix}response_length_clip_ratio"] = bucket["response_clipped"] / n
        summary[f"{key_prefix}turns_clip_ratio"] = bucket["turn_clipped"] / n
        summary[f"{key_prefix}compacted_trajectory_ratio"] = (
            sum(1 for v in bucket["compactions_per_trajectory"] if v > 0) / n
        )
        summary[f"{key_prefix}compaction_count"] = sum(bucket["compactions_per_trajectory"])

    group_total = bucket["group_total"]
    if group_total > 0:
        summary[f"{key_prefix}all_correct_ratio"] = bucket["group_all_correct"] / group_total
        summary[f"{key_prefix}all_wrong_ratio"] = bucket["group_all_wrong"] / group_total
    return summary


class RolloutSampler:
    """
    Rollout sampling strategy with multi-data-source support.

    Each data source has its own num_prompts_per_step and datasource instance.
    The sampler launches generation tasks from all data sources concurrently.
    """
    def __init__(
        self,
        config: DictConfig,
        datasources: list[RolloutDataSourceWithBuffer],
        agentflow: AgentFlow
    ):
        self.config = config
        self.datasources = datasources
        self.agentflow = agentflow
        self.traj_queue = agentflow.traj_queue
        self.data_filter = DataFilter(config.rollout.filter)
        self.timeout = config.rollout.sampler.timeout
        self.metrics = {}
        self.is_fully_async = config.fully_async.enable

        if self.is_fully_async:
            num_trajectories_per_prompt = int(config.data_sources[0].num_trajectories_per_prompt)
            mini_batch_size = config.trainer.mini_batch_size

            self._groups_per_mini_batch = mini_batch_size // num_trajectories_per_prompt
            self._stopped = False
            self._run_gate = threading.Event()
            self._run_gate.set()  # initially not paused
            self._cleanup_done = threading.Event()
            self._cleanup_done.set()  # initially "done" (no cleanup pending)

            self.step = 0

            # Collect all trajs during one step, to calculate partial rate.
            # Add when trainer get trajs from buf, clear when pause. Should have no competition.
            self._this_step_traj: list[Trajectory] = []
            self._dropped_count = 0

            # Original None. If any thread encounter an error, pass it to trainer via this.
            self._fatal_error: Optional[BaseException] = None

            self.pipeline_buf = PipelineBuffer(is_stopped=self.is_stopped)

            # Background thread that continuously scans traj_queue for complete groups
            self._collector_thread = threading.Thread(target=self._collect_groups_loop, daemon=True)

    def _need_eval(self, step: int) -> bool:
        """Whether this rollout round should co-dispatch eval.

        The first and last steps always do, plus every ``eval.interval`` steps.
        """
        interval = int(self.config.rollout.eval.interval)
        return interval > 0 and (step % interval == 0 or step == self.config.total_steps or step == 1)

    def _trigger_eval_generation(self, step, task_set, ds_index: int):
        """Call agenticflow to generate the whole eval set of a specific data source."""
        datasource = RolloutDataSource(
            self.config.data_sources[ds_index], self.config, ds_index=ds_index, is_eval=True
        )
        datasource.step = step  # eval get() takes no step; without this ids all say step0
        traj_groups = datasource.get(len(datasource))
        for traj_group in traj_groups:
            task = asyncio.create_task(self.agentflow.generate(traj_group.trajectories))
            task_set.add(task)
            task.add_done_callback(task_set.discard)
        return traj_groups

    async def __call__(
        self,
        step: int,
    ) -> list[TrajectoryGroup]:
        """Execute rollout based on configured sampler."""
        num_oversample = self.config.rollout.sampler.num_oversample

        if self.is_fully_async:
            # fully_async runs rollout in a background thread; skip in-round eval
            # to avoid contending with it for the shared inference engine.
            groups = []
            for _ in range(self._groups_per_mini_batch):
                item = await self.pipeline_buf.async_get()
                if item is None:
                    self._check_error()
                    break
                groups.append(item)
            self._this_step_traj.extend(t for g in groups for t in g.trajectories)
            return groups
        elif self.config.rollout.sampler.name == "dynamic":
            logger.info("[step %d] Rollout sampler=%s", step, self.config.rollout.sampler.name)
            return await asyncio.wait_for(
                self.dynamic_rollout(num_oversample, step),
                timeout=self.timeout,
            )
        raise ValueError(f"Unknown sampler: {self.config.rollout.sampler.name}")

    def is_paused(self) -> bool:
        """Whether the rollout loop is currently paused."""
        return not self._run_gate.is_set()

    def is_stopped(self) -> bool:
        """Whether the rollout loop has been stopped."""
        return self._stopped

    def wait_if_paused(self) -> bool:
        """Block until resumed or stopped (called from producer thread).

        Returns False if stopped (producer should exit).
        """
        if self._stopped:
            return False
        self._run_gate.wait()
        return not self._stopped

    def pause(self) -> None:
        """Pause rollout loop and block until cleanup completes."""
        self._cleanup_done.clear()
        self._check_error()  # check before wait
        self._run_gate.clear()
        self._cleanup_done.wait()
        self._check_error()  # check before return
        self._flush_rollout_metrics()

    def _flush_rollout_metrics(self) -> None:
        """Report accumulated rollout metrics for the current step (fully async only)."""
        metrics = {'rollout/pipeline_buf_size': self.pipeline_buf.qsize,
                   'rollout/filter_drop': self._dropped_count}
        self._dropped_count = 0
        if self._this_step_traj:
            metrics['rollout/partial_ratio'] = self.calculate_partial_ratio(self._this_step_traj)
            metrics['rollout/partial_span_max'] = self.calculate_max_partial_span(self._this_step_traj)
            self._this_step_traj = []

        if self.metrics.get(self.step):
            rollout_metrics = {k: v for k, v in self.metrics[self.step].items()
                               if k.startswith(_TRACKED_PREFIXES)}
            metrics.update(rollout_metrics)

        if metrics:
            track(metrics, self.step)

    def snapshot_pipeline_buf(self) -> list[TrajectoryGroup]:
        """Return a non-destructive snapshot of every completed pipeline group.

        Called by Trainer.save_ckpt (fully_async only) to persist prompts that
        the trainer has not consumed yet. The pipeline continues to own and
        serve the original groups after the checkpoint is written.
        """
        return self.pipeline_buf.snapshot()

    def resume(self) -> None:
        """Resume rollout loop. Advances step (new weight version)."""
        self.step += 1
        self._run_gate.set()

    def stop(self) -> None:
        """Stop rollout loop, signaling exit."""
        self._stopped = True
        self._run_gate.set()  # unblock any paused producer

    def _set_fatal_error(self, exc: BaseException) -> None:
        """Called by any background thread on crash. Unblocks all waiting points."""
        self._fatal_error = exc
        self._stopped = True
        self._run_gate.set()
        self._cleanup_done.set()

    def _check_error(self) -> None:
        """Called by main thread to re-raise background thread errors."""
        if self._fatal_error:
            raise RuntimeError("Background thread crashed") from self._fatal_error

    def _collect_groups_loop(self) -> None:
        """Background thread: continuously scan traj_queue for complete groups and put them into pipeline_buf.

        This thread runs independently of the rollout_loop. It pops complete groups from traj_queue,
        applies pre-processing (data filter, metrics), and puts valid groups into pipeline_buf.

        Thread safety: strategy.on_collected() is protected by the lock inside _ThreadSafeStrategy,
        so calling it directly here is safe without additional synchronization.
        """
        logger.info("[fully_async] _collect_groups_loop started")
        try:
            while not self._stopped:
                group = self.traj_queue.wait_for_group(timeout=0.9, will_collect=self._strategy.will_collect)
                if group is None:
                    continue

                prompt_id = group.prompt_id

                # Pre-process (filter, metrics)
                group = self._pre_process(group, self.step)
                if group is None:
                    self._dropped_count += 1
                    self._strategy.on_collected(prompt_id)
                    continue

                ok = self.pipeline_buf.put(group)
                self._strategy.on_collected(prompt_id)
                if not ok:
                    break
        except Exception as e:
            self._set_fatal_error(e)

    async def rollout_loop(self) -> None:
        """Continuously generate trajectories; complete groups are collected by _collect_groups_loop.

        Dispatch logic is delegated to a SlidingWindowStrategy instance, making the loop
        strategy-agnostic. New strategies only need to implement the strategy interface.
        """
        running_tasks: set = set()
        self._strategy = create_strategy(self.config)

        logger.info("[fully_async] rollout_loop started at step=%d", self.step)

        # Start the collector thread
        self._collector_thread.start()

        try:
            while not self._stopped:
                if self.is_paused():
                    await self._cleanup(running_tasks, self.step, wait=not self.config.rollout.partial)
                    running_tasks = set()
                    self._strategy.on_reset()
                    self._cleanup_done.set()

                    # wait here until resume, and then return true. return false when stop, so break
                    if not self.wait_if_paused():
                        break

                dispatch_count = self._strategy.compute_dispatch_count(self.pipeline_buf.qsize)
                if dispatch_count > 0:
                    # fully_async invariant: single data source enforced by Trainer._validate_config
                    dispatched = self._trigger_generation(dispatch_count, self.step, running_tasks, ds_index=0)
                    self._strategy.on_dispatched(dispatched)

                # Sleep briefly to avoid busy-spinning; the collector thread handles queue draining
                await asyncio.sleep(1.0)

        except Exception as e:
            logger.exception("[fully_async] rollout_loop encountered an error")
            self._set_fatal_error(e)
        finally:
            await self._cleanup(running_tasks, self.step)
            logger.info("[fully_async] rollout_loop exited at step=%d", self.step)

    def _trigger_generation(self, num, step, task_set, ds_index: int):
        """Call agenticflow to generate trajectory from a specific data source."""
        datasource = self.datasources[ds_index]
        traj_groups = datasource.get(num, step)
        for traj_group in traj_groups:
            task = asyncio.create_task(self.agentflow.generate(traj_group.trajectories))
            task_set.add(task)
            task.add_done_callback(task_set.discard)
        return traj_groups

    async def _pop_group(self):
        """try to pop a group from traj_queue"""
        group = self.traj_queue.pop_group()
        if group:
            return group
        else:
            return await asyncio.to_thread(self.traj_queue.wait_for_group, timeout=5.0)

    def calculate_partial_ratio(
        self,
        trajectories: list[Trajectory],
    ) -> float:
        """Calculate the ratio of trajectories whose rollout crossed weight versions."""
        if not trajectories:
            return 0.0

        count = 0
        for traj in trajectories:
            start, end = traj.start_rollout_weight_version, traj.end_rollout_weight_version
            if start != -1 and end != start:
                count += 1
        return count / len(trajectories)

    def calculate_max_partial_span(
        self,
        trajectories: list[Trajectory],
    ) -> int:
        """Calculate the maximum weight version span across all trajectories."""
        max_span = 0
        for traj in trajectories:
            start, end = traj.start_rollout_weight_version, traj.end_rollout_weight_version
            if start != -1:
                max_span = max(max_span, end - start)
        return max_span

    def _stat_metrics(self, traj_group: TrajectoryGroup, step):
        """Accumulate rollout metrics for the given step across multiple trajectory groups.

        ``rollout/*`` always covers the whole batch. With more than one data source
        configured, each source additionally reports ``rollout_per_ds/ds{ds_index}_*``.
        """
        # Initialize metrics for this step (and free memory from previous steps).
        # Raw accumulators live at the top level; "per_ds" holds one bucket per source.
        if step not in self.metrics:
            self.metrics = {step: {**_ds_bucket(), "per_ds": {}}}
        metrics = self.metrics[step]

        _accumulate_ds_metrics(metrics, traj_group, self.config.data_sources)

        # Summary keys are recomputed from the accumulators on every group.
        metrics.update(_summary_metrics(metrics, "rollout/"))
        n = len(metrics["rewards"])
        if n > 0:
            metrics["rollout/prompt_length_mean"] = sum(metrics["prompt_length"]) / n
            metrics["rollout/prompt_length_max"]  = max(metrics["prompt_length"])
            metrics["rollout/prompt_length_min"]  = min(metrics["prompt_length"])
        metrics["rollout/completed_count"] = n

        # Per-data-source breakdown; with a single source it only duplicates rollout/*.
        if len(self.config.data_sources) > 1 and traj_group.trajectories:
            # Every trajectory of a prompt group shares one data source, so the whole
            # group folds into a single bucket.
            per_ds = metrics["per_ds"]
            bucket = per_ds.setdefault(traj_group.trajectories[0].ds_index, _ds_bucket())
            _accumulate_ds_metrics(bucket, traj_group, self.config.data_sources)
            for ds_index, ds_bucket in sorted(per_ds.items()):
                metrics.update(_summary_metrics(ds_bucket, f"rollout_per_ds/ds{ds_index}_"))

    async def dynamic_rollout(
        self,
        num_oversample: int,
        step: int,
    ) -> list[TrajectoryGroup]:
        """
        Dynamic sampling with multi-data-source support.

        Launches generation tasks from all data sources concurrently,
        each contributing its own num_prompts_per_step share.
        Refill only targets data sources that haven't met their quota.
        """
        accepted_groups: list[TrajectoryGroup] = []
        overflow_groups: list[TrajectoryGroup] = []
        running_tasks = set()
        dropped_count = 0
        refill_count = 0
        running_count = 0
        rollout_version_mismatch_ratio = 0.0
        max_partial_span = 0
        refill_ratio = self.config.rollout.sampler.refill_ratio
        max_refill_count = self.config.rollout.sampler.max_refill_count

        # Per-data-source tracking
        ds_targets: dict[int, int] = {}
        ds_accepted: dict[int, int] = {}
        ds_running: dict[int, int] = {}

        # Initial trigger: dispatch from each data source, plus its eval set on an eval round
        need_eval = self._need_eval(step)
        eval_targets: dict[int, int] = {}
        eval_accepted: dict[int, int] = {}
        for ds_index, ds_cfg in enumerate(self.config.data_sources):
            ds_target = int(ds_cfg.num_prompts_per_step)
            ds_targets[ds_index] = ds_target
            ds_accepted[ds_index] = 0
            ds_running[ds_index] = 0
            ds_num = ds_target + num_oversample
            self._trigger_generation(ds_num, step, running_tasks, ds_index)
            ds_running[ds_index] += ds_num

            if need_eval and ds_cfg.dataset.eval_prompt_data_path:
                eval_dispatched = self._trigger_eval_generation(step, running_tasks, ds_index)
                eval_targets[ds_index] = len(eval_dispatched)
                eval_accepted[ds_index] = 0
        eval_traj_groups: list[TrajectoryGroup] = []

        try:
            while (any(ds_accepted[i] < ds_targets[i] for i in ds_targets)
                   or any(eval_accepted[i] < eval_targets[i] for i in eval_targets)):
                # 1. check if need to refill — only training sources under quota
                for ds_index in ds_targets:
                    ds_need = ds_targets[ds_index] - ds_accepted[ds_index] - ds_running[ds_index]
                    if ds_need > 0:
                        ds_refill = int(min(ds_need * refill_ratio, max_refill_count - refill_count))
                        if ds_refill > 0:
                            self._trigger_generation(ds_refill, step, running_tasks, ds_index)
                            ds_running[ds_index] += ds_refill
                            refill_count += ds_refill
                running_count = sum(ds_running.values())

                # 2. pop group from traj_queue (training or eval)
                group = await self._pop_group()
                if not group:
                    if running_count == 0 and sum(eval_accepted.values()) == sum(eval_targets.values()):
                        raise OverflowError("no more trajectory left to rollout")
                    logger.info("no group popped, continue, running_count: %d", running_count)
                    continue

                # 3. route eval groups out of the training batch
                if group.trajectories[0].is_eval:
                    eval_traj_groups.append(group)
                    eval_ds = group.trajectories[0].ds_index
                    eval_accepted[eval_ds] = eval_accepted.get(eval_ds, 0) + 1
                    continue

                ds_index = group.trajectories[0].ds_index
                ds_running[ds_index] -= 1
                # 4. pre_process training group
                group = self._pre_process(group, step)
                if group:
                    if ds_accepted[ds_index] < ds_targets[ds_index]:
                        logger.info("[step %d] add %s to accepted_groups.", step, group.prompt_id)
                        accepted_groups.append(group)
                        ds_accepted[ds_index] += 1
                    else:
                        logger.info("[step %d] ds[%d] already met quota, overflow %s.", step, ds_index, group.prompt_id)
                        overflow_groups.append(group)
                else:
                    dropped_count += 1
                logger.info("[step %d] dropped is %d, refilled is %d accepted is %d running is %d",
                            step, dropped_count, refill_count, len(accepted_groups), sum(ds_running.values()))

            trajectories = [t for g in accepted_groups for t in g.trajectories]
            trajectory_ids = [t.trajectory_id for t in trajectories]
            rollout_version_mismatch_ratio = self.calculate_partial_ratio(trajectories)
            max_partial_span = self.calculate_max_partial_span(trajectories)
            logger.info(
                "[step %d] dynamic_rollout: accepted %d trajectories, "
                "rollout_version_switch_ratio: %.2f max_partial_span: %d",
                step, len(trajectory_ids), rollout_version_mismatch_ratio, max_partial_span,
            )
            # 5. compute + report eval metrics from the collected eval groups
            if need_eval:
                eval_metrics = eval_utils.compute_eval_metrics(eval_traj_groups)
                if eval_metrics:
                    track(eval_metrics, step)
                    logger.info("[eval] step=%d %s", step, eval_metrics)
            return accepted_groups
        except Exception as e:
            logger.error("[step %d] dynamic_rollout failed: %s", step, e)
            raise
        finally:
            # only send the metrics needed, other type of data like list will got an mlflow exception
            metrics = {
                'rollout/filter_drop': dropped_count,
                'rollout/filter_refill': refill_count,
                'rollout/partial_ratio': rollout_version_mismatch_ratio,
                'rollout/partial_span_max': max_partial_span,
            }
            if self.metrics.get(step):
                rollout_metrics = {k: v for k, v in self.metrics[step].items()
                                   if k.startswith(_TRACKED_PREFIXES)}
                metrics.update(rollout_metrics)
            track(metrics, step)
            await self._cleanup(running_tasks, step, overflow_groups)


    async def _cleanup(self, running_tasks, step, overflow_groups: list = None, wait=False):
        """Cleanup running tasks.

        Behavior depends on wait and partial settings:
        - wait=True (partial=False): wait for all tasks to complete, then drain queue
          and restore complete groups to datasource to avoid wasting computed results.
        - wait=False (partial=True): cancel running tasks immediately, then drain queue
          and restore complete groups to datasource for reuse in the next step.
        - wait=False (partial=False): cancel running tasks and discard all remaining
          data (used in final cleanup or non-fully-async paths).
        """
        active_tasks = list(running_tasks)
        if wait:
            # In fully async mode, when partial=false, data is not discarded.
            # Instead, synchronously wait for the task to finish.
            logger.info("[step %d] cleanup: waiting for %d running tasks to complete", step, len(active_tasks))
            await asyncio.gather(*active_tasks, return_exceptions=True)
        else:
            for t in active_tasks:
                t.cancel()
            await asyncio.gather(*active_tasks, return_exceptions=True)
            await self.agentflow.abort()

        # Drain remaining items from queue
        remaining_items = []
        if overflow_groups:
            for g in overflow_groups:
                remaining_items.extend(g.trajectories)
        while not self.traj_queue.is_empty():
            item = self.traj_queue.pop_all()
            if item is not None:
                remaining_items.extend(item)

        logger.info("[step %d] cleanup: drained %d remaining trajectories from queue", step, len(remaining_items))
        # Drop eval trajectories: restoring them would feed the eval set into training.
        remaining_items = [t for t in remaining_items if not t.is_eval]
        # Handle partial rollout items
        if (self.config.rollout.partial or wait) and remaining_items:
            # group trajectories by prompt_id for datasource.add()
            groups_dict = defaultdict(list)
            complete_groups_by_ds: dict[int, list[TrajectoryGroup]] = defaultdict(list)
            incomplete_count = 0
            for traj in remaining_items:
                groups_dict[traj.prompt_id].append(traj)
            for prompt_id, trajs in groups_dict.items():
                ds_index = trajs[0].ds_index
                if any(traj.ds_index != ds_index for traj in trajs):
                    raise ValueError(f"mixed ds_index in prompt group {prompt_id}")
                expected_size = int(self.config.data_sources[ds_index].num_trajectories_per_prompt)
                if len(trajs) == expected_size:
                    complete_groups_by_ds[ds_index].append(TrajectoryGroup(prompt_id=prompt_id, trajectories=trajs))
                else:
                    incomplete_count += len(trajs)
                    logger.warning(
                        "[step %d] cleanup: dropping incomplete group %s (%d/%d trajectories)",
                        step, prompt_id, len(trajs), expected_size,
                    )
            restored_count = 0
            restored_group_count = 0
            for ds_index, complete_groups in complete_groups_by_ds.items():
                self.datasources[ds_index].add(complete_groups)
                restored_group_count += len(complete_groups)
                restored_count += sum(len(g.trajectories) for g in complete_groups)
            logger.info("[step %d] cleanup: restored %d partial rollout groups (%d trajectories) to buffer",
                        step, restored_group_count, restored_count)
            metrics = {
                'rollout/partial_restored_count': restored_count,
                'rollout/partial_dropped_incomplete_count': incomplete_count,
            }
            track(metrics, step)

    def _pre_process(self, traj_group: TrajectoryGroup, step) -> TrajectoryGroup | None:
        """Collect metrics for a TrajectoryGroup and apply the data filter.

        Raises ValueError if a surviving trajectory is FAILED, because dirty
        trajectories lead to wrong gradients. Configure ``rollout.filter.status``
        to drop such groups instead.
        """
        self._stat_metrics(traj_group, step)
        processed_traj_group = self.data_filter.apply(traj_group)
        if processed_traj_group:
            for traj in processed_traj_group.trajectories:
                if traj.status == TrajectoryStatus.FAILED:
                    logger.warning("preprocess failed by dirty traj: %s", traj.trajectory_id)
                    raise ValueError(f"preprocess failed by dirty traj: {traj.trajectory_id}")
        return processed_traj_group


"""
Sliding window dispatch strategies for fully-async rollout loop.

Each strategy controls how many new generation tasks to dispatch per iteration,
based on its own internal state (sequence tracking, capacity, etc.).

To add a new strategy:
1. Subclass SlidingWindowStrategy
2. Decorate it with @register_sliding_window_strategy("your-name")

"""


class SlidingWindowStrategy(abc.ABC):
    """Interface for sliding window dispatch strategies.

    Implementations are stateful — they track in-flight sequences or capacity
    and expose a uniform API consumed by rollout_loop.

    running_count is managed by the base class: incremented on dispatch, decremented on collect.
    """

    def __init__(self):
        self._running_count = 0

    @abc.abstractmethod
    def compute_dispatch_count(self, buf_qsize: int) -> int:
        """Return how many new groups to dispatch this iteration.

        Args:
            buf_qsize: number of completed items sitting in pipeline buffer.
        """

    def on_dispatched(self, groups: list[TrajectoryGroup]) -> None:
        """Called after groups are dispatched. Update internal bookkeeping."""
        self._running_count += len(groups)

    def on_collected(self, prompt_id: str) -> None:
        """Called when a prompt_id has been collected (retired). Update internal state."""
        self._running_count -= 1

    def on_reset(self) -> None:
        """Reset internal state (called on pause/cleanup)."""
        self._running_count = 0

    def will_collect(self, group: dict[str, Trajectory]) -> bool:
        """Check whether a complete group should be collected.

        Return True to allow the group to be collected, False to leave it in the traj_queue.
        Groups containing aborted trajectories are not collectable; failed ones are
        still collected and handled downstream by the data filter.

        WARNING: This method is called while holding TrajQueue._cond lock, and wrapped
        by _ThreadSafeStrategy._lock. To avoid deadlock/performance issues:
        - MUST NOT perform expensive operations (network, IO, sleep, etc.)
        - MUST NOT hold or access TrajQueue objects or its lock
        - MUST NOT acquire any additional locks
        - MUST NOT modify this "group" parameter
        """
        return not any(
            traj.status == TrajectoryStatus.ABORTED
            for traj in group.values()
        )


@register_sliding_window_strategy("no-window")
class NoWindowStrategy(SlidingWindowStrategy):
    """Keep total (in-flight + buffered) at or below max_inflight."""

    def __init__(self, config):
        """Initialize with maximum inflight capacity."""
        super().__init__()
        batch_size = int(config.data_sources[0].num_prompts_per_step)
        stale_steps = config.fully_async.stale_steps
        self._max_inflight = int(batch_size * (1 + stale_steps))

    def compute_dispatch_count(self, buf_qsize: int) -> int:
        """Return available slots: max_inflight minus running and buffered."""
        return max(0, self._max_inflight - self._running_count - buf_qsize)


@register_sliding_window_strategy("window-gated")
class WindowGatedStrategy(SlidingWindowStrategy):
    """Constrain spread between oldest in-flight seq and next dispatch seq."""

    def __init__(self, config):
        """Initialize with window size constraint."""
        super().__init__()
        batch_size = int(config.data_sources[0].num_prompts_per_step)
        stale_steps = config.fully_async.stale_steps
        self._window_size = int(batch_size * (1 + stale_steps))
        self._next_seq = 0
        self._generating_seqs: set[int] = set()
        self._prompt_to_seq_id: dict[str, int] = {}

    def compute_dispatch_count(self, buf_qsize: int) -> int:
        """Return dispatch count bounded by window spread and capacity."""
        min_seq = min(self._generating_seqs) if self._generating_seqs else self._next_seq
        return min(
            min_seq + self._window_size - self._next_seq,
            self._window_size - self._running_count - buf_qsize,
        )

    def on_dispatched(self, groups: list[TrajectoryGroup]) -> None:
        """Assign sequence numbers to dispatched groups."""
        super().on_dispatched(groups)
        for g in groups:
            self._prompt_to_seq_id[g.prompt_id] = self._next_seq
            self._generating_seqs.add(self._next_seq)
            self._next_seq += 1

    def on_collected(self, prompt_id: str) -> None:
        """Remove collected prompt_id from inflight tracking."""
        super().on_collected(prompt_id)
        seq = self._prompt_to_seq_id.pop(prompt_id, None)
        if seq is not None:
            self._generating_seqs.discard(seq)

    def on_reset(self) -> None:
        """Clear all inflight sequence tracking state."""
        super().on_reset()
        self._generating_seqs.clear()
        self._prompt_to_seq_id.clear()


@register_sliding_window_strategy("windowed-fifo")
class WindowedFifoStrategy(SlidingWindowStrategy):
    """Capacity-based dispatch (like NoWindow) with window-gated collection ordering."""

    def __init__(self, config):
        super().__init__()
        batch_size = int(config.data_sources[0].num_prompts_per_step)
        stale_steps = config.fully_async.stale_steps
        self._max_inflight = int(batch_size * (1 + stale_steps))
        self._window_size = batch_size
        self._next_seq = 0
        self._generating_seqs: set[int] = set()
        self._prompt_to_seq_id: dict[str, int] = {}

    def compute_dispatch_count(self, buf_qsize: int) -> int:
        return max(0, self._max_inflight - self._running_count - buf_qsize)

    def on_dispatched(self, groups: list[TrajectoryGroup]) -> None:
        super().on_dispatched(groups)
        for g in groups:
            self._prompt_to_seq_id[g.prompt_id] = self._next_seq
            self._generating_seqs.add(self._next_seq)
            self._next_seq += 1

    def on_collected(self, prompt_id: str) -> None:
        super().on_collected(prompt_id)
        seq = self._prompt_to_seq_id.pop(prompt_id, None)
        if seq is not None:
            self._generating_seqs.discard(seq)

    def on_reset(self) -> None:
        super().on_reset()
        self._generating_seqs.clear()
        self._prompt_to_seq_id.clear()

    def will_collect(self, group: dict[str, Trajectory]) -> bool:
        if not super().will_collect(group):
            return False

        prompt_id = next(iter(group.values())).prompt_id
        seq = self._prompt_to_seq_id.get(prompt_id)
        if seq is None:
            raise RuntimeError(f"group: {prompt_id} is not in window mapping. check execution sequence or lock")
        if not self._generating_seqs:
            return True
        # NOTE: min(set()) is O(n). And this func is under lock.
        # Shall performance bottlenecks be observed, consider replacing it with "heapq + lazy_delete".
        min_inflight_seq = min(self._generating_seqs)
        return (seq - min_inflight_seq) < self._window_size


class _ThreadSafeStrategy(SlidingWindowStrategy):
    """Proxy that serializes all method calls with a lock."""

    def __init__(self, inner: SlidingWindowStrategy):
        """Proxy that serializes all method calls with a lock."""
        # Skip super().__init__() — we delegate everything to inner
        self._inner = inner
        self._lock = threading.Lock()

    def compute_dispatch_count(self, buf_qsize: int) -> int:
        """Proxy that serializes all method calls with a lock."""
        with self._lock:
            return self._inner.compute_dispatch_count(buf_qsize)

    def on_dispatched(self, groups: list[TrajectoryGroup]) -> None:
        """Proxy that serializes all method calls with a lock."""
        with self._lock:
            self._inner.on_dispatched(groups)

    def on_collected(self, prompt_id: str) -> None:
        """Proxy that serializes all method calls with a lock."""
        with self._lock:
            self._inner.on_collected(prompt_id)

    def on_reset(self) -> None:
        """Proxy that serializes all method calls with a lock."""
        with self._lock:
            self._inner.on_reset()

    def will_collect(self, group: dict[str, Trajectory]) -> bool:
        """Proxy that serializes all method calls with a lock."""
        with self._lock:
            return self._inner.will_collect(group)


def create_strategy(config) -> SlidingWindowStrategy:
    """Factory: instantiate a strategy by config name (thread-safe)."""
    name = config.fully_async.sliding_window
    if name not in SLIDING_WINDOW_STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown sliding_window strategy '{name}'. "
            f"Available: {list_sliding_window_strategies()}"
        )
    return _ThreadSafeStrategy(get_sliding_window_strategy(name)(config))
