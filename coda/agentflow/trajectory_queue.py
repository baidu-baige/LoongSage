"""Trajectory Queue between SingleController and Agentflow"""
import threading
import time
from typing import Callable

from coda.agentflow.trajectory_store import Trajectory, TrajectoryGroup, TrajectoryStatus

# Set on every trajectory of a group once group_rm has scored it, so a group
# restored by partial rollout and re-emitted unchanged is not scored twice.
GROUP_RM_SCORED_KEY = "group_rm_scored"


class TrajQueue:
    """Thread-safe queue for managing Trajectory objects grouped by prompt_id.

    Uses threading.Condition for producer-consumer signaling (wait_for_group)
    and an optional maxsize for backpressure.

    Internally, each group stores trajectories in an insertion-ordered dict
    (trajectory_id -> Trajectory), giving O(1) lookup and removal by id.

    For data sources in *group_rm_ds*, a complete group only becomes ready
    after AgentFlow has scored it: claim_group_rm -> commit_group_rm.
    """
    def __init__(self, group_sizes: dict[int, int], maxsize: int = 0,
                 group_rm_ds: frozenset[int] | set[int] = frozenset()):
        self.group_sizes = {int(k): int(v) for k, v in group_sizes.items()}
        self.maxsize = maxsize  # 0 means unlimited
        # prompt_id -> {trajectory_id -> Trajectory} (dict preserves insertion order)
        self.groups: dict[str, dict[str, Trajectory]] = {}
        # trajectory_id -> prompt_id, for quick lookup
        self.tid_pid_map: dict[str, str] = {}
        self.group_rm_ds = {int(ds) for ds in group_rm_ds}
        # prompt_ids of group_rm data sources being scored / already scored
        self._scoring: set[str] = set()
        self._scored: set[str] = set()
        self._cond = threading.Condition()
        if self.maxsize < 0:
            raise ValueError(f"maxsize must be >= 0, got {maxsize}")
        if not self.group_sizes:
            raise ValueError("group_sizes must not be empty")
        for ds_index, group_size in self.group_sizes.items():
            if group_size < 1:
                raise ValueError(f"group_sizes[{ds_index}] must be >= 1, got {group_size}")

    # ------------------------------------------------------------------
    # Internal helpers (caller must hold self._cond)
    # ------------------------------------------------------------------

    def _total_count(self) -> int:
        """Return total trajectory count. Caller must hold lock."""
        return len(self.tid_pid_map)

    def _group_size_for(self, ds_index: int) -> int:
        """Return configured group size for a data source."""
        if ds_index not in self.group_sizes:
            raise ValueError(f"No group size configured for ds_index={ds_index}")
        return self.group_sizes[ds_index]

    def _group_size_for_prompt(self, prompt_id: str) -> int:
        """Return configured group size for a prompt group. Caller must hold lock."""
        group = self.groups[prompt_id]
        first_traj = next(iter(group.values()))
        return self._group_size_for(first_traj.ds_index)

    def _cleanup_group(self, prompt_id: str) -> None:
        """Remove group entry if empty. Caller must hold lock."""
        if prompt_id in self.groups and not self.groups[prompt_id]:
            del self.groups[prompt_id]
            self._scoring.discard(prompt_id)
            self._scored.discard(prompt_id)

    def _pop_n_from_group(self, prompt_id: str, n: int) -> TrajectoryGroup:
        """Pop the first *n* trajectories from a group. Caller must hold lock."""
        group = self.groups[prompt_id]
        keys = list(group.keys())[:n]
        items = []
        for k in keys:
            items.append(group.pop(k))
            self.tid_pid_map.pop(k)
        # Leftover trajectories (if any) form a new, unscored group.
        self._scored.discard(prompt_id)
        self._cleanup_group(prompt_id)
        return TrajectoryGroup(prompt_id=prompt_id, trajectories=items)

    def _find_and_pop_ready_group(
        self, will_collect: Callable[[dict[str, Trajectory]], bool] | None = None
    ) -> TrajectoryGroup | None:
        """Find and pop the first ready group whose full prompt group satisfies will_collect. Caller must hold lock."""
        for pid, group in list(self.groups.items()):
            group_size = self._group_size_for_prompt(pid)
            if len(group) >= group_size and (
                next(iter(group.values())).ds_index not in self.group_rm_ds or pid in self._scored
            ):
                if will_collect is not None and not will_collect(group):
                    continue
                return self._pop_n_from_group(pid, group_size)
        return None

    # ------------------------------------------------------------------
    # Public API — single-item operations
    # ------------------------------------------------------------------

    def add(self, trajectory: Trajectory) -> None:
        """Add a single Trajectory to the queue.

        Blocks if *maxsize* is set and the queue is full.
        """
        if not trajectory.trajectory_id or not trajectory.prompt_id:
            raise ValueError("trajectory_id and prompt_id must not be empty")
        self._group_size_for(trajectory.ds_index)

        with self._cond:
            if self.maxsize > 0:
                while self._total_count() >= self.maxsize:
                    self._cond.wait()

            if trajectory.trajectory_id in self.tid_pid_map:
                raise ValueError(f"Duplicate trajectory_id {trajectory.trajectory_id}")

            if trajectory.prompt_id not in self.groups:
                self.groups[trajectory.prompt_id] = {}
            elif next(iter(self.groups[trajectory.prompt_id].values())).ds_index != trajectory.ds_index:
                raise ValueError("All trajectories in a group must have the same ds_index")
            self.groups[trajectory.prompt_id][trajectory.trajectory_id] = trajectory
            self.tid_pid_map[trajectory.trajectory_id] = trajectory.prompt_id
            self._cond.notify()

    def pop(self, trajectory_id: str | None = None) -> Trajectory | None:
        """Pop a single Trajectory.

        If *trajectory_id* is given, pop that specific trajectory (O(1)).
        Otherwise pop the first available trajectory from any group.
        """
        with self._cond:
            if trajectory_id is not None:
                if trajectory_id not in self.tid_pid_map:
                    return None
                prompt_id = self.tid_pid_map.pop(trajectory_id)
                traj = self.groups[prompt_id].pop(trajectory_id)
                self._scored.discard(prompt_id)
                self._cleanup_group(prompt_id)
                self._cond.notify()
                return traj
            else:
                for prompt_id, group in self.groups.items():
                    if group:
                        first_tid = next(iter(group))
                        traj = group.pop(first_tid)
                        self.tid_pid_map.pop(first_tid)
                        self._scored.discard(prompt_id)
                        self._cleanup_group(prompt_id)
                        self._cond.notify()
                        return traj
                return None

    # ------------------------------------------------------------------
    # Public API — group operations
    # ------------------------------------------------------------------

    def add_group(self, traj_group: TrajectoryGroup) -> None:
        """Add a TrajectoryGroup to the queue.

        *len(traj_group.trajectories)* must be a positive multiple of *group_size*.
        Blocks if *maxsize* is set and there is not enough room.
        """
        trajectories = traj_group.trajectories
        if not trajectories:
            return
        group_size = self._group_size_for(trajectories[0].ds_index)
        if len(trajectories) % group_size != 0:
            raise ValueError(
                f"Group length must be a multiple of group_size ({group_size}), "
                f"got {len(trajectories)}"
            )

        with self._cond:
            if self.maxsize > 0:
                while self._total_count() + len(trajectories) > self.maxsize:
                    self._cond.wait()

            try:
                # Validate entire list before mutating any state
                prompt_id = trajectories[0].prompt_id
                seen: set[str] = set()
                for traj in trajectories:
                    if not traj.trajectory_id or not traj.prompt_id:
                        raise ValueError("trajectory_id and prompt_id must not be empty")
                    if traj.prompt_id != prompt_id:
                        raise ValueError("All trajectories in a group must have the same prompt_id")
                    if traj.ds_index != trajectories[0].ds_index:
                        raise ValueError("All trajectories in a group must have the same ds_index")
                    if traj.trajectory_id in self.tid_pid_map or traj.trajectory_id in seen:
                        raise ValueError(f"Duplicate trajectory_id {traj.trajectory_id}")
                    seen.add(traj.trajectory_id)

                # All valid — commit to state
                # Write groups first so a partial failure in map update
                # leaves the map clean rather than orphaned entries.
                if prompt_id not in self.groups:
                    self.groups[prompt_id] = {}
                elif next(iter(self.groups[prompt_id].values())).ds_index != trajectories[0].ds_index:
                    raise ValueError("All trajectories in a group must have the same ds_index")
                for traj in trajectories:
                    self.groups[prompt_id][traj.trajectory_id] = traj
                    self.tid_pid_map[traj.trajectory_id] = prompt_id
            finally:
                # Always notify other waiters after we've potentially waited for maxsize.
                # Use notify_all() because multiple complete groups may become ready.
                self._cond.notify_all()

    def pop_group(self, prompt_id: str | None = None) -> TrajectoryGroup | None:
        """Pop a group of *group_size* Trajectories as a TrajectoryGroup.

        If *prompt_id* is ``None``, pop the first ready group from any prompt.
        """
        with self._cond:
            if prompt_id is None:
                # Use list() to safely iterate even though we return immediately,
                # guarding against future refactors that might continue the loop.
                for pid, group in list(self.groups.items()):
                    group_size = self._group_size_for_prompt(pid)
                    if len(group) >= group_size and (
                        next(iter(group.values())).ds_index not in self.group_rm_ds or pid in self._scored
                    ):
                        result = self._pop_n_from_group(pid, group_size)
                        self._cond.notify()
                        return result
            elif prompt_id in self.groups:
                group = self.groups[prompt_id]
                group_size = self._group_size_for_prompt(prompt_id)
                if len(group) >= group_size and (
                    next(iter(group.values())).ds_index not in self.group_rm_ds or prompt_id in self._scored
                ):
                    result = self._pop_n_from_group(prompt_id, group_size)
                    self._cond.notify()
                    return result
            return None

    def wait_for_group(
        self,
        timeout: float | None = None,
        will_collect: Callable[[dict[str, Trajectory]], bool] | None = None,
    ) -> TrajectoryGroup | None:
        """Block until a ready group satisfying *will_collect* is available, then pop and return it.

        If *will_collect* is provided, only groups whose ``will_collect(group)`` returns True
        will be popped. The group passed to the callback is the internal trajectory_id -> Trajectory
        mapping and must not be mutated. Groups that do not satisfy the condition remain in the queue.

        Returns ``None`` if *timeout* expires before a qualifying group is ready.
        """
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be >= 0")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while True:
                result = self._find_and_pop_ready_group(will_collect)
                if result is not None:
                    return result
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                else:
                    remaining = None
                if not self._cond.wait(remaining):
                    return None  # timed out

    # ------------------------------------------------------------------
    # Public API — group reward model
    # ------------------------------------------------------------------

    def claim_group_rm(self, prompt_id: str) -> list[Trajectory] | None:
        """Claim a complete, unscored group of a group_rm data source for scoring.

        Returns deep copies of the group's first *group_size* trajectories and
        adds *prompt_id* to ``_scoring``. Returns ``None`` when there is nothing to score:

        - The data source does not use group_rm.
        - The group is not complete yet, i.e. the first n-1 emits of the group.
        - The group is already in ``_scoring`` (another emit is scoring it) or
          in ``_scored``.
        - The group has an ABORTED or FAILED member. It is marked scored and
          released to consumers without scoring. After partial rollout resumes
          it, ``pop_all`` has cleared that mark, so the fully COMPLETED group is
          scored then.
        - Every member carries ``GROUP_RM_SCORED_KEY``: a group scored before an
          abort and re-emitted unchanged on resume. It is marked scored and keeps
          its previous rewards.
        """
        with self._cond:
            group = self.groups.get(prompt_id)
            if not group or prompt_id in self._scoring or prompt_id in self._scored:
                return None
            ds_index = next(iter(group.values())).ds_index
            group_size = self._group_size_for(ds_index)
            if ds_index not in self.group_rm_ds or len(group) < group_size:
                return None
            trajectories = list(group.values())[:group_size]
            if (any(t.status != TrajectoryStatus.COMPLETED for t in trajectories)
                    or all(t.metadata.get(GROUP_RM_SCORED_KEY) for t in trajectories)):
                self._scored.add(prompt_id)
                self._cond.notify_all()
                return None
            self._scoring.add(prompt_id)
            return [t.model_copy(deep=True) for t in trajectories]

    def commit_group_rm(self, prompt_id: str, trajectories: list[Trajectory]) -> None:
        """Replace a claimed group's trajectories with their scored copies and mark it ready."""
        with self._cond:
            if prompt_id not in self._scoring:
                return  # drained while scoring; drop the result
            self._scoring.discard(prompt_id)
            group = self.groups[prompt_id]
            for traj in trajectories:
                if traj.trajectory_id in group:
                    group[traj.trajectory_id] = traj
            self._scored.add(prompt_id)
            self._cond.notify_all()

    def release_group_rm(self, prompt_id: str) -> None:
        """Give up a claim without scoring, so the group can be claimed again."""
        with self._cond:
            self._scoring.discard(prompt_id)

    # ------------------------------------------------------------------
    # Public API — bulk operations
    # ------------------------------------------------------------------

    def pop_all(self) -> list[Trajectory]:
        """Pop and return all trajectories from the queue."""
        with self._cond:
            all_items = []
            for group in self.groups.values():
                all_items.extend(group.values())
            self.groups.clear()
            self.tid_pid_map.clear()
            self._scoring.clear()
            self._scored.clear()
            self._cond.notify_all()
            return all_items

    def pop_all_group(self) -> list[TrajectoryGroup]:
        """Pop and return all ready groups (groups with >= group_size trajectories)."""
        with self._cond:
            all_groups: list[TrajectoryGroup] = []
            for prompt_id in list(self.groups.keys()):
                while prompt_id in self.groups:
                    group = self.groups[prompt_id]
                    group_size = self._group_size_for_prompt(prompt_id)
                    if len(group) < group_size or (
                        next(iter(group.values())).ds_index in self.group_rm_ds and prompt_id not in self._scored
                    ):
                        break
                    all_groups.append(self._pop_n_from_group(prompt_id, group_size))
            if all_groups:
                self._cond.notify_all()
            return all_groups

    # ------------------------------------------------------------------
    # Public API — introspection / observability
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the total number of trajectories in the queue."""
        with self._cond:
            return self._total_count()

    def is_empty(self) -> bool:
        """Return True if the queue has no trajectories."""
        with self._cond:
            return self._total_count() == 0

    def group_count(self) -> int:
        """Return the number of distinct prompt_id groups."""
        with self._cond:
            return len(self.groups)

    def pending_prompt_ids(self) -> list[str]:
        """Return list of prompt_ids that have pending trajectories."""
        with self._cond:
            return list(self.groups.keys())

    def group_size_of(self, prompt_id: str) -> int:
        """Return the number of trajectories for a given prompt_id."""
        with self._cond:
            return len(self.groups.get(prompt_id, {}))
