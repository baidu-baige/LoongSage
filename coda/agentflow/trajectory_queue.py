"""Trajectory Queue between SingleController and Agentflow"""
import threading
import time
from typing import Callable

from coda.agentflow.trajectory_store import Trajectory, TrajectoryGroup


class TrajQueue:
    """Thread-safe queue for managing Trajectory objects grouped by prompt_id.

    Uses threading.Condition for producer-consumer signaling (wait_for_group)
    and an optional maxsize for backpressure.

    Internally, each group stores trajectories in an insertion-ordered dict
    (trajectory_id -> Trajectory), giving O(1) lookup and removal by id.
    """
    def __init__(self, group_sizes: dict[int, int], maxsize: int = 0):
        self.group_sizes = {int(k): int(v) for k, v in group_sizes.items()}
        self.maxsize = maxsize  # 0 means unlimited
        # prompt_id -> {trajectory_id -> Trajectory} (dict preserves insertion order)
        self.groups: dict[str, dict[str, Trajectory]] = {}
        # trajectory_id -> prompt_id, for quick lookup
        self.tid_pid_map: dict[str, str] = {}
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

    def _pop_n_from_group(self, prompt_id: str, n: int) -> TrajectoryGroup:
        """Pop the first *n* trajectories from a group. Caller must hold lock."""
        group = self.groups[prompt_id]
        keys = list(group.keys())[:n]
        items = []
        for k in keys:
            items.append(group.pop(k))
            self.tid_pid_map.pop(k)
        self._cleanup_group(prompt_id)
        return TrajectoryGroup(prompt_id=prompt_id, trajectories=items)

    def _find_and_pop_ready_group(
        self, will_collect: Callable[[dict[str, Trajectory]], bool] | None = None
    ) -> TrajectoryGroup | None:
        """Find and pop the first ready group whose full prompt group satisfies will_collect. Caller must hold lock."""
        for pid, group in list(self.groups.items()):
            group_size = self._group_size_for_prompt(pid)
            if len(group) >= group_size:
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
                self._cleanup_group(prompt_id)
                self._cond.notify()
                return traj
            else:
                for prompt_id, group in self.groups.items():
                    if group:
                        first_tid = next(iter(group))
                        traj = group.pop(first_tid)
                        self.tid_pid_map.pop(first_tid)
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
                    if len(group) >= group_size:
                        result = self._pop_n_from_group(pid, group_size)
                        self._cond.notify()
                        return result
            elif prompt_id in self.groups:
                group_size = self._group_size_for_prompt(prompt_id)
                if len(self.groups[prompt_id]) >= group_size:
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
            self._cond.notify_all()
            return all_items

    def pop_all_group(self) -> list[TrajectoryGroup]:
        """Pop and return all ready groups (groups with >= group_size trajectories)."""
        with self._cond:
            all_groups: list[TrajectoryGroup] = []
            for prompt_id in list(self.groups.keys()):
                while prompt_id in self.groups:
                    group_size = self._group_size_for_prompt(prompt_id)
                    if len(self.groups.get(prompt_id, {})) < group_size:
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
