"""Filter chain for post-processing TrajectoryGroup batches.

DataFilter applies a configurable sequence of filter functions to a TrajectoryGroup.
Each filter receives a TrajectoryGroup and returns either the group (keep) or None (drop).
Filters are identified by name in a config map and may carry arbitrary keyword parameters.

Built-in filters
----------------
- reward: Keep a trajectory group only when its rewards are not all identical.
- status: Keep a trajectory group only when none of its trajectories have FAILED status.

Registering custom filters
--------------------------
Use the package-level register_data_filter decorator to register additional filters:

    from coda.data_factory import register_data_filter

    @register_data_filter("my_filter")
    def my_filter(traj_group: TrajectoryGroup, **kwargs) -> TrajectoryGroup | None:
        return traj_group if <keep_condition> else None

Configuration example (YAML):
    rollout:
      filter:
        status:              # enabled, no params (null/empty = enabled)
        reward: false        # explicitly disabled
        custom_filter:
          threshold: 0.5    # params forwarded as kwargs
"""

import inspect
import logging
from typing import Any, Callable

from coda.agentflow.trajectory_store import TrajectoryGroup, TrajectoryStatus
from coda.data_factory import get_data_filter, register_data_filter

logger = logging.getLogger(__name__)


@register_data_filter("reward")
def _filter_by_reward(traj_group: TrajectoryGroup) -> TrajectoryGroup | None:
    """Keep a trajectory group only when its rewards are not all identical.

    A group whose trajectories all carry the same reward (all failed, all succeeded, or
    any other uniform value) carries no contrastive training signal and is dropped.  A
    group is also dropped if any trajectory's reward is None.

    Args:
        traj_group: A group of Trajectory objects(i.e. one prompt's n rollouts).

    Returns:
        The original group unchanged if it passes the filter, or None if the group is dropped.
    """
    if not traj_group.trajectories:
        logger.info("Traj_group's trajectories is empty, thus drop this traj_group")
        return None

    rewards = [t.reward for t in traj_group.trajectories]
    if any(r is None for r in rewards):
        logger.info("Some trajectory's reward is None, thus drop this traj_group")
        return None

    # if rewards are same return None
    if len(set(rewards)) <= 1:
        logger.info("prompt id %s rewards are same %s drop it", traj_group.prompt_id, rewards)
        return None

    return traj_group

@register_data_filter("status")
def _filter_by_status(traj_group: TrajectoryGroup) -> TrajectoryGroup | None:
    """Keep a trajectory group only when its status are not failed.

    Args:
        traj_group: A group of Trajectory objects(i.e. one prompt's n rollouts).

    Returns:
        The original group unchanged if it passes the filter, or None if the group is dropped.
    """
    if not traj_group.trajectories:
        logger.info("Traj_group's trajectories is empty, thus drop this traj_group")
        return None

    statuses = [t.status for t in traj_group.trajectories]
    if any(r is TrajectoryStatus.FAILED for r in statuses):
        logger.warning("Some trajectory's status is Failed, thus drop traj_group: %s", traj_group.prompt_id)
        return None

    return traj_group

class _BoundFilter:
    """A filter function with pre-bound keyword parameters.

    Wraps a registered filter callable together with its parameter dict so that the filter can be invoked with a single
    traj_group argument while retaining its registry name for logging and debugging purposes.

    Args:
        name: Registry key used for this filter (preserved for logging).
        func: The underlying filter callable.
        params: Keyword arguments forwarded to *func* on every call.
    """

    def __init__(
        self,
        name: str,
        func: Callable[..., TrajectoryGroup | None],
        params: dict[str, Any],
    ) -> None:
        self.name = name
        self._func = func
        self._params = dict(params)  # shallow copy — insulates from caller mutations

    def __call__(self, traj_group: TrajectoryGroup) -> TrajectoryGroup | None:
        return self._func(traj_group, **self._params)


class DataFilter:
    """Filter chain that processes a TrajectoryGroup.

    On construction the caller passes a map of filter configs. Each key is a filter name
    registered in DATA_FILTER_REGISTRY, and the value controls behavior:
      - None (or any non-dict/non-False): enabled with no params
      - dict: enabled with params forwarded as kwargs
      - False: explicitly disabled (skipped)

    Hydra merge semantics: default.yaml defines the baseline filters. User configs
    override same-name keys (set to false to disable) and add new keys.

    Attributes:
        chain: Ordered list of _BoundFilter callables built from *filter_configs*.

    Args:
        filter_configs: Dict mapping filter names to param dicts, None, or False::
            {
                "status": None,              # enabled, no params
                "reward": False,             # disabled
                "my_filter": {"param_key": "param_value"},     # enabled with kwargs
            }

    Raises:
        KeyError: If a config entry references an unregistered filter name.
        TypeError: If a config entry's params are incompatible with the filter signature.

    Example::
        df = DataFilter({"reward": None, "status": None})
        result = df.apply(traj_group)  # returns TrajectoryGroup or None
    """

    def __init__(self, filter_configs: dict[str, dict[str, Any] | bool | None] | None):
        self.chain: list[_BoundFilter] = []
        if not filter_configs:
            logger.info("no data filter needed to apply")
            return None
        for name, params in filter_configs.items():
            if params is False:
                # False value means the filter is explicitly disabled (e.g. user override)
                logger.info(f"disabled filter {name}")
                continue
            params = params if isinstance(params, dict) else {}
            func = get_data_filter(name)
            # Validate params statically via signature inspection — avoids calling the function body and thus
            # sidesteps any side-effects custom filters might have (logging, counters, remote calls, etc.).
            try:
                inspect.signature(func).bind([], **params)
            except TypeError as exc:
                raise TypeError(f"[data_filter] Invalid params for filter '{name}': {exc}") from exc
            logger.info(f"apply filter {name}")
            self.chain.append(_BoundFilter(name, func, params))

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def apply(self, traj_group: TrajectoryGroup) -> TrajectoryGroup | None:
        """Apply the filter chain in order and return the surviving group.

        Each filter in chain receives the output of the previous one, so filters compose sequentially.
        Returns None if any filter drops the group.

        Args:
            traj_group: Input TrajectoryGroup to filter.

        Returns:
            The TrajectoryGroup if it passes all filters, or None if dropped.
        """
        result: TrajectoryGroup | None = traj_group
        for bound_filter in self.chain:
            if result is None:
                break

            before = len(result.trajectories)
            result = bound_filter(result)
            if result is None:
                logger.info("Filter %s() dropped all %d trajectories.", bound_filter.name, before)

        return result
