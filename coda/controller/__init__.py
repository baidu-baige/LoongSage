"""Controller components and its pluggable registries."""

from coda.utils.registry import Registry

# Package-level registry: built-in and user-registered sliding window strategies
# share one namespace. Registered classes must subclass SlidingWindowStrategy.
SLIDING_WINDOW_STRATEGY_REGISTRY = Registry("sliding_window_strategy")

register_sliding_window_strategy = SLIDING_WINDOW_STRATEGY_REGISTRY.register
get_sliding_window_strategy = SLIDING_WINDOW_STRATEGY_REGISTRY.get
list_sliding_window_strategies = SLIDING_WINDOW_STRATEGY_REGISTRY.keys

__all__ = [
    "SLIDING_WINDOW_STRATEGY_REGISTRY",
    "register_sliding_window_strategy",
    "get_sliding_window_strategy",
    "list_sliding_window_strategies",
]

# Import built-in strategies so their register decorators execute.
# Must stay below the aliases above: rollout_sampler imports them from this package.
from coda.controller import rollout_sampler  # noqa: E402,F401
