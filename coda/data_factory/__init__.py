"""Data factory components and its pluggable registries."""

from coda.utils.registry import Registry

# Package-level registry: built-in and user-registered filters share one namespace.
DATA_FILTER_REGISTRY = Registry("data_filter")

register_data_filter = DATA_FILTER_REGISTRY.register
get_data_filter = DATA_FILTER_REGISTRY.get
list_data_filters = DATA_FILTER_REGISTRY.keys

# Registry for pluggable buffer-replay-strategy functions.
# Signature: (config, step, buffer: list, num: int) -> list
BUFFER_REPLAY_STRATEGY_REGISTRY = Registry("buffer_replay_strategy")

register_buffer_replay_strategy = BUFFER_REPLAY_STRATEGY_REGISTRY.register
get_buffer_replay_strategy = BUFFER_REPLAY_STRATEGY_REGISTRY.get
list_buffer_replay_strategies = BUFFER_REPLAY_STRATEGY_REGISTRY.keys

# Registry for pluggable raw-record pre-processors, applied by `Dataset` to every raw dataset
# record before it is turned into messages.
# Signature: (data: dict, prompt_key: str) -> dict
DATA_PRE_PROCESSOR_REGISTRY = Registry("data_pre_processor")

register_data_pre_processor = DATA_PRE_PROCESSOR_REGISTRY.register
get_data_pre_processor = DATA_PRE_PROCESSOR_REGISTRY.get
list_data_pre_processors = DATA_PRE_PROCESSOR_REGISTRY.keys

__all__ = [
    "DATA_FILTER_REGISTRY",
    "register_data_filter",
    "get_data_filter",
    "list_data_filters",
    "BUFFER_REPLAY_STRATEGY_REGISTRY",
    "register_buffer_replay_strategy",
    "get_buffer_replay_strategy",
    "list_buffer_replay_strategies",
    "DATA_PRE_PROCESSOR_REGISTRY",
    "register_data_pre_processor",
    "get_data_pre_processor",
    "list_data_pre_processors",
]

# Import built-in filters, pre-processors and replay strategies so their register decorators execute.
# Must stay below the aliases above: these modules import them from this package.
from coda.data_factory import data_filter  # noqa: E402,F401
from coda.data_factory import data_pre_processor  # noqa: E402,F401
from coda.data_factory import data_source  # noqa: E402,F401
