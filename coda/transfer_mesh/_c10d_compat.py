"""Adapter layer for torch.distributed.distributed_c10d private APIs.

Centralizes loading and validation of private symbols so that the rest of the
package never imports from distributed_c10d directly.  Any breakage caused by
an upstream PyTorch refactor surfaces here with a clear, version-stamped error.
"""

from typing import Any

import torch
import torch.distributed as dist


def _get_distributed_c10d_symbols() -> tuple[Any, Any, Any]:
    """Load and validate private symbols from torch.distributed.distributed_c10d.

    Returns:
        (PrefixStore, _new_process_group_helper, _world)

    Raises:
        RuntimeError: if any required symbol is missing, with torch.__version__.
    """
    required = ("PrefixStore", "_new_process_group_helper", "_world")
    try:
        import torch.distributed.distributed_c10d as _c10d
        missing = [sym for sym in required if not hasattr(_c10d, sym)]
        if missing:
            raise AttributeError(f"Missing symbols: {missing}")
        return (
            getattr(_c10d, "PrefixStore"),
            getattr(_c10d, "_new_process_group_helper"),
            getattr(_c10d, "_world"),
        )
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            f"torch.distributed.distributed_c10d private API unavailable "
            f"(torch=={torch.__version__}): {exc}"
        ) from exc


def _register_process_group_ranks(
    _world: Any,
    pg: dist.ProcessGroup,
    ranks_map: dict[int, int],
) -> None:
    """Write rank mapping into _world.pg_group_ranks.

    Args:
        _world: The _world singleton from distributed_c10d.
        pg: The process group to register.
        ranks_map: Mapping of {local_rank: global_rank}.

    Raises:
        RuntimeError: if _world.pg_group_ranks does not exist or is not a dict.
    """
    if not hasattr(_world, "pg_group_ranks"):
        raise RuntimeError(
            f"torch.distributed._world.pg_group_ranks is unavailable "
            f"(torch=={torch.__version__}). The internal structure may have changed."
        )
    store = getattr(_world, "pg_group_ranks")
    if not isinstance(store, dict):
        raise RuntimeError(
            f"torch.distributed._world.pg_group_ranks is not a dict "
            f"(got {type(store).__name__}, torch=={torch.__version__})."
        )
    store[pg] = ranks_map
