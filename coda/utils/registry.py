"""Generic named registry for pluggable components.

Provides a `Registry` class that any subsystem can instantiate to create its own isolated namespace of named
callables (or other objects).

Typical usage::

    # In the subsystem module
    from coda.utils.registry import Registry

    MY_REGISTRY = Registry("my_component")

    @MY_REGISTRY.register("foo")
    def foo_impl(...):
        ...

    # At call site
    fn = MY_REGISTRY.get("foo")
"""
from __future__ import annotations

from typing import Any, Callable


class Registry:
    """A simple name-to-object registry.

    Each Registry instance maintains its own isolated mapping, so different subsystems do not interfere with each other

    Args:
        name: Human-readable label for this registry, used in error messages.

    Example::
        BUFFER_REPLAY_STRATEGY_REGISTRY = Registry("buffer_replay_strategy")

        register_buffer_replay_strategy = BUFFER_REPLAY_STRATEGY_REGISTRY.register
        get_buffer_replay_strategy = BUFFER_REPLAY_STRATEGY_REGISTRY.get

        @register_buffer_replay_strategy("fifo")
        def fifo(config, step, buffer, num):
            ...

        fn = get_buffer_replay_strategy("fifo")
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._registry: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, name: str) -> Callable:
        """Return a decorator that registers the decorated object under *name*.

        Args:
            name: The lookup key.  Must be unique within this registry.

        Returns:
            A decorator that stores the object and returns it unchanged.

        Raises:
            ValueError: If *name* is already registered.
        """
        def decorator(obj: Any) -> Any:
            if name in self._registry:
                raise ValueError(f"[{self._name}] '{name}' is already registered.")
            self._registry[name] = obj

            return obj

        return decorator

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> Any:
        """Return the object registered under *name*.

        Args:
            name: The lookup key.

        Returns:
            The registered object.

        Raises:
            KeyError: If *name* is not found.
        """
        if name not in self._registry:
            available = list(self._registry)
            raise KeyError(
                f"[{self._name}] '{name}' is not registered. "
                f"Available: {available}"
            )
        return self._registry[name]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __contains__(self, name: str) -> bool:
        """Support `'foo' in registry` membership tests."""
        return name in self._registry

    def __len__(self) -> int:
        """Return the number of registered entries."""
        return len(self._registry)

    def keys(self) -> list[str]:
        """Return all registered names."""
        return list(self._registry)

    def __repr__(self) -> str:
        return f"Registry(name={self._name!r}, keys={self.keys()})"
