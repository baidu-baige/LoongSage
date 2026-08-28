"""Router package: lookup helpers for Router middleware classes."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping
from typing import Any

DEFAULT_MIDDLEWARE_NAME = "parser"

logger = logging.getLogger(__name__)

_MIDDLEWARE_CLASSES: dict[str, type[Any] | str] = {}


def add_middleware_class(name: str, middleware_cls: type[Any] | str) -> None:
    """Bind a Router middleware class to a stable config name."""
    if not name:
        raise ValueError("Middleware name must be a non-empty string")
    _MIDDLEWARE_CLASSES[name] = middleware_cls


def get_middleware_class(name: str) -> type[Any]:
    """Look up a Router middleware class by its config name."""
    try:
        middleware = _MIDDLEWARE_CLASSES[name]
    except KeyError as exc:
        known = ", ".join(sorted(_MIDDLEWARE_CLASSES)) or "<empty>"
        raise ValueError(
            f"Unknown router middleware {name!r}. Known middlewares: {known}"
        ) from exc

    if isinstance(middleware, str):
        module_path, _, attr_name = middleware.rpartition(".")
        if not module_path or not attr_name:
            raise ValueError(f"Invalid middleware import path: {middleware!r}")
        module = importlib.import_module(module_path)
        middleware_cls = getattr(module, attr_name)
        if not isinstance(middleware_cls, type):
            raise TypeError(
                f"Middleware {name!r} did not resolve to a class: {middleware!r}"
            )
        _MIDDLEWARE_CLASSES[name] = middleware_cls
        return middleware_cls

    return middleware


def resolve_middleware_chain(
    middleware_configs: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Normalize YAML-friendly middleware config into an ordered list of specs.

    Config is a mapping of ``{middleware_name: params}``, e.g.::

        middleware:
          parser:
            reasoning_parser: qwen3
            tool_call_parser: qwen3_coder

    ``params`` may be a mapping of constructor kwargs, or ``null`` for no params.
    Mapping insertion order determines the wrapping order.
    """
    if middleware_configs is None:
        raw_configs: Mapping[str, Any] = {DEFAULT_MIDDLEWARE_NAME: None}
    elif isinstance(middleware_configs, Mapping):
        raw_configs = middleware_configs
    else:
        raise TypeError(
            "agentflow.router.middleware must be a mapping of {name: params}, "
            f"got {type(middleware_configs).__name__}"
        )

    resolved: list[dict[str, Any]] = []
    for name, params in raw_configs.items():
        if not name or not isinstance(name, str):
            raise ValueError(
                f"agentflow.router.middleware key must be a non-empty string, got {name!r}"
            )
        if params is None:
            params = {}
        elif not isinstance(params, Mapping):
            raise TypeError(
                f"agentflow.router.middleware[{name!r}] must be a mapping or null, "
                f"got {type(params).__name__}"
            )
        resolved.append(
            {
                "name": name,
                "middleware_cls": get_middleware_class(name),
                "params": dict(params),
            }
        )
    return resolved

def __getattr__(name: str) -> Any:
    """Lazily expose built-in middleware classes."""
    if name == "ParserMiddleware":
        return get_middleware_class(DEFAULT_MIDDLEWARE_NAME)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


add_middleware_class(
    DEFAULT_MIDDLEWARE_NAME,
    "coda.agentflow.router.parser_middleware.ParserMiddleware",
)

__all__ = [
    "DEFAULT_MIDDLEWARE_NAME",
    "ParserMiddleware",
    "add_middleware_class",
    "get_middleware_class",
    "resolve_middleware_chain",
]
