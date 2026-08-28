"""Sandbox client registry and factory helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from coda.agentflow.sandbox.base import SandboxClient
from coda.utils.registry import Registry

SANDBOX_REGISTRY = Registry("sandbox")

register_sandbox = SANDBOX_REGISTRY.register
get_sandbox_class = SANDBOX_REGISTRY.get
list_sandboxes = SANDBOX_REGISTRY.keys


def create_sandbox_client(
    sandbox_config: Mapping[str, Any] | None,
    **kwargs: Any,
) -> SandboxClient | None:
    """Create a sandbox client from config without exposing concrete classes."""
    config = dict(sandbox_config or {})
    sandbox_type = str(config.get("type", "none")).strip().lower()
    if sandbox_type in {"", "none"}:
        return None
    sandbox_cls = get_sandbox_class(sandbox_type)
    return sandbox_cls.from_config(config, **kwargs)


__all__ = [
    "SandboxClient",
    "SANDBOX_REGISTRY",
    "create_sandbox_client",
    "get_sandbox_class",
    "list_sandboxes",
    "register_sandbox",
]

# Auto-discover built-in sandbox implementations so their @register_sandbox decorators execute.
import pkgutil as _pkgutil  # noqa: E402
for _, _name, _ in _pkgutil.walk_packages(__path__, __name__ + "."):
    __import__(_name)
