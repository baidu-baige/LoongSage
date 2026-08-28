"""Agent interfaces and helpers."""

from coda.agentflow.agent.base_agent import BaseAgent
from coda.utils.registry import Registry

AGENT_REGISTRY = Registry("agent")

register_agent = AGENT_REGISTRY.register
get_agent_class = AGENT_REGISTRY.get
list_agents = AGENT_REGISTRY.keys

__all__ = [
    "BaseAgent",
    "AGENT_REGISTRY",
    "register_agent",
    "get_agent_class",
    "list_agents",
]

# Auto-discover built-in agents so their @register_agent decorators execute.
# Each module is imported independently so one agent's missing dependency won't
# prevent other agents from loading.
import pkgutil as _pkgutil  # noqa: E402
import logging as _logging  # noqa: E402
_logger = _logging.getLogger(__name__)
for _, _name, _ in _pkgutil.walk_packages(__path__, __name__ + "."):
    try:
        __import__(_name)
    except ImportError as _e:
        _logger.debug("Skip agent module %s (missing dependency): %s", _name, _e)
