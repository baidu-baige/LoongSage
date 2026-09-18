"""Protocol adapter registry."""

from .base import ProtocolAdapter
from .openai import OpenAIChatCompletionsAdapter, OpenAIResponsesAdapter

PROTOCOL_ADAPTERS: tuple[ProtocolAdapter, ...] = (
    OpenAIChatCompletionsAdapter(),
    OpenAIResponsesAdapter(),
)

__all__ = [
    "OpenAIChatCompletionsAdapter",
    "OpenAIResponsesAdapter",
    "PROTOCOL_ADAPTERS",
    "ProtocolAdapter",
]
