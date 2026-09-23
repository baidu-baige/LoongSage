"""Protocol adapter registry."""

from .base import ProtocolAdapter
from .anthropic_messages import AnthropicMessagesAdapter
from .openai import OpenAIChatCompletionsAdapter, OpenAIResponsesAdapter

PROTOCOL_ADAPTERS: tuple[ProtocolAdapter, ...] = (
    OpenAIChatCompletionsAdapter(),
    OpenAIResponsesAdapter(),
    AnthropicMessagesAdapter(),
)

__all__ = [
    "OpenAIChatCompletionsAdapter",
    "OpenAIResponsesAdapter",
    "AnthropicMessagesAdapter",
    "PROTOCOL_ADAPTERS",
    "ProtocolAdapter",
]
