"""Conversational agent with interchangeable text and voice frontends."""

from __future__ import annotations

from conversational_agent.cache import LRUCache
from conversational_agent.core.engine import ConversationEngine
from conversational_agent.core.models import (
    Conversation,
    FrontendType,
    LLMAction,
    Turn,
    TurnRequest,
    TurnStatus,
    ValidationError,
)
from conversational_agent.fakes import FakeRAGProvider, FakeTextLLM
from conversational_agent.providers import ProviderValidationError, RAGProvider
from conversational_agent.storage import (
    FunctionConversationStore,
    InMemoryConversationStore,
)
from conversational_agent.support import (
    FakeAttachmentGuard,
    FakeOrderRepository,
    SupportScenarioLLM,
    support_tools,
)

__all__ = [
    'Conversation',
    'ConversationEngine',
    'FakeAttachmentGuard',
    'FakeOrderRepository',
    'FakeRAGProvider',
    'FakeTextLLM',
    'FrontendType',
    'FunctionConversationStore',
    'InMemoryConversationStore',
    'LLMAction',
    'LRUCache',
    'ProviderValidationError',
    'RAGProvider',
    'SupportScenarioLLM',
    'Turn',
    'TurnRequest',
    'TurnStatus',
    'ValidationError',
    'support_tools',
]
