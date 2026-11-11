from __future__ import annotations

from typing import Any

import pytest

from conversational_agent import ConversationEngine
from conversational_agent.contracts import is_llm_client
from conversational_agent.providers import (
    ProviderValidationError,
    is_rag_provider,
    validate_providers,
)
from conversational_agent.storage import InMemoryConversationStore, is_conversation_store


class GoodProvider:
    provider_id = "good"
    timeout = 1.0

    def retrieve(self, query: str, **kwargs: Any) -> list[Any]:
        return []


class BadProvider:
    provider_id = "bad"
    timeout = 1.0
    retrieve = 42


class LLM:
    def next_action(self, **kwargs: Any) -> Any:
        return None


def test_type_guards_validate_runtime_contracts() -> None:
    assert is_rag_provider(GoodProvider())
    assert not is_rag_provider(BadProvider())
    assert is_llm_client(LLM())
    assert is_conversation_store(InMemoryConversationStore())
    assert not is_conversation_store(object())


def test_invalid_provider_has_actionable_error() -> None:
    with pytest.raises(ProviderValidationError) as exc:
        validate_providers([BadProvider()])
    assert exc.value.code == "INVALID_PROVIDER"


def test_engine_rejects_invalid_adapters_early() -> None:
    with pytest.raises(TypeError, match="next_action"):
        ConversationEngine(providers=[GoodProvider()], llm=object())
    with pytest.raises(TypeError, match="save and load"):
        ConversationEngine(
            providers=[GoodProvider()],
            llm=LLM(),
            store=object(),
        )
