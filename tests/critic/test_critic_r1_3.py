# REQ-2 regression: RAG provider timeout must be a positive number.

from __future__ import annotations

from typing import Any

import pytest

from conversational_agent.core.engine import ConversationEngine
from conversational_agent.core.models import LLMAction
from conversational_agent.providers import ProviderValidationError


class NanTimeoutProvider:
    provider_id = 'nan-provider'
    timeout = float('nan')

    def retrieve(self, query: str, **kwargs: Any) -> list[Any]:
        return []


class AnswerLLM:
    def next_action(self, *args: Any, **kwargs: Any) -> LLMAction:
        return LLMAction(action='answer', text='ok')


def test_nan_rag_timeout_is_rejected_at_startup() -> None:
    with pytest.raises(ProviderValidationError):
        ConversationEngine(providers=[NanTimeoutProvider()], llm=AnswerLLM())
