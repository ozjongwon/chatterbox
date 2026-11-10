# Edge-case regression: an LLM tool request with a whitespace-only retrieval
# query is malformed and SHALL be rejected.  The current action validator uses
# bool(query), so "   " passes validation and the core makes a real RAG call.

from __future__ import annotations

from typing import Any

from conversational_agent.core.engine import ConversationEngine
from conversational_agent.core.models import LLMAction, TurnRequest, TurnStatus


class Provider:
    provider_id = 'p1'
    timeout = 1.0

    def __init__(self) -> None:
        self.calls: list[str] = []

    def retrieve(self, query: str, **kwargs: Any) -> list[Any]:
        self.calls.append(query)
        return [{'evidence_id': 'e1'}]


class ScriptedLLM:
    def __init__(self) -> None:
        self.actions = [
            LLMAction(action='retrieve', provider_id='p1', query='   '),
            LLMAction(action='answer', text='ok'),
        ]

    def next_action(self, *args: Any, **kwargs: Any) -> LLMAction:
        return self.actions.pop(0)


def test_whitespace_only_retrieval_query_is_rejected_as_malformed() -> None:
    provider = Provider()
    engine = ConversationEngine(
        providers=[provider],
        llm=ScriptedLLM(),
        cache_capacity=2,
        max_steps=2,
        max_rag_calls=2,
    )

    events = list(
        engine.run_turn(
            TurnRequest(
                conversation_id='c',
                idempotency_key='k',
                text='hello',
            )
        )
    )

    final = events[-1]
    assert final['type'] == 'turn.error'
    assert final['status'] == TurnStatus.ERROR
    assert final['error_code'] == 'INVALID_LLM_ACTION'
    assert provider.calls == []
