# REQ-5 regression: a failed eviction write must not be discarded.

from __future__ import annotations

from typing import Any

from conversational_agent.core.engine import ConversationEngine
from conversational_agent.core.models import FrontendType, LLMAction, TurnRequest


class Provider:
    provider_id = 'p1'
    timeout = 1.0

    def retrieve(self, query: str, **kwargs: Any) -> list[Any]:
        return []


class AnswerLLM:
    def next_action(self, *args: Any, **kwargs: Any) -> LLMAction:
        return LLMAction(action='answer', text='ok')


class FailingStore:
    def __init__(self) -> None:
        self.save_attempts = 0

    def save(self, snapshot: dict[str, Any]) -> None:
        self.save_attempts += 1
        raise RuntimeError('database unavailable')

    def load(self, conversation_id: str) -> dict[str, Any] | None:
        return None


def test_failed_eviction_write_keeps_dirty_conversation() -> None:
    engine = ConversationEngine(
        providers=[Provider()],
        llm=AnswerLLM(),
        store=FailingStore(),
        cache_capacity=1,
    )

    list(
        engine.run_turn(
            TurnRequest(
                conversation_id='A',
                idempotency_key='a1',
                text='hello A',
                frontend=FrontendType.TEXT,
            )
        )
    )
    list(
        engine.run_turn(
            TurnRequest(
                conversation_id='B',
                idempotency_key='b1',
                text='hello B',
                frontend=FrontendType.TEXT,
            )
        )
    )

    # REQ-5: when the DB write fails during eviction the conversation must be
    # retained in memory (or dirty), never silently discarded.  With capacity 1,
    # inserting B should not have removed A after the failed A persistence.
    conversation_a = engine.get_conversation('A')
    assert len(conversation_a.turns) == 1
    assert conversation_a.turns[0].user_text == 'hello A'
