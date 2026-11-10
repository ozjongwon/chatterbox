# REQ-10 regression: cancellation during an in-flight LLM call.

from __future__ import annotations

import threading
from typing import Any

from conversational_agent.core.engine import ConversationEngine
from conversational_agent.core.models import (
    FrontendType,
    LLMAction,
    TurnRequest,
    TurnStatus,
)
from conversational_agent.storage import InMemoryConversationStore


class Provider:
    provider_id = 'p1'
    timeout = 1.0

    def retrieve(self, query: str, **kwargs: Any) -> list[Any]:
        return []


class BlockingLLM:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def next_action(self, *args: Any, **kwargs: Any) -> LLMAction:
        self.entered.set()
        if not self.release.wait(10):
            raise RuntimeError('test LLM was not released')
        return LLMAction(action='answer', text='finished after cancellation')


def test_cancellation_during_llm_marks_cancelled_and_persists_nothing() -> None:
    llm = BlockingLLM()
    engine = ConversationEngine(
        providers=[Provider()],
        llm=llm,
        store=InMemoryConversationStore(),
        cache_capacity=2,
        llm_first_token_timeout=10.0,
        total_turn_timeout=10.0,
    )

    gen = engine.run_turn(
        TurnRequest(
            conversation_id='c',
            idempotency_key='k',
            text='hello',
            frontend=FrontendType.TEXT,
        )
    )
    accepted = next(gen)
    assert accepted['type'] == 'turn.accepted'
    turn_id = accepted['turn_id']

    received: list[dict[str, Any]] = []

    def consume() -> None:
        received.extend(gen)

    worker = threading.Thread(target=consume)
    worker.start()

    assert llm.entered.wait(2)
    engine.cancel('c', turn_id)
    llm.release.set()
    worker.join(5)

    assert not worker.is_alive(), 'turn processing did not terminate'
    final = received[-1]
    assert final['type'] == 'turn.cancelled'
    assert final['status'] == TurnStatus.CANCELLED
    assert len(engine.get_conversation('c').turns) == 0
