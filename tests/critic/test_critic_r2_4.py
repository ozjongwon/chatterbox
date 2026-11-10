# REQ-8 regression: the engine's configurable RAG timeout is ignored.
# _retrieve uses provider.timeout instead of self._rag_timeout, so a RAG call
# can outlive the timeout supplied to ConversationEngine and still produce a
# successful, non-degraded turn.

from __future__ import annotations

import time
from typing import Any

from conversational_agent.core.engine import ConversationEngine
from conversational_agent.core.models import LLMAction, TurnRequest, TurnStatus


class SlowProvider:
    provider_id = 'p1'
    timeout = 10.0

    def retrieve(self, query: str, **kwargs: Any) -> list[Any]:
        time.sleep(0.3)
        return [{'evidence_id': 'e1'}]


class ScriptedLLM:
    def __init__(self) -> None:
        self.actions = [
            LLMAction(action='retrieve', provider_id='p1', query='q'),
            LLMAction(action='answer', text='answered'),
        ]

    def next_action(self, *args: Any, **kwargs: Any) -> LLMAction:
        return self.actions.pop(0)


def test_engine_rag_timeout_is_enforced() -> None:
    engine = ConversationEngine(
        providers=[SlowProvider()],
        llm=ScriptedLLM(),
        cache_capacity=2,
        max_steps=2,
        max_rag_calls=1,
        rag_timeout=0.1,
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
    assert final['type'] == 'turn.final'
    assert final['status'] == TurnStatus.SUCCESS
    assert final['degraded'] is True

    rag_trace_items = [
        item for item in final['trace'] if item.get('type') == 'rag_call'
    ]
    assert len(rag_trace_items) == 1
    assert rag_trace_items[0]['status'] == 'degraded'
