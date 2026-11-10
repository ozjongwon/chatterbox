# REQ-8 regression: total_turn_timeout must bound blocking LLM work too.
# The implementation checks the deadline only at the top of the step loop;
# a blocking LLM call that returns after the deadline still becomes a SUCCESS.

from __future__ import annotations

import time
from typing import Any

from conversational_agent.core.engine import ConversationEngine
from conversational_agent.core.models import LLMAction, TurnRequest, TurnStatus


class Provider:
    provider_id = 'p1'
    timeout = 1.0

    def retrieve(self, query: str, **kwargs: Any) -> list[Any]:
        return []


class SlowButSuccessfulLLM:
    def next_action(self, *args: Any, **kwargs: Any) -> LLMAction:
        time.sleep(0.3)
        return LLMAction(action='answer', text='finished late')


def test_total_turn_timeout_rejects_llm_that_finishes_after_deadline() -> None:
    engine = ConversationEngine(
        providers=[Provider()],
        llm=SlowButSuccessfulLLM(),
        cache_capacity=2,
        max_steps=1,
        total_turn_timeout=0.1,
        llm_first_token_timeout=5.0,
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
    assert final['error_code'] == 'TURN_TIMEOUT'
    assert len(engine.get_conversation('c').turns) == 0
