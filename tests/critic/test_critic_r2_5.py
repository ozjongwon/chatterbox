# REQ-5 regression: graceful shutdown SHALL persist all cached conversations
# before exiting.  The current shutdown implementation makes one attempt per
# conversation and silently returns if that write fails; it does not retry,
# so a transient database failure during shutdown loses the latest in-memory
# state.

from __future__ import annotations

from typing import Any

from conversational_agent.core.engine import ConversationEngine
from conversational_agent.core.models import LLMAction, TurnRequest


class Provider:
    provider_id = 'p1'
    timeout = 1.0

    def retrieve(self, query: str, **kwargs: Any) -> list[Any]:
        return []


class AnswerLLM:
    def next_action(self, *args: Any, **kwargs: Any) -> LLMAction:
        return LLMAction(action='answer', text='ok')


class FailOnceStore:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.fail_next_save = False

    def save(self, snapshot: dict[str, Any]) -> None:
        if self.fail_next_save:
            self.fail_next_save = False
            raise RuntimeError('transient database failure')
        self.rows[snapshot['conversation_id']] = snapshot

    def load(self, conversation_id: str) -> dict[str, Any] | None:
        return self.rows.get(conversation_id)


def test_shutdown_retries_transient_failure_and_persists_latest_snapshot() -> None:
    store = FailOnceStore()
    engine = ConversationEngine(
        providers=[Provider()],
        llm=AnswerLLM(),
        store=store,
        cache_capacity=2,
    )

    list(
        engine.run_turn(
            TurnRequest(
                conversation_id='c',
                idempotency_key='k',
                text='hello',
            )
        )
    )

    # Make the cached conversation newer than the last successful DB snapshot.
    engine.get_conversation('c').metadata['flush_marker'] = 'latest'
    store.fail_next_save = True

    engine.shutdown()

    assert store.rows['c']['metadata'].get('flush_marker') == 'latest'
