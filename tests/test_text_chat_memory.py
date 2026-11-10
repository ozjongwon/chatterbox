from __future__ import annotations

from copy import deepcopy

from conversational_agent import (
    ConversationEngine,
    FakeRAGProvider,
    FakeTextLLM,
    FunctionConversationStore,
)
from conversational_agent.frontends.text import TextFrontend


def _store(rows: dict[str, dict]) -> FunctionConversationStore:
    return FunctionConversationStore(
        save_fn=lambda snapshot: rows.__setitem__(
            snapshot["conversation_id"], deepcopy(snapshot)
        ),
        load_fn=lambda conversation_id: deepcopy(rows.get(conversation_id)),
    )


def test_ten_turn_conversation_history_and_store_reload() -> None:
    rows: dict[str, dict] = {}
    provider = FakeRAGProvider("knowledge-base")
    first_llm = FakeTextLLM()
    first_engine = ConversationEngine(
        providers=[provider],
        llm=first_llm,
        store=_store(rows),
    )
    text = TextFrontend(first_engine)

    answers: list[str] = []
    for turn_number in range(1, 11):
        events = list(text.send(
            conversation_id="memory-demo",
            idempotency_key=f"turn-{turn_number}",
            text=f"Question {turn_number}: tell me about the refund policy",
            metadata={"provider_id": "knowledge-base"},
        ))
        final = next(event for event in events if event["type"] == "turn.final")
        answers.append(final["text"])

    # The first LLM call of each turn sees only completed prior turns.
    first_calls = [call for call in first_llm.calls if call["step"] == 0]
    assert [len(call["history"]) for call in first_calls] == list(range(10))
    assert len(rows["memory-demo"]["turns"]) == 10
    assert all("refund" in answer.lower() for answer in answers)

    # A new engine restores the same conversation from the function store.
    second_llm = FakeTextLLM()
    second_engine = ConversationEngine(
        providers=[provider],
        llm=second_llm,
        store=_store(rows),
    )
    second_events = list(TextFrontend(second_engine).send(
        conversation_id="memory-demo",
        idempotency_key="turn-11",
        text="Question 11: what did we discuss before?",
        metadata={"provider_id": "knowledge-base"},
    ))
    second_final = next(
        event for event in second_events if event["type"] == "turn.final"
    )
    assert len(second_llm.calls[0]["history"]) == 10
    assert "refund" in second_final["text"].lower()
    assert len(rows["memory-demo"]["turns"]) == 11
