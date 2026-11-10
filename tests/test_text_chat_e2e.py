from __future__ import annotations

from copy import deepcopy

from conversational_agent import (
    ConversationEngine,
    FakeRAGProvider,
    FakeTextLLM,
    FunctionConversationStore,
)
from conversational_agent.frontends.text import TextFrontend


def test_text_chat_uses_selected_fake_rag_and_function_store() -> None:
    rows: dict[str, dict] = {}
    store = FunctionConversationStore(
        save_fn=lambda snapshot: rows.__setitem__(
            snapshot["conversation_id"], deepcopy(snapshot)
        ),
        load_fn=lambda conversation_id: deepcopy(rows.get(conversation_id)),
    )
    provider = FakeRAGProvider(
        "knowledge-base",
        documents=[
            {
                "evidence_id": "refund-30d",
                "text": "Refunds are available within 30 days.",
            }
        ],
    )
    llm = FakeTextLLM()
    engine = ConversationEngine(providers=[provider], llm=llm, store=store)
    text = TextFrontend(engine)

    events = list(text.send(
        conversation_id="conv-1",
        idempotency_key="turn-1",
        text="What is the refund policy?",
        metadata={"provider_id": "knowledge-base"},
    ))

    rag_events = [event for event in events if event["type"] == "rag.call"]
    final = next(event for event in events if event["type"] == "turn.final")
    assert rag_events[0]["provider_id"] == "knowledge-base"
    assert rag_events[0]["evidence_ids"] == ["refund-30d"]
    assert "Refunds are available within 30 days." in final["text"]
    assert "conv-1" in rows
    assert rows["conv-1"]["turns"][-1]["assistant_text"] == final["text"]
