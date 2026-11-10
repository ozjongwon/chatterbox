from __future__ import annotations

from copy import deepcopy

from conversational_agent import (
    ConversationEngine,
    FakeRAGProvider,
    FakeTextLLM,
    FunctionConversationStore,
)
from conversational_agent.frontends.text import TextFrontend


def store_for(rows: dict[str, dict]) -> FunctionConversationStore:
    return FunctionConversationStore(
        save_fn=lambda snapshot: rows.__setitem__(
            snapshot["conversation_id"], deepcopy(snapshot)
        ),
        load_fn=lambda conversation_id: deepcopy(rows.get(conversation_id)),
    )


def final_text(events: list[dict]) -> str:
    return next(event for event in events if event["type"] == "turn.final")["text"]


def main() -> None:
    rows: dict[str, dict] = {}
    provider = FakeRAGProvider("knowledge-base")
    llm = FakeTextLLM()
    engine = ConversationEngine(
        providers=[provider],
        llm=llm,
        store=store_for(rows),
    )
    text = TextFrontend(engine)

    for number in range(1, 11):
        question = f"Turn {number}: what is the refund policy?"
        events = list(text.send(
            conversation_id="demo",
            idempotency_key=f"turn-{number}",
            text=question,
            metadata={"provider_id": "knowledge-base"},
        ))
        history = llm.calls[(number - 1) * 2]["history"]
        print(f"{number:02d} USER : {question}")
        print(f"   history_before_turn: {len(history)}")
        print(f"   ASSISTANT: {final_text(events)}")

    print("\n--- reload from FunctionConversationStore ---")
    restored_llm = FakeTextLLM()
    restored_engine = ConversationEngine(
        providers=[provider],
        llm=restored_llm,
        store=store_for(rows),
    )
    restored_text = TextFrontend(restored_engine)
    restored_events = list(restored_text.send(
        conversation_id="demo",
        idempotency_key="turn-11",
        text="Turn 11: what did we discuss before?",
        metadata={"provider_id": "knowledge-base"},
    ))
    print("11 USER : Turn 11: what did we discuss before?")
    print(f"   history_before_turn: {len(restored_llm.calls[0]['history'])}")
    print(f"   ASSISTANT: {final_text(restored_events)}")
    print(f"\npersisted_turns: {len(rows['demo']['turns'])}")


if __name__ == "__main__":
    main()
