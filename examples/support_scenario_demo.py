from __future__ import annotations

from pathlib import Path

from conversational_agent import (
    ConversationEngine,
    FakeAttachmentGuard,
    FakeOrderRepository,
    FakeRAGProvider,
    SupportScenarioLLM,
    WorkflowRegistry,
    support_tools,
)
from conversational_agent.frontends.text import TextFrontend


def final_text(events: list[dict]) -> str:
    return next(event for event in events if event["type"] == "turn.final")["text"]


def main() -> None:
    provider = FakeRAGProvider(
        "support-policy",
        documents=[
            {
                "evidence_id": "policy-ata-1",
                "text": "Items damaged on arrival may be exchanged or refunded with photo evidence.",
            }
        ],
    )
    workflows = WorkflowRegistry()
    workflows.load_directory(
        Path(__file__).resolve().parents[1] / "workflows"
    )
    engine = ConversationEngine(
        providers=[provider],
        llm=SupportScenarioLLM(),
        tools=support_tools(FakeOrderRepository(), FakeAttachmentGuard()),
        workflows=workflows,
        max_steps=6,
    )
    text = TextFrontend(engine)
    turns = [
        (
            "I purchased an item and got a broken one.",
            {"customer_id": "customer-1", "provider_id": "support-policy"},
        ),
        (
            "The order number is 12345.",
            {"customer_id": "customer-1", "provider_id": "support-policy"},
        ),
        (
            "The photo frame is broken and arrived that way.",
            {
                "customer_id": "customer-1",
                "provider_id": "support-policy",
                "attachments": [
                    {"attachment_id": "photo-1", "content_type": "image/jpeg"}
                ],
            },
        ),
        (
            "I want a refund.",
            {"customer_id": "customer-1", "provider_id": "support-policy"},
        ),
        (
            "What proof do I need?",
            {"customer_id": "customer-1", "provider_id": "support-policy"},
        ),
    ]

    for number, (message, metadata) in enumerate(turns, 1):
        events = list(text.send(
            "support-demo",
            f"turn-{number}",
            message,
            metadata=metadata,
        ))
        print(f"\n{number}. USER: {message}")
        for event in events:
            if event["type"] == "workflow.selected":
                print(
                    "   FLOW:",
                    event["workflow_id"],
                    f"v{event['workflow_version']}",
                    event["workflow_state"],
                )
            elif event["type"] == "rag.call":
                print(
                    "   RAG :",
                    event["provider_id"],
                    event["evidence_ids"],
                )
            elif event["type"] == "tool.call":
                print(
                    "   TOOL:",
                    event["tool_name"],
                    "ok=" + str(event["result"].get("ok")),
                )
        print("   SYSTEM:", final_text(events))


if __name__ == "__main__":
    main()
