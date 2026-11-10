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


def test_realistic_damage_support_scenario() -> None:
    repository = FakeOrderRepository()
    guard = FakeAttachmentGuard()
    provider = FakeRAGProvider(
        "support-policy",
        documents=[
            {
                "evidence_id": "policy-ata-1",
                "text": "Items damaged on arrival may be exchanged or refunded with photo evidence.",
            }
        ],
    )
    llm = SupportScenarioLLM()
    workflows = WorkflowRegistry()
    workflows.load_directory(Path(__file__).resolve().parents[1] / "workflows")
    engine = ConversationEngine(
        providers=[provider],
        llm=llm,
        tools=support_tools(repository, guard),
        workflows=workflows,
        max_steps=6,
    )
    text = TextFrontend(engine)

    # 1) Policy lookup, then ask for the order number.
    events1 = list(text.send(
        "support-1", "t1",
        "I purchased an item and got a broken one.",
        metadata={"customer_id": "customer-1", "provider_id": "support-policy"},
    ))
    assert any(event["type"] == "rag.call" for event in events1)
    assert any(event["type"] == "workflow.selected" for event in events1)
    assert "주문번호" in final_text(events1)

    # 2) Order DB lookup, then ask which item.
    events2 = list(text.send(
        "support-1", "t2",
        "The order number is 12345.",
        metadata={"customer_id": "customer-1", "provider_id": "support-policy"},
    ))
    assert any(
        event["type"] == "tool.call" and event["tool_name"] == "lookup_order"
        for event in events2
    )
    assert "어느 상품" in final_text(events2)

    # 3) Item DB lookup + image proof validation + policy lookup.
    events3 = list(text.send(
        "support-1", "t3",
        "The photo frame is broken and arrived that way.",
        metadata={
            "customer_id": "customer-1",
            "provider_id": "support-policy",
            "attachments": [
                {"attachment_id": "photo-1", "content_type": "image/jpeg"}
            ],
        },
    ))
    tool_names = [
        event["tool_name"] for event in events3 if event["type"] == "tool.call"
    ]
    assert tool_names == ["lookup_order", "lookup_order_item", "validate_attachment"]
    assert any(event["type"] == "rag.call" for event in events3)
    assert "사진 증빙" in final_text(events3)
    assert "교환 또는 환불" in final_text(events3)

    # 4) User chooses refund.
    events4 = list(text.send(
        "support-1", "t4",
        "I want a refund.",
        metadata={"customer_id": "customer-1", "provider_id": "support-policy"},
    ))
    assert "환불 접수" in final_text(events4)

    # 5) Follow-up information request.
    events5 = list(text.send(
        "support-1", "t5",
        "What proof do I need?",
        metadata={"customer_id": "customer-1", "provider_id": "support-policy"},
    ))
    assert "사진" in final_text(events5)
