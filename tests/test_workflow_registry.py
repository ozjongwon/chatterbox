from __future__ import annotations

from pathlib import Path

from conversational_agent.core.models import Conversation
from conversational_agent.workflow import (
    WorkflowRegistry,
    WorkflowSelector,
    validate_workflow_slots,
)


WORKFLOWS = Path(__file__).resolve().parents[1] / "workflows"


def test_registry_loads_and_selects_explicit_workflow() -> None:
    registry = WorkflowRegistry(known_tools={"retrieve_policy", "lookup_order",
                                              "lookup_order_item", "validate_attachment"})
    registry.load_directory(WORKFLOWS)
    workflow = registry.get("damaged_on_arrival")
    assert workflow is not None
    assert workflow.version == 3
    assert workflow.initial_state == "need_order"


def test_selector_uses_keyword_and_pins_version() -> None:
    registry = WorkflowRegistry()
    registry.load_directory(WORKFLOWS)
    selector = WorkflowSelector(registry)
    workflow = selector.select({}, "My photo frame arrived broken")
    assert workflow is not None
    assert workflow.workflow_id == "damaged_on_arrival"

    conversation = Conversation("c1")
    selector.pin(conversation, workflow)
    assert conversation.metadata["workflow_id"] == "damaged_on_arrival"
    assert conversation.metadata["workflow_version"] == 3
    assert selector.workflow_context(conversation)["workflow_state"] == "need_order"


def test_explicit_workflow_choice_wins_and_slots_are_validated() -> None:
    registry = WorkflowRegistry()
    registry.load_directory(WORKFLOWS)
    selector = WorkflowSelector(registry)
    workflow = selector.select(
        {"workflow_id": "order_status"},
        "My item arrived broken",
    )
    assert workflow is not None
    assert workflow.workflow_id == "order_status"
    assert validate_workflow_slots(workflow, {"order_id": "12345"}) == []
    assert validate_workflow_slots(workflow, {"order_id": "bad"}) == ["order_id"]
