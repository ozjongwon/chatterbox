from __future__ import annotations

import json
import tempfile
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


def test_reload_requires_approval_and_rollback_restores_previous() -> None:
    registry = WorkflowRegistry()
    registry.load_directory(WORKFLOWS)
    source = (WORKFLOWS / "damaged_on_arrival.v3.json").read_text()
    candidate = json.loads(source)
    candidate["version"] = 4

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "damaged_on_arrival.v4.json"
        path.write_text(json.dumps(candidate))
        registry.reload(path.parent, activate=False)
        assert registry.get("damaged_on_arrival").version == 3

        registry.activate("damaged_on_arrival", 4)
        assert registry.get("damaged_on_arrival").version == 4
        assert registry.rollback("damaged_on_arrival") == 3
        assert registry.get("damaged_on_arrival").version == 3


def test_workflow_handoff_preserves_previous_context() -> None:
    registry = WorkflowRegistry()
    registry.load_directory(WORKFLOWS)
    conversation = Conversation("c2")
    first = registry.get("damaged_on_arrival")
    assert first is not None
    WorkflowSelector.pin(conversation, first)
    conversation.metadata["workflow_slots"] = {"order_id": "12345"}

    target = WorkflowSelector.handoff(conversation, registry, "order_status")
    assert target.workflow_id == "order_status"
    assert conversation.metadata["workflow_handoff_from"] == "damaged_on_arrival"
    assert conversation.metadata["workflow_history"][0]["workflow_id"] == (
        "damaged_on_arrival"
    )
