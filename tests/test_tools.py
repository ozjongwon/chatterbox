from __future__ import annotations

from conversational_agent.tools import ToolRegistry, ToolSpec


def test_tool_schema_and_postconditions() -> None:
    registry = ToolRegistry()
    registry.register(ToolSpec(
        name="lookup",
        description="lookup",
        input_schema={
            "type": "object",
            "properties": {"order_id": {"type": "string", "pattern": "^\\d{5}$"}},
            "required": ["order_id"],
        },
        handler=lambda order_id, **kwargs: {"ok": True, "order_id": order_id},
        postconditions=(
            lambda result, context: result["order_id"] == "12345",
        ),
    ))
    tool = registry.get("lookup")
    assert tool is not None
    assert not tool.invoke({"order_id": "bad"}, {})["ok"]
    assert tool.invoke({"order_id": "12345"}, {})["ok"]
    assert not tool.invoke({"order_id": "54321"}, {})["ok"]
