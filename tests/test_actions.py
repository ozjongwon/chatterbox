from __future__ import annotations

import pytest

from conversational_agent.core.models import (
    ActionType,
    AnswerAction,
    LLMAction,
    RetrieveAction,
    ToolAction,
    ValidationError,
    parse_llm_action,
)


def test_typed_actions_and_compatibility_factory() -> None:
    answer = LLMAction(action="answer", text="hello")
    retrieve = LLMAction(action="retrieve", provider_id="p1", query="q")
    tool = LLMAction(action="tool", tool_name="lookup", arguments={"id": "1"})
    assert isinstance(answer, AnswerAction)
    assert isinstance(retrieve, RetrieveAction)
    assert isinstance(tool, ToolAction)
    assert answer.action == ActionType.ANSWER
    assert retrieve.action == ActionType.RETRIEVE
    assert tool.action == ActionType.TOOL


def test_action_parser_rejects_unknown_and_invalid_payloads() -> None:
    with pytest.raises(ValidationError):
        parse_llm_action({"action": "answr", "text": "x"})
    with pytest.raises(ValidationError):
        parse_llm_action({"action": "tool", "tool_name": "lookup"})
