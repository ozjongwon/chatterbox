"""Static LLM contract plus explicit runtime adapter validation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeGuard

if TYPE_CHECKING:
    from conversational_agent.core.models import LLMAction


class LLMClient(Protocol):
    def next_action(self, **kwargs: Any) -> LLMAction:
        ...


def is_llm_client(value: object) -> TypeGuard[LLMClient]:
    return callable(getattr(value, "next_action", None))
