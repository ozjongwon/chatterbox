"""Text frontend adapter for the shared core turn contract."""

from __future__ import annotations

from collections.abc import Iterator

from conversational_agent.core.engine import ConversationEngine
from conversational_agent.core.models import CoreEvent, FrontendType, TurnRequest


class TextFrontend:
    """Converts text input into core turn requests and streams core events."""

    def __init__(self, engine: ConversationEngine) -> None:
        self._engine = engine

    def send(
        self,
        conversation_id: str,
        idempotency_key: str,
        text: str,
    ) -> Iterator[CoreEvent]:
        request = TurnRequest(
            conversation_id=conversation_id,
            idempotency_key=idempotency_key,
            text=text,
            frontend=FrontendType.TEXT,
        )
        yield from self._engine.run_turn(request)
