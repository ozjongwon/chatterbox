"""Text frontend adapter for the shared core turn contract."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from conversational_agent.core.engine import ConversationEngine
from conversational_agent.core.models import CoreEvent, FrontendType, TurnRequest
from conversational_agent.ordering import ReorderingBuffer


class TextFrontend:
    """Converts text input into core turn requests and streams core events."""

    def __init__(self, engine: ConversationEngine) -> None:
        self._engine = engine

    def send(
        self,
        conversation_id: str,
        idempotency_key: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[CoreEvent]:
        request = TurnRequest(
            conversation_id=conversation_id,
            idempotency_key=idempotency_key,
            text=text,
            frontend=FrontendType.TEXT,
            metadata=dict(metadata or {}),
        )
        yield from self._engine.run_turn(request)


class OrderedTextFrontend:
    """Text frontend with per-conversation client sequence ordering."""

    def __init__(self, frontend: TextFrontend, gap_timeout: float = 1.0) -> None:
        self._frontend = frontend
        self._gap_timeout = gap_timeout
        self._buffers: dict[str, ReorderingBuffer] = {}

    def _buffer(self, conversation_id: str) -> ReorderingBuffer:
        return self._buffers.setdefault(
            conversation_id,
            ReorderingBuffer(conversation_id, gap_timeout=self._gap_timeout),
        )

    def send(
        self,
        conversation_id: str,
        idempotency_key: str,
        text: str,
        client_seq: int,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[CoreEvent]:
        item = {
            "conversation_id": conversation_id,
            "idempotency_key": idempotency_key,
            "text": text,
            "metadata": dict(metadata or {}),
        }
        ready = self._buffer(conversation_id).accept(client_seq, item)
        if not ready:
            gap = self._buffer(conversation_id).expire()
            if gap is not None:
                yield gap.to_event()
            return
        for request in ready:
            yield from self._frontend.send(**request)

    def poll(self, conversation_id: str) -> Iterator[CoreEvent]:
        gap = self._buffer(conversation_id).expire()
        if gap is not None:
            yield gap.to_event()
