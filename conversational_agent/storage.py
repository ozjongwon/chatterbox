"""Conversation storage abstractions and an in-memory implementation."""

from __future__ import annotations

import copy
import threading
from typing import Any, Protocol


class ConversationStore(Protocol):
    """Transactional snapshot store keyed by conversation_id."""

    def save(self, snapshot: dict[str, Any]) -> None:
        ...

    def load(self, conversation_id: str) -> dict[str, Any] | None:
        ...


class InMemoryConversationStore:
    """Simple deterministic store used in tests and local development."""

    def __init__(self) -> None:
        self._rows: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def save(self, snapshot: dict[str, Any]) -> None:
        conversation_id = snapshot['conversation_id']
        with self._lock:
            self._rows[conversation_id] = copy.deepcopy(snapshot)

    def load(self, conversation_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._rows.get(conversation_id)
            return copy.deepcopy(row) if row is not None else None

    # Conventional aliases used by simple database adapters.
    def upsert(self, snapshot: dict[str, Any]) -> None:
        self.save(snapshot)

    def get(self, conversation_id: str) -> dict[str, Any] | None:
        return self.load(conversation_id)

    def __contains__(self, conversation_id: object) -> bool:
        with self._lock:
            return conversation_id in self._rows
