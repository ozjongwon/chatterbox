"""Shared, frontend-agnostic conversation and turn models."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

SCHEMA_VERSION = 1


class FrontendType(str, Enum):
    TEXT = 'text'
    VOICE = 'voice'


class TurnStatus(str, Enum):
    SUCCESS = 'success'
    ERROR = 'error'
    CANCELLED = 'cancelled'


class ValidationError(ValueError):
    """Client-facing validation failure with a stable error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass
class LLMAction:
    """A single step decision produced by the LLM adapter."""

    action: str
    text: str | None = None
    provider_id: str | None = None
    query: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'action': self.action,
            'text': self.text,
            'provider_id': self.provider_id,
            'query': self.query,
        }


@dataclass
class TurnRequest:
    """The one turn contract shared by text and voice adapters."""

    conversation_id: str
    idempotency_key: str
    text: str
    frontend: FrontendType = FrontendType.TEXT
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Turn:
    turn_id: str
    idempotency_key: str
    frontend: FrontendType
    user_text: str
    assistant_text: str = ''
    status: TurnStatus = TurnStatus.SUCCESS
    trace: list[dict[str, Any]] = field(default_factory=list)
    error_code: str | None = None
    degraded: bool = False
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            'turn_id': self.turn_id,
            'idempotency_key': self.idempotency_key,
            'frontend': self.frontend.value,
            'user_text': self.user_text,
            'assistant_text': self.assistant_text,
            'status': self.status.value,
            'trace': list(self.trace),
            'error_code': self.error_code,
            'degraded': self.degraded,
            'created_at': self.created_at,
            'completed_at': self.completed_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Turn:
        return cls(
            turn_id=str(data['turn_id']),
            idempotency_key=str(data['idempotency_key']),
            frontend=FrontendType(data.get('frontend', FrontendType.TEXT.value)),
            user_text=str(data.get('user_text', '')),
            assistant_text=str(data.get('assistant_text', '')),
            status=TurnStatus(data.get('status', TurnStatus.SUCCESS.value)),
            trace=list(data.get('trace') or []),
            error_code=data.get('error_code'),
            degraded=bool(data.get('degraded', False)),
            created_at=float(data.get('created_at', time.time())),
            completed_at=data.get('completed_at'),
        )


@dataclass
class Conversation:
    conversation_id: str
    turns: list[Turn] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    dirty: bool = False
    schema_version: int = SCHEMA_VERSION

    def find_turn(self, idempotency_key: str) -> Turn | None:
        for turn in self.turns:
            if turn.idempotency_key == idempotency_key:
                return turn
        return None

    def to_snapshot(self) -> dict[str, Any]:
        return {
            'conversation_id': self.conversation_id,
            'schema_version': self.schema_version,
            'turns': [turn.to_dict() for turn in self.turns],
            'metadata': dict(self.metadata),
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'dirty': self.dirty,
        }

    @classmethod
    def from_snapshot(cls, data: dict[str, Any]) -> Conversation:
        return cls(
            conversation_id=str(data['conversation_id']),
            turns=[Turn.from_dict(item) for item in data.get('turns', [])],
            metadata=dict(data.get('metadata') or {}),
            created_at=float(data.get('created_at', time.time())),
            updated_at=float(data.get('updated_at', time.time())),
            dirty=bool(data.get('dirty', False)),
            schema_version=int(data.get('schema_version', SCHEMA_VERSION)),
        )


CoreEvent = dict[str, Any]
