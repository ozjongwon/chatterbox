"""Client sequence ordering, gap detection, and resend notifications."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GapNotice:
    session_id: str
    expected_seq: int
    buffered_sequences: tuple[int, ...]
    resend_from: int
    message: str = "A message was lost or delayed. Please resend from the indicated sequence."

    def to_event(self) -> dict[str, Any]:
        return {
            "type": "sequence.gap",
            "session_id": self.session_id,
            "expected_seq": self.expected_seq,
            "buffered_sequences": list(self.buffered_sequences),
            "resend_from": self.resend_from,
            "message": self.message,
        }


@dataclass
class _Pending:
    item: Any
    arrived_at: float = field(default_factory=time.monotonic)


class ReorderingBuffer:
    """Holds out-of-order client messages until the missing sequence arrives or expires."""

    def __init__(
        self,
        session_id: str,
        gap_timeout: float = 1.0,
        max_pending: int = 100,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if gap_timeout < 0:
            raise ValueError("gap_timeout must be non-negative")
        self.session_id = session_id
        self.gap_timeout = gap_timeout
        self.max_pending = max(1, max_pending)
        self.clock = clock
        self._next_seq = 1
        self._pending: dict[int, _Pending] = {}

    @property
    def next_seq(self) -> int:
        return self._next_seq

    @property
    def pending_sequences(self) -> tuple[int, ...]:
        return tuple(sorted(self._pending))

    def accept(self, sequence: int, item: Any) -> list[Any]:
        if not isinstance(sequence, int) or sequence < 1:
            raise ValueError("client sequence must be a positive integer")
        if sequence < self._next_seq:
            return []
        if sequence >= self._next_seq:
            self._pending[sequence] = _Pending(item, arrived_at=self.clock())
            if len(self._pending) > self.max_pending:
                oldest = min(self._pending)
                del self._pending[oldest]
                self._next_seq = max(self._next_seq, oldest + 1)
        return self._release_ready()

    def expire(self) -> GapNotice | None:
        if not self._pending or self._next_seq in self._pending:
            return None
        now = self.clock()
        first = self._pending[min(self._pending)]
        if now - first.arrived_at < self.gap_timeout:
            return None
        return GapNotice(
            session_id=self.session_id,
            expected_seq=self._next_seq,
            buffered_sequences=self.pending_sequences,
            resend_from=self._next_seq,
        )

    def reset(self, next_seq: int = 1) -> None:
        if next_seq < 1:
            raise ValueError("next_seq must be positive")
        self._next_seq = next_seq
        self._pending.clear()

    def _release_ready(self) -> list[Any]:
        ready: list[Any] = []
        while self._next_seq in self._pending:
            pending = self._pending.pop(self._next_seq)
            ready.append(pending.item)
            self._next_seq += 1
        return ready
