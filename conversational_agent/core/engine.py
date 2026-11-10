"""Frontend-agnostic conversational orchestration engine."""

from __future__ import annotations

import re
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from conversational_agent.cache import LRUCache
from conversational_agent.core.models import (
    Conversation,
    CoreEvent,
    LLMAction,
    Turn,
    TurnRequest,
    TurnStatus,
    ValidationError,
)
from conversational_agent.providers import validate_providers
from conversational_agent.storage import ConversationStore, InMemoryConversationStore


class _CancelledError(Exception):
    """Internal signal used to unwind cancelled turns."""


class _LLMFailure(Exception):
    """Internal signal raised after the LLM retry budget is exhausted."""


_SAFE_ERROR_MESSAGES = {
    'EMPTY_INPUT': 'The input text must not be empty.',
    'INVALID_LLM_ACTION': 'The assistant produced an invalid action.',
    'STEP_LIMIT_EXCEEDED': 'The assistant could not complete the request within its step budget.',
    'MAX_RAG_CALLS_EXCEEDED': 'The retrieval budget for this turn was exhausted.',
    'LLM_FAILURE': 'The assistant is temporarily unavailable. Please try again.',
    'TURN_TIMEOUT': 'The request took too long to complete. Please try again.',
    'CONVERSATION_BUSY': 'Another turn for this conversation is still in progress.',
    'CANCELLED': 'The turn was cancelled.',
    'INTERNAL_ERROR': 'An unexpected error occurred while processing the request.',
}


def _chunks(text: str) -> Iterator[str]:
    """Split text into deterministic incrementally streamable pieces."""
    for match in re.finditer(r'\S+\s*', text):
        yield match.group(0)


def _evidence_ids(items: Any) -> list[str]:
    evidence: list[str] = []
    for item in items or []:
        value: Any = None
        if isinstance(item, dict):
            value = item.get('evidence_id', item.get('id'))
        else:
            value = getattr(item, 'evidence_id', None)
            if value is None:
                value = getattr(item, 'id', None)
        evidence.append(str(value) if value is not None else str(item))
    return evidence


def _evidence_payload(items: Any) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for item in items or []:
        if isinstance(item, dict):
            row = dict(item)
        else:
            row = {
                'evidence_id': getattr(item, 'evidence_id', getattr(item, 'id', None)),
                'text': getattr(item, 'text', getattr(item, 'content', None)),
            }
        if row.get('evidence_id') is None:
            row['evidence_id'] = row.get('id')
        if row.get('text') is None:
            row['text'] = row.get('content')
        payload.append(row)
    return payload


class ConversationEngine:
    """Owns turns, the bounded LLM/RAG loop, cache, and persistence."""

    def __init__(
        self,
        providers: list[Any] | tuple[Any, ...],
        llm: Any,
        store: ConversationStore | None = None,
        cache_capacity: int = 100,
        max_steps: int = 4,
        max_rag_calls: int = 4,
        rag_timeout: float = 2.0,
        llm_first_token_timeout: float = 5.0,
        total_turn_timeout: float = 30.0,
        history_limit: int = 50,
    ) -> None:
        self._providers = validate_providers(providers)
        self._llm = llm
        self._store: ConversationStore = store or InMemoryConversationStore()
        self._cache: LRUCache[str, Conversation] = LRUCache(capacity=cache_capacity)
        self._cache_lock = threading.RLock()
        self._max_steps = max(1, max_steps)
        self._max_rag_calls = max(0, max_rag_calls)
        self._rag_timeout = rag_timeout
        self._llm_timeout = llm_first_token_timeout
        self._total_turn_timeout = total_turn_timeout
        self._history_limit = history_limit
        # A plain (non-reentrant) lock may be released by a different thread
        # than the one that acquired it, which matters because callers may
        # drive the streaming generator from another thread.
        self._locks: dict[str, Any] = {}
        self._locks_guard = threading.Lock()
        self._cancelled: set[tuple[str, str]] = set()
        self._cancelled_guard = threading.Lock()
        self._health_errors: list[str] = []

    # ------------------------------------------------------------------
    # Public turn contract
    # ------------------------------------------------------------------
    def run_turn(self, request: TurnRequest) -> Iterator[CoreEvent]:
        """Process one turn, yielding streaming core events.

        Validation happens eagerly so empty input is rejected before any
        conversation state, LLM call, or RAG call is created.
        """
        text = request.text.strip()
        if not text:
            raise ValidationError('EMPTY_INPUT', _SAFE_ERROR_MESSAGES['EMPTY_INPUT'])
        return self._run_turn(request, text)

    def cancel(self, conversation_id: str, turn_id: str | None = None) -> None:
        """Request cancellation of an in-flight turn."""
        with self._cancelled_guard:
            self._cancelled.add((conversation_id, turn_id or '*'))

    def get_conversation(self, conversation_id: str) -> Conversation:
        with self._cache_lock:
            conversation, _ = self._obtain(conversation_id)
            return conversation

    def shutdown(self) -> None:
        """Persist every cached conversation before exiting.

        A transient write failure is retried immediately; if it still fails the
        conversation stays resident and dirty, the failure is surfaced through
        the health signal, and the next access/eviction will retry it.
        """
        with self._cache_lock:
            failures: list[str] = []
            for conversation_id in self._cache:
                conversation = self._cache.get(conversation_id)
                if conversation is None:
                    continue
                if not self._persist_with_retry(conversation):
                    failures.append(conversation_id)
            if failures and 'SHUTDOWN_PERSIST_FAILURE' not in self._health_errors:
                self._health_errors.append('SHUTDOWN_PERSIST_FAILURE')

    def flush(self) -> None:
        self.shutdown()

    def health(self) -> dict[str, Any]:
        database_ok = True
        try:
            self._store.load('__health_probe__')
        except Exception:  # noqa: BLE001 - health probes must never raise
            database_ok = False
        return {
            'status': 'ok' if database_ok else 'degraded',
            'database': {'connected': database_ok},
            'providers': {'valid': True, 'count': len(self._providers)},
            'cache': {'size': len(self._cache), 'capacity': self._cache.capacity},
            'errors': list(self._health_errors),
        }

    # ------------------------------------------------------------------
    # Turn execution
    # ------------------------------------------------------------------
    def _run_turn(self, request: TurnRequest, text: str) -> Iterator[CoreEvent]:
        lock = self._lock_for(request.conversation_id)
        if not lock.acquire(timeout=self._total_turn_timeout):
            # Turns on the same conversation are serialized; a caller that
            # cannot be serialized within the turn budget gets a documented
            # conflict error instead of interleaving writes.
            yield {
                'type': 'turn.error',
                'conversation_id': request.conversation_id,
                'turn_id': '',
                'status': TurnStatus.ERROR,
                'text': '',
                'trace': [],
                'degraded': False,
                'cache_status': 'unknown',
                'error_code': 'CONVERSATION_BUSY',
                'message': _SAFE_ERROR_MESSAGES['CONVERSATION_BUSY'],
            }
            return
        try:
            yield from self._execute(request, text)
        finally:
            lock.release()

    def _execute(self, request: TurnRequest, text: str) -> Iterator[CoreEvent]:
        conversation_id = request.conversation_id
        turn_id = ''
        with self._cache_lock:
            conversation, cache_status = self._obtain(conversation_id)
            self._cache.pin(conversation_id)

        try:
            existing = conversation.find_turn(request.idempotency_key)
            if existing is not None:
                yield {
                    'type': 'turn.accepted',
                    'conversation_id': conversation_id,
                    'turn_id': existing.turn_id,
                    'frontend': request.frontend.value,
                    'cache_status': cache_status,
                    'replay': True,
                }
                yield {
                    'type': 'turn.final',
                    'conversation_id': conversation_id,
                    'turn_id': existing.turn_id,
                    'status': existing.status,
                    'text': existing.assistant_text,
                    'trace': list(existing.trace),
                    'degraded': existing.degraded,
                    'cache_status': cache_status,
                    'replay': True,
                }
                return

            turn_id = uuid.uuid4().hex
            started = time.perf_counter()
            deadline = started + self._total_turn_timeout
            yield {
                'type': 'turn.accepted',
                'conversation_id': conversation_id,
                'turn_id': turn_id,
                'frontend': request.frontend.value,
                'cache_status': cache_status,
            }

            trace: list[dict[str, Any]] = []
            evidence: list[dict[str, Any]] = []
            degraded = False
            answer_text: str | None = None
            rag_calls = 0
            error_code: str | None = None
            cancelled = False

            try:
                for step in range(self._max_steps):
                    if self._is_cancelled(conversation_id, turn_id):
                        cancelled = True
                        break
                    if time.perf_counter() >= deadline:
                        error_code = 'TURN_TIMEOUT'
                        break

                    step_started = time.perf_counter()
                    try:
                        action = self._call_llm(
                            request, conversation, evidence, step, turn_id
                        )
                    except _CancelledError:
                        cancelled = True
                        break
                    except _LLMFailure:
                        error_code = (
                            'TURN_TIMEOUT'
                            if time.perf_counter() >= deadline
                            else 'LLM_FAILURE'
                        )
                        break

                    # The model call may itself have blocked past the turn
                    # deadline; a late result is a timeout, not a success.
                    if time.perf_counter() >= deadline:
                        error_code = 'TURN_TIMEOUT'
                        break

                    if not self._is_valid_action(action):
                        trace.append(
                            {
                                'type': 'model_action',
                                'action': getattr(action, 'action', 'invalid'),
                                'step': step,
                                'duration_ms': (time.perf_counter() - step_started) * 1000.0,
                                'status': 'error',
                            }
                        )
                        error_code = 'INVALID_LLM_ACTION'
                        break

                    assert isinstance(action, LLMAction)
                    trace.append(
                        {
                            'type': 'model_action',
                            'action': action.action,
                            'step': step,
                            'duration_ms': (time.perf_counter() - step_started) * 1000.0,
                        }
                    )
                    yield {
                        'type': 'llm.action',
                        'conversation_id': conversation_id,
                        'turn_id': turn_id,
                        'action': action.action,
                        'provider_id': action.provider_id,
                        'query': action.query,
                    }

                    if action.action == 'answer':
                        answer_text = action.text or ''
                        break

                    if rag_calls >= self._max_rag_calls:
                        error_code = 'MAX_RAG_CALLS_EXCEEDED'
                        break

                    provider = self._providers.get(action.provider_id or '')
                    rag_started = time.perf_counter()
                    if provider is None:
                        degraded = True
                        trace.append(
                            {
                                'type': 'rag_call',
                                'provider_id': action.provider_id,
                                'query': action.query,
                                'evidence_ids': [],
                                'duration_ms': (time.perf_counter() - rag_started) * 1000.0,
                                'status': 'error',
                                'error_code': 'UNKNOWN_PROVIDER',
                                'degraded': True,
                            }
                        )
                        yield {
                            'type': 'rag.call',
                            'conversation_id': conversation_id,
                            'turn_id': turn_id,
                            'provider_id': action.provider_id,
                            'query': action.query,
                            'evidence_ids': [],
                            'degraded': True,
                        }
                        rag_calls += 1
                        continue

                    try:
                        results = self._retrieve(
                            provider, action.query or '', conversation_id, turn_id
                        )
                    except _CancelledError:
                        cancelled = True
                        break
                    except Exception:  # noqa: BLE001 - degraded turn boundary
                        degraded = True
                        trace.append(
                            {
                                'type': 'rag_call',
                                'provider_id': provider.provider_id,
                                'query': action.query,
                                'evidence_ids': [],
                                'duration_ms': (time.perf_counter() - rag_started) * 1000.0,
                                'status': 'degraded',
                                'error_code': 'RAG_FAILURE',
                                'degraded': True,
                            }
                        )
                        yield {
                            'type': 'rag.call',
                            'conversation_id': conversation_id,
                            'turn_id': turn_id,
                            'provider_id': provider.provider_id,
                            'query': action.query,
                            'evidence_ids': [],
                            'degraded': True,
                        }
                        rag_calls += 1
                        if time.perf_counter() >= deadline:
                            error_code = 'TURN_TIMEOUT'
                            break
                        continue

                    if time.perf_counter() >= deadline:
                        error_code = 'TURN_TIMEOUT'
                        break

                    ids = _evidence_ids(results)
                    if not ids:
                        degraded = True
                    trace.append(
                        {
                            'type': 'rag_call',
                            'provider_id': provider.provider_id,
                            'query': action.query,
                            'evidence_ids': ids,
                            'duration_ms': (time.perf_counter() - rag_started) * 1000.0,
                            'status': 'ok',
                            'degraded': not ids,
                        }
                    )
                    evidence.append(
                        {
                            'provider_id': provider.provider_id,
                            'query': action.query,
                            'evidence_ids': ids,
                            'items': _evidence_payload(results),
                        }
                    )
                    yield {
                        'type': 'rag.call',
                        'conversation_id': conversation_id,
                        'turn_id': turn_id,
                        'provider_id': provider.provider_id,
                        'query': action.query,
                        'evidence_ids': ids,
                        'degraded': not ids,
                    }
                    rag_calls += 1

                if not cancelled and error_code is None and answer_text is None:
                    error_code = (
                        'TURN_TIMEOUT'
                        if time.perf_counter() >= deadline
                        else 'STEP_LIMIT_EXCEEDED'
                    )

                # Cancellation may have arrived while the LLM/RAG call was
                # blocking; a cancelled turn never becomes a success turn.
                if (
                    not cancelled
                    and error_code is None
                    and self._is_cancelled(conversation_id, turn_id)
                ):
                    cancelled = True

                if cancelled:
                    yield self._terminal_event(
                        'turn.cancelled',
                        TurnStatus.CANCELLED,
                        conversation_id,
                        turn_id,
                        cache_status,
                        trace=trace,
                        degraded=degraded,
                        error_code='CANCELLED',
                        started=started,
                    )
                    return

                if error_code is None and time.perf_counter() >= deadline:
                    # The deadline is a hard bound on the whole turn, including
                    # time spent streaming the final answer.
                    error_code = 'TURN_TIMEOUT'

                if error_code is not None:
                    yield self._terminal_event(
                        'turn.error',
                        TurnStatus.ERROR,
                        conversation_id,
                        turn_id,
                        cache_status,
                        trace=trace,
                        degraded=degraded,
                        error_code=error_code,
                        started=started,
                    )
                    return

                assert answer_text is not None
                for piece in _chunks(answer_text):
                    if self._is_cancelled(conversation_id, turn_id):
                        cancelled = True
                        break
                    yield {
                        'type': 'token',
                        'conversation_id': conversation_id,
                        'turn_id': turn_id,
                        'text': piece,
                    }

                if cancelled:
                    # Partial output is never a completed turn, so nothing is
                    # appended to the conversation or persisted.
                    yield self._terminal_event(
                        'turn.cancelled',
                        TurnStatus.CANCELLED,
                        conversation_id,
                        turn_id,
                        cache_status,
                        trace=trace,
                        degraded=degraded,
                        error_code='CANCELLED',
                        started=started,
                    )
                    return

                if time.perf_counter() >= deadline:
                    yield self._terminal_event(
                        'turn.error',
                        TurnStatus.ERROR,
                        conversation_id,
                        turn_id,
                        cache_status,
                        trace=trace,
                        degraded=degraded,
                        error_code='TURN_TIMEOUT',
                        started=started,
                    )
                    return

                turn = Turn(
                    turn_id=turn_id,
                    idempotency_key=request.idempotency_key,
                    frontend=request.frontend,
                    user_text=text,
                    assistant_text=answer_text,
                    status=TurnStatus.SUCCESS,
                    trace=trace,
                    degraded=degraded,
                    completed_at=time.time(),
                )
                conversation.turns.append(turn)
                conversation.updated_at = time.time()
                self._persist(conversation)
                yield {
                    'type': 'turn.final',
                    'conversation_id': conversation_id,
                    'turn_id': turn_id,
                    'status': TurnStatus.SUCCESS,
                    'text': answer_text,
                    'trace': trace,
                    'degraded': degraded,
                    'cache_status': cache_status,
                    'latency_ms': (time.perf_counter() - started) * 1000.0,
                }
            except _CancelledError:
                yield self._terminal_event(
                    'turn.cancelled',
                    TurnStatus.CANCELLED,
                    conversation_id,
                    turn_id,
                    cache_status,
                    trace=trace,
                    degraded=degraded,
                    error_code='CANCELLED',
                    started=started,
                )
            except Exception:  # noqa: BLE001 - user-safe terminal boundary
                self._health_errors.append('INTERNAL_ERROR')
                yield self._terminal_event(
                    'turn.error',
                    TurnStatus.ERROR,
                    conversation_id,
                    turn_id,
                    cache_status,
                    trace=trace,
                    degraded=degraded,
                    error_code='INTERNAL_ERROR',
                    started=started,
                )
        finally:
            with self._cache_lock:
                try:
                    self._cache.unpin(conversation_id)
                except KeyError:
                    pass
            with self._cancelled_guard:
                self._cancelled.discard((conversation_id, turn_id))
                self._cancelled.discard((conversation_id, '*'))

    def _terminal_event(
        self,
        event_type: str,
        status: TurnStatus,
        conversation_id: str,
        turn_id: str,
        cache_status: str,
        *,
        trace: list[dict[str, Any]],
        degraded: bool,
        error_code: str | None,
        started: float,
    ) -> CoreEvent:
        event: CoreEvent = {
            'type': event_type,
            'conversation_id': conversation_id,
            'turn_id': turn_id,
            'status': status,
            'text': '',
            'trace': trace,
            'degraded': degraded,
            'cache_status': cache_status,
            'latency_ms': (time.perf_counter() - started) * 1000.0,
        }
        if error_code is not None:
            event['error_code'] = error_code
            event['message'] = _SAFE_ERROR_MESSAGES.get(error_code, 'Request failed.')
        return event

    # ------------------------------------------------------------------
    # LLM / RAG calls
    # ------------------------------------------------------------------
    def _call_llm(
        self,
        request: TurnRequest,
        conversation: Conversation,
        evidence: list[dict[str, Any]],
        step: int,
        turn_id: str,
    ) -> Any:
        history = [turn.to_dict() for turn in conversation.turns[-self._history_limit:]]
        kwargs: dict[str, Any] = {
            'conversation_id': request.conversation_id,
            'turn_id': turn_id,
            'user_text': request.text,
            'history': history,
            'evidence': evidence,
            'step': step,
            'frontend': request.frontend.value,
            'metadata': dict(request.metadata),
        }
        last_error: Exception | None = None
        for _ in range(2):
            if self._is_cancelled(request.conversation_id, turn_id):
                raise _CancelledError()
            try:
                result = self._call_with_timeout(
                    self._llm.next_action, self._llm_timeout, **kwargs
                )
            except Exception as exc:
                if self._is_cancelled(request.conversation_id, turn_id):
                    raise _CancelledError() from exc
                last_error = exc
                continue
            if self._is_cancelled(request.conversation_id, turn_id):
                # Cancellation arrived while the model call was in flight.
                raise _CancelledError()
            return result
        raise _LLMFailure('llm failed') from last_error

    def _retrieve(
        self,
        provider: Any,
        query: str,
        conversation_id: str,
        turn_id: str,
    ) -> list[Any]:
        # The engine's configured RAG timeout bounds every call, even when a
        # provider advertises a longer (but still valid) timeout.
        timeout = min(self._rag_timeout, float(provider.timeout))
        last_error: Exception | None = None
        for _ in range(2):
            if self._is_cancelled(conversation_id, turn_id):
                raise _CancelledError()
            try:
                result = self._call_with_timeout(
                    provider.retrieve,
                    timeout,
                    query,
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                )
                return list(result or [])
            except Exception as exc:
                if self._is_cancelled(conversation_id, turn_id):
                    raise _CancelledError() from exc
                last_error = exc
        raise RuntimeError('rag provider failed') from last_error

    def _call_with_timeout(
        self,
        fn: Callable[..., Any],
        timeout: float,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _is_valid_action(action: Any) -> bool:
        if not isinstance(action, LLMAction):
            return False
        if action.action == 'answer':
            return True
        if action.action == 'retrieve':
            return bool(action.provider_id) and bool(
                action.query and action.query.strip()
            )
        return False

    def _is_cancelled(self, conversation_id: str, turn_id: str) -> bool:
        with self._cancelled_guard:
            return (
                (conversation_id, turn_id) in self._cancelled
                or (conversation_id, '*') in self._cancelled
            )

    # ------------------------------------------------------------------
    # Cache and persistence
    # ------------------------------------------------------------------
    def _obtain(self, conversation_id: str) -> tuple[Conversation, str]:
        existing = self._cache.get(conversation_id)
        if existing is not None:
            if existing.dirty:
                # REQ-5: retry a failed write on the next access.
                self._persist(existing)
            return existing, 'hit'

        snapshot = self._store.load(conversation_id)
        if snapshot is None:
            conversation = Conversation(conversation_id=conversation_id)
        else:
            conversation = Conversation.from_snapshot(snapshot)
        self._admit(conversation)
        return conversation, 'miss'

    def _admit(self, conversation: Conversation) -> None:
        """Insert a conversation, persisting each victim before removal."""
        conversation_id = conversation.conversation_id
        with self._cache_lock:
            if conversation_id in self._cache:
                self._cache.insert(conversation_id, conversation)
                return
            while len(self._cache) >= self._cache.capacity:
                victim_id = self._cache.lru_key()
                if victim_id is None:
                    break
                victim = self._cache.peek(victim_id)
                if isinstance(victim, Conversation) and not self._persist(victim):
                    # Never silently discard history on a failed write: the
                    # dirty conversation stays resident (the cache is allowed
                    # to exceed capacity) and the write is retried on the next
                    # eviction or access.
                    break
                self._cache.delete(victim_id)
            # Insert without evicting: eviction is owned by this loop so that a
            # failed write can never drop a conversation from memory.
            self._cache.insert(conversation_id, conversation)

    def _persist_with_retry(self, conversation: Conversation, attempts: int = 3) -> bool:
        for _ in range(max(1, attempts)):
            if self._persist(conversation):
                return True
        return False

    def _persist(self, conversation: Conversation) -> bool:
        conversation.updated_at = time.time()
        snapshot = conversation.to_snapshot()
        try:
            self._store.save(snapshot)
        except Exception:  # noqa: BLE001 - mark dirty and retry on next eviction
            conversation.dirty = True
            if 'PERSIST_FAILURE' not in self._health_errors:
                self._health_errors.append('PERSIST_FAILURE')
            return False
        conversation.dirty = False
        return True

    def _lock_for(self, conversation_id: str) -> Any:
        with self._locks_guard:
            lock = self._locks.get(conversation_id)
            if lock is None:
                lock = threading.Lock()
                self._locks[conversation_id] = lock
            return lock
