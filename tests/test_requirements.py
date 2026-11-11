'''Contract tests for approved batch REQ-1 through REQ-4.'''
from __future__ import annotations

from typing import Any

import pytest

from conversational_agent.cache import LRUCache
from conversational_agent.core.engine import ConversationEngine
from conversational_agent.core.models import (
    FrontendType,
    LLMAction,
    LLMActionType,
    TurnRequest,
    TurnStatus,
    ValidationError,
)
from conversational_agent.frontends.text import TextFrontend
from conversational_agent.frontends.voice import STTResult, VoiceFrontend
from conversational_agent.providers import ProviderValidationError
from conversational_agent.storage import InMemoryConversationStore


class Evidence:
    def __init__(self, evidence_id: str) -> None:
        self.evidence_id = evidence_id


class MockRAGProvider:
    def __init__(
        self,
        provider_id: str,
        evidence_ids: tuple[str, ...] = ('e1',),
        timeout: float = 1.0,
        fail: bool = False,
    ) -> None:
        self.provider_id = provider_id
        self.timeout = timeout
        self.evidence_ids = tuple(evidence_ids)
        self.fail = fail
        self.calls: list[tuple[str, Any, Any]] = []

    def retrieve(self, query: str, **kwargs: Any) -> list[Evidence]:
        self.calls.append((query, kwargs.get('conversation_id'), kwargs.get('turn_id')))
        if self.fail:
            raise RuntimeError('provider failure')
        return [Evidence(evidence_id) for evidence_id in self.evidence_ids]


class ScriptedLLM:
    def __init__(self, actions: list[LLMActionType]) -> None:
        self.actions = list(actions)
        self.calls: list[dict[str, Any]] = []

    def next_action(self, *args: Any, **kwargs: Any) -> LLMAction:
        self.calls.append(kwargs)
        if not self.actions:
            return LLMAction(action='answer', text='default answer')
        return self.actions.pop(0)


class AnswerLLM:
    def __init__(self, text: str = 'answer') -> None:
        self.text = text
        self.calls = 0

    def next_action(self, *args: Any, **kwargs: Any) -> LLMAction:
        self.calls += 1
        return LLMAction(action='answer', text=self.text)


class StubSTT:
    def __init__(self, text: str, confidence: float = 0.99) -> None:
        self.text = text
        self.confidence = confidence

    def transcribe(self, audio: bytes, **kwargs: Any) -> STTResult:
        return STTResult(text=self.text, confidence=self.confidence)


class StubTTS:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def synthesize(self, text: str, **kwargs: Any) -> list[bytes]:
        self.texts.append(text)
        return [b'audio-1', b'audio-2']


def make_engine(
    providers: list[Any] | None = None,
    llm: Any | None = None,
    cache_capacity: int = 100,
    store: Any | None = None,
    **kwargs: Any,
) -> ConversationEngine:
    return ConversationEngine(
        providers=list(providers or []),
        llm=llm or AnswerLLM(),
        store=store or InMemoryConversationStore(),
        cache_capacity=cache_capacity,
        **kwargs,
    )


def req(
    conversation_id: str,
    idempotency_key: str,
    text: str = 'hello',
    frontend: FrontendType = FrontendType.TEXT,
) -> TurnRequest:
    return TurnRequest(
        conversation_id=conversation_id,
        idempotency_key=idempotency_key,
        text=text,
        frontend=frontend,
    )


def as_dict(event: Any) -> dict[str, Any]:
    if isinstance(event, dict):
        return dict(event)
    if hasattr(event, 'to_dict'):
        return event.to_dict()
    return dict(vars(event))


def event_type(event: Any) -> str | None:
    data = as_dict(event)
    return data.get('type') or data.get('event_type')


def status_value(value: Any) -> Any:
    return getattr(value, 'value', value)


def final_event(events: list[Any]) -> dict[str, Any]:
    for event in reversed(events):
        data = as_dict(event)
        if event_type(data) in {'turn.final', 'turn.error', 'turn.cancelled'}:
            return data
    raise AssertionError(f'no final event in {events!r}')


def final_text(events: list[Any]) -> str:
    data = final_event(events)
    return data.get('text') or data.get('response_text') or ''


def trace_of(final: dict[str, Any]) -> list[dict[str, Any]]:
    return final.get('trace') or (final.get('payload') or {}).get('trace') or []


CORE_EVENT_TYPES = {
    'turn.accepted',
    'llm.action',
    'rag.call',
    'token',
    'turn.final',
    'turn.error',
    'turn.cancelled',
}


def normalize_core_events(events: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for event in events:
        data = as_dict(event)
        kind = event_type(data)
        if kind not in CORE_EVENT_TYPES:
            continue
        item = {
            key: data.get(key)
            for key in (
                'type',
                'event_type',
                'text',
                'status',
                'provider_id',
                'query',
                'evidence_ids',
                'error_code',
                'degraded',
                'action',
            )
            if data.get(key) is not None
        }
        if 'type' not in item and 'event_type' in item:
            item['type'] = item.pop('event_type')
        item.pop('event_type', None)
        if 'status' in item:
            item['status'] = status_value(item['status'])
        if 'evidence_ids' in item:
            item['evidence_ids'] = tuple(item['evidence_ids'])
        normalized.append(item)
    return normalized


def test_req_1_primary_success_path() -> None:
    llm = AnswerLLM('Hello from core')
    engine = make_engine(providers=[MockRAGProvider('p1')], llm=llm)
    text_frontend = TextFrontend(engine)
    voice_frontend = VoiceFrontend(engine, stt=StubSTT('Hello from core'), tts=StubTTS())

    text_events = list(
        text_frontend.send(
            conversation_id='conv-text',
            idempotency_key='t1',
            text='hello',
        )
    )
    voice_events = list(
        voice_frontend.handle_audio(
            conversation_id='conv-voice',
            idempotency_key='v1',
            audio=b'audio',
            end_of_speech=True,
        )
    )

    assert normalize_core_events(text_events) == normalize_core_events(voice_events)
    assert final_text(text_events) == 'Hello from core'
    assert final_text(voice_events) == 'Hello from core'
    assert status_value(final_event(text_events)['status']) == TurnStatus.SUCCESS.value
    assert status_value(final_event(voice_events)['status']) == TurnStatus.SUCCESS.value

    text_state = engine.get_conversation('conv-text')
    voice_state = engine.get_conversation('conv-voice')
    assert len(text_state.turns) == 1
    assert len(voice_state.turns) == 1
    assert text_state.turns[-1].assistant_text == 'Hello from core'
    assert voice_state.turns[-1].assistant_text == 'Hello from core'

    shared_text_events = list(
        text_frontend.send(
            conversation_id='conv-shared',
            idempotency_key='s1',
            text='first',
        )
    )
    shared_voice_events = list(
        voice_frontend.handle_audio(
            conversation_id='conv-shared',
            idempotency_key='s2',
            audio=b'second',
            end_of_speech=True,
        )
    )
    shared_state = engine.get_conversation('conv-shared')
    assert len(shared_state.turns) == 2
    assert [turn.frontend for turn in shared_state.turns] == [
        FrontendType.TEXT,
        FrontendType.VOICE,
    ]
    assert final_text(shared_text_events) == 'Hello from core'
    assert final_text(shared_voice_events) == 'Hello from core'


def test_req_1_invalid_input_handling() -> None:
    llm = AnswerLLM()
    engine = make_engine(providers=[MockRAGProvider('p1')], llm=llm)
    text_frontend = TextFrontend(engine)

    with pytest.raises(ValidationError) as exc_info:
        list(
            text_frontend.send(
                conversation_id='empty',
                idempotency_key='k1',
                text='   ',
            )
        )
    assert getattr(exc_info.value, 'code', '') == 'EMPTY_INPUT'
    assert llm.calls == 0
    assert len(engine.get_conversation('empty').turns) == 0

    voice_frontend = VoiceFrontend(
        engine,
        stt=StubSTT('', confidence=0.05),
        tts=StubTTS(),
    )
    events = list(
        voice_frontend.handle_audio(
            conversation_id='low-conf',
            idempotency_key='k2',
            audio=b'noise',
            end_of_speech=True,
        )
    )
    types = [event_type(event) for event in events]
    assert any(
        kind in {'audio.repeat_requested', 'turn.repeat_requested', 'voice.repeat_requested'}
        for kind in types
    )
    assert llm.calls == 0
    assert len(engine.get_conversation('low-conf').turns) == 0


def test_req_2_primary_success_path() -> None:
    provider_one = MockRAGProvider('p1', evidence_ids=('e1',))
    provider_two = MockRAGProvider('p2', evidence_ids=('e2',))
    llm = ScriptedLLM(
        [
            LLMAction(action='retrieve', provider_id='p1', query='q1'),
            LLMAction(action='retrieve', provider_id='p2', query='q2'),
            LLMAction(action='answer', text='both'),
        ]
    )
    engine = make_engine(providers=[provider_one, provider_two], llm=llm)
    events = list(engine.run_turn(req('c2', 'k2')))

    assert [call[0] for call in provider_one.calls] == ['q1']
    assert [call[0] for call in provider_two.calls] == ['q2']
    final = final_event(events)
    assert status_value(final['status']) == TurnStatus.SUCCESS.value
    assert final_text(events) == 'both'
    trace = trace_of(final)
    rag_provider_ids = [
        item.get('provider_id')
        for item in trace
        if item.get('type') in {'rag_call', 'rag.call'}
    ]
    assert rag_provider_ids == ['p1', 'p2']


@pytest.mark.parametrize(
    ('providers', 'expected_code'),
    [
        ([], 'NO_RAG_PROVIDERS'),
        ([object()], 'INVALID_PROVIDER'),
        ([MockRAGProvider('bad-timeout', timeout=0)], 'INVALID_PROVIDER_TIMEOUT'),
        ([MockRAGProvider('dup'), MockRAGProvider('dup')], 'DUPLICATE_PROVIDER_ID'),
    ],
)
def test_req_2_invalid_input_handling(
    providers: list[Any],
    expected_code: str,
) -> None:
    with pytest.raises(ProviderValidationError) as exc_info:
        ConversationEngine(
            providers=providers,
            llm=AnswerLLM(),
            store=InMemoryConversationStore(),
            cache_capacity=2,
        )
    assert getattr(exc_info.value, 'code', None) == expected_code
    assert str(exc_info.value).strip()


@pytest.mark.parametrize(
    ('actions', 'expected_provider_calls', 'expected_trace_types'),
    [
        (
            [LLMAction(action='answer', text='direct')],
            [],
            ['model_action'],
        ),
        (
            [
                LLMAction(action='retrieve', provider_id='p1', query='q1'),
                LLMAction(action='answer', text='one'),
            ],
            ['p1'],
            ['model_action', 'rag_call', 'model_action'],
        ),
        (
            [
                LLMAction(action='retrieve', provider_id='p1', query='q1'),
                LLMAction(action='retrieve', provider_id='p2', query='q2'),
                LLMAction(action='answer', text='two'),
            ],
            ['p1', 'p2'],
            ['model_action', 'rag_call', 'model_action', 'rag_call', 'model_action'],
        ),
    ],
)
def test_req_3_primary_success_path(
    actions: list[LLMActionType],
    expected_provider_calls: list[str],
    expected_trace_types: list[str],
) -> None:
    provider_one = MockRAGProvider('p1', evidence_ids=('e1',))
    provider_two = MockRAGProvider('p2', evidence_ids=('e2',))
    llm = ScriptedLLM(actions)
    engine = make_engine(
        providers=[provider_one, provider_two],
        llm=llm,
        max_steps=4,
        max_rag_calls=4,
    )
    events = list(engine.run_turn(req('c3', 'k3')))
    final = final_event(events)

    assert status_value(final['status']) == TurnStatus.SUCCESS.value
    actual_provider_ids: list[str] = []
    for provider in (provider_one, provider_two):
        actual_provider_ids.extend([provider.provider_id] * len(provider.calls))
    assert sorted(actual_provider_ids) == sorted(expected_provider_calls)

    trace = trace_of(final)
    assert [item.get('type') for item in trace] == expected_trace_types
    for item in trace:
        if item.get('type') == 'rag_call':
            assert item.get('provider_id') in {'p1', 'p2'}
            assert item.get('query')
            assert isinstance(item.get('evidence_ids'), (list, tuple))
            assert item.get('duration_ms', 0) >= 0


def test_req_3_invalid_input_handling() -> None:
    provider = MockRAGProvider('p1')
    llm = ScriptedLLM(
        [
            LLMAction(action='retrieve', provider_id='p1', query=f'q{index}')
            for index in range(10)
        ]
    )
    engine = make_engine(
        providers=[provider],
        llm=llm,
        max_steps=2,
        max_rag_calls=2,
    )
    events = list(engine.run_turn(req('c3-invalid', 'k3-invalid')))
    final = final_event(events)

    assert status_value(final['status']) == TurnStatus.ERROR.value
    assert final.get('error_code') in {'STEP_LIMIT_EXCEEDED', 'MAX_STEPS_EXCEEDED'}
    assert len(llm.calls) <= 2
    assert len(provider.calls) <= 2
    assert len(engine.get_conversation('c3-invalid').turns) == 0

    class MalformedLLM:
        def __init__(self) -> None:
            self.calls = 0

        def next_action(self, *args: Any, **kwargs: Any) -> Any:
            self.calls += 1
            return object()

    bad_llm = MalformedLLM()
    engine_two = make_engine(
        providers=[MockRAGProvider('p1')],
        llm=bad_llm,
        max_steps=2,
        max_rag_calls=2,
    )
    events_two = list(engine_two.run_turn(req('c3-malformed', 'k3-malformed')))
    final_two = final_event(events_two)

    assert status_value(final_two['status']) == TurnStatus.ERROR.value
    assert final_two.get('error_code') == 'INVALID_LLM_ACTION'
    assert len(engine_two.get_conversation('c3-malformed').turns) == 0


def test_req_4_primary_success_path() -> None:
    cache = LRUCache(capacity=2)
    cache.put('A', 'a')
    cache.put('B', 'b')

    assert cache.put('C', 'c') == ['A']
    assert cache.get('A') is None
    assert cache.get('B') == 'b'
    assert cache.put('D', 'd') == ['C']
    assert cache.get('B') == 'b'

    cache.pin('B')
    evicted = cache.put('E', 'e')
    assert 'B' not in evicted
    assert cache.get('B') == 'b'

    cache.unpin('B')
    assert 'B' in cache.put('F', 'f')


def test_req_4_invalid_input_handling() -> None:
    with pytest.raises(ValueError):
        LRUCache(capacity=0)

    cache = LRUCache(capacity=1)
    with pytest.raises(KeyError):
        cache.pin('missing')
    with pytest.raises(KeyError):
        cache.unpin('missing')

    cache.put('A', 'a')
    assert cache.put('B', 'b') == ['A']
