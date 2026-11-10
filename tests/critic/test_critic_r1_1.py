# REQ-7 regression: voice frontend must stream audio before the whole answer.

from __future__ import annotations

from typing import Any

from conversational_agent.core.engine import ConversationEngine
from conversational_agent.core.models import LLMAction
from conversational_agent.frontends.voice import STTResult, VoiceFrontend


class Provider:
    provider_id = 'p1'
    timeout = 1.0

    def retrieve(self, query: str, **kwargs: Any) -> list[Any]:
        return []


class AnswerLLM:
    def next_action(self, *args: Any, **kwargs: Any) -> LLMAction:
        return LLMAction(action='answer', text='hello world')


class StubSTT:
    def transcribe(self, audio: bytes, **kwargs: Any) -> STTResult:
        return STTResult(text='hello world', confidence=0.99)


class StubTTS:
    def __init__(self) -> None:
        self.calls = 0

    def synthesize(self, text: str, **kwargs: Any) -> list[bytes]:
        self.calls += 1
        return [b'audio-1', b'audio-2']


def test_voice_emits_first_audio_before_core_final() -> None:
    engine = ConversationEngine(providers=[Provider()], llm=AnswerLLM())
    voice = VoiceFrontend(engine, stt=StubSTT(), tts=StubTTS())

    events = list(
        voice.handle_audio(
            conversation_id='c',
            idempotency_key='k',
            audio=b'audio',
            end_of_speech=True,
        )
    )
    types = [event['type'] for event in events]

    assert 'audio.chunk' in types
    # REQ-7: voice SHALL NOT wait for the entire response before emitting the
    # first audio chunk.
    assert types.index('audio.chunk') < types.index('turn.final')
