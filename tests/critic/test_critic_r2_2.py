# REQ-7 regression: voice must not wait for the entire short response before
# emitting the first audio chunk.  The default 40-character flush threshold is
# never reached for a short answer with no sentence terminator, so the first
# audio chunk is delayed until the core has already emitted every token and is
# about to emit turn.final.

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
    def synthesize(self, text: str, **kwargs: Any) -> list[bytes]:
        return [b'audio-1', b'audio-2']


def test_first_audio_chunk_is_emitted_before_entire_response_has_streamed() -> None:
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
    token_indices = [index for index, kind in enumerate(types) if kind == 'token']
    first_audio_index = types.index('audio.chunk')

    assert token_indices, 'expected streamed text tokens'
    assert first_audio_index < token_indices[-1], (
        'first audio chunk must be emitted before the final text token; '
        'otherwise the voice frontend waited for the whole response'
    )
