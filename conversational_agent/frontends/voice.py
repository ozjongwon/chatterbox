"""Voice frontend adapter with stubbed STT/TTS integration points."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from conversational_agent.core.engine import ConversationEngine
from conversational_agent.core.models import CoreEvent, FrontendType, TurnRequest

_SENTENCE_END = '.!?;:\n'
_TERMINAL_TYPES = {'turn.final', 'turn.error', 'turn.cancelled'}


@dataclass
class STTResult:
    text: str
    confidence: float = 1.0


class VoiceFrontend:
    """Converts audio into the same core turn contract used by text.

    Speech-to-text and text-to-speech are injected at startup. Audio transport
    and playback are intentionally out of scope; this adapter emits audio chunk
    events and stops yielding on cancellation (barge-in).

    Audio is synthesized incrementally while the core streams assistant tokens:
    the first word boundary is flushed immediately, later text at sentence
    boundaries or after a configurable amount of buffered text, and any
    remainder before the turn's final event. The first audio chunk therefore
    never waits for the whole response.
    """

    def __init__(
        self,
        engine: ConversationEngine,
        stt: Any,
        tts: Any,
        min_confidence: float = 0.5,
        min_chunk_chars: int = 40,
    ) -> None:
        self._engine = engine
        self._stt = stt
        self._tts = tts
        self._min_confidence = min_confidence
        self._min_chunk_chars = min_chunk_chars

    def handle_audio(
        self,
        conversation_id: str,
        idempotency_key: str,
        audio: bytes,
        end_of_speech: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[CoreEvent]:
        if not end_of_speech:
            return

        try:
            result = self._stt.transcribe(
                audio,
                conversation_id=conversation_id,
                idempotency_key=idempotency_key,
            )
        except Exception:  # noqa: BLE001 - STT boundary emits a user-safe error
            yield {
                'type': 'voice.error',
                'conversation_id': conversation_id,
                'error_code': 'STT_FAILURE',
                'message': 'Speech recognition failed. Please try again.',
            }
            return

        text = getattr(result, 'text', '') or ''
        confidence = float(getattr(result, 'confidence', 1.0))
        if not text.strip() or confidence < self._min_confidence:
            yield {
                'type': 'audio.repeat_requested',
                'conversation_id': conversation_id,
                'reason': 'EMPTY_OR_LOW_CONFIDENCE_TRANSCRIPT',
            }
            return

        request = TurnRequest(
            conversation_id=conversation_id,
            idempotency_key=idempotency_key,
            text=text,
            frontend=FrontendType.VOICE,
            metadata=dict(metadata or {}),
        )

        buffer: list[str] = []
        turn_id = ''
        tts_failed = False
        emitted_audio = False

        def flush() -> Iterator[CoreEvent]:
            nonlocal tts_failed, emitted_audio
            pending = ''.join(buffer)
            buffer.clear()
            if not pending.strip() or tts_failed:
                return
            try:
                chunks = self._tts.synthesize(
                    pending,
                    conversation_id=conversation_id,
                    turn_id=turn_id,
                )
                for chunk in chunks:
                    emitted_audio = True
                    yield {
                        'type': 'audio.chunk',
                        'conversation_id': conversation_id,
                        'turn_id': turn_id,
                        'audio': chunk,
                    }
            except Exception:  # noqa: BLE001 - TTS boundary preserves text output
                tts_failed = True
                yield {
                    'type': 'audio.error',
                    'conversation_id': conversation_id,
                    'turn_id': turn_id,
                    'error_code': 'TTS_FAILURE',
                    'message': 'Audio synthesis failed; the text response is still available.',
                    'text': pending,
                }

        for event in self._engine.run_turn(request):
            kind = event.get('type')
            if kind in _TERMINAL_TYPES or (
                kind == 'turn.accepted' and not event.get('replay')
            ):
                turn_id = str(event.get('turn_id') or turn_id)

            if kind == 'token':
                buffer.append(str(event.get('text') or ''))
                yield event
                pending_text = ''.join(buffer)
                if self._should_flush(pending_text, first_chunk=not emitted_audio):
                    yield from flush()
                continue

            if kind == 'turn.final':
                # Emit any buffered audio before the turn is announced as final.
                yield from flush()
                yield event
                return

            if kind == 'turn.cancelled':
                # Barge-in: stop playback and drop unsynthesized text.
                buffer.clear()
                yield event
                return

            yield event

    def _should_flush(self, text: str, *, first_chunk: bool) -> bool:
        stripped = text.rstrip()
        if not stripped:
            return False
        if stripped[-1] in _SENTENCE_END:
            return True
        # Emit the very first chunk at the first natural word boundary so the
        # first audio never waits for the complete response.
        if first_chunk and text[-1].isspace():
            return True
        return len(text) >= self._min_chunk_chars and text[-1].isspace()

    def cancel(self, conversation_id: str, turn_id: str | None = None) -> None:
        """Barge-in: cancel the in-flight turn and stop playback."""
        self._engine.cancel(conversation_id, turn_id)
