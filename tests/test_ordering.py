from __future__ import annotations

from conversational_agent import ConversationEngine, FakeRAGProvider, FakeTextLLM
from conversational_agent.frontends.text import OrderedTextFrontend, TextFrontend
from conversational_agent.ordering import ReorderingBuffer


def test_out_of_order_messages_wait_for_missing_message() -> None:
    now = [0.0]
    buffer = ReorderingBuffer(
        "session-1",
        gap_timeout=0.05,
        clock=lambda: now[0],
    )
    assert buffer.accept(2, "r2") == []
    assert buffer.accept(3, "r3") == []
    assert buffer.next_seq == 1
    assert buffer.expire() is None

    now[0] = 0.1
    gap = buffer.expire()
    assert gap is not None
    assert gap.expected_seq == 1
    assert gap.buffered_sequences == (2, 3)
    assert gap.to_event()["resend_from"] == 1
    assert gap.to_event()["type"] == "sequence.gap"

    assert buffer.accept(1, "r1") == ["r1", "r2", "r3"]
    assert buffer.next_seq == 4


def test_duplicate_and_invalid_sequences() -> None:
    buffer = ReorderingBuffer("s")
    assert buffer.accept(1, "r1") == ["r1"]
    assert buffer.accept(1, "r1-retry") == []
    try:
        buffer.accept(0, "bad")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid sequence was accepted")


def test_ordered_frontend_notifies_client_and_releases_buffered_turns() -> None:
    engine = ConversationEngine(
        providers=[FakeRAGProvider("fake")],
        llm=FakeTextLLM(),
    )
    ordered = OrderedTextFrontend(TextFrontend(engine), gap_timeout=0)

    events2 = list(ordered.send(
        "conversation-1", "r2", "second", client_seq=2,
        metadata={"provider_id": "fake"},
    ))
    assert events2[0]["type"] == "sequence.gap"
    assert events2[0]["resend_from"] == 1

    events1 = list(ordered.send(
        "conversation-1", "r1", "first", client_seq=1,
        metadata={"provider_id": "fake"},
    ))
    finals = [event for event in events1 if event["type"] == "turn.final"]
    assert len(finals) == 2
    assert all(event["status"].value == "success" for event in finals)
