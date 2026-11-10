# Text Chat RAG Implementation Plan

## Goal

Implement a working local text-chat path with deterministic fake adapters while keeping the
core frontend-agnostic.

## Architecture

- `provider_id` identifies a RAG provider, not text/voice.
- The frontend selects a provider using `TurnRequest.metadata["provider_id"]`.
- Text and voice share the same `ConversationEngine` and `FakeTextLLM`.
- Voice adds STT/TTS adapters around the shared core.
- `FunctionConversationStore` wraps caller-provided save/load functions.
- `InMemoryConversationStore` remains the default when no store is provided.

## Steps

1. Add `FakeRAGProvider`.
   - Deterministic documents.
   - `provider_id`, `timeout`, and `retrieve(query)` contract.
2. Add `FakeTextLLM`.
   - First action retrieves from the provider selected by frontend metadata.
   - Second action answers using normalized RAG evidence.
3. Add `FunctionConversationStore`.
   - `save_fn(snapshot)` and `load_fn(conversation_id)`.
   - Keep `InMemoryConversationStore` as the default.
4. Add Text Chat E2E coverage.
   - TextFrontend → core → FakeRAGProvider → FakeTextLLM → FunctionStore.
   - Assert RAG event, final answer, and persisted conversation.
5. Preserve voice compatibility.
   - Voice passes the same provider metadata to the shared engine after STT.

## Usage

```python
rows = {}
store = FunctionConversationStore(
    save_fn=lambda snapshot: rows.__setitem__(snapshot["conversation_id"], snapshot),
    load_fn=lambda conversation_id: rows.get(conversation_id),
)
engine = ConversationEngine(
    providers=[FakeRAGProvider("knowledge-base")],
    llm=FakeTextLLM(),
    store=store,
)
text = TextFrontend(engine)
events = text.send(
    conversation_id="c1",
    idempotency_key="turn-1",
    text="What is the refund policy?",
    metadata={"provider_id": "knowledge-base"},
)
```

## Later Real Integrations

- Replace `FakeRAGProvider` with vector DB or search API adapter.
- Replace `FakeTextLLM` with a real `next_action()` adapter.
- Replace `FunctionConversationStore` callbacks with SQL/NoSQL operations.
- Add real STT/TTS adapters for voice; keep the same engine and LLM.

## Verification

```bash
python3 -m pytest -q -x
ruff check . --no-cache
pyright
```
