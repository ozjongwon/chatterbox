# SPEC.md — Conversational Agentic System with Text and Voice Frontends

## 1. Purpose and Scope

Build one conversational core engine that serves two interchangeable frontends: text and voice. Both frontends use the same conversation model and core turn contract. The core performs agentic orchestration using an LLM plus one or more modular RAG providers supplied at startup. Active conversations are held in an in-memory LRU cache and persisted to a database when evicted. The system must meet measurable interactive-latency targets.

## 2. Definitions

- **Conversation**: An ordered set of turns and associated metadata, identified by an opaque conversation_id.
- **Turn**: One user input plus the assistant response, including streaming output events, RAG/LLM trace, status, and timestamps.
- **Core engine**: Frontend-agnostic orchestration service that manages turns, RAG calls, LLM calls, cache, persistence, and conversation state.
- **Frontend adapter**: Text or voice implementation that converts client input into the shared turn contract and converts core output events into text tokens or audio.
- **RAG provider**: A startup-supplied retrieval component implementing a common provider interface with a unique provider_id, retrieve operation, and timeout.
- **LRU capacity**: Maximum number of conversations held in memory.
- **In-flight turn**: A turn that has been accepted by the core but has not reached a final success, error, or cancellation status.
- **Reference benchmark**: The documented acceptance profile in REQ-7, using 4 vCPU, 8 GB RAM, one service instance, 20 concurrent sessions, 200 turns, and configured provider stubs.
- **Interactive latency SLO**: The measurable thresholds defined in REQ-7.

## 3. Assumptions

- A single service instance owns the in-memory LRU cache. Multi-instance cache coordination is out of scope.
- The database is transactional and supports upsert/read by conversation_id.
- RAG providers, LLM adapter, STT, and TTS are configured at startup with credentials and timeouts.
- The provider set is immutable after startup; dynamic hot-swapping is out of scope.
- Frontends supply an opaque conversation_id and a per-turn idempotency key.
- Authentication, authorization, tenant management, and PII retention policy are deployment concerns, not defined here.
- Existing libraries and frameworks may be used, but this specification defines behavior, not implementation.
- Voice transport, telephony, audio codecs, echo cancellation, and full-duplex acoustics are out of scope; voice barge-in is represented by a cancellation event.

## 4. Requirements

### REQ-1 — Interchangeable frontend contract

The core SHALL expose one frontend-agnostic turn contract used by both text and voice adapters. A conversation started through one frontend SHALL be resumable through the other using the same conversation_id and SHALL preserve prior history. The core SHALL NOT contain modality-specific branching based on whether the caller is text or voice.

**Test:** Run an identical scripted scenario through the text adapter and through the voice adapter with stubbed STT/TTS; assert equivalent normalized core event sequences, equivalent final conversation state, and successful cross-modality continuation. Use a dependency/architecture test to verify the core module has no dependency on either frontend implementation.

### REQ-2 — Startup RAG provider injection and validation

The core SHALL accept one or more RAG providers at startup through configuration or dependency injection. It SHALL validate each provider has a unique provider_id, a callable retrieve operation, and a positive timeout. Startup SHALL fail with a non-zero exit and an actionable error if no provider is supplied, a provider is invalid, or provider IDs are duplicated. Runtime retrieval SHALL use only the validated startup providers.

**Test:** Start with valid mock providers and verify both can be selected and called. Start with zero providers, a provider missing retrieve, a provider with non-positive timeout, and duplicate provider IDs; each must fail startup with the corresponding error and must not serve requests.

### REQ-3 — Agentic orchestration with bounded RAG/LLM loop

For each turn, the core SHALL use the LLM to decide whether to answer or request retrieval, SHALL invoke zero or more RAG providers through the provider interface, and SHALL produce a final assistant response. The loop SHALL be bounded by a configurable maximum number of model/action steps, default 4, and a maximum number of RAG calls per turn, default 4. The core SHALL record an ordered trace containing each model action, provider_id, retrieval query, returned evidence identifiers, and timing.

**Test:** Use scripted mock LLMs to produce: no retrieval, one retrieval, multiple retrievals, and more actions than the step limit. Assert the expected provider calls occur, the trace is ordered and complete, and every scenario terminates with a final status within the configured bound.

### REQ-4 — In-memory LRU cache semantics

The core SHALL keep active conversations in memory keyed by conversation_id with a configurable capacity of at least 1. Reads and writes SHALL update recency. When inserting a new conversation exceeds capacity, the least recently used conversation SHALL be selected for eviction. A conversation with an in-flight turn SHALL be pinned and SHALL NOT be evicted until that turn reaches success, error, or cancellation.

**Test:** With capacity 2, access conversations A, B, then insert C; assert A is selected for eviction. Read A then insert D; assert B is selected. Start a slow turn on B, insert enough conversations to exceed capacity, and assert B is not evicted until the turn completes or cancels.

### REQ-5 — Persist before eviction, with failure safety and graceful shutdown

Before removing an evicted conversation from memory, the core SHALL persist a complete snapshot to the database containing ordered turns, metadata, RAG/LLM trace, schema version, and updated_at. The removal SHALL occur only after the database write succeeds. If the write fails, the core SHALL retain the conversation in memory or mark it dirty, SHALL NOT silently discard it, SHALL expose a health/error signal, and SHALL retry on the next eviction or access. On graceful shutdown, the core SHALL persist all cached conversations before exiting.

**Test:** Use a mock database to succeed and fail during eviction. On success, assert the row exists before the cache entry is removed. On failure, assert the cache entry remains or is marked dirty, an error signal is emitted, and a later successful retry persists it. Trigger graceful shutdown and assert all cached conversations are persisted.

### REQ-6 — Cache-miss restore and exactly-once turn append

If a conversation_id is not present in cache, the core SHALL load the latest persisted snapshot from the database before processing the new turn. If no snapshot exists, it SHALL create a new conversation. Each completed turn SHALL be appended exactly once using the client-supplied idempotency key, so duplicate client retries do not create duplicate turns. Restored conversations SHALL retain prior turn order and metadata.

**Test:** Evict conversation A, then submit a new turn for A; assert the persisted history is loaded, the new turn is appended, and the combined order is correct. Submit the same turn twice with the same idempotency key; assert only one completed turn exists and the second request returns the same final outcome.

### REQ-7 — Streaming and interactive latency SLOs

Both frontends SHALL stream incremental output and SHALL NOT wait for the entire response before emitting the first text token or first audio chunk. In the reference benchmark with provider stubs configured as STT final 400 ms, LLM first token 600 ms, each RAG call 250 ms, and TTS first audio 400 ms, the system SHALL meet these p95 thresholds over at least 200 turns:

- Text: first visible token within 1.5 s of turn acceptance; final token within 8 s for responses up to 500 tokens.
- Voice: first audible output within 2.5 s of the end-of-speech event; final audible output within 10 s for responses up to 500 tokens.
- Core-plus-adapter overhead excluding provider delays: p95 at most 300 ms for text first token and at most 500 ms for voice first audio.

**Test:** Run the documented reference benchmark harness, capture timestamps at turn acceptance, end-of-speech, first output, and final output, and compute p95 values. Assert all thresholds are met.

### REQ-8 — Timeouts, retries, degradation, and user-safe errors

Every RAG, LLM, STT, and TTS call SHALL have a configurable timeout. Defaults SHALL be RAG 2 s per call, LLM first token 5 s, and total turn 30 s. The core SHALL retry a failed provider or LLM call at most once where the provider is idempotent. On RAG failure or timeout, the core SHALL continue with remaining providers or with no retrieval and SHALL mark the turn as degraded in its trace. On final LLM failure after retry, the core SHALL emit a final error event and SHALL NOT persist a completed assistant turn for that failed attempt. Error payloads SHALL NOT expose stack traces, credentials, or raw provider internals.

**Test:** Use mock providers that fail, time out, and recover on retry. Assert the specified retry count, degraded status on RAG failure, final error status on LLM failure, no completed assistant turn for failed attempts, and absence of sensitive fields in error output.

### REQ-9 — Concurrency and conversation isolation

The system SHALL support at least 20 concurrent sessions without data leakage between conversations. Operations on the same conversation SHALL be serialized or rejected with a documented conflict error; interleaved writes to one conversation SHALL NOT corrupt turn order. Cache and persistence operations SHALL be safe under concurrent access.

**Test:** Run 20 concurrent scripted sessions with unique content and conversation IDs, then verify every final transcript contains only its own session content. Run concurrent turns against one conversation ID and assert either serialized order or a documented conflict error, with no interleaved or duplicated turns.

### REQ-10 — Cancellation and voice interruption

Each frontend SHALL be able to cancel an in-flight turn through a cancellation signal. On cancellation, the core SHALL stop further LLM and RAG work, and the voice frontend SHALL stop playback on barge-in. A cancelled partial assistant response SHALL NOT be persisted as a completed turn. The conversation SHALL remain usable for subsequent turns.

**Test:** Start a turn against a slow mock LLM or RAG provider, send cancellation, and assert provider work stops, no completed assistant turn is persisted, cancellation status is reported, and a subsequent turn on the same conversation succeeds.

### REQ-11 — Observability and health

For every turn, the system SHALL emit structured events containing conversation_id, frontend type, cache hit/miss/eviction status, provider calls, step count, final status, and latency measurements. It SHALL expose a health endpoint reporting database connectivity, provider registry validity, and cache size versus capacity.

**Test:** Capture structured events for successful, degraded, failed, and cancelled turns; assert required fields are present. Simulate database failure and invalid provider state; assert the health endpoint reports each condition and cache metrics reflect current size and capacity.

## 5. Edge Cases

- Empty or whitespace-only text input SHALL be rejected with a validation error and no LLM call.
- Empty or low-confidence STT transcript SHALL cause the voice frontend to ask the user to repeat and SHALL NOT create an assistant turn.
- A conversation requested during cache miss with no persisted snapshot SHALL be created as a new conversation.
- A conversation with an in-flight turn SHALL be pinned and not evicted until the turn finishes or cancels.
- A RAG provider returning no evidence SHALL be treated as a successful empty retrieval with degraded or no-evidence metadata.
- An LLM producing an invalid action or malformed tool request SHALL be rejected by the core and either retried within the step bound or terminated with a documented error.
- TTS failure SHALL emit an error event; the text transcript of the assistant response SHALL remain available to the voice adapter and SHALL be persisted according to REQ-6 if the turn otherwise completed.
- A database outage during eviction SHALL retain or mark the conversation dirty and SHALL NOT silently lose history.
- A process crash may lose conversations that were active but not evicted; graceful shutdown flush is required by REQ-5.
- Duplicate idempotency keys SHALL return the recorded outcome rather than creating a second turn.

## 6. Error Behavior

- Invalid startup configuration or invalid RAG providers SHALL fail startup and prevent serving.
- RAG timeout or failure SHALL degrade the turn but not necessarily fail it.
- LLM failure after retry SHALL produce a user-safe final error event and SHALL NOT persist a completed assistant turn for that attempt.
- STT failure SHALL produce a voice-frontend error or repeat request and SHALL NOT create an assistant turn.
- TTS failure SHALL produce an audio error event while preserving available text output.
- Database write failure during eviction SHALL retain the conversation in memory or mark it dirty, alert health, and retry.
- Cancellation SHALL stop generation and mark the turn cancelled; partial output SHALL not be treated as a completed turn.
- All client-facing errors SHALL be documented, non-sensitive, and associated with a stable error code.

## 7. Non-Goals

- No UI design, telephony integration, audio transport, codec handling, echo cancellation, or full-duplex voice implementation.
- No implementation of STT, TTS, LLM, vector database, or RAG algorithms.
- No model training, prompt-engineering standard, or guarantee of factual correctness beyond recorded retrieval evidence.
- No horizontal scaling, distributed cache, or multi-instance LRU coordination.
- No authentication, authorization, tenant isolation, billing, analytics, or compliance certification.
- No dynamic RAG provider hot-swapping after startup.
- No long-term retention or PII policy definition.
- No mobile SDK, web framework, or deployment topology mandate.

## 8. Real Ambiguities

- The exact agent policy and prompt strategy are intentionally unspecified; the contract only requires a bounded LLM/RAG loop with traceability.
- RAG provider selection policy is unspecified; implementations may use priority, routing, or ensemble behavior as long as it is deterministic, configurable, and recorded.
- Database technology and schema are unspecified; the required behavior is transactional snapshot persistence and restore by conversation_id.
- “Barge-in” is defined as cancellation of an in-flight turn and playback stop, not as full acoustic echo cancellation or full-duplex speech.
- The reference benchmark hardware and provider stub delays are acceptance parameters; changing them requires an explicit SLO revision.
- Multi-instance deployment and shared cache behavior are out of scope; a single-instance ownership assumption applies.
- Retention duration, data residency, and PII handling are deployment-specific and not fixed by this specification.