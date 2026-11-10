# Chatbox Implementation

## 1. Purpose

This document describes the current chatbox implementation from the top down. It explains
the runtime architecture, primary entry points, conversation lifecycle, workflow selection,
RAG and database tools, policy validation, memory, persistence, and voice integration.

The central design principle is that the language model proposes actions and generates
language, while deterministic workflow, tool, policy, and response guards control what the
system is allowed to do and claim.

## 2. High-Level Architecture

The system has five main layers.

The frontend layer accepts text or audio and converts either input into the shared turn
contract. Its primary entry points are `TextFrontend.send` and `VoiceFrontend.handle_audio`.

The orchestration layer owns turn execution, conversation locking, memory, cancellation,
RAG calls, tool calls, workflow state, and persistence. Its primary entry point is
`ConversationEngine.run_turn`, which validates input and delegates to
`ConversationEngine._run_turn` and `ConversationEngine._execute`.

The intelligence adapters provide language-model decisions and RAG results. The LLM adapter
exposes `next_action`, while RAG providers implement `retrieve` and conform to the
`RAGProvider` contract. The current local implementation uses `FakeTextLLM` and
`FakeRAGProvider`.

The control layer contains declarative workflows, business tools, execution policies, and
final-response validation. Important entry points are `WorkflowSelector.select`,
`WorkflowRegistry.get`, `ToolRegistry.get`, `ToolSpec.invoke`, `PolicyRegistry.get`, and
`ResponseGuard.validate`.

The persistence and memory layer stores completed turns, workflow slots, selected workflow
versions, and policy references. It uses `ConversationStore`, with
`InMemoryConversationStore` as the default and `FunctionConversationStore` as the
function-backed adapter.

## 3. Primary Runtime Flow

A text conversation starts when the host calls `TextFrontend.send`. That method creates a
`TurnRequest`, attaches frontend metadata, and delegates to
`ConversationEngine.run_turn`.

A voice conversation starts when the host calls `VoiceFrontend.handle_audio`. The frontend
calls the injected STT adapter, converts the transcript into the same `TurnRequest` type,
and delegates to the same `ConversationEngine.run_turn` entry point.

`ConversationEngine.run_turn` rejects empty input immediately. Valid input passes to
`ConversationEngine._run_turn`, which acquires a per-conversation lock so concurrent turns
for the same conversation cannot interleave.

The locked execution path calls `ConversationEngine._execute`. This method obtains the
conversation from the LRU cache or storage, handles idempotent turn replay, selects and pins
a workflow when necessary, and runs the bounded action loop.

The action loop repeatedly calls `ConversationEngine._call_llm`. The current LLM adapter
returns an `LLMAction` with one of three action types: retrieve, tool, or answer.

Retrieve actions call `ConversationEngine._retrieve`. Tool actions call
`ConversationEngine._call_tool`. Answer actions pass through `ResponseGuard.validate`
when a policy or structured claims are present.

Successful turns are appended to `Conversation.turns`, persisted through
`ConversationEngine._persist`, and returned as streaming core events ending with
`turn.final`.

## 4. Entry Points

The main text entry point is `TextFrontend.send`.

The main voice entry point is `VoiceFrontend.handle_audio`.

The shared conversation entry point is `ConversationEngine.run_turn`.

The internal turn execution entry point is `ConversationEngine._execute`.

The LLM decision entry point is the injected adapter's `next_action` method.

The RAG entry point is the selected provider's `retrieve` method through
`ConversationEngine._retrieve`.

The tool entry point is `ConversationEngine._call_tool`, which invokes either a registered
`ToolSpec.invoke` implementation or a legacy callable tool.

Workflow loading starts at `WorkflowRegistry.load_directory`. Safe updates use
`WorkflowRegistry.reload`, explicit approval uses `WorkflowRegistry.activate`, and
reversion uses `WorkflowRegistry.rollback`.

Workflow selection and pinning start at `WorkflowSelector.select` and
`WorkflowSelector.pin`. Cross-workflow transitions use `WorkflowSelector.handoff`.

Policy loading starts at `PolicyRegistry.load_directory`. Final answer safety is checked
through `ResponseGuard.validate`.

Cancellation enters through `ConversationEngine.cancel` or `VoiceFrontend.cancel`.
Flush and shutdown enter through `ConversationEngine.flush`,
`ConversationEngine.shutdown`, and the store adapter.

## 5. Conversation Lifecycle

Every successful turn records the user text, assistant text, frontend type, status, trace,
degraded flag, timestamps, and idempotency key.

Before processing a turn, the engine checks the idempotency key with
`Conversation.find_turn`. If the key was already completed, the engine emits replayed
accepted and final events instead of calling the LLM or tools again.

During LLM invocation, the engine builds a context containing the current user text,
recent history, current-turn RAG evidence, current-turn tool results, frontend type,
request metadata, workflow state, workflow slots, and any pinned execution-policy
references.

The default history window is fifty completed turns. The LLM therefore sees prior user and
assistant messages, including their traces, but the current turn is not appended to history
until it succeeds.

The action loop is bounded by maximum steps, maximum RAG calls, maximum tool calls,
per-call timeouts, and a total turn timeout. Cancellation is checked before model calls,
after blocking calls, during RAG and tools, and while streaming output.

## 6. Workflow Selection

Workflow definitions are loaded from JSON files into `WorkflowRegistry`.

Each definition contains an identifier, version, entry intents, entry keywords, slot
definitions, states, allowed tools, and transition declarations.

At the start of a conversation, `WorkflowSelector.select` first honors an explicit
`workflow_id` in request metadata. If no explicit workflow is present, it scores active
workflow definitions using their entry keywords against the first user message.

Once selected, `WorkflowSelector.pin` writes the workflow identifier, version, initial
state, and empty slot map into conversation metadata. The version is pinned for the life
of the conversation, so a newly deployed workflow version does not alter a conversation
that is already in progress.

Workflow state transitions update conversation metadata after successful tool or RAG
results. The current transition evaluator uses named predicates such as order identifier
presence, item identifier presence, photo verification, and policy evidence presence.

`WorkflowSelector.handoff` supports explicit movement to another workflow. It stores the
previous workflow context in conversation metadata, pins the target workflow, and records
the source workflow. Automatic handoff triggering is currently limited to explicit code
paths rather than an arbitrary condition-expression engine.

## 7. Workflow Administration

`WorkflowRegistry` maintains all loaded versions and a separately active version for each
workflow identifier.

`WorkflowRegistry.reload` validates a candidate directory and merges new versions without
immediately activating them. `WorkflowRegistry.activate` represents explicit administrator
approval. `WorkflowRegistry.rollback` activates the previous lower version.

This provides version staging, approval, and rollback behavior without an external workflow
framework. A management UI or multi-approver workflow is not yet implemented.

## 8. RAG Execution

RAG providers are validated by `validate_providers` when the engine is constructed. Each
provider must expose a provider identifier, a positive finite timeout, and a callable
`retrieve` operation.

The LLM selects a provider by returning a retrieve action with a provider identifier and
query. The engine resolves the provider from its registry and calls
`ConversationEngine._retrieve`.

Provider calls have bounded timeout and retry behavior. Successful results are converted
into evidence identifiers and normalized evidence payloads. RAG events and trace entries
include provider identifier, query, evidence identifiers, duration, and degradation state.

The RAG provider selection is separate from frontend type. Both text and voice can use the
same provider, or the frontend can pass different provider metadata based on user choice.

## 9. RAG-to-Policy Linking

RAG results may include a policy identifier. When that identifier appears,
`ConversationEngine._update_workflow_from_rag` resolves it through `PolicyRegistry` and pins
the execution policy identifier and version into conversation metadata.

This separates retrieved policy text from executable policy rules. RAG supplies policy
evidence, while the versioned execution policy supplies required slots, forbidden claims,
and safe fallback behavior.

If RAG results do not contain a policy identifier, no execution policy is linked and normal
workflow validation remains in effect.

## 10. Tool Execution

Business tools are registered in `ToolRegistry`. Each `ToolSpec` declares a name,
description, input schema, handler, optional preconditions, and optional postconditions.

`ConversationEngine._call_tool` passes the tool arguments and execution context to
`ToolSpec.invoke`. Argument validation checks required properties, primitive types, and
regular-expression formats.

Preconditions run before the handler. Handler exceptions are converted into structured
failure results. Postconditions run only after a successful handler result. Validation
failures become structured tool errors rather than uncaught exceptions.

The engine records every tool call and result in the turn trace and emits a tool event.
Failed tools mark the turn as degraded and return their structured error to the LLM.

Workflow state also constrains which tools may run. `ConversationEngine._workflow_allows`
checks the current workflow state's allowed tool list before execution.

## 11. Final Response Guard

The LLM may attach structured claims to an answer action.

`ResponseGuard.validate` retrieves the conversation's pinned execution policy and checks
required slots and forbidden claims. Forbidden claims can require a slot such as a refund
identifier or verified refund eligibility. They can also require evidence identifiers.

When validation fails, the unsafe answer is not returned. The engine marks the response as
degraded and uses the policy safe fallback. A response-guard trace entry and event record
the result.

Answers without an execution policy and without structured claims continue through the
normal response path for backward compatibility.

## 12. Conversation Memory

Short-term turn context consists of history, evidence, tool results, trace, workflow state,
slots, and policy references.

Long-term conversation memory consists of completed turns stored in the `Conversation`
object. Each turn contains the user text, assistant text, frontend type, status, trace,
error information, degraded state, and completion time.

The LRU cache stores active conversations. Conversations are pinned while a turn is in
flight. Cache eviction attempts persistence first, and failed writes keep the dirty
conversation resident for later retry.

The default persistence adapter is `InMemoryConversationStore`.
`FunctionConversationStore` wraps caller-provided save and load functions for SQL,
document stores, or external databases.

## 13. Voice Frontend

`VoiceFrontend.handle_audio` accepts audio and an end-of-speech signal.

The injected STT adapter produces a transcript and confidence score. Empty or
low-confidence transcripts produce a repeat request without invoking the core.

Valid transcripts become the same `TurnRequest` used by text chat. The voice and text
frontends therefore share workflow selection, RAG, tools, policy checks, memory, and
persistence.

Assistant tokens are buffered and passed to the injected TTS adapter incrementally. The
first chunk is emitted at the first natural word boundary, with later flushes at sentence
or punctuation boundaries and configurable minimum text length.

Barge-in cancellation flows through `VoiceFrontend.cancel` and
`ConversationEngine.cancel`.

## 14. Fake Support Scenario

The current local support scenario demonstrates a realistic damaged-on-arrival flow.

The first user message triggers the policy RAG search. The model asks for an order number.

The second user message supplies order number 12345. The LLM issues a lookup-order tool
call. The fake order repository validates the order format and customer ownership, then
returns three items and asks which item was damaged.

The third user message identifies the photo frame and includes image metadata. The engine
performs order lookup, item lookup, attachment validation, and RAG policy lookup. Workflow
slots are updated and the workflow advances toward a decision state.

The fourth turn accepts a refund request. The fifth turn answers a proof follow-up
question. The support demo shows workflow selection, RAG calls, tool calls, final answers,
and decision output.

The primary local entries are `SupportScenarioLLM.next_action`,
`FakeOrderRepository.lookup_order`, `FakeOrderRepository.lookup_order_item`,
`FakeAttachmentGuard.validate_attachment`, and `support_tools`.

## 15. Evidence and Tracing

Each turn trace records model actions, RAG calls, tool calls, duration, and guard outcomes.
RAG evidence includes provider identifier, query, evidence identifiers, and normalized
evidence items.

Tool trace includes the tool name, arguments, structured result, duration, and status.
Workflow selection, state transitions, slots, response-guard results, and pinned versions
are available through events and conversation metadata.

This trace is used by tests and can be exposed by a production observability layer.

## 16. Current Tests and Validation

The current suite covers text frontend operation, voice frontend operation, idempotency,
cancellation, RAG provider validation, workflow loading and selection, workflow reload and
rollback, workflow handoff, tool schema and postconditions, policy forbidden claims,
ten-turn conversation memory, store reload, function-backed storage, and the realistic
five-turn support scenario.

Current verification status is thirty-four passing pytest tests, a clean Ruff check, and a
clean Pyright check.

## 17. Current Limitations

The implementation still uses deterministic fake services in the main local scenarios.
Real STT, TTS, LLM, RAG, and database adapters are not yet wired into production transport
paths.

The workflow transition language currently supports simple named predicates. Complex
boolean expressions, automatic subflow scheduling, and a full handoff graph are future work.

Administrator approval is represented by explicit registry activation rather than a
multi-user approval UI or signed approval workflow.

The response guard validates structured claims and policy requirements, but it does not
perform semantic natural-language entailment over arbitrary prose. Production responses
should continue using structured decisions and evidence references.

The current support tools are read-only fakes. Real refund, replacement, or fulfillment
side effects will require idempotency, authorization, durable execution, retries, and
compensating transactions.

The voice frontend handles logical streaming and interruption but does not include WebRTC,
WebSocket transport, acoustic echo cancellation, or production audio playback.

## 18. Recommended Next Steps

Implement real adapters for the LLM, RAG, order repository, attachment verification, STT,
and TTS.

Replace fake evidence with versioned policy identifiers and production policy storage.

Add a management API or UI for workflow staging, approval, activation, and rollback.

Expand the workflow expression language to support compound conditions, subflow scheduling,
and explicit handoff transitions.

Add durable execution for business side effects and human approval steps.

Add a low-latency voice transport layer when audio streaming becomes the primary interface.
