# Chatterbox

Chatterbox is a Python framework for building stateful, tool-using conversational
applications across text and voice frontends. It keeps language-model output behind typed
actions, deterministic workflows, validated tools, versioned policies, and persistence.

The current repository is a reference implementation and development harness. It includes
deterministic fakes for RAG, LLM, order lookup, attachment validation, and storage so the
full conversation lifecycle can be tested without external services.

## Highlights

- Shared core for text and voice conversations
- Streaming text and incremental TTS-friendly voice output
- Pluggable RAG providers with timeouts and evidence tracing
- Typed actions for answers, retrieval, and business tools
- JSON-defined workflows with version pinning and explicit tool allowlists
- Tool input schemas plus preconditions and postconditions
- Versioned execution policies and final-response guards
- Conversation history, LRU caching, and pluggable persistence
- Client sequence reordering with gap notifications and resend support
- Deterministic end-to-end scenarios for local development

## Architecture

```text
TextFrontend ───────────────┐
                            ├── ConversationEngine ── LLMClient
VoiceFrontend ── STT/TTS ───┘          │
                                       ├── RAGProvider
                                       ├── ToolRegistry
                                       ├── WorkflowRegistry
                                       ├── PolicyRegistry
                                       ├── ResponseGuard
                                       └── ConversationStore + LRU Cache
```

The LLM proposes one of three typed actions:

- `AnswerAction` returns a final answer and optional evidence claims.
- `RetrieveAction` searches a selected RAG provider.
- `ToolAction` calls a whitelisted business tool.

Raw LLM JSON is validated through `parse_llm_action`. Workflow and policy checks determine
whether an action is allowed; the model never directly executes a database operation.

## Requirements

- Python 3.11 or newer
- No runtime third-party dependencies
- `pytest`, `ruff`, and `pyright` for development

## Quick Start

```bash
cd /home/jc/Projects/chatterbox
PYTHONPATH="$PWD" python3 - <<'PY'
from conversational_agent import ConversationEngine, FakeRAGProvider, FakeTextLLM
from conversational_agent.frontends.text import TextFrontend

engine = ConversationEngine(
    providers=[FakeRAGProvider("knowledge-base")],
    llm=FakeTextLLM(),
)
text = TextFrontend(engine)
events = list(
    text.send(
        conversation_id="demo",
        idempotency_key="turn-1",
        text="What is the refund policy?",
        metadata={"provider_id": "knowledge-base"},
    )
)

print(next(event for event in events if event["type"] == "turn.final")["text"])
PY
```

The example uses deterministic fakes. The same frontend and conversation engine can be
used after replacing the RAG and LLM adapters with production implementations.

## Examples

Run the ten-turn memory and store-reload example:

```bash
PYTHONPATH="$PWD" python3 examples/text_chat_memory_demo.py
```

Run the five-turn damaged-item support scenario:

```bash
PYTHONPATH="$PWD" python3 examples/support_scenario_demo.py
```

The support scenario demonstrates:

1. RAG policy lookup and order-number collection
2. Customer-owned order lookup
3. Item selection
4. Image attachment validation
5. Policy-based exchange or refund eligibility
6. Follow-up proof requirements

## Conversation Memory

Completed turns are stored in `Conversation.turns`. The LLM receives recent completed
turns through the configured history window, which defaults to fifty turns.

Active conversations are held in an LRU cache. Conversations are pinned while a turn is
running, persisted through `ConversationStore`, and reloaded by a new engine when the same
conversation ID is used again.

The default store is `InMemoryConversationStore`. For an external database, use
`FunctionConversationStore` with save and load callbacks.

## Dynamic Workflows

Workflows are loaded from JSON files in `workflows/`. A workflow declares:

- Entry intents and keywords
- Slot definitions and validation patterns
- States and transitions
- Allowed tools per state
- Optional subflow and handoff metadata

At chat start, `WorkflowSelector` uses an explicit `workflow_id` from frontend metadata or
selects a workflow from the first user message. The selected workflow version and state are
pinned in conversation metadata for the duration of the conversation.

The registry supports:

- Atomic reload of new workflow versions
- Explicit activation as an administrator approval step
- Rollback to the previous active version
- Runtime handoff to another workflow

## Tools and Validation

Business tools are registered in `ToolRegistry` using `ToolSpec`. Every tool definition
contains an input schema, a handler, and optional preconditions and postconditions.

The tool boundary validates:

- Required arguments
- Primitive value types
- Format patterns
- Tool preconditions
- Handler results and postconditions
- Workflow tool allowlists
- Tool-call limits and timeouts

Database access is exposed through explicit read-only tools. The model cannot generate
arbitrary SQL or invoke an unregistered tool.

## RAG and Business Data

Use RAG for unstructured knowledge such as policies, procedures, FAQs, and product
documentation. Use managed tools for customer-specific facts such as orders, item
ownership, delivery state, and verified attachments.

RAG evidence is passed to the LLM and recorded in the turn trace. A retrieved policy
identifier can be linked automatically to a versioned `ExecutionPolicy`.

The separation is intentional:

- RAG supplies policy text and evidence.
- Versioned policy supplies executable constraints.
- Tools supply transactional facts.
- `ResponseGuard` checks final claims against required slots and policy rules.

## Final Response Guards

The LLM can attach structured claims to an answer. `ResponseGuard` checks those claims
against the pinned execution policy before the answer is returned.

For example, a `refund_completed` claim is rejected unless the required refund state exists.
Unsafe answers are replaced with the policy safe fallback and recorded as degraded.

The guard does not perform semantic entailment over arbitrary prose. Production responses
should expose structured decisions and evidence references alongside generated text.

## Client Message Ordering

`ReorderingBuffer` supports `client_seq` values for text turns. Out-of-order requests wait
for the missing sequence until the configured timeout. A timeout produces a
`sequence.gap` event containing:

- Expected sequence
- Buffered sequences
- `resend_from`

When the client resends the missing request, contiguous buffered requests are released in
order. Duplicate and already-delivered sequences are ignored.

`OrderedTextFrontend` connects this buffer to the normal text frontend. Audio packet
reordering belongs in the audio transport layer; Chatterbox provides semantic turn
ordering after STT.

## Voice Frontend

`VoiceFrontend` accepts audio and an end-of-speech signal, calls an injected STT adapter,
and sends the transcript through the same conversation core used by text.

Assistant tokens are buffered into incremental TTS requests. The first audio chunk can be
emitted at the first word boundary, and cancellation supports barge-in.

The repository does not include production WebRTC, WebSocket, acoustic echo cancellation,
or audio playback. Those belong in a transport adapter.

## Development

Install development tools:

```bash
python3 -m pip install pytest ruff "pyright[nodejs]"
```

Run tests:

```bash
python3 -m pytest -q -x
```

Run static checks:

```bash
ruff check . --no-cache
pyright
```

Current repository status:

```text
pytest: 42 passing tests
ruff: pass
pyright: 0 errors
```

## Project Layout

```text
conversational_agent/
  core/                 conversation engine and typed models
  frontends/            text and voice adapters
  contracts.py          LLM protocol and runtime adapter validation
  providers.py          RAG provider contract and validation
  storage.py            memory and function-backed stores
  tools.py              tool registry and validation middleware
  workflow.py           workflow loading, selection, reload, rollback, handoff
  policy.py             execution policies and response guards
  ordering.py           client sequence reordering and gap recovery
  fakes.py              deterministic RAG and LLM adapters
  support.py            order DB, attachment guard, and support scenario
workflows/              declarative workflow definitions
policies/               versioned execution policies
examples/               executable local demos
tests/                  unit and end-to-end contract tests
```

## Reference Documentation

- [design.md](design.md) explains the architectural decisions and workflow model.
- [implementation.md](implementation.md) describes the current runtime from the top down.
- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) records the implementation sequence and
  integration plan.

## Current Limitations

- The local demos use deterministic fake services.
- Production LLM, RAG, order DB, STT, and TTS adapters are not included.
- The workflow condition language supports simple named predicates, not arbitrary
  expression graphs.
- Administrator approval is represented by explicit registry activation, not a multi-user
  approval UI.
- Response validation covers structured claims, not semantic verification of arbitrary
  prose.
- Refund and replacement tools are read-only fakes; production side effects require
  authorization, idempotency, retries, durable execution, and compensating transactions.

## Suggested Next Steps

- Add production adapters for LLM, RAG, order storage, attachment verification, STT, and
  TTS.
- Add workflow staging and approval APIs or a management UI.
- Expand workflow conditions, subflow scheduling, and handoff rules.
- Add durable execution for business side effects and human approval.
- Add a low-latency voice transport layer such as WebRTC, Pipecat, or LiveKit.
