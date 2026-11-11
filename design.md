# Chatbox Design

작성 시점: 2026-11-10

이 문서는 현재 chatterbox 구현과 이후 실제 고객지원 서비스로 확장하기 위한 설계 결정을
정리한다.

## 1. 목표

Text와 Voice frontend가 동일한 core engine과 LLM을 사용하면서 다음 기능을 제공한다.

- 텍스트 채팅
- 음성 STT/TTS adapter
- startup 시 주입되는 modular RAG
- 주문/상품 조회를 위한 read-only DB tool
- 사진 등 사용자 증빙 검증
- conversation history와 persistence
- 업무 조건과 최종 결정을 강제하는 validation layer

핵심 원칙:

> LLM은 의도 이해와 자연스러운 표현을 담당하고, 업무상 허용되는 action과 최종 결정은
> deterministic workflow/validator가 통제한다.

## 2. 전체 구조

```text
Text 사용자
   └─ TextFrontend ───────────────┐
                                  │
Voice 사용자                      ├─ ConversationEngine ─ LLMAdapter
   └─ STT → VoiceFrontend ────────┤          │
      TTS ← audio.chunk           │          ├─ RAGProvider
                                  │          ├─ ToolRegistry
                                  │          ├─ ConversationStore
                                  │          └─ LRU Cache
                                  │
                          Workflow / Validator / ResponseGuard
```

Text와 Voice는 입력·출력 형식만 다르고, 대화 상태와 LLM/RAG/tool loop는 공유한다.

## 3. Frontend

### TextFrontend

- 문자열 입력을 `TurnRequest`로 변환한다.
- `metadata`에 사용자가 선택한 RAG `provider_id` 등을 전달할 수 있다.
- core event를 text 응답으로 streaming한다.

### VoiceFrontend

- STT adapter가 audio를 text로 변환한다.
- 이후 `ConversationEngine`은 Text와 동일한 경로를 사용한다.
- assistant token을 TTS adapter로 streaming해 `audio.chunk` event를 생성한다.
- barge-in은 진행 중인 turn을 cancel한다.

Voice adapter는 실제 음성 전송/재생을 포함하지 않는다. WebSocket/WebRTC와 실제 STT/TTS
서비스는 이후 adapter로 추가한다.

## 4. Provider ID의 의미

`provider_id`는 text/voice를 구분하는 값이 아니다.

```text
Frontend 구분      → FrontendType.TEXT | FrontendType.VOICE
RAG provider 구분  → provider_id="product-policy" | "faq-kb" | ...
```

사용자는 frontend에서 RAG provider를 선택할 수 있다.

```python
metadata = {"provider_id": "product-policy"}
```

하나의 provider를 Text와 Voice가 공유할 수 있다. 필요하면 frontend별로 다른 provider를
선택할 수도 있지만 이는 core의 고정 규칙이 아니라 선택 정책이다.

## 5. RAG와 업무 DB의 역할

### RAG 사용 대상

- 파손/환불/교환 정책
- 보증과 반품 기간
- 고객지원 절차
- 제품 사용법과 FAQ
- 일반적인 보상 기준

### DB 사용 대상

- 주문번호와 주문 상태
- 주문에 포함된 상품
- 고객-주문 소유권
- 배송 상태
- 환불/교환 처리 상태
- 인증된 사진 증빙

### 판단 원칙

```text
일반 지식·정책·절차   → RAG
사용자별 사실·상태    → DB/tool
정책 + 사용자 사실    → DB 확인 후 RAG 정책 적용
```

RAG는 문서 설명과 정책 근거를 제공한다. 실제 환불 가능 여부, tool 허용, 최종 승인은
versioned workflow policy와 validator가 결정한다.

## 6. Conversation Memory

### In-turn memory

한 turn 안에서 다음 정보가 유지된다.

- `history`
- `evidence`
- `tool_results`
- trace

### Conversation history

성공한 turn은 `Conversation.turns`에 user/assistant 메시지와 trace를 저장한다.

LLM 호출 시 최근 `history_limit`개의 완료 turn이 전달된다. 기본값은 50이다.

```text
1번째 turn history = 0
2번째 turn history = 1
...
11번째 turn history = 10
```

### Persistence

- 현재 conversation은 LRU cache에 유지된다.
- turn 완료 후 `ConversationStore.save(snapshot)`를 호출한다.
- cache에서 evict되는 conversation도 persist한다.
- 새 engine이 같은 store를 사용하면 이전 conversation을 복원한다.

### Client sequence ordering

Text 메시지는 `client_seq`를 함께 전달할 수 있다.

`ReorderingBuffer`는 conversation/session별로 기대 sequence를 추적한다.

- 기대보다 큰 sequence는 pending buffer에 보관한다.
- 기대 sequence가 도착하면 pending 항목까지 연속으로 release한다.
- 누락 sequence가 `gap_timeout`을 넘으면 `sequence.gap` 이벤트를 생성한다.
- 이벤트에는 `expected_seq`, `buffered_sequences`, `resend_from`이 포함된다.
- client가 누락 sequence를 다시 보내면 buffered 요청들을 순서대로 실행한다.
- 이미 처리된 sequence는 idempotency key와 sequence comparison으로 중복 제거한다.

`OrderedTextFrontend.send`는 sequence-aware text ingress entry point이고,
`OrderedTextFrontend.poll`은 gap timeout 알림 entry point다.

음성에서는 packet-level jitter buffer와 transcript 이후 semantic turn ordering을
분리해야 한다. 현재 구현은 semantic turn ordering이며 실제 RTP jitter buffer는
transport layer에서 추가해야 한다.

기본 store:

```python
InMemoryConversationStore()
```

함수 기반 adapter:

```python
FunctionConversationStore(
    save_fn=my_save,
    load_fn=my_load,
)
```

### RAG evidence memory

현재 turn의 RAG 원문은 LLM `evidence`로 전달된다.

이전 turn history에는 RAG `evidence_ids`와 trace가 남지만 원문 전체는 저장하지 않는다.
후속 turn에서 원문이 필요하면 다음 중 하나를 선택한다.

- 다시 RAG 호출
- RAG 결과를 `Turn.metadata` 또는 별도 evidence table에 저장
- 요약을 assistant turn에 포함
- conversation-level RAG memory 유지

## 7. LLM Action

LLM은 자연어를 직접 실행하지 않고 명시적 typed action을 제안한다.

```python
RetrieveAction(
    provider_id="product-policy",
    query="damaged on arrival refund policy",
)
```

```python
ToolAction(
    tool_name="lookup_order",
    arguments={"order_id": "12345", "customer_id": "customer-1"},
)
```

```python
AnswerAction(
    text="세 상품 중 어떤 상품이 파손됐나요?",
)
```

내부 Engine은 action class 타입으로 분기하며, 기존 adapter 호환을 위해 `LLMAction` factory를
제공한다. 외부 LLM JSON은 `parse_llm_action`이 검증 후 typed action으로 변환한다.

지원 action:

- `retrieve`: RAG provider 호출
- `tool`: whitelist business tool 호출
- `answer`: assistant 응답

## 8. 업무 Workflow와 Slot

업무 전체를 문장 순서로 hardcode하지 않는다. 상태·slot·허용 action을 선언한다.

예시 상태:

```python
state = {
    "intent": "damaged_on_arrival",
    "stage": "need_order",
    "slots": {
        "order_id": None,
        "item_id": None,
        "photo_verified": False,
        "policy_evidence_id": None,
        "refund_eligible": None,
    },
}
```

예시 정책:

```python
DAMAGE_CLAIM_POLICY = {
    "need_order": {
        "required": ["order_id"],
        "allowed_tools": ["retrieve_policy", "lookup_order"],
        "next": "need_item",
    },
    "need_item": {
        "required": ["order_id", "item_id"],
        "allowed_tools": ["lookup_order_item"],
        "next": "need_photo",
    },
    "need_photo": {
        "required": ["item_id", "photo_verified"],
        "allowed_tools": ["validate_attachment"],
        "next": "policy_check",
    },
    "policy_check": {
        "required": ["photo_verified", "policy_evidence_id"],
        "allowed_tools": ["retrieve_policy"],
        "next": "ready_for_decision",
    },
}
```

### LLM과 Workflow의 역할 분리

LLM:

- 사용자의 자연어 의도 이해
- slot에 해당하는 정보 추출
- 자연스러운 질문과 설명 생성
- 후보 tool 선택

Workflow:

- 다음에 받아야 할 필수 slot 결정
- 현재 state에서 허용되는 tool 결정
- 최종 결정 가능 여부 판단
- 누락 정보가 있으면 질문 단계로 되돌림

예를 들어 주문번호와 상품이 모두 없을 때 LLM이 문장을 선택할 수는 있지만,
workflow의 우선순위에 따라 주문번호를 먼저 요청할 수 있다.

## 9. Guided Slot Filling

고객지원·거래 업무에서는 시스템이 필요한 순서를 안내하고 slot을 수집하는 방식이
더 안정적이다.

```text
1. 상품 파손 신고
   → 주문번호 요청

2. 주문번호 12345
   → 주문 조회
   → 파손 상품 요청

3. 사진 액자
   → 상품 소속 검증
   → 파손 사진 요청

4. 사진 첨부
   → 이미지 검증
   → RAG 정책 조회
   → 교환/환불 가능 여부 제시
```

사용자가 여러 정보를 한 번에 제공하면 이미 채워진 slot 질문은 생략한다.

```text
User: "12345 주문의 사진 액자가 깨졌고 사진도 첨부했어요."
```

Workflow는 주문번호·상품·사진 요청을 건너뛰고 검증 단계로 진행할 수 있다.

## 10. Tool Registry

LLM이 직접 DB를 조회하거나 SQL을 생성하지 않는다. Whitelist tool만 호출한다.

```python
tools = {
    "lookup_order": repository.lookup_order,
    "lookup_order_item": repository.lookup_order_item,
    "validate_attachment": guard.validate_attachment,
}
```

Tool 호출은 다음 순서를 따른다.

```text
1. tool whitelist 확인
2. arguments schema 확인
3. 호출 권한 확인
4. 현재 workflow state에서 허용된 tool인지 확인
5. tool별 precondition 확인
6. tool 실행
7. result schema와 postcondition 확인
8. trace와 idempotency 기록
```

공통 검사는 middleware/wrapper가 담당하고, tool에는 해당 업무의 pre/post condition만
정의하는 것이 좋다.

## 11. Validation 예시

### 주문 조회

Precondition:

- 5자리 주문번호
- 로그인 고객 ID 존재
- 현재 workflow에서 주문 조회 허용

Postcondition:

- 주문이 존재
- 주문 소유자가 현재 고객과 일치

### 상품 조회

Precondition:

- 주문 ID와 상품 ID 존재

Postcondition:

- 상품이 해당 주문에 포함

### 사진 증빙

Precondition:

- attachment ID 존재

Postcondition:

- `image/*` content type
- 증빙 검증 완료

### RAG 정책

Precondition:

- query가 비어 있지 않음

Postcondition:

- evidence ID가 1개 이상
- 정책 version 또는 policy ID가 확인됨

## 12. 최종 결정 검증

Tool과 RAG가 성공해도 최종 답변이 자동으로 증명되는 것은 아니다.

예:

```text
refund_eligible = true
```

이것은 환불 가능성을 의미하지 실제 환불 완료를 의미하지 않는다.

최종 응답은 구조화된 claim을 포함하는 것이 좋다.

```json
{
  "answer": "환불 접수가 가능합니다.",
  "claims": [
    {
      "type": "decision",
      "value": "refund_eligible",
      "evidence_ids": ["policy-ata-1", "photo-1"]
    }
  ]
}
```

금지 예:

- 환불 tool 없이 `refund_completed`
- 사진 검증 없이 `damage_verified`
- 주문 소유권 확인 없이 주문 상세 공개
- RAG 근거 없이 특정 정책 인용
- 승인되지 않은 교환/환불 확정

## 13. Policy는 RAG에서 직접 실행하지 않는다

RAG는 정책 문서와 근거를 검색한다. 하지만 RAG 결과가 tool 허용 여부나 환불 결정을 직접
바꾸면 안 된다.

구분:

```text
RAG 문서
  - 정책 설명
  - 고객 안내 문구
  - 적용 조건
  - 절차

Versioned Policy
  - allowed actions
  - required slots
  - allowed tools
  - forbidden claims
  - escalation 조건
```

RAG가 `policy_id`를 찾으면 versioned `PolicyRegistry`에서 실행 정책을 로드한다.

```text
RAG → policy_id 검색
PolicyRegistry → 검증된 정책 version 로드
Workflow/Validator → tool과 최종 결정 통제
```

## 14. 현재 구현된 Fake Scenario

현실적인 파손 상품 support 시나리오가 구현돼 있다.

```text
1. 고객: 상품이 깨져서 도착
   → RAG 정책 검색
   → 주문번호 요청

2. 고객: 주문번호 12345
   → lookup_order
   → 상품 선택 요청

3. 고객: photo frame, 사진 첨부
   → lookup_order
   → lookup_order_item
   → validate_attachment
   → RAG 정책
   → 교환/환불 가능성 제시

4. 고객: 환불 요청
   → 환불 접수 진행 안내

5. 고객: 필요한 증빙 질문
   → 사진 요구사항 안내
```

관련 파일:

- `conversational_agent/support.py`
- `tests/test_support_scenario.py`
- `examples/support_scenario_demo.py`
- `conversational_agent/workflow.py`
- `workflows/damaged_on_arrival.v3.json`
- `workflows/order_status.v2.json`

## 15. 현재 구현 파일

```text
conversational_agent/core/engine.py       core loop, tool/RAG 실행
conversational_agent/core/models.py       Turn, LLMAction, Conversation
conversational_agent/providers.py         RAGProvider contract
conversational_agent/storage.py           ConversationStore, InMemory/Function store
conversational_agent/cache.py             LRU conversation cache
conversational_agent/fakes.py             FakeRAGProvider, FakeTextLLM
conversational_agent/support.py           order DB, attachment guard, support LLM
conversational_agent/workflow.py          workflow load/select/validate/pin
conversational_agent/frontends/text.py    TextFrontend
conversational_agent/frontends/voice.py   STT/TTS VoiceFrontend
```

### 15.1 현재 구현 상태

구현 완료:

- JSON workflow definition loading
- `WorkflowRegistry`
- explicit `workflow_id` 선택
- 첫 메시지 keyword 기반 workflow 선택
- conversation 시작 시 workflow ID/version/initial state pinning
- 선택된 workflow의 `allowed_tools` 검사
- tool/RAG 결과를 `workflow_slots`에 반영
- 단순 predicate 기반 state transition
- workflow reload, explicit activate approval, rollback
- subflow 선언 필드와 runtime handoff API
- 공통 `ToolSpec`/`ToolRegistry`
- tool input schema, precondition, postcondition 검증
- `ExecutionPolicy`/`PolicyRegistry`
- RAG 결과의 `policy_id`를 versioned execution policy에 자동 연결
- `ResponseGuard`와 forbidden claim 검증
- support 시나리오 5턴 E2E
- text memory 10턴 + store reload E2E
- FunctionConversationStore
- generic `tool` action

부분 구현:

- tool별 owner/item/photo 검증은 support adapter에 구현됨
- workflow state transition은 현재 `order_id_present`, `item_id_present`,
  `photo_verified`, `policy_evidence_id_present` 수준
- subflow는 선언 가능하지만 자동 실행 graph scheduler는 아직 없음
- handoff는 API 수준으로 구현됐지만 자동 trigger 규칙은 단순함

미구현:

- workflow 관리자 UI와 다단계 approval workflow
- 복잡한 JSON 조건식 DSL
- 실제 STT/TTS/LLM/DB adapter

## 16. 직접 구현과 Framework 선택

현재는 custom lightweight core를 유지하는 것이 적합하다.

이유:

- core가 작고 deterministic하다.
- RAG, tool, memory, persistence, validation을 테스트로 확보했다.
- framework를 빨리 도입하면 문제 원인이 framework와 자체 코드에 분산된다.

동적 workflow loading, 승인/rollback, policy guard의 1차 구현은 이미 완료됐다.
다음 단계는 framework 도입보다
현재 DSL과 검증 계층을 완성하는 것이다.

```text
1. workflow DSL 조건식/subflow/handoff 확장
2. 실제 LLM/RAG/DB adapter
3. workflow 관리자 UI와 운영 approval pipeline
```

Framework 도입을 고려할 조건:

- workflow 5개 이상
- tool 15개 이상
- human-in-the-loop 승인
- 장시간 실행과 재시작/복구
- multi-agent handoff
- 복잡한 branch/join graph
- 강한 감사·정책 준수·observability 요구
- payment/refund 같은 side effect의 보상 transaction

후보:

| Framework | 적합한 용도 |
|---|---|
| LangGraph | state graph, branching, checkpoint, multi-agent |
| PydanticAI | typed LLM/tool output과 agent 구조 |
| OpenAI Agents SDK | tool loop, tracing, OpenAI 중심 agent |
| Temporal/DBOS | durable workflow, 승인, 재시도, side effect |

권장:

```text
현재 ConversationEngine 유지
+ JSON 기반 workflow DSL과 registry (1차 구현)
+ ToolSpec/ToolRegistry + pre/post validation (구현)
+ ResponseGuard + versioned PolicyRegistry (구현)

복잡도 증가 시 orchestration은 LangGraph 검토
실제 결제/환불 side effect는 Temporal/DBOS 검토
```

## 17. 테스트

```bash
cd ~/Projects/chatterbox
python3 -m pytest -q -x
ruff check . --no-cache
pyright
```

현재 검증:

```text
pytest: 42 passed
ruff: All checks passed
pyright: 0 errors
```

현재 포함된 테스트:

- 선언형 workflow state/slot 전이
- workflow explicit/keyword selection과 version pinning
- workflow reload/activate/rollback
- client sequence reordering, gap notification, resend 처리
- subflow 선언과 workflow handoff
- ToolSpec schema/pre/post validation
- ResponseGuard forbidden claim 검증
- RAG policy_id → versioned policy 연결
- 10턴 conversation history와 store reload
- function store 기반 conversation persistence
- 선택된 RAG provider와 fake text chat E2E
- 주문 DB, 상품 DB, 사진 증빙, RAG 정책을 결합한 5턴 support E2E

추가 예정 테스트:

- 복잡한 조건식/subflow scheduler
- 다단계 관리자 approval workflow
- human-in-the-loop escalation
- 실제 LLM tool-calling 품질 평가
- 실제 DB transaction/idempotency
