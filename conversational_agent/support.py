"""Deterministic support scenario with RAG, order DB tools, and attachments."""

from __future__ import annotations

import re
from typing import Any

from conversational_agent.core.models import LLMAction, LLMActionType
from conversational_agent.tools import ToolRegistry, ToolSpec


class FakeOrderRepository:
    """Read-only order repository with customer ownership checks."""

    def __init__(self) -> None:
        self._orders: dict[str, dict[str, Any]] = {
            "12345": {
                "order_id": "12345",
                "customer_id": "customer-1",
                "status": "delivered",
                "items": [
                    {"item_id": "photo-frame", "name": "photo frame", "status": "delivered"},
                    {"item_id": "lamp", "name": "table lamp", "status": "delivered"},
                    {"item_id": "mug", "name": "mug", "status": "delivered"},
                ],
            }
        }

    def lookup_order(
        self,
        order_id: str,
        customer_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not re.fullmatch(r"\d{5}", str(order_id)):
            return {
                "ok": False,
                "error_code": "INVALID_ORDER_NUMBER",
                "message": "Order number must contain exactly five digits.",
            }
        order = self._orders.get(str(order_id))
        if order is None:
            return {
                "ok": False,
                "error_code": "ORDER_NOT_FOUND",
                "message": "Order not found.",
            }
        if customer_id != order["customer_id"]:
            return {
                "ok": False,
                "error_code": "ORDER_ACCESS_DENIED",
                "message": "Order does not belong to this customer.",
            }
        return {"ok": True, "order": dict(order)}

    def lookup_order_item(
        self,
        order_id: str,
        item_id: str,
        customer_id: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        order = self.lookup_order(order_id, customer_id, **kwargs)
        if not order.get("ok"):
            return order
        for item in order["order"]["items"]:
            if item["item_id"] == item_id:
                return {"ok": True, "item": dict(item), "order_id": str(order_id)}
        return {
            "ok": False,
            "error_code": "ITEM_NOT_FOUND",
            "message": "Item is not part of this order.",
        }


class FakeAttachmentGuard:
    """Validates that a user-provided damage photo is attached."""

    def validate_attachment(
        self,
        attachment_id: str,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        attachments = (metadata or {}).get("attachments") or []
        for attachment in attachments:
            if str(attachment.get("attachment_id")) != str(attachment_id):
                continue
            if str(attachment.get("content_type", "")).startswith("image/"):
                return {
                    "ok": True,
                    "attachment_id": attachment_id,
                    "content_type": attachment.get("content_type"),
                    "verified": True,
                }
            return {
                "ok": False,
                "error_code": "INVALID_ATTACHMENT_TYPE",
                "message": "Damage evidence must be an image.",
            }
        return {
            "ok": False,
            "error_code": "ATTACHMENT_MISSING",
            "message": "No matching damage photo was attached.",
        }


class SupportScenarioLLM:
    """Scripted realistic policy: RAG, order DB, item DB, and proof validation."""

    def __init__(self, provider_id: str = "support-policy") -> None:
        self.provider_id = provider_id
        self.calls: list[dict[str, Any]] = []

    def next_action(self, **kwargs: Any) -> LLMActionType:
        self.calls.append(dict(kwargs))
        text = str(kwargs.get("user_text") or "")
        history = kwargs.get("history") or []
        evidence = kwargs.get("evidence") or []
        tool_results = kwargs.get("tool_results") or []
        metadata = kwargs.get("metadata") or {}
        customer_id = metadata.get("customer_id")
        attachments = metadata.get("attachments") or []
        last_assistant = next(
            (str(t.get("assistant_text") or "") for t in reversed(history)
             if t.get("assistant_text")),
            "",
        )
        if "refund" in text.lower() and "교환 또는 환불" in last_assistant:
            return LLMAction(
                action="answer",
                text="환불 접수를 진행하겠습니다. 검증된 사진 증빙과 주문 12345를 기준으로 처리합니다.",
            )
        if ("proof" in text.lower() or "증빙" in text) and "사진" in last_assistant:
            return LLMAction(
                action="answer",
                text="상품 전체와 파손 부위가 보이는 사진, 포장 상태 사진을 첨부해 주세요.",
            )

        order_id = re.search(r"\b\d{5}\b", text)
        if order_id is None:
            for turn in reversed(history):
                match = re.search(r"\b\d{5}\b", str(turn.get("user_text") or ""))
                if match:
                    order_id = match
                    break

        tool_names = [result.get("tool_name") for result in tool_results]
        if not evidence and not tool_results and order_id is None:
            if not history:
                return LLMAction(
                    action="retrieve",
                    provider_id=self.provider_id,
                    query="damaged item on arrival exchange refund policy",
                )
            return LLMAction(
                action="answer",
                text="주문번호를 알려주시겠어요?",
            )

        if order_id is not None and "lookup_order" not in tool_names:
            return LLMAction(
                action="tool",
                tool_name="lookup_order",
                arguments={
                    "order_id": order_id.group(0),
                    "customer_id": customer_id,
                },
            )

        if order_id is None:
            return LLMAction(
                action="answer",
                text="주문번호를 알려주시겠어요?",
            )

        order_result = next(
            (r for r in tool_results if r.get("tool_name") == "lookup_order"), None
        )
        if order_result is not None and not order_result.get("ok"):
            return LLMAction(
                action="answer",
                text=(
                    "주문을 확인하지 못했습니다. "
                    f"{order_result.get('message', '주문번호를 다시 확인해 주세요.')}"
                ),
            )
        if order_result is not None and "lookup_order_item" not in tool_names:
            names = {
                str(item.get("name", "")).lower()
                for item in order_result.get("order", {}).get("items", [])
            }
            selected = next((name for name in names if name in text.lower()), None)
            if selected is None:
                names_text = ", ".join(sorted(names))
                return LLMAction(
                    action="answer",
                    text=f"주문에 {names_text}가 있습니다. 어느 상품이 파손됐나요?",
                )
            return LLMAction(
                action="tool",
                tool_name="lookup_order_item",
                arguments={
                    "order_id": order_id.group(0),
                    "item_id": selected.replace(" ", "-"),
                    "customer_id": customer_id,
                },
            )

        item_result = next(
            (r for r in tool_results if r.get("tool_name") == "lookup_order_item"), None
        )
        if item_result is not None and not item_result.get("ok"):
            return LLMAction(action="answer", text=str(item_result.get("message")))

        if item_result is not None and "validate_attachment" not in tool_names:
            if not attachments:
                return LLMAction(
                    action="answer",
                    text="파손 상태를 확인할 수 있도록 상품과 포장 사진을 첨부해 주세요.",
                )
            return LLMAction(
                action="tool",
                tool_name="validate_attachment",
                arguments={"attachment_id": attachments[0].get("attachment_id")},
            )

        attachment_result = next(
            (r for r in tool_results if r.get("tool_name") == "validate_attachment"),
            None,
        )
        if attachment_result is not None and not attachment_result.get("ok"):
            return LLMAction(
                action="answer",
                text=(
                    "사진 증빙을 확인하지 못했습니다. "
                    f"{attachment_result.get('message', '이미지 파일을 다시 첨부해 주세요.')}"
                ),
            )

        if attachment_result is not None and not evidence:
            return LLMAction(
                action="retrieve",
                provider_id=self.provider_id,
                query="damaged item on arrival replacement refund policy",
            )

        policy = "; ".join(
            str(item.get("text"))
            for group in evidence
            for item in group.get("items", [])
            if item.get("text")
        )
        return LLMAction(
            action="answer",
            text=(
                "확인 결과: 도착 시 파손으로 접수할 수 있습니다. "
                "사진 증빙이 검증되었습니다. "
                "선택 가능한 처리: 교환 또는 환불. "
                f"정책 근거: {policy}"
            ),
            claims=[
                {
                    "value": "damage_verified",
                    "evidence_ids": ["policy-ata-1", "photo-1"],
                }
            ],
        )


def support_tools(repository: FakeOrderRepository, guard: FakeAttachmentGuard) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ToolSpec(
        name="lookup_order",
        description="Look up a customer-owned order.",
        input_schema={
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "pattern": "^\\d{5}$"},
                "customer_id": {"type": "string"},
            },
            "required": ["order_id", "customer_id"],
        },
        handler=repository.lookup_order,
    ))
    registry.register(ToolSpec(
        name="lookup_order_item",
        description="Verify an item belongs to an order.",
        input_schema={
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "pattern": "^\\d{5}$"},
                "item_id": {"type": "string"},
                "customer_id": {"type": "string"},
            },
            "required": ["order_id", "item_id", "customer_id"],
        },
        handler=repository.lookup_order_item,
    ))
    registry.register(ToolSpec(
        name="validate_attachment",
        description="Validate an image attachment used as damage proof.",
        input_schema={
            "type": "object",
            "properties": {"attachment_id": {"type": "string"}},
            "required": ["attachment_id"],
        },
        handler=guard.validate_attachment,
    ))
    return registry
