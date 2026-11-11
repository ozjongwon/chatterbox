"""Deterministic fakes for local text-chat development."""

from __future__ import annotations

from typing import Any

from conversational_agent.core.models import LLMAction, LLMActionType


class FakeRAGProvider:
    """Small in-memory retrieval provider selected by provider_id."""

    def __init__(
        self,
        provider_id: str = "fake",
        documents: list[dict[str, Any]] | None = None,
        timeout: float = 1.0,
        top_k: int = 2,
    ) -> None:
        self.provider_id = provider_id
        self.timeout = timeout
        self._documents = list(documents or [
            {
                "evidence_id": "doc-1",
                "text": "The fake knowledge base says refunds are available within 30 days.",
            },
            {
                "evidence_id": "doc-2",
                "text": "Voice and text frontends share the same conversation engine.",
            },
        ])
        self._top_k = max(0, top_k)
        self.calls: list[str] = []

    def retrieve(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(query)
        terms = {term.lower() for term in query.split() if term.strip()}
        ranked = sorted(
            self._documents,
            key=lambda doc: sum(
                term in str(doc.get("text", "")).lower()
                for term in terms
            ),
            reverse=True,
        )
        return [dict(doc) for doc in ranked[:self._top_k]]


class FakeTextLLM:
    """Text/voice-shared LLM fake that retrieves once, then answers."""

    def __init__(self, default_provider_id: str = "fake") -> None:
        self.default_provider_id = default_provider_id
        self.calls: list[dict[str, Any]] = []

    def next_action(self, **kwargs: Any) -> LLMActionType:
        self.calls.append(dict(kwargs))
        user_text = str(kwargs.get("user_text") or "")
        evidence = kwargs.get("evidence") or []
        metadata = kwargs.get("metadata") or {}
        provider_id = str(metadata.get("provider_id") or self.default_provider_id)

        if not evidence:
            return LLMAction(
                action="retrieve",
                provider_id=provider_id,
                query=user_text,
            )

        items: list[dict[str, Any]] = []
        for group in evidence:
            items.extend(group.get("items") or [])
        relevant = "; ".join(
            str(item.get("text"))
            for item in items
            if item.get("text")
        )
        return LLMAction(
            action="answer",
            text=relevant or "No matching fake evidence was found.",
        )
