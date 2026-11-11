"""Startup-supplied RAG provider contract and validation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Protocol, TypeGuard


class ProviderValidationError(ValueError):
    """Raised when the configured RAG provider registry is invalid."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class RAGProvider(Protocol):
    """Common interface every startup-supplied retrieval provider implements."""

    provider_id: str
    timeout: float

    def retrieve(self, query: str, **kwargs: Any) -> list[Any]:
        ...


def is_rag_provider(value: object) -> TypeGuard[RAGProvider]:
    """Validate structural runtime requirements for the static protocol."""

    retrieve = getattr(value, "retrieve", None)
    provider_id = getattr(value, "provider_id", None)
    timeout = getattr(value, "timeout", None)
    return (
        callable(retrieve)
        and isinstance(provider_id, str)
        and bool(provider_id.strip())
        and not isinstance(timeout, bool)
        and isinstance(timeout, (int, float))
        and math.isfinite(timeout)
        and timeout > 0
    )


def validate_providers(providers: Sequence[object]) -> dict[str, RAGProvider]:
    """Validate providers and return an immutable-by-convention registry.

    Validation is intentionally strict: a partially valid registry is not
    worth serving traffic with, so startup fails with an actionable code.
    """

    if not providers:
        raise ProviderValidationError(
            'NO_RAG_PROVIDERS',
            'At least one RAG provider must be supplied at startup.',
        )

    registry: dict[str, RAGProvider] = {}
    for provider in providers:
        if not is_rag_provider(provider):
            retrieve = getattr(provider, 'retrieve', None)
            provider_id = getattr(provider, 'provider_id', None)
            if not callable(retrieve) or not isinstance(provider_id, str) or not provider_id.strip():
                raise ProviderValidationError(
                    'INVALID_PROVIDER',
                    'Each RAG provider must expose a non-empty provider_id and a callable retrieve operation.',
                )
            raise ProviderValidationError(
                'INVALID_PROVIDER_TIMEOUT',
                f"RAG provider {provider_id!r} must have a positive, finite timeout in seconds.",
            )

        provider_id = provider.provider_id
        if provider_id in registry:
            raise ProviderValidationError(
                'DUPLICATE_PROVIDER_ID',
                f'Duplicate RAG provider_id {provider_id!r}; provider IDs must be unique.',
            )
        registry[provider_id] = provider

    return registry
