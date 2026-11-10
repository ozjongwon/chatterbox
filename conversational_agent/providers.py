"""Startup-supplied RAG provider contract and validation."""

from __future__ import annotations

import math
from typing import Any, Protocol, runtime_checkable


class ProviderValidationError(ValueError):
    """Raised when the configured RAG provider registry is invalid."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@runtime_checkable
class RAGProvider(Protocol):
    """Common interface every startup-supplied retrieval provider implements."""

    provider_id: str
    timeout: float

    def retrieve(self, query: str, **kwargs: Any) -> list[Any]:
        ...


def validate_providers(providers: list[Any] | tuple[Any, ...]) -> dict[str, Any]:
    """Validate providers and return an immutable-by-convention registry.

    Validation is intentionally strict: a partially valid registry is not
    worth serving traffic with, so startup fails with an actionable code.
    """

    if not providers:
        raise ProviderValidationError(
            'NO_RAG_PROVIDERS',
            'At least one RAG provider must be supplied at startup.',
        )

    registry: dict[str, Any] = {}
    for provider in providers:
        retrieve = getattr(provider, 'retrieve', None)
        provider_id = getattr(provider, 'provider_id', None)
        if not callable(retrieve) or not isinstance(provider_id, str) or not provider_id.strip():
            raise ProviderValidationError(
                'INVALID_PROVIDER',
                'Each RAG provider must expose a non-empty provider_id and a callable retrieve operation.',
            )

        timeout = getattr(provider, 'timeout', None)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ProviderValidationError(
                'INVALID_PROVIDER_TIMEOUT',
                f"RAG provider {provider_id!r} must have a positive, finite timeout in seconds.",
            )

        if provider_id in registry:
            raise ProviderValidationError(
                'DUPLICATE_PROVIDER_ID',
                f'Duplicate RAG provider_id {provider_id!r}; provider IDs must be unique.',
            )
        registry[provider_id] = provider

    return registry
