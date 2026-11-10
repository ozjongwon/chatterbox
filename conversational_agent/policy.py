"""Versioned execution policies and final-response guards."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class PolicyError(ValueError):
    pass


@dataclass(frozen=True)
class ExecutionPolicy:
    policy_id: str
    version: int
    required_slots: tuple[str, ...] = ()
    forbidden_claims: dict[str, dict[str, Any]] = field(default_factory=dict)
    safe_fallback: str = "현재 요청을 처리하기 위한 확인이 더 필요합니다."

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionPolicy:
        policy_id = data.get("id")
        version = data.get("version")
        if not isinstance(policy_id, str) or not policy_id:
            raise PolicyError("policy id is required")
        if not isinstance(version, int) or version < 1:
            raise PolicyError("policy version must be >= 1")
        forbidden = data.get("forbidden_claims") or {}
        if not isinstance(forbidden, dict):
            raise PolicyError("forbidden_claims must be an object")
        return cls(
            policy_id=policy_id,
            version=version,
            required_slots=tuple(str(x) for x in data.get("required_slots", [])),
            forbidden_claims=dict(forbidden),
            safe_fallback=str(data.get("safe_fallback") or
                              "현재 요청을 처리하기 위한 확인이 더 필요합니다."),
        )


class PolicyRegistry:
    def __init__(self) -> None:
        self._versions: dict[str, list[ExecutionPolicy]] = {}
        self._active: dict[str, int] = {}

    def load_directory(self, path: Path) -> None:
        for file in sorted(path.glob("*.json")):
            policy = ExecutionPolicy.from_dict(
                json.loads(file.read_text(encoding="utf-8"))
            )
            self.register(policy)

    def register(self, policy: ExecutionPolicy) -> None:
        self._versions.setdefault(policy.policy_id, []).append(policy)
        self._versions[policy.policy_id].sort(key=lambda item: item.version)
        self._active[policy.policy_id] = max(
            self._active.get(policy.policy_id, 0), policy.version
        )

    def get(self, policy_id: str, version: int | None = None) -> ExecutionPolicy | None:
        versions = self._versions.get(policy_id) or []
        target = version if version is not None else self._active.get(policy_id)
        return next((item for item in versions if item.version == target), None)


class ResponseGuard:
    """Rejects unsupported final claims before text is returned."""

    def validate(
        self,
        policy: ExecutionPolicy | None,
        claims: list[dict[str, Any]],
        slots: dict[str, Any],
    ) -> tuple[bool, str | None]:
        if policy is None or not claims:
            return True, None
        for requirement in policy.required_slots:
            if not slots.get(requirement):
                return False, f"MISSING_REQUIRED_SLOT:{requirement}"
        for claim in claims:
            value = str(claim.get("value") or "")
            rule = policy.forbidden_claims.get(value)
            if not rule:
                continue
            required_slot = rule.get("requires_slot")
            if required_slot and not slots.get(required_slot):
                return False, f"FORBIDDEN_CLAIM:{value}"
            required_evidence = set(rule.get("requires_evidence") or [])
            evidence = {str(x) for x in slots.get("evidence_ids", [])}
            if required_evidence and not required_evidence.issubset(evidence):
                return False, f"FORBIDDEN_CLAIM:{value}"
        return True, None
