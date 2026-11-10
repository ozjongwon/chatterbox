from __future__ import annotations

from pathlib import Path

from conversational_agent.policy import PolicyRegistry, ResponseGuard


POLICIES = Path(__file__).resolve().parents[1] / "policies"


def test_forbidden_claim_requires_slot() -> None:
    registry = PolicyRegistry()
    registry.load_directory(POLICIES)
    policy = registry.get("damaged_on_arrival")
    assert policy is not None
    guard = ResponseGuard()

    slots = {
        "order_id": "12345",
        "item_id": "photo-frame",
        "photo_verified": True,
        "policy_evidence_id": "policy-1",
    }
    ok, error = guard.validate(
        policy,
        [{"value": "refund_completed"}],
        slots,
    )
    assert not ok
    assert error == "FORBIDDEN_CLAIM:refund_completed"

    ok, error = guard.validate(policy, [], slots)
    assert ok
    assert error is None


def test_policy_version_is_resolved_from_rag_id() -> None:
    registry = PolicyRegistry()
    registry.load_directory(POLICIES)
    policy = registry.get("damaged_on_arrival")
    assert policy is not None
    assert policy.version == 3
