"""Declarative workflow loading, selection, and conversation pinning."""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class WorkflowError(ValueError):
    pass


@dataclass(frozen=True)
class WorkflowDefinition:
    workflow_id: str
    version: int
    entry_intents: tuple[str, ...]
    entry_keywords: tuple[str, ...]
    slots: dict[str, dict[str, Any]]
    states: dict[str, dict[str, Any]]
    subflows: tuple[str, ...] = ()

    @property
    def initial_state(self) -> str:
        if not self.states:
            raise WorkflowError(f"workflow {self.workflow_id!r} has no states")
        return next(iter(self.states))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowDefinition:
        workflow_id = data.get("id")
        version = data.get("version")
        states = data.get("states")
        if not isinstance(workflow_id, str) or not workflow_id.strip():
            raise WorkflowError("workflow id must be a non-empty string")
        if not isinstance(version, int) or version < 1:
            raise WorkflowError(f"workflow {workflow_id!r} version must be >= 1")
        if not isinstance(states, dict) or not states:
            raise WorkflowError(f"workflow {workflow_id!r} needs states")
        return cls(
            workflow_id=workflow_id,
            version=version,
            entry_intents=tuple(str(x) for x in data.get("entry_intents", [])),
            entry_keywords=tuple(str(x).lower() for x in data.get("entry_keywords", [])),
            slots=dict(data.get("slots") or {}),
            states=dict(states),
            subflows=tuple(str(x) for x in data.get("subflows", [])),
        )


class WorkflowRegistry:
    """Holds versioned workflow definitions; selection uses the latest version."""

    def __init__(self, known_tools: set[str] | None = None) -> None:
        self._known_tools = known_tools
        self._workflows: dict[str, list[WorkflowDefinition]] = {}
        self._active: dict[str, int] = {}
        self._lock = threading.RLock()

    def register(self, workflow: WorkflowDefinition) -> None:
        self._validate(workflow)
        with self._lock:
            self._workflows.setdefault(workflow.workflow_id, []).append(workflow)
            self._workflows[workflow.workflow_id].sort(key=lambda item: item.version)
            self._active[workflow.workflow_id] = max(
                self._active.get(workflow.workflow_id, 0), workflow.version
            )

    def load_directory(self, path: Path) -> None:
        for file in sorted(path.glob("*.json")):
            data = json.loads(file.read_text(encoding="utf-8"))
            self.register(WorkflowDefinition.from_dict(data))

    def reload(self, path: Path, activate: bool = False) -> None:
        candidate = WorkflowRegistry(self._known_tools)
        candidate.load_directory(path)
        with self._lock:
            for workflow_id, versions in candidate._workflows.items():
                merged = {
                    item.version: item
                    for item in self._workflows.get(workflow_id, [])
                }
                merged.update({item.version: item for item in versions})
                self._workflows[workflow_id] = [
                    merged[version] for version in sorted(merged)
                ]
            if activate:
                self._active.update(candidate._active)
            else:
                for workflow_id in candidate._active:
                    self._active.setdefault(
                        workflow_id, candidate._active[workflow_id]
                    )

    def activate(self, workflow_id: str, version: int) -> None:
        with self._lock:
            if self.get(workflow_id, version) is None:
                raise WorkflowError(
                    f"cannot activate unknown workflow {workflow_id!r} v{version}"
                )
            self._active[workflow_id] = version

    def rollback(self, workflow_id: str) -> int:
        with self._lock:
            versions = [item.version for item in self._workflows.get(workflow_id, [])]
            current = self._active.get(workflow_id)
            previous = [version for version in versions if current is None or version < current]
            if not previous:
                raise WorkflowError(f"no previous version for workflow {workflow_id!r}")
            self._active[workflow_id] = previous[-1]
            return previous[-1]

    def get(self, workflow_id: str, version: int | None = None) -> WorkflowDefinition | None:
        candidates = self._workflows.get(workflow_id) or []
        if version is not None:
            return next((item for item in candidates if item.version == version), None)
        active = self._active.get(workflow_id)
        return next((item for item in candidates if item.version == active), None)

    def all(self) -> list[WorkflowDefinition]:
        workflows: list[WorkflowDefinition] = []
        for workflow_id in self._workflows:
            workflow = self.get(workflow_id)
            if workflow is not None:
                workflows.append(workflow)
        return workflows

    def _validate(self, workflow: WorkflowDefinition) -> None:
        for name, state in workflow.states.items():
            if not isinstance(state, dict):
                raise WorkflowError(f"{workflow.workflow_id}.{name} must be an object")
            for tool in state.get("allowed_tools", []):
                if self._known_tools is not None and tool not in self._known_tools:
                    raise WorkflowError(
                        f"{workflow.workflow_id}.{name}: unknown tool {tool!r}"
                    )
            for transition in state.get("transitions", []):
                target = transition.get("to")
                if target not in workflow.states:
                    raise WorkflowError(
                        f"{workflow.workflow_id}.{name}: unknown target {target!r}"
                    )


class WorkflowSelector:
    """Selects a workflow from explicit metadata or first-message keywords."""

    def __init__(self, registry: WorkflowRegistry) -> None:
        self.registry = registry

    def select(
        self,
        metadata: dict[str, Any],
        user_text: str,
    ) -> WorkflowDefinition | None:
        explicit = metadata.get("workflow_id")
        if explicit:
            return self.registry.get(str(explicit))
        text = user_text.lower()
        scored: list[tuple[int, WorkflowDefinition]] = []
        for workflow in self.registry.all():
            score = sum(1 for keyword in workflow.entry_keywords if keyword in text)
            if score:
                scored.append((score, workflow))
        if not scored:
            return None
        return max(scored, key=lambda item: (item[0], item[1].version))[1]

    @staticmethod
    def pin(conversation: Any, workflow: WorkflowDefinition) -> None:
        conversation.metadata.update({
            "workflow_id": workflow.workflow_id,
            "workflow_version": workflow.version,
            "workflow_state": workflow.initial_state,
            "workflow_slots": {},
        })

    @staticmethod
    def workflow_context(conversation: Any) -> dict[str, Any]:
        metadata = conversation.metadata
        workflow_id = metadata.get("workflow_id")
        version = metadata.get("workflow_version")
        return {
            "workflow_id": workflow_id,
            "workflow_version": version,
            "workflow_state": metadata.get("workflow_state"),
            "workflow_slots": dict(metadata.get("workflow_slots") or {}),
        }

    @staticmethod
    def handoff(
        conversation: Any,
        registry: WorkflowRegistry,
        workflow_id: str,
    ) -> WorkflowDefinition:
        target = registry.get(workflow_id)
        if target is None:
            raise WorkflowError(f"unknown handoff workflow {workflow_id!r}")
        previous = WorkflowSelector.workflow_context(conversation)
        if previous.get("workflow_id"):
            conversation.metadata.setdefault("workflow_history", []).append(previous)
        WorkflowSelector.pin(conversation, target)
        conversation.metadata["workflow_handoff_from"] = previous.get("workflow_id")
        return target


def validate_workflow_slots(workflow: WorkflowDefinition, slots: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for name, spec in workflow.slots.items():
        if name not in slots or slots[name] is None:
            missing.append(name)
            continue
        pattern = spec.get("pattern")
        if pattern and not re.fullmatch(pattern, str(slots[name])):
            missing.append(name)
    return missing
