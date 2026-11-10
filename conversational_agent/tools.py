"""Typed business-tool contracts and validation middleware."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any


class ToolValidationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def validate_arguments(schema: Mapping[str, Any], args: Mapping[str, Any]) -> None:
    if not isinstance(args, Mapping):
        raise ToolValidationError("INVALID_ARGUMENTS", "Tool arguments must be an object.")
    required = schema.get("required") or []
    for name in required:
        if name not in args:
            raise ToolValidationError(
                "MISSING_ARGUMENT", f"Missing required argument: {name}"
            )
    properties = schema.get("properties") or {}
    types: dict[str, type | tuple[type, ...]] = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    for name, value in args.items():
        spec = properties.get(name) or {}
        expected = spec.get("type")
        if expected and expected in types and not isinstance(value, types[expected]):
            raise ToolValidationError(
                "INVALID_ARGUMENT_TYPE",
                f"Argument {name!r} must be {expected}.",
            )
        pattern = spec.get("pattern")
        if pattern and not re.fullmatch(str(pattern), str(value)):
            raise ToolValidationError(
                "INVALID_ARGUMENT_FORMAT",
                f"Argument {name!r} does not match the required format.",
            )


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Any]
    preconditions: tuple[Callable[[dict[str, Any], dict[str, Any]], Any], ...] = field(
        default_factory=tuple
    )
    postconditions: tuple[Callable[[dict[str, Any], dict[str, Any]], Any], ...] = field(
        default_factory=tuple
    )

    def invoke(self, args: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        try:
            validate_arguments(self.input_schema, args)
            for check in self.preconditions:
                outcome = check(args, context)
                if outcome is False:
                    raise ToolValidationError(
                        "PRECONDITION_FAILED",
                        f"Precondition failed for tool {self.name!r}.",
                    )
                if isinstance(outcome, str):
                    raise ToolValidationError("PRECONDITION_FAILED", outcome)
            result = self.handler(**args, **context)
            if not isinstance(result, dict):
                result = {"ok": True, "data": result}
            if result.get("ok"):
                for check in self.postconditions:
                    outcome = check(result, context)
                    if outcome is False:
                        raise ToolValidationError(
                            "POSTCONDITION_FAILED",
                            f"Postcondition failed for tool {self.name!r}.",
                        )
                    if isinstance(outcome, str):
                        raise ToolValidationError("POSTCONDITION_FAILED", outcome)
            return result
        except ToolValidationError as exc:
            return {"ok": False, "error_code": exc.code, "message": str(exc)}


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if not spec.name or spec.name in self._tools:
            raise ValueError(f"duplicate or empty tool name: {spec.name!r}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def names(self) -> set[str]:
        return set(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools
