"""Serializable execution plans and recovery decisions for workspace tools."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from contextopt.runtime.protocol import ToolCall, ToolOutcome

ReplayPolicy = Literal["safe", "reconcile", "never"]
RecoveryAction = Literal["completed", "retry", "paused", "divergence"]

PLAN_SCHEMA_VERSION = "1"
_REPLAY_POLICIES = frozenset({"safe", "reconcile", "never"})
_RECOVERY_ACTIONS = frozenset({"completed", "retry", "paused", "divergence"})
_SAFE_REPLAY_TOOLS = frozenset(
    {
        "list_files",
        "search_text",
        "read_file",
        "memory_search",
        "memory_save",
        "memory_invalidate",
        "memory_feedback",
    }
)
_RECONCILED_WRITE_TOOLS = frozenset({"create_file", "replace_text"})
_NEVER_REPLAY_TOOLS = frozenset({"run_tests"})


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def tool_call_fingerprint(call: ToolCall) -> str:
    """Return the same canonical call fingerprint used by the agent runner."""

    return hashlib.sha256(
        _canonical_json({"arguments_json": call.arguments_json, "name": call.name})
    ).hexdigest()


def tool_replay_policy(tool_name: str) -> ReplayPolicy:
    """Return the conservative cross-restart replay policy for a built-in tool."""

    if tool_name in _SAFE_REPLAY_TOOLS:
        return "safe"
    if tool_name in _RECONCILED_WRITE_TOOLS:
        return "reconcile"
    if tool_name in _NEVER_REPLAY_TOOLS:
        return "never"
    raise ValueError(f"unknown tool: {tool_name}")


def _outcome_from_dict(value: Any) -> ToolOutcome:
    if not isinstance(value, dict):
        raise ValueError("planned_outcome must be an object")
    required = {"call_id", "tool_name", "ok", "content"}
    allowed = required | {"error_code", "metadata", "truncated"}
    unknown = set(value) - allowed
    missing = required - set(value)
    if unknown or missing:
        raise ValueError(
            "planned_outcome has invalid fields: "
            f"missing={sorted(missing)!r}, unknown={sorted(unknown)!r}"
        )
    if not isinstance(value["ok"], bool):
        raise ValueError("planned_outcome.ok must be a boolean")
    if not isinstance(value.get("truncated", False), bool):
        raise ValueError("planned_outcome.truncated must be a boolean")
    metadata = value.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("planned_outcome.metadata must be an object")
    error_code = value.get("error_code")
    if error_code is not None and not isinstance(error_code, str):
        raise ValueError("planned_outcome.error_code must be a string or null")
    return ToolOutcome(
        call_id=str(value["call_id"]),
        tool_name=str(value["tool_name"]),
        ok=value["ok"],
        content=str(value["content"]),
        error_code=error_code,
        metadata=dict(metadata),
        truncated=value.get("truncated", False),
    )


def _plan_payload(plan: ToolExecutionPlan) -> dict[str, Any]:
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "operation_id": plan.operation_id,
        "call": plan.call.to_dict(),
        "fingerprint": plan.fingerprint,
        "replay_policy": plan.replay_policy,
        "preconditions": dict(plan.preconditions),
        "postconditions": dict(plan.postconditions),
        "planned_outcome": plan.planned_outcome.to_dict(),
    }


def _plan_checksum(plan: ToolExecutionPlan) -> str:
    return hashlib.sha256(_canonical_json(_plan_payload(plan))).hexdigest()


@dataclass(frozen=True, slots=True)
class ToolExecutionPlan:
    """A checksummed JSON description of one intended tool execution."""

    operation_id: str
    call: ToolCall
    fingerprint: str
    replay_policy: ReplayPolicy
    preconditions: Mapping[str, Any] = field(default_factory=dict)
    postconditions: Mapping[str, Any] = field(default_factory=dict)
    planned_outcome: ToolOutcome = field(
        default_factory=lambda: ToolOutcome("", "", True, "")
    )
    plan_sha256: str = ""

    def __post_init__(self) -> None:
        if not self.operation_id.strip():
            raise ValueError("operation_id must not be empty")
        if self.replay_policy not in _REPLAY_POLICIES:
            raise ValueError(f"unsupported replay policy: {self.replay_policy}")
        expected_fingerprint = tool_call_fingerprint(self.call)
        if not hmac.compare_digest(self.fingerprint, expected_fingerprint):
            raise ValueError("tool call fingerprint does not match the plan call")
        if not self.planned_outcome.ok or self.planned_outcome.error_code is not None:
            raise ValueError("planned_outcome must describe a successful tool outcome")
        if self.planned_outcome.call_id != self.call.id:
            raise ValueError("planned_outcome call_id does not match the plan call")
        if self.planned_outcome.tool_name != self.call.name:
            raise ValueError("planned_outcome tool_name does not match the plan call")
        object.__setattr__(self, "preconditions", dict(self.preconditions))
        object.__setattr__(self, "postconditions", dict(self.postconditions))
        _canonical_json(_plan_payload(self))
        expected_checksum = _plan_checksum(self)
        if self.plan_sha256:
            if not hmac.compare_digest(self.plan_sha256, expected_checksum):
                raise ValueError(
                    "tool execution plan checksum does not match its payload"
                )
        else:
            object.__setattr__(self, "plan_sha256", expected_checksum)

    def validate(self) -> None:
        """Recheck integrity in case a nested mapping was mutated after construction."""

        if not hmac.compare_digest(self.fingerprint, tool_call_fingerprint(self.call)):
            raise ValueError("tool call fingerprint does not match the plan call")
        if not hmac.compare_digest(self.plan_sha256, _plan_checksum(self)):
            raise ValueError("tool execution plan checksum does not match its payload")

    def to_dict(self) -> dict[str, Any]:
        return {**_plan_payload(self), "plan_sha256": self.plan_sha256}

    def to_json(self) -> str:
        return _canonical_json(self.to_dict()).decode("utf-8")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ToolExecutionPlan:
        required = {
            "schema_version",
            "operation_id",
            "call",
            "fingerprint",
            "replay_policy",
            "preconditions",
            "postconditions",
            "planned_outcome",
            "plan_sha256",
        }
        unknown = set(value) - required
        missing = required - set(value)
        if unknown or missing:
            raise ValueError(
                "tool execution plan has invalid fields: "
                f"missing={sorted(missing)!r}, unknown={sorted(unknown)!r}"
            )
        if value["schema_version"] != PLAN_SCHEMA_VERSION:
            raise ValueError("unsupported tool execution plan schema")
        call_value = value["call"]
        if not isinstance(call_value, dict):
            raise ValueError("call must be an object")
        if set(call_value) != {"id", "name", "arguments_json"}:
            raise ValueError("call has invalid fields")
        preconditions = value["preconditions"]
        postconditions = value["postconditions"]
        if not isinstance(preconditions, dict) or not isinstance(postconditions, dict):
            raise ValueError("plan conditions must be objects")
        policy = value["replay_policy"]
        if not isinstance(policy, str) or policy not in _REPLAY_POLICIES:
            raise ValueError(f"unsupported replay policy: {policy}")
        return cls(
            operation_id=str(value["operation_id"]),
            call=ToolCall(
                id=str(call_value["id"]),
                name=str(call_value["name"]),
                arguments_json=str(call_value["arguments_json"]),
            ),
            fingerprint=str(value["fingerprint"]),
            replay_policy=cast(ReplayPolicy, policy),
            preconditions=dict(preconditions),
            postconditions=dict(postconditions),
            planned_outcome=_outcome_from_dict(value["planned_outcome"]),
            plan_sha256=str(value["plan_sha256"]),
        )

    @classmethod
    def from_json(cls, value: str) -> ToolExecutionPlan:
        try:
            decoded: Any = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"tool execution plan is not valid JSON: {exc.msg}"
            ) from None
        if not isinstance(decoded, dict):
            raise ValueError("tool execution plan must be a JSON object")
        return cls.from_dict(decoded)


@dataclass(frozen=True, slots=True)
class ToolReconciliation:
    """The deterministic action to take for an interrupted tool execution."""

    action: RecoveryAction
    reason: str
    outcome: ToolOutcome | None = None

    def __post_init__(self) -> None:
        if self.action not in _RECOVERY_ACTIONS:
            raise ValueError(f"unsupported recovery action: {self.action}")
        if self.action == "completed" and self.outcome is None:
            raise ValueError("completed reconciliation requires an outcome")
        if self.action != "completed" and self.outcome is not None:
            raise ValueError("only completed reconciliation may carry an outcome")

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "outcome": None if self.outcome is None else self.outcome.to_dict(),
        }
