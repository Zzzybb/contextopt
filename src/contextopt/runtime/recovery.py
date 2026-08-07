"""Pure event reduction and validated JSON projection checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

from contextopt.runtime.protocol import (
    AgentMessage,
    ModelResponse,
    RunLimits,
    RunPermissions,
    RunStatus,
    TokenUsage,
    ToolCall,
    ToolOutcome,
)
from contextopt.runtime.tool_state import ToolExecutionPlan

ProjectionPhase = Literal["running", "paused", "interrupted", "terminal"]
CHECKPOINT_SCHEMA_VERSION = "1"
EVENT_SCHEMA_VERSION = "2"
GENESIS_EVENT_SHA256 = "0" * 64


class RecoveryError(ValueError):
    """An event stream violates the recoverable runtime contract."""


def canonical_sha256(value: Any) -> str:
    """Return a stable SHA-256 for a JSON-compatible value."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RecoveryError(f"value is not canonical JSON: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def tool_call_fingerprint(call: ToolCall) -> str:
    """Fingerprint exactly the logical input whose result may be reused."""

    return canonical_sha256({"name": call.name, "arguments_json": call.arguments_json})


def event_sha256(event: Mapping[str, Any]) -> str:
    """Hash an event, validating an optional embedded event hash."""

    value = dict(event)
    declared = value.pop("event_sha256", None)
    digest = canonical_sha256(value)
    if declared is not None:
        _sha256(declared, "event event_sha256")
        if declared != digest:
            raise RecoveryError("event_sha256 does not match event content")
    return digest


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise RecoveryError(f"{label} must be an object")
    return value


def _array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise RecoveryError(f"{label} must be an array")
    return value


def _fields(
    value: Mapping[str, Any],
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
    label: str,
) -> Mapping[str, Any]:
    keys = set(value)
    missing = required - keys
    unknown = keys - required - optional
    if missing:
        raise RecoveryError(f"{label} is missing fields: {sorted(missing)!r}")
    if unknown:
        raise RecoveryError(f"{label} has unknown fields: {sorted(unknown)!r}")
    return value


def _string(value: Any, label: str, *, allow_empty: bool = True) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "a non-empty string" if not allow_empty else "a string"
        raise RecoveryError(f"{label} must be {qualifier}")
    return value


def _optional_string(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _string(value, label)


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise RecoveryError(f"{label} must be an integer >= {minimum}")
    return value


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise RecoveryError(f"{label} must be a boolean")
    return value


def _sha256(value: Any, label: str) -> str:
    digest = _string(value, label, allow_empty=False)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise RecoveryError(f"{label} must be a lowercase SHA-256")
    return digest


def _protocol(callable_: Any, data: Mapping[str, Any], label: str) -> Any:
    try:
        return callable_(data)
    except ValueError as exc:
        raise RecoveryError(f"invalid {label}: {exc}") from exc


@dataclass(frozen=True, slots=True)
class RunConfigSnapshot:
    task: str
    initial_messages: tuple[AgentMessage, ...]
    model_fingerprint: str
    tool_fingerprint: str
    limits: RunLimits
    permissions: RunPermissions
    config_sha256: str

    @classmethod
    def create(
        cls,
        *,
        task: str,
        initial_messages: Sequence[AgentMessage],
        model_fingerprint: str,
        tool_fingerprint: str,
        limits: RunLimits,
        permissions: RunPermissions,
    ) -> RunConfigSnapshot:
        base = {
            "task": task,
            "initial_messages": [message.to_dict() for message in initial_messages],
            "model_fingerprint": model_fingerprint,
            "tool_fingerprint": tool_fingerprint,
            "limits": limits.to_dict(),
            "permissions": permissions.to_dict(),
        }
        return cls.from_dict({**base, "config_sha256": canonical_sha256(base)})

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunConfigSnapshot:
        value = _fields(
            _object(data, "run config"),
            required=frozenset(
                {
                    "task",
                    "initial_messages",
                    "model_fingerprint",
                    "tool_fingerprint",
                    "limits",
                    "permissions",
                    "config_sha256",
                }
            ),
            label="run config",
        )
        messages = tuple(
            _protocol(
                AgentMessage.from_dict,
                _object(item, f"initial_messages[{index}]"),
                f"initial_messages[{index}]",
            )
            for index, item in enumerate(
                _array(value["initial_messages"], "initial_messages")
            )
        )
        if not messages:
            raise RecoveryError("initial_messages must not be empty")
        limits = _protocol(
            RunLimits.from_dict,
            _object(value["limits"], "run config limits"),
            "run config limits",
        )
        permissions = _protocol(
            RunPermissions.from_dict,
            _object(value["permissions"], "run config permissions"),
            "run config permissions",
        )
        snapshot = cls(
            task=_string(value["task"], "run config task", allow_empty=False),
            initial_messages=messages,
            model_fingerprint=_sha256(
                value["model_fingerprint"], "run config model_fingerprint"
            ),
            tool_fingerprint=_sha256(
                value["tool_fingerprint"], "run config tool_fingerprint"
            ),
            limits=cast(RunLimits, limits),
            permissions=cast(RunPermissions, permissions),
            config_sha256=_sha256(value["config_sha256"], "run config config_sha256"),
        )
        normalized = snapshot.to_dict()
        normalized.pop("config_sha256")
        if canonical_sha256(normalized) != snapshot.config_sha256:
            raise RecoveryError("config_sha256 does not match normalized run config")
        return snapshot

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "initial_messages": [
                message.to_dict() for message in self.initial_messages
            ],
            "model_fingerprint": self.model_fingerprint,
            "tool_fingerprint": self.tool_fingerprint,
            "limits": self.limits.to_dict(),
            "permissions": self.permissions.to_dict(),
            "config_sha256": self.config_sha256,
        }


@dataclass(frozen=True, slots=True)
class PendingModelCall:
    turn: int
    request_sha256: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PendingModelCall:
        value = _fields(
            _object(data, "pending model call"),
            required=frozenset({"turn", "request_sha256"}),
            label="pending model call",
        )
        return cls(
            turn=_integer(value["turn"], "pending model turn", minimum=1),
            request_sha256=_sha256(
                value["request_sha256"], "pending model request_sha256"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"turn": self.turn, "request_sha256": self.request_sha256}


@dataclass(frozen=True, slots=True)
class PendingToolCall:
    call: ToolCall
    turn: int
    call_index: int
    fingerprint: str
    operation_id: str | None = None
    plan: ToolExecutionPlan | None = None
    started: bool = False

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PendingToolCall:
        value = _fields(
            _object(data, "pending tool call"),
            required=frozenset(
                {
                    "call",
                    "turn",
                    "call_index",
                    "fingerprint",
                    "operation_id",
                    "plan",
                    "started",
                }
            ),
            label="pending tool call",
        )
        call = cast(
            ToolCall,
            _protocol(
                ToolCall.from_dict,
                _object(value["call"], "pending tool call payload"),
                "pending tool call payload",
            ),
        )
        raw_plan = value["plan"]
        try:
            plan = (
                None
                if raw_plan is None
                else ToolExecutionPlan.from_dict(
                    _object(raw_plan, "pending tool execution plan")
                )
            )
        except ValueError as exc:
            raise RecoveryError(f"invalid pending tool execution plan: {exc}") from exc
        pending = cls(
            call=call,
            turn=_integer(value["turn"], "pending tool turn", minimum=1),
            call_index=_integer(value["call_index"], "pending tool call_index"),
            fingerprint=_sha256(value["fingerprint"], "pending tool fingerprint"),
            operation_id=_optional_string(
                value["operation_id"], "pending tool operation_id"
            ),
            plan=plan,
            started=_boolean(value["started"], "pending tool started"),
        )
        if pending.fingerprint != tool_call_fingerprint(call):
            raise RecoveryError("pending tool fingerprint does not match call")
        if pending.started != (pending.operation_id is not None):
            raise RecoveryError("started pending tool requires an operation_id")
        if plan is not None and (
            plan.operation_id != pending.operation_id
            or plan.call != pending.call
            or plan.fingerprint != pending.fingerprint
        ):
            raise RecoveryError("pending execution plan does not match pending call")
        return pending

    def to_dict(self) -> dict[str, Any]:
        return {
            "call": self.call.to_dict(),
            "turn": self.turn,
            "call_index": self.call_index,
            "fingerprint": self.fingerprint,
            "operation_id": self.operation_id,
            "plan": None if self.plan is None else self.plan.to_dict(),
            "started": self.started,
        }


@dataclass(frozen=True, slots=True)
class CompletedToolCall:
    call: ToolCall
    operation_id: str
    turn: int
    call_index: int
    fingerprint: str
    outcome: ToolOutcome

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CompletedToolCall:
        value = _fields(
            _object(data, "completed tool call"),
            required=frozenset(
                {
                    "call",
                    "operation_id",
                    "turn",
                    "call_index",
                    "fingerprint",
                    "outcome",
                }
            ),
            label="completed tool call",
        )
        call = cast(
            ToolCall,
            _protocol(
                ToolCall.from_dict,
                _object(value["call"], "completed tool call payload"),
                "completed tool call payload",
            ),
        )
        outcome = cast(
            ToolOutcome,
            _protocol(
                ToolOutcome.from_dict,
                _object(value["outcome"], "completed tool outcome"),
                "completed tool outcome",
            ),
        )
        completed = cls(
            call=call,
            operation_id=_string(
                value["operation_id"],
                "completed tool operation_id",
                allow_empty=False,
            ),
            turn=_integer(value["turn"], "completed tool turn", minimum=1),
            call_index=_integer(value["call_index"], "completed tool call_index"),
            fingerprint=_sha256(value["fingerprint"], "completed tool fingerprint"),
            outcome=outcome,
        )
        if completed.fingerprint != tool_call_fingerprint(call):
            raise RecoveryError("completed tool fingerprint does not match call")
        if outcome.call_id != call.id or outcome.tool_name != call.name:
            raise RecoveryError("completed tool outcome does not match call")
        return completed

    def to_dict(self) -> dict[str, Any]:
        return {
            "call": self.call.to_dict(),
            "operation_id": self.operation_id,
            "turn": self.turn,
            "call_index": self.call_index,
            "fingerprint": self.fingerprint,
            "outcome": self.outcome.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class TerminalState:
    status: RunStatus
    reason: str
    final_text: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TerminalState:
        value = _fields(
            _object(data, "terminal state"),
            required=frozenset({"status", "reason", "final_text"}),
            label="terminal state",
        )
        status = _string(value["status"], "terminal status", allow_empty=False)
        if status not in {"completed", "stopped", "failed", "cancelled"}:
            raise RecoveryError(f"invalid terminal status: {status!r}")
        return cls(
            status=cast(RunStatus, status),
            reason=_string(value["reason"], "terminal reason"),
            final_text=_string(value["final_text"], "terminal final_text"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "final_text": self.final_text,
        }


@dataclass(frozen=True, slots=True)
class RunProjection:
    run_id: str
    config: RunConfigSnapshot
    messages: tuple[AgentMessage, ...]
    usage: TokenUsage
    turn: int
    tool_calls: int
    completed_calls: Mapping[str, CompletedToolCall]
    pending_model: PendingModelCall | None
    pending_tools: tuple[PendingToolCall, ...]
    phase: ProjectionPhase
    awaiting_terminal: bool
    terminal: TerminalState | None
    through_seq: int
    through_event_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "completed_calls",
            MappingProxyType(dict(sorted(self.completed_calls.items()))),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunProjection:
        value = _fields(
            _object(data, "run projection"),
            required=frozenset(
                {
                    "run_id",
                    "config",
                    "messages",
                    "usage",
                    "turn",
                    "tool_calls",
                    "completed_calls",
                    "pending_model",
                    "pending_tools",
                    "phase",
                    "awaiting_terminal",
                    "terminal",
                    "through_seq",
                    "through_event_sha256",
                }
            ),
            label="run projection",
        )
        config = RunConfigSnapshot.from_dict(
            _object(value["config"], "projection config")
        )
        messages = tuple(
            cast(
                AgentMessage,
                _protocol(
                    AgentMessage.from_dict,
                    _object(item, f"projection messages[{index}]"),
                    f"projection messages[{index}]",
                ),
            )
            for index, item in enumerate(
                _array(value["messages"], "projection messages")
            )
        )
        raw_completed = _object(value["completed_calls"], "projection completed_calls")
        completed = {
            key: CompletedToolCall.from_dict(
                _object(item, f"projection completed_calls[{key!r}]")
            )
            for key, item in raw_completed.items()
            if isinstance(key, str)
        }
        if len(completed) != len(raw_completed):
            raise RecoveryError("completed call cache keys must be strings")
        pending_value = value["pending_model"]
        pending_model = (
            None
            if pending_value is None
            else PendingModelCall.from_dict(
                _object(pending_value, "projection pending_model")
            )
        )
        pending_tools = tuple(
            PendingToolCall.from_dict(
                _object(item, f"projection pending_tools[{index}]")
            )
            for index, item in enumerate(
                _array(value["pending_tools"], "projection pending_tools")
            )
        )
        terminal_value = value["terminal"]
        terminal = (
            None
            if terminal_value is None
            else TerminalState.from_dict(_object(terminal_value, "projection terminal"))
        )
        phase = _string(value["phase"], "projection phase", allow_empty=False)
        if phase not in {"running", "paused", "interrupted", "terminal"}:
            raise RecoveryError(f"invalid projection phase: {phase!r}")
        projection = cls(
            run_id=_string(value["run_id"], "projection run_id", allow_empty=False),
            config=config,
            messages=messages,
            usage=cast(
                TokenUsage,
                _protocol(
                    TokenUsage.from_dict,
                    _object(value["usage"], "projection usage"),
                    "projection usage",
                ),
            ),
            turn=_integer(value["turn"], "projection turn"),
            tool_calls=_integer(value["tool_calls"], "projection tool_calls"),
            completed_calls=completed,
            pending_model=pending_model,
            pending_tools=pending_tools,
            phase=cast(ProjectionPhase, phase),
            awaiting_terminal=_boolean(
                value["awaiting_terminal"], "projection awaiting_terminal"
            ),
            terminal=terminal,
            through_seq=_integer(value["through_seq"], "projection through_seq"),
            through_event_sha256=_sha256(
                value["through_event_sha256"],
                "projection through_event_sha256",
            ),
        )
        projection._validate()
        return projection

    def _validate(self) -> None:
        if (
            self.messages[: len(self.config.initial_messages)]
            != self.config.initial_messages
        ):
            raise RecoveryError("projection messages lost the initial message prefix")
        if self.phase == "terminal" and self.terminal is None:
            raise RecoveryError("terminal projection requires terminal state")
        if self.phase != "terminal" and self.terminal is not None:
            raise RecoveryError("non-terminal projection cannot have terminal state")
        if self.pending_model is not None:
            if self.pending_model.turn != self.turn + 1:
                raise RecoveryError("pending model turn must be the next turn")
            if self.pending_tools or self.awaiting_terminal:
                raise RecoveryError("pending model conflicts with pending tool state")
        if self.awaiting_terminal and self.pending_tools:
            raise RecoveryError("terminal wait cannot contain pending tools")
        positions = {
            (pending.turn, pending.call_index) for pending in self.pending_tools
        }
        if len(positions) != len(self.pending_tools):
            raise RecoveryError("pending tool positions must be unique")
        for pending in self.pending_tools:
            has_execution_identity = pending.operation_id is not None
            if pending.started != has_execution_identity:
                raise RecoveryError("started pending tool requires an operation id")
            if pending.plan is not None:
                try:
                    pending.plan.validate()
                except ValueError as exc:
                    raise RecoveryError(
                        f"pending tool plan failed validation: {exc}"
                    ) from exc
                if (
                    pending.plan.operation_id != pending.operation_id
                    or pending.plan.call != pending.call
                    or pending.plan.fingerprint != pending.fingerprint
                ):
                    raise RecoveryError("pending tool plan does not match pending call")
        operations = [
            pending.operation_id
            for pending in self.pending_tools
            if pending.operation_id is not None
        ]
        operations.extend(item.operation_id for item in self.completed_calls.values())
        if len(set(operations)) != len(operations):
            raise RecoveryError("tool operation ids must be unique")
        for call_id, completed in self.completed_calls.items():
            if call_id != completed.call.id:
                raise RecoveryError("completed call cache key does not match call id")
            if completed.fingerprint != tool_call_fingerprint(completed.call):
                raise RecoveryError("completed call fingerprint does not match call")
            if (
                completed.outcome.call_id != completed.call.id
                or completed.outcome.tool_name != completed.call.name
            ):
                raise RecoveryError("completed outcome does not match call")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config": self.config.to_dict(),
            "messages": [message.to_dict() for message in self.messages],
            "usage": self.usage.to_dict(),
            "turn": self.turn,
            "tool_calls": self.tool_calls,
            "completed_calls": {
                call_id: completed.to_dict()
                for call_id, completed in sorted(self.completed_calls.items())
            },
            "pending_model": (
                None if self.pending_model is None else self.pending_model.to_dict()
            ),
            "pending_tools": [pending.to_dict() for pending in self.pending_tools],
            "phase": self.phase,
            "awaiting_terminal": self.awaiting_terminal,
            "terminal": None if self.terminal is None else self.terminal.to_dict(),
            "through_seq": self.through_seq,
            "through_event_sha256": self.through_event_sha256,
        }


@dataclass(frozen=True, slots=True)
class ProjectionCheckpoint:
    through_seq: int
    through_event_sha256: str
    state_sha256: str
    state: RunProjection

    @classmethod
    def create(cls, state: RunProjection) -> ProjectionCheckpoint:
        state._validate()
        state_data = state.to_dict()
        return cls(
            through_seq=state.through_seq,
            through_event_sha256=state.through_event_sha256,
            state_sha256=canonical_sha256(state_data),
            state=state,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProjectionCheckpoint:
        value = _fields(
            _object(data, "projection checkpoint"),
            required=frozenset(
                {
                    "checkpoint_schema_version",
                    "through_seq",
                    "through_event_sha256",
                    "state_sha256",
                    "state",
                }
            ),
            label="projection checkpoint",
        )
        if value["checkpoint_schema_version"] != CHECKPOINT_SCHEMA_VERSION:
            raise RecoveryError("unsupported checkpoint schema version")
        state_data = _object(value["state"], "checkpoint state")
        state_sha = _sha256(value["state_sha256"], "checkpoint state_sha256")
        if canonical_sha256(state_data) != state_sha:
            raise RecoveryError("checkpoint state_sha256 does not match state")
        state = RunProjection.from_dict(state_data)
        checkpoint = cls(
            through_seq=_integer(value["through_seq"], "checkpoint through_seq"),
            through_event_sha256=_sha256(
                value["through_event_sha256"],
                "checkpoint through_event_sha256",
            ),
            state_sha256=state_sha,
            state=state,
        )
        if checkpoint.through_seq != state.through_seq:
            raise RecoveryError("checkpoint sequence does not match state")
        if checkpoint.through_event_sha256 != state.through_event_sha256:
            raise RecoveryError("checkpoint event hash does not match state")
        return checkpoint

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "through_seq": self.through_seq,
            "through_event_sha256": self.through_event_sha256,
            "state_sha256": self.state_sha256,
            "state": self.state.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class _Event:
    run_id: str
    seq: int
    type: str
    data: Mapping[str, Any]
    sha256: str
    previous_sha256: str | None


def _parse_event(raw: Mapping[str, Any]) -> _Event:
    value = _fields(
        _object(raw, "run event"),
        required=frozenset({"schema_version", "run_id", "seq", "type", "data"}),
        optional=frozenset({"timestamp", "event_sha256", "prev_event_sha256"}),
        label="run event",
    )
    if value["schema_version"] != EVENT_SCHEMA_VERSION:
        raise RecoveryError("recovery requires schema version 2 events")
    if "timestamp" in value:
        _string(value["timestamp"], "event timestamp", allow_empty=False)
    previous = value.get("prev_event_sha256")
    return _Event(
        run_id=_string(value["run_id"], "event run_id", allow_empty=False),
        seq=_integer(value["seq"], "event seq"),
        type=_string(value["type"], "event type", allow_empty=False),
        data=_object(value["data"], "event data"),
        sha256=event_sha256(value),
        previous_sha256=(
            None if previous is None else _sha256(previous, "event prev_event_sha256")
        ),
    )


def _advance(
    state: RunProjection,
    event: _Event,
    **changes: Any,
) -> RunProjection:
    advanced = replace(
        state,
        through_seq=event.seq,
        through_event_sha256=event.sha256,
        **changes,
    )
    advanced._validate()
    return advanced


def _pending_at(
    state: RunProjection, turn: int, call_index: int
) -> tuple[int, PendingToolCall]:
    if not state.pending_tools:
        raise RecoveryError("tool event has no pending call")
    pending = state.pending_tools[0]
    if pending.turn != turn or pending.call_index != call_index:
        raise RecoveryError(
            "tool events must resolve pending calls in model-declared order"
        )
    return 0, pending


def _tool_identity(data: Mapping[str, Any], label: str) -> tuple[str, int, int, str]:
    operation_id = _string(
        data["operation_id"], f"{label} operation_id", allow_empty=False
    )
    turn = _integer(data["turn"], f"{label} turn", minimum=1)
    call_index = _integer(data["call_index"], f"{label} call_index")
    fingerprint = _sha256(data["fingerprint"], f"{label} fingerprint")
    return operation_id, turn, call_index, fingerprint


def _assert_tool_event_matches(
    data: Mapping[str, Any], pending: PendingToolCall, fingerprint: str
) -> None:
    if pending.fingerprint != fingerprint:
        raise RecoveryError("tool event fingerprint does not match pending call")
    if _string(data["call_id"], "tool event call_id") != pending.call.id:
        raise RecoveryError("tool event call_id does not match pending call")
    if _string(data["tool_name"], "tool event tool_name") != pending.call.name:
        raise RecoveryError("tool event tool_name does not match pending call")


def _outcome_from_event(data: Mapping[str, Any]) -> ToolOutcome:
    outcome_data = {
        name: data[name]
        for name in (
            "call_id",
            "tool_name",
            "ok",
            "content",
            "error_code",
            "metadata",
            "truncated",
        )
    }
    return cast(
        ToolOutcome,
        _protocol(ToolOutcome.from_dict, outcome_data, "tool outcome"),
    )


def _tool_message(outcome: ToolOutcome) -> AgentMessage:
    return AgentMessage(
        role="tool",
        content=json.dumps(
            outcome.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        tool_call_id=outcome.call_id,
        tool_name=outcome.tool_name,
    )


def reduce_event(
    state: RunProjection | None, raw_event: Mapping[str, Any]
) -> RunProjection:
    """Apply one schema-v2 event without performing I/O or external side effects."""

    event = _parse_event(raw_event)
    if state is None:
        if event.seq != 0 or event.type != "run.started":
            raise RecoveryError("an event stream must begin with run.started at seq 0")
        if event.previous_sha256 not in {None, GENESIS_EVENT_SHA256}:
            raise RecoveryError("run.started must reference the genesis event hash")
        data = _fields(
            event.data,
            required=frozenset({"config"}),
            optional=frozenset({"status"}),
            label="run.started data",
        )
        if data.get("status", "running") != "running":
            raise RecoveryError("run.started status must be running")
        config = RunConfigSnapshot.from_dict(
            _object(data["config"], "run.started config")
        )
        projection = RunProjection(
            run_id=event.run_id,
            config=config,
            messages=config.initial_messages,
            usage=TokenUsage(),
            turn=0,
            tool_calls=0,
            completed_calls={},
            pending_model=None,
            pending_tools=(),
            phase="running",
            awaiting_terminal=False,
            terminal=None,
            through_seq=event.seq,
            through_event_sha256=event.sha256,
        )
        projection._validate()
        return projection

    if event.run_id != state.run_id:
        raise RecoveryError("event run_id does not match projection")
    if event.seq != state.through_seq + 1:
        raise RecoveryError(
            f"expected event seq {state.through_seq + 1}, got {event.seq}"
        )
    if (
        event.previous_sha256 is not None
        and event.previous_sha256 != state.through_event_sha256
    ):
        raise RecoveryError("event prev_event_sha256 does not match projection")
    if state.phase == "terminal":
        raise RecoveryError("no event is valid after a terminal event")

    if event.type == "model.requested":
        if state.phase != "running":
            raise RecoveryError("model request requires a running projection")
        if state.pending_model is not None or state.pending_tools:
            raise RecoveryError("model request cannot overlap pending work")
        if state.awaiting_terminal:
            raise RecoveryError("model request cannot follow a final model response")
        data = _fields(
            event.data,
            required=frozenset({"turn", "request_sha256"}),
            optional=frozenset(
                {
                    "message_count",
                    "message_roles",
                    "max_output_tokens",
                }
            ),
            label="model.requested data",
        )
        turn = _integer(data["turn"], "model request turn", minimum=1)
        if turn != state.turn + 1:
            raise RecoveryError("model request turn is not the next turn")
        if "message_count" in data and _integer(
            data["message_count"], "model request message_count"
        ) != len(state.messages):
            raise RecoveryError("model request message_count does not match projection")
        if "message_roles" in data:
            roles = _array(data["message_roles"], "model request message_roles")
            if roles != [message.role for message in state.messages]:
                raise RecoveryError(
                    "model request message_roles do not match projection"
                )
        if "max_output_tokens" in data:
            _integer(
                data["max_output_tokens"],
                "model request max_output_tokens",
                minimum=1,
            )
        return _advance(
            state,
            event,
            pending_model=PendingModelCall(
                turn=turn,
                request_sha256=_sha256(
                    data["request_sha256"], "model request request_sha256"
                ),
            ),
        )

    if event.type == "model.responded":
        if state.phase != "running" or state.pending_model is None:
            raise RecoveryError("model response requires a pending model request")
        data = _fields(
            event.data,
            required=frozenset(
                {
                    "turn",
                    "content",
                    "tool_calls",
                    "finish_reason",
                    "usage",
                    "response_id",
                }
            ),
            label="model.responded data",
        )
        turn = _integer(data["turn"], "model response turn", minimum=1)
        if turn != state.pending_model.turn:
            raise RecoveryError("model response turn does not match pending request")
        response_data = {
            name: data[name]
            for name in (
                "content",
                "tool_calls",
                "finish_reason",
                "usage",
                "response_id",
            )
        }
        response = cast(
            ModelResponse,
            _protocol(ModelResponse.from_dict, response_data, "model response"),
        )
        call_ids = [call.id for call in response.tool_calls]
        if len(set(call_ids)) != len(call_ids):
            raise RecoveryError("model response tool call ids must be unique")
        pending_tools = tuple(
            PendingToolCall(
                call=call,
                turn=turn,
                call_index=index,
                fingerprint=tool_call_fingerprint(call),
            )
            for index, call in enumerate(response.tool_calls)
        )
        return _advance(
            state,
            event,
            messages=(
                *state.messages,
                AgentMessage(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                ),
            ),
            usage=state.usage + response.usage,
            turn=turn,
            pending_model=None,
            pending_tools=pending_tools,
            awaiting_terminal=not pending_tools,
        )

    if event.type == "budget.updated":
        data = _fields(
            event.data,
            required=frozenset({"turn", "turns", "tool_calls", "usage"}),
            optional=frozenset({"limit", "overage"}),
            label="budget.updated data",
        )
        if _integer(data["turn"], "budget turn") != state.turn:
            raise RecoveryError("budget turn does not match projection")
        if _integer(data["turns"], "budget turns") != state.turn:
            raise RecoveryError("budget turns does not match projection")
        if _integer(data["tool_calls"], "budget tool_calls") != state.tool_calls:
            raise RecoveryError("budget tool_calls does not match projection")
        usage = cast(
            TokenUsage,
            _protocol(
                TokenUsage.from_dict,
                _object(data["usage"], "budget usage"),
                "budget usage",
            ),
        )
        if usage != state.usage:
            raise RecoveryError("budget usage does not match projection")
        return _advance(state, event)

    tool_base = frozenset(
        {
            "operation_id",
            "turn",
            "call_index",
            "fingerprint",
            "call_id",
            "tool_name",
        }
    )
    if event.type == "tool.started":
        data = _fields(
            event.data,
            required=tool_base | frozenset({"arguments_json", "plan"}),
            label="tool.started data",
        )
        if state.phase != "running" or state.pending_model is not None:
            raise RecoveryError("tool start requires a running projection")
        operation_id, turn, call_index, fingerprint = _tool_identity(data, "tool start")
        index, pending = _pending_at(state, turn, call_index)
        if pending.started:
            raise RecoveryError("pending tool was already started")
        _assert_tool_event_matches(data, pending, fingerprint)
        if (
            _string(data["arguments_json"], "tool start arguments_json")
            != pending.call.arguments_json
        ):
            raise RecoveryError("tool start arguments do not match pending call")
        raw_plan = data["plan"]
        try:
            plan = (
                None
                if raw_plan is None
                else ToolExecutionPlan.from_dict(
                    _object(raw_plan, "tool start execution plan")
                )
            )
        except ValueError as exc:
            raise RecoveryError(f"invalid tool start execution plan: {exc}") from exc
        if plan is not None and (
            plan.operation_id != operation_id
            or plan.call != pending.call
            or plan.fingerprint != fingerprint
        ):
            raise RecoveryError("tool start execution plan does not match pending call")
        if operation_id in {
            item.operation_id for item in state.completed_calls.values()
        } or operation_id in {
            item.operation_id
            for item in state.pending_tools
            if item.operation_id is not None
        }:
            raise RecoveryError("tool operation_id was already used")
        updated = list(state.pending_tools)
        updated[index] = replace(
            pending,
            operation_id=operation_id,
            plan=plan,
            started=True,
        )
        return _advance(
            state,
            event,
            pending_tools=tuple(updated),
            tool_calls=state.tool_calls + 1,
        )

    outcome_fields = frozenset(
        {
            "ok",
            "content",
            "error_code",
            "metadata",
            "truncated",
        }
    )
    if event.type in {"tool.completed", "tool.failed", "tool.reused"}:
        data = _fields(
            event.data,
            required=tool_base | outcome_fields,
            label=f"{event.type} data",
        )
        operation_id, turn, call_index, fingerprint = _tool_identity(data, event.type)
        index, pending = _pending_at(state, turn, call_index)
        _assert_tool_event_matches(data, pending, fingerprint)
        outcome = _outcome_from_event(data)
        updated = list(state.pending_tools)
        del updated[index]
        if event.type == "tool.reused":
            if pending.started:
                raise RecoveryError("a started tool cannot be satisfied by reuse")
            cached = state.completed_calls.get(pending.call.id)
            if cached is None:
                raise RecoveryError("tool reuse has no completed cached call")
            if (
                cached.operation_id != operation_id
                or cached.fingerprint != fingerprint
                or cached.outcome != outcome
            ):
                raise RecoveryError("tool reuse does not match completed cache")
            return _advance(
                state,
                event,
                messages=(*state.messages, _tool_message(outcome)),
                pending_tools=tuple(updated),
            )

        if not pending.started or pending.operation_id != operation_id:
            raise RecoveryError("tool result requires its matching started operation")
        if event.type == "tool.completed" and not outcome.ok:
            raise RecoveryError("tool.completed outcome must have ok=true")
        if event.type == "tool.failed" and outcome.ok:
            raise RecoveryError("tool.failed outcome must have ok=false")
        if pending.call.id in state.completed_calls:
            raise RecoveryError("completed call id must use tool.reused")
        completed = CompletedToolCall(
            call=pending.call,
            operation_id=operation_id,
            turn=turn,
            call_index=call_index,
            fingerprint=fingerprint,
            outcome=outcome,
        )
        cache = dict(state.completed_calls)
        cache[pending.call.id] = completed
        return _advance(
            state,
            event,
            messages=(*state.messages, _tool_message(outcome)),
            completed_calls=cache,
            pending_tools=tuple(updated),
        )

    if event.type == "model.failed":
        data = _fields(
            event.data,
            required=frozenset({"code", "retryable", "message"}),
            optional=frozenset({"turn"}),
            label="model.failed data",
        )
        if state.phase != "running" or state.pending_model is None:
            raise RecoveryError("model.failed requires a pending model request")
        _string(data["code"], "model failure code", allow_empty=False)
        _boolean(data["retryable"], "model failure retryable")
        _string(data["message"], "model failure message")
        if (
            "turn" in data
            and _integer(data["turn"], "model failure turn", minimum=1)
            != state.pending_model.turn
        ):
            raise RecoveryError("model failure turn does not match pending request")
        return _advance(state, event, pending_model=None)

    if event.type == "runtime.failed":
        data = _fields(
            event.data,
            required=frozenset({"error_type", "message"}),
            optional=frozenset({"turn", "operation_id"}),
            label="runtime.failed data",
        )
        _string(data["error_type"], "runtime failure error_type", allow_empty=False)
        _string(data["message"], "runtime failure message")
        if "turn" in data:
            _integer(data["turn"], "runtime failure turn")
        if "operation_id" in data:
            _string(
                data["operation_id"],
                "runtime failure operation_id",
                allow_empty=False,
            )
        return _advance(state, event)

    if event.type in {"run.paused", "run.interrupted"}:
        data = _fields(
            event.data,
            required=frozenset({"reason"}),
            optional=frozenset({"status"}),
            label=f"{event.type} data",
        )
        _string(data["reason"], f"{event.type} reason")
        expected_status = "paused" if event.type == "run.paused" else "interrupted"
        if data.get("status", expected_status) != expected_status:
            raise RecoveryError(f"{event.type} status is inconsistent")
        valid_phases = (
            {"running", "interrupted"} if event.type == "run.paused" else {"running"}
        )
        if state.phase not in valid_phases:
            raise RecoveryError(f"{event.type} is invalid from {state.phase}")
        if event.type == "run.paused":
            phase: ProjectionPhase = "paused"
        else:
            phase = "interrupted"
        return _advance(state, event, phase=phase)

    if event.type == "run.resumed":
        data = _fields(
            event.data,
            required=frozenset({"config_sha256"}),
            optional=frozenset({"model_fingerprint", "tool_fingerprint", "reason"}),
            label="run.resumed data",
        )
        if state.phase not in {"paused", "interrupted"}:
            raise RecoveryError("run.resumed requires a paused or interrupted run")
        if (
            _sha256(data["config_sha256"], "resume config_sha256")
            != state.config.config_sha256
        ):
            raise RecoveryError("resume config fingerprint changed")
        if (
            "model_fingerprint" in data
            and _sha256(data["model_fingerprint"], "resume model_fingerprint")
            != state.config.model_fingerprint
        ):
            raise RecoveryError("resume model fingerprint changed")
        if (
            "tool_fingerprint" in data
            and _sha256(data["tool_fingerprint"], "resume tool_fingerprint")
            != state.config.tool_fingerprint
        ):
            raise RecoveryError("resume tool fingerprint changed")
        if "reason" in data:
            _string(data["reason"], "resume reason")
        return _advance(state, event, phase="running")

    terminal_types: dict[str, RunStatus] = {
        "run.completed": "completed",
        "run.stopped": "stopped",
        "run.failed": "failed",
        "run.cancelled": "cancelled",
    }
    if event.type in terminal_types:
        data = _fields(
            event.data,
            required=frozenset(
                {"status", "reason", "turns", "tool_calls", "usage", "final_text"}
            ),
            optional=frozenset({"elapsed_seconds"}),
            label=f"{event.type} data",
        )
        status = terminal_types[event.type]
        if data["status"] != status:
            raise RecoveryError("terminal event status does not match event type")
        if status == "completed" and (
            not state.awaiting_terminal
            or state.pending_model is not None
            or state.pending_tools
        ):
            raise RecoveryError("run.completed requires a final model response")
        if _integer(data["turns"], "terminal turns") != state.turn:
            raise RecoveryError("terminal turns does not match projection")
        if _integer(data["tool_calls"], "terminal tool_calls") != state.tool_calls:
            raise RecoveryError("terminal tool_calls does not match projection")
        terminal_usage = cast(
            TokenUsage,
            _protocol(
                TokenUsage.from_dict,
                _object(data["usage"], "terminal usage"),
                "terminal usage",
            ),
        )
        if terminal_usage != state.usage:
            raise RecoveryError("terminal usage does not match projection")
        final_text = _string(data["final_text"], "terminal final_text")
        latest_assistant = next(
            (
                message.content
                for message in reversed(state.messages)
                if message.role == "assistant"
            ),
            "",
        )
        if final_text != latest_assistant:
            raise RecoveryError("terminal final_text does not match model history")
        return _advance(
            state,
            event,
            phase="terminal",
            terminal=TerminalState(
                status=status,
                reason=_string(data["reason"], "terminal reason"),
                final_text=final_text,
            ),
        )

    raise RecoveryError(f"unsupported event type: {event.type!r}")


def replay_events(events: Iterable[Mapping[str, Any]]) -> RunProjection:
    """Rebuild one projection from a complete event stream."""

    state: RunProjection | None = None
    for event in events:
        state = reduce_event(state, event)
    if state is None:
        raise RecoveryError("cannot replay an empty event stream")
    return state


def write_projection_checkpoint(
    path: str | Path, state: RunProjection
) -> ProjectionCheckpoint:
    """Atomically replace a JSON checkpoint cache for ``state``."""

    checkpoint = ProjectionCheckpoint.create(state)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            checkpoint.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return checkpoint


def load_projection_checkpoint(
    path: str | Path,
    *,
    through_event: Mapping[str, Any] | None = None,
) -> ProjectionCheckpoint | None:
    """Load an internally consistent cache record.

    Authoritative validation against the event prefix is performed by
    :func:`replay_events_with_checkpoint`; this parser alone does not authenticate
    state.
    """

    try:
        source = Path(path)
        if not source.is_file() or source.stat().st_size > 32 * 1024 * 1024:
            return None
        decoded = json.loads(source.read_text(encoding="utf-8"))
        checkpoint = ProjectionCheckpoint.from_dict(
            _object(decoded, "projection checkpoint")
        )
        if through_event is not None:
            event = _parse_event(through_event)
            if event.seq != checkpoint.through_seq:
                return None
            if event.run_id != checkpoint.state.run_id:
                return None
            if event.sha256 != checkpoint.through_event_sha256:
                return None
        return checkpoint
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecoveryError):
        return None


def replay_from_checkpoint(
    checkpoint: ProjectionCheckpoint,
    suffix_events: Iterable[Mapping[str, Any]],
) -> RunProjection:
    """Replay only events after a previously validated checkpoint."""

    state = checkpoint.state
    if (
        checkpoint.through_seq != state.through_seq
        or checkpoint.through_event_sha256 != state.through_event_sha256
        or checkpoint.state_sha256 != canonical_sha256(state.to_dict())
    ):
        raise RecoveryError("checkpoint object failed integrity validation")
    for event in suffix_events:
        state = reduce_event(state, event)
    return state


def replay_events_with_checkpoint(
    events: Sequence[Mapping[str, Any]], checkpoint_path: str | Path
) -> RunProjection:
    """Use a matching checkpoint only after verifying it against its event prefix.

    The checkpoint checksum is not an authentication mechanism. Replaying the anchored
    prefix prevents a self-consistent but forged cache from refunding budgets, changing
    permissions, or inventing pending work while the authoritative log is unchanged.
    """

    checkpoint = load_projection_checkpoint(checkpoint_path)
    if checkpoint is None or checkpoint.through_seq >= len(events):
        return replay_events(events)
    through_event = events[checkpoint.through_seq]
    verified = load_projection_checkpoint(checkpoint_path, through_event=through_event)
    if verified is None:
        return replay_events(events)
    prefix_state = replay_events(events[: verified.through_seq + 1])
    if prefix_state != verified.state:
        return replay_events(events)
    return replay_from_checkpoint(verified, events[verified.through_seq + 1 :])
