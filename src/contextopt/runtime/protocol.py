"""Provider-neutral messages, tool calls, budgets, and run results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, cast

RunStatus = Literal["completed", "stopped", "failed", "cancelled", "paused"]
MessageRole = Literal["system", "user", "assistant", "tool"]


def _fields(
    value: Mapping[str, Any],
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    keys = set(value)
    missing = required - keys
    unknown = keys - required - optional
    if missing:
        raise ValueError(f"{label} is missing fields: {sorted(missing)!r}")
    if unknown:
        raise ValueError(f"{label} has unknown fields: {sorted(unknown)!r}")
    return value


def _string(value: Any, label: str, *, allow_empty: bool = True) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "a non-empty string" if not allow_empty else "a string"
        raise ValueError(f"{label} must be {qualifier}")
    return value


def _optional_string(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _string(value, label)


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str, *, minimum: float = 0.0) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or float(value) < minimum
    ):
        raise ValueError(f"{label} must be a number >= {minimum}")
    return float(value)


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments_json: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ToolCall:
        value = _fields(
            data,
            required=frozenset({"id", "name", "arguments_json"}),
            label="tool call",
        )
        return cls(
            id=_string(value["id"], "tool call id", allow_empty=False),
            name=_string(value["name"], "tool call name", allow_empty=False),
            arguments_json=_string(value["arguments_json"], "tool call arguments_json"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "arguments_json": self.arguments_json,
        }


@dataclass(frozen=True, slots=True)
class AgentMessage:
    role: MessageRole
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    tool_name: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AgentMessage:
        value = _fields(
            data,
            required=frozenset({"role", "content"}),
            optional=frozenset({"tool_calls", "tool_call_id", "tool_name"}),
            label="agent message",
        )
        role_value = _string(value["role"], "message role", allow_empty=False)
        if role_value not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported message role: {role_value!r}")
        raw_calls = _array(value.get("tool_calls", []), "message tool_calls")
        calls = tuple(
            ToolCall.from_dict(_object(item, f"message tool_calls[{index}]"))
            for index, item in enumerate(raw_calls)
        )
        tool_call_id = _optional_string(
            value.get("tool_call_id"), "message tool_call_id"
        )
        tool_name = _optional_string(value.get("tool_name"), "message tool_name")
        if role_value == "assistant":
            if tool_call_id is not None or tool_name is not None:
                raise ValueError("assistant message cannot identify a tool result")
        elif role_value == "tool":
            if calls:
                raise ValueError("tool message cannot contain tool calls")
            if not tool_call_id or not tool_name:
                raise ValueError("tool message requires tool_call_id and tool_name")
        elif calls or tool_call_id is not None or tool_name is not None:
            raise ValueError(f"{role_value} message cannot contain tool metadata")
        return cls(
            role=cast(MessageRole, role_value),
            content=_string(value["content"], "message content"),
            tool_calls=calls,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
        }


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: Mapping[str, Any]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ToolDefinition:
        value = _fields(
            data,
            required=frozenset({"name", "description", "input_schema"}),
            label="tool definition",
        )
        return cls(
            name=_string(value["name"], "tool name", allow_empty=False),
            description=_string(value["description"], "tool description"),
            input_schema=dict(_object(value["input_schema"], "tool input_schema")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": dict(self.input_schema),
        }


@dataclass(frozen=True, slots=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TokenUsage:
        value = _fields(
            data,
            required=frozenset(),
            optional=frozenset(
                {
                    "input_tokens",
                    "output_tokens",
                    "cached_input_tokens",
                    "reasoning_tokens",
                    "total_tokens",
                }
            ),
            label="token usage",
        )
        usage = cls(
            input_tokens=_integer(value.get("input_tokens", 0), "usage input_tokens"),
            output_tokens=_integer(
                value.get("output_tokens", 0), "usage output_tokens"
            ),
            cached_input_tokens=_integer(
                value.get("cached_input_tokens", 0),
                "usage cached_input_tokens",
            ),
            reasoning_tokens=_integer(
                value.get("reasoning_tokens", 0), "usage reasoning_tokens"
            ),
        )
        if "total_tokens" in value:
            declared = _integer(value["total_tokens"], "usage total_tokens")
            if declared != usage.total_tokens:
                raise ValueError("usage total_tokens does not match token counts")
        return usage

    def __post_init__(self) -> None:
        for field_name in (
            "input_tokens",
            "output_tokens",
            "cached_input_tokens",
            "reasoning_tokens",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative")

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_input_tokens=(self.cached_input_tokens + other.cached_input_tokens),
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True, slots=True)
class ModelRequest:
    run_id: str
    turn: int
    messages: tuple[AgentMessage, ...]
    tools: tuple[ToolDefinition, ...]
    max_output_tokens: int


@dataclass(frozen=True, slots=True)
class ModelResponse:
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str = "stop"
    usage: TokenUsage = field(default_factory=TokenUsage)
    response_id: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelResponse:
        value = _fields(
            data,
            required=frozenset(
                {"content", "tool_calls", "finish_reason", "usage", "response_id"}
            ),
            label="model response",
        )
        raw_calls = _array(value["tool_calls"], "model response tool_calls")
        return cls(
            content=_string(value["content"], "model response content"),
            tool_calls=tuple(
                ToolCall.from_dict(_object(item, f"tool_calls[{index}]"))
                for index, item in enumerate(raw_calls)
            ),
            finish_reason=_string(
                value["finish_reason"], "model response finish_reason"
            ),
            usage=TokenUsage.from_dict(_object(value["usage"], "model response usage")),
            response_id=_optional_string(
                value["response_id"], "model response response_id"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "finish_reason": self.finish_reason,
            "usage": self.usage.to_dict(),
            "response_id": self.response_id,
        }


class ModelClient(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def configuration_fingerprint(self) -> str: ...

    def resume_from_turn(self, completed_turns: int) -> None: ...

    async def complete(self, request: ModelRequest) -> ModelResponse: ...


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    call_id: str
    tool_name: str
    ok: bool
    content: str
    error_code: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    truncated: bool = False

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ToolOutcome:
        value = _fields(
            data,
            required=frozenset(
                {
                    "call_id",
                    "tool_name",
                    "ok",
                    "content",
                    "error_code",
                    "metadata",
                    "truncated",
                }
            ),
            label="tool outcome",
        )
        return cls(
            call_id=_string(value["call_id"], "outcome call_id", allow_empty=False),
            tool_name=_string(
                value["tool_name"], "outcome tool_name", allow_empty=False
            ),
            ok=_boolean(value["ok"], "outcome ok"),
            content=_string(value["content"], "outcome content"),
            error_code=_optional_string(value["error_code"], "outcome error_code"),
            metadata=dict(_object(value["metadata"], "outcome metadata")),
            truncated=_boolean(value["truncated"], "outcome truncated"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "ok": self.ok,
            "content": self.content,
            "error_code": self.error_code,
            "metadata": dict(self.metadata),
            "truncated": self.truncated,
        }


@dataclass(frozen=True, slots=True)
class RunLimits:
    max_turns: int = 20
    max_tool_calls: int = 50
    max_total_tokens: int | None = 100_000
    max_output_tokens_per_call: int = 2_048
    wall_timeout_seconds: float = 900.0
    command_timeout_seconds: float = 120.0
    max_tool_output_bytes: int = 256 * 1024

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunLimits:
        names = frozenset(
            {
                "max_turns",
                "max_tool_calls",
                "max_total_tokens",
                "max_output_tokens_per_call",
                "wall_timeout_seconds",
                "command_timeout_seconds",
                "max_tool_output_bytes",
            }
        )
        value = _fields(data, required=names, label="run limits")
        raw_total = value["max_total_tokens"]
        max_total_tokens = (
            None
            if raw_total is None
            else _integer(raw_total, "limits max_total_tokens", minimum=1)
        )
        return cls(
            max_turns=_integer(value["max_turns"], "limits max_turns", minimum=1),
            max_tool_calls=_integer(
                value["max_tool_calls"], "limits max_tool_calls", minimum=1
            ),
            max_total_tokens=max_total_tokens,
            max_output_tokens_per_call=_integer(
                value["max_output_tokens_per_call"],
                "limits max_output_tokens_per_call",
                minimum=1,
            ),
            wall_timeout_seconds=_number(
                value["wall_timeout_seconds"],
                "limits wall_timeout_seconds",
                minimum=0.000001,
            ),
            command_timeout_seconds=_number(
                value["command_timeout_seconds"],
                "limits command_timeout_seconds",
                minimum=0.000001,
            ),
            max_tool_output_bytes=_integer(
                value["max_tool_output_bytes"],
                "limits max_tool_output_bytes",
                minimum=1,
            ),
        )

    def __post_init__(self) -> None:
        positive = (
            "max_turns",
            "max_tool_calls",
            "max_output_tokens_per_call",
            "wall_timeout_seconds",
            "command_timeout_seconds",
            "max_tool_output_bytes",
        )
        for field_name in positive:
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.max_total_tokens is not None and self.max_total_tokens <= 0:
            raise ValueError("max_total_tokens must be positive when set")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_turns": self.max_turns,
            "max_tool_calls": self.max_tool_calls,
            "max_total_tokens": self.max_total_tokens,
            "max_output_tokens_per_call": self.max_output_tokens_per_call,
            "wall_timeout_seconds": self.wall_timeout_seconds,
            "command_timeout_seconds": self.command_timeout_seconds,
            "max_tool_output_bytes": self.max_tool_output_bytes,
        }


@dataclass(frozen=True, slots=True)
class RunPermissions:
    allow_write: bool = False
    allow_command: bool = False

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunPermissions:
        value = _fields(
            data,
            required=frozenset({"allow_write", "allow_command"}),
            label="run permissions",
        )
        return cls(
            allow_write=_boolean(value["allow_write"], "permissions allow_write"),
            allow_command=_boolean(value["allow_command"], "permissions allow_command"),
        )

    def to_dict(self) -> dict[str, bool]:
        return {
            "allow_write": self.allow_write,
            "allow_command": self.allow_command,
        }


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    run_id: str
    status: RunStatus
    reason: str
    final_text: str
    turns: int
    tool_calls: int
    usage: TokenUsage
    event_log: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AgentRunResult:
        value = _fields(
            data,
            required=frozenset(
                {
                    "run_id",
                    "status",
                    "reason",
                    "final_text",
                    "turns",
                    "tool_calls",
                    "usage",
                    "event_log",
                }
            ),
            label="agent run result",
        )
        status_value = _string(value["status"], "run status", allow_empty=False)
        if status_value not in {
            "completed",
            "stopped",
            "failed",
            "cancelled",
            "paused",
        }:
            raise ValueError(f"unsupported run status: {status_value!r}")
        return cls(
            run_id=_string(value["run_id"], "run id", allow_empty=False),
            status=cast(RunStatus, status_value),
            reason=_string(value["reason"], "run reason"),
            final_text=_string(value["final_text"], "run final_text"),
            turns=_integer(value["turns"], "run turns"),
            tool_calls=_integer(value["tool_calls"], "run tool_calls"),
            usage=TokenUsage.from_dict(_object(value["usage"], "run usage")),
            event_log=_string(value["event_log"], "run event_log"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "reason": self.reason,
            "final_text": self.final_text,
            "turns": self.turns,
            "tool_calls": self.tool_calls,
            "usage": self.usage.to_dict(),
            "event_log": self.event_log,
        }
