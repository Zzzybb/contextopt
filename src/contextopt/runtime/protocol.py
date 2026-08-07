"""Provider-neutral messages, tool calls, budgets, and run results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

RunStatus = Literal["completed", "stopped", "failed", "cancelled"]
MessageRole = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments_json: str

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
