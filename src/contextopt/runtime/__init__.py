"""Auditable runtime primitives for long-running coding agents."""

from contextopt.runtime.events import EventLog, read_events, render_trace
from contextopt.runtime.model import OpenAICompatibleModel, ScriptedModel
from contextopt.runtime.protocol import (
    AgentMessage,
    AgentRunResult,
    ModelClient,
    ModelRequest,
    ModelResponse,
    RunLimits,
    RunPermissions,
    TokenUsage,
    ToolCall,
    ToolDefinition,
    ToolOutcome,
)
from contextopt.runtime.runner import AgentRunner
from contextopt.runtime.tools import WorkspaceTools

__all__ = [
    "AgentMessage",
    "AgentRunResult",
    "AgentRunner",
    "EventLog",
    "ModelClient",
    "ModelRequest",
    "ModelResponse",
    "OpenAICompatibleModel",
    "RunLimits",
    "RunPermissions",
    "ScriptedModel",
    "TokenUsage",
    "ToolCall",
    "ToolDefinition",
    "ToolOutcome",
    "WorkspaceTools",
    "read_events",
    "render_trace",
]
