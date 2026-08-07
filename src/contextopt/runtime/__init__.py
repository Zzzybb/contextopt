"""Auditable runtime primitives for long-running coding agents."""

from contextopt.runtime.events import (
    EventLog,
    EventScan,
    RunLease,
    read_events,
    render_trace,
    scan_events,
)
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
from contextopt.runtime.recovery import (
    RecoveryError,
    RunProjection,
    replay_events,
    replay_events_with_checkpoint,
)
from contextopt.runtime.runner import AgentRunner, PendingToolResolution
from contextopt.runtime.tool_state import ToolExecutionPlan, ToolReconciliation
from contextopt.runtime.tools import WorkspaceTools

__all__ = [
    "AgentMessage",
    "AgentRunResult",
    "AgentRunner",
    "EventLog",
    "EventScan",
    "ModelClient",
    "ModelRequest",
    "ModelResponse",
    "OpenAICompatibleModel",
    "PendingToolResolution",
    "RecoveryError",
    "RunLease",
    "RunLimits",
    "RunPermissions",
    "RunProjection",
    "ScriptedModel",
    "TokenUsage",
    "ToolCall",
    "ToolDefinition",
    "ToolExecutionPlan",
    "ToolOutcome",
    "ToolReconciliation",
    "WorkspaceTools",
    "read_events",
    "render_trace",
    "replay_events",
    "replay_events_with_checkpoint",
    "scan_events",
]
