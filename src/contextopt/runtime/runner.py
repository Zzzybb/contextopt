"""A bounded, auditable single-agent tool loop."""

from __future__ import annotations

import asyncio
import hashlib
import json
from time import monotonic
from typing import Any

from contextopt.runtime.errors import ModelError, RuntimeContractError
from contextopt.runtime.events import EventLog
from contextopt.runtime.prompts import DEFAULT_CODING_SYSTEM_PROMPT
from contextopt.runtime.protocol import (
    AgentMessage,
    AgentRunResult,
    ModelClient,
    ModelRequest,
    RunLimits,
    RunStatus,
    TokenUsage,
    ToolOutcome,
)
from contextopt.runtime.tools import WorkspaceTools


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tool_message(outcome: ToolOutcome) -> AgentMessage:
    content = json.dumps(
        outcome.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return AgentMessage(
        role="tool",
        content=content,
        tool_call_id=outcome.call_id,
        tool_name=outcome.tool_name,
    )


class AgentRunner:
    """Run one model against one workspace until it stops or reaches a limit."""

    def __init__(
        self,
        *,
        model: ModelClient,
        tools: WorkspaceTools,
        event_log: EventLog,
        limits: RunLimits | None = None,
        system_prompt: str = DEFAULT_CODING_SYSTEM_PROMPT,
    ) -> None:
        self.model = model
        self.tools = tools
        self.event_log = event_log
        self.limits = limits or tools.limits
        self.system_prompt = system_prompt

    async def run(self, task: str) -> AgentRunResult:
        if not task.strip():
            raise ValueError("task must not be empty")
        if self.event_log.event_count:
            raise ValueError("a new run requires an empty event log")

        run_id = self.event_log.run_id
        messages = [
            AgentMessage(role="system", content=self.system_prompt),
            AgentMessage(role="user", content=task),
        ]
        turns = 0
        tool_calls = 0
        usage = TokenUsage()
        final_text = ""
        started = monotonic()
        cached_calls: dict[str, tuple[str, ToolOutcome]] = {}
        terminal_written = False

        self.event_log.append(
            "run.started",
            {
                "status": "running",
                "task": task,
                "model": self.model.name,
                "workspace": ".",
                "permissions": self.tools.permissions.to_dict(),
                "limits": self.limits.to_dict(),
                "tool_names": [tool.name for tool in self.tools.definitions],
            },
        )

        def finish(status: RunStatus, reason: str) -> AgentRunResult:
            nonlocal terminal_written
            if terminal_written:
                raise RuntimeContractError("run emitted more than one terminal event")
            terminal_written = True
            self.event_log.append(
                f"run.{status}",
                {
                    "status": status,
                    "reason": reason,
                    "turns": turns,
                    "tool_calls": tool_calls,
                    "usage": usage.to_dict(),
                    "elapsed_seconds": round(monotonic() - started, 6),
                    "final_text": final_text,
                },
            )
            return AgentRunResult(
                run_id=run_id,
                status=status,
                reason=reason,
                final_text=final_text,
                turns=turns,
                tool_calls=tool_calls,
                usage=usage,
                event_log=self.event_log.path.as_posix(),
            )

        try:
            while turns < self.limits.max_turns:
                elapsed = monotonic() - started
                remaining_wall = self.limits.wall_timeout_seconds - elapsed
                if remaining_wall <= 0:
                    return finish("stopped", "wall_timeout")
                if (
                    self.limits.max_total_tokens is not None
                    and usage.total_tokens >= self.limits.max_total_tokens
                ):
                    return finish("stopped", "token_limit")

                remaining_tokens = (
                    self.limits.max_output_tokens_per_call
                    if self.limits.max_total_tokens is None
                    else min(
                        self.limits.max_output_tokens_per_call,
                        self.limits.max_total_tokens - usage.total_tokens,
                    )
                )
                request = ModelRequest(
                    run_id=run_id,
                    turn=turns + 1,
                    messages=tuple(messages),
                    tools=self.tools.definitions,
                    max_output_tokens=remaining_tokens,
                )
                request_hash = _stable_hash(
                    {
                        "messages": [message.to_dict() for message in request.messages],
                        "tools": [tool.to_dict() for tool in request.tools],
                        "max_output_tokens": request.max_output_tokens,
                    }
                )
                self.event_log.append(
                    "model.requested",
                    {
                        "turn": request.turn,
                        "message_count": len(request.messages),
                        "message_roles": [message.role for message in request.messages],
                        "request_sha256": request_hash,
                        "max_output_tokens": request.max_output_tokens,
                    },
                )
                try:
                    response = await asyncio.wait_for(
                        self.model.complete(request), timeout=remaining_wall
                    )
                except TimeoutError:
                    return finish("stopped", "wall_timeout")
                turns += 1
                usage = usage + response.usage
                final_text = response.content
                messages.append(
                    AgentMessage(
                        role="assistant",
                        content=response.content,
                        tool_calls=response.tool_calls,
                    )
                )
                self.event_log.append(
                    "model.responded",
                    {
                        "turn": turns,
                        **response.to_dict(),
                    },
                )
                self.event_log.append(
                    "budget.updated",
                    {
                        "turn": turns,
                        "turns": turns,
                        "tool_calls": tool_calls,
                        "usage": usage.to_dict(),
                        "limit": self.limits.max_total_tokens,
                        "overage": (
                            0
                            if self.limits.max_total_tokens is None
                            else max(
                                0, usage.total_tokens - self.limits.max_total_tokens
                            )
                        ),
                    },
                )
                if (
                    self.limits.max_total_tokens is not None
                    and usage.total_tokens > self.limits.max_total_tokens
                ):
                    return finish("stopped", "token_limit_after_response")
                if not response.tool_calls:
                    return finish("completed", "model_stopped")

                for call in response.tool_calls:
                    if tool_calls >= self.limits.max_tool_calls:
                        return finish("stopped", "tool_call_limit")
                    fingerprint = _stable_hash(
                        {
                            "name": call.name,
                            "arguments_json": call.arguments_json,
                        }
                    )
                    cached = cached_calls.get(call.id)
                    if cached is not None:
                        cached_fingerprint, outcome = cached
                        if cached_fingerprint != fingerprint:
                            message = f"tool call id {call.id!r} was reused with new "
                            message += "arguments"
                            raise RuntimeContractError(message)
                        self.event_log.append(
                            "tool.reused",
                            {
                                **outcome.to_dict(),
                                "fingerprint": fingerprint,
                            },
                        )
                        messages.append(_tool_message(outcome))
                        continue

                    tool_calls += 1
                    self.event_log.append(
                        "tool.started",
                        {
                            "call_id": call.id,
                            "tool_name": call.name,
                            "arguments_json": call.arguments_json,
                            "fingerprint": fingerprint,
                        },
                    )
                    outcome = await self.tools.execute(call)
                    cached_calls[call.id] = (fingerprint, outcome)
                    self.event_log.append(
                        "tool.completed" if outcome.ok else "tool.failed",
                        outcome.to_dict(),
                    )
                    messages.append(_tool_message(outcome))

            return finish("stopped", "turn_limit")
        except asyncio.CancelledError:
            if not terminal_written:
                finish("cancelled", "cancelled")
            raise
        except ModelError as exc:
            self.event_log.append(
                "model.failed",
                {"code": exc.code, "retryable": exc.retryable, "message": str(exc)},
            )
            return finish("failed", f"model_error:{exc.code}")
        except (RuntimeContractError, ValueError, OSError) as exc:
            self.event_log.append(
                "runtime.failed",
                {"error_type": type(exc).__name__, "message": str(exc)},
            )
            return finish("failed", "runtime_contract_error")


def require_completed(result: AgentRunResult) -> AgentRunResult:
    """Narrow a result for callers that intentionally require normal model stop."""

    if result.status != "completed":
        raise RuntimeError(f"agent run did not complete: {result.reason}")
    return result
