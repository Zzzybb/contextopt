"""Bounded coding-agent execution driven by a recoverable event projection."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from contextlib import suppress
from pathlib import Path
from time import monotonic
from typing import Any, Literal, TypeVar

from contextopt.runtime.context import (
    CompiledContext,
    ContextBudgetError,
    ContextCompiler,
    compile_runtime_context,
)
from contextopt.runtime.errors import ModelError, RuntimeContractError
from contextopt.runtime.events import EventLog, read_events
from contextopt.runtime.identity import stable_hash
from contextopt.runtime.prompts import DEFAULT_CODING_SYSTEM_PROMPT
from contextopt.runtime.protocol import (
    AgentMessage,
    AgentRunResult,
    ModelClient,
    ModelRequest,
    ModelResponse,
    RunLimits,
    RunStatus,
    ToolOutcome,
)
from contextopt.runtime.recovery import (
    RecoveryError,
    RunConfigSnapshot,
    RunProjection,
    reduce_event,
    replay_events_with_checkpoint,
    write_projection_checkpoint,
)
from contextopt.runtime.tool_state import tool_replay_policy
from contextopt.runtime.tools import WorkspaceTools

PendingToolResolution = Literal["retry", "mark_failed"]
_T = TypeVar("_T")


def _latest_assistant_text(state: RunProjection) -> str:
    return next(
        (
            message.content
            for message in reversed(state.messages)
            if message.role == "assistant"
        ),
        "",
    )


class AgentRunner:
    """Run or resume one model against one workspace under durable limits."""

    def __init__(
        self,
        *,
        model: ModelClient,
        tools: WorkspaceTools,
        event_log: EventLog,
        limits: RunLimits | None = None,
        system_prompt: str = DEFAULT_CODING_SYSTEM_PROMPT,
        checkpoint_path: str | Path | None = None,
        context_compiler: ContextCompiler | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.event_log = event_log
        self.limits = limits or tools.limits
        if self.limits != tools.limits:
            raise ValueError("runner limits must match workspace tool limits")
        self.system_prompt = system_prompt
        self.context_compiler = context_compiler
        self.checkpoint_path = (
            Path(checkpoint_path)
            if checkpoint_path is not None
            else event_log.path.with_name(event_log.path.name + ".checkpoint.json")
        )

    def _projection(self) -> RunProjection:
        events = read_events(self.event_log.path)
        return replay_events_with_checkpoint(events, self.checkpoint_path)

    def _append(
        self,
        state: RunProjection | None,
        event_type: str,
        data: dict[str, object],
    ) -> RunProjection:
        event = self.event_log.append(event_type, data)
        updated = reduce_event(state, event.to_dict())
        # A checkpoint is a disposable acceleration cache. The fsynced event
        # stream remains authoritative and can always be replayed in full.
        with suppress(OSError):
            write_projection_checkpoint(self.checkpoint_path, updated)
        return updated

    def _result(
        self, state: RunProjection, status: RunStatus, reason: str
    ) -> AgentRunResult:
        return AgentRunResult(
            run_id=state.run_id,
            status=status,
            reason=reason,
            final_text=_latest_assistant_text(state),
            turns=state.turn,
            tool_calls=state.tool_calls,
            usage=state.usage,
            event_log=self.event_log.path.as_posix(),
        )

    def _finish(
        self,
        state: RunProjection,
        status: Literal["completed", "stopped", "failed", "cancelled"],
        reason: str,
        session_started: float,
    ) -> AgentRunResult:
        state = self._append(
            state,
            f"run.{status}",
            {
                "status": status,
                "reason": reason,
                "turns": state.turn,
                "tool_calls": state.tool_calls,
                "usage": state.usage.to_dict(),
                "elapsed_seconds": round(monotonic() - session_started, 6),
                "final_text": _latest_assistant_text(state),
            },
        )
        return self._result(state, status, reason)

    def _pause(self, state: RunProjection, reason: str) -> AgentRunResult:
        state = self._append(
            state,
            "run.paused",
            {"status": "paused", "reason": reason},
        )
        self.event_log.close()
        return self._result(state, "paused", reason)

    def _compile_context(self, state: RunProjection) -> CompiledContext | None:
        if self.context_compiler is None:
            return None
        return compile_runtime_context(
            self.context_compiler,
            state.messages,
            task=state.config.task,
        )

    def _request(
        self, state: RunProjection, tools: WorkspaceTools
    ) -> tuple[ModelRequest, CompiledContext | None]:
        limits = state.config.limits
        remaining_tokens = (
            limits.max_output_tokens_per_call
            if limits.max_total_tokens is None
            else min(
                limits.max_output_tokens_per_call,
                limits.max_total_tokens - state.usage.total_tokens,
            )
        )
        compiled = self._compile_context(state)
        request = ModelRequest(
            run_id=state.run_id,
            turn=state.turn + 1,
            messages=(state.messages if compiled is None else compiled.messages),
            tools=tools.definitions,
            max_output_tokens=remaining_tokens,
        )
        return request, compiled

    @staticmethod
    def _request_sha256(request: ModelRequest) -> str:
        return stable_hash(
            {
                "messages": [message.to_dict() for message in request.messages],
                "tools": [tool.to_dict() for tool in request.tools],
                "max_output_tokens": request.max_output_tokens,
            }
        )

    def _append_budget(self, state: RunProjection) -> RunProjection:
        limit = state.config.limits.max_total_tokens
        return self._append(
            state,
            "budget.updated",
            {
                "turn": state.turn,
                "turns": state.turn,
                "tool_calls": state.tool_calls,
                "usage": state.usage.to_dict(),
                "limit": limit,
                "overage": (
                    0 if limit is None else max(0, state.usage.total_tokens - limit)
                ),
            },
        )

    @staticmethod
    def _budget_event_missing(events: tuple[dict[str, object], ...]) -> bool:
        last_response = -1
        last_budget = -1
        for index, event in enumerate(events):
            if event.get("type") == "model.responded":
                last_response = index
            elif event.get("type") == "budget.updated":
                last_budget = index
        return last_response > last_budget

    def _tool_result_event(
        self,
        state: RunProjection,
        *,
        operation_id: str,
        turn: int,
        call_index: int,
        fingerprint: str,
        outcome: ToolOutcome,
    ) -> RunProjection:
        return self._append(
            state,
            "tool.completed" if outcome.ok else "tool.failed",
            {
                "operation_id": operation_id,
                "turn": turn,
                "call_index": call_index,
                "fingerprint": fingerprint,
                **outcome.to_dict(),
            },
        )

    @staticmethod
    async def _settle_tool_before_cancellation(
        operation: Coroutine[Any, Any, _T],
    ) -> _T:
        """Keep the run lease until an in-flight tool reaches a durable boundary.

        Cancelling ``asyncio.to_thread`` does not stop its worker thread. Shielding the
        complete prepare/execute/event sequence prevents another process from resuming
        against a workspace mutation that is still running in this process.
        """

        task = asyncio.create_task(operation)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError as cancellation:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    # Repeated cancellation requests remain deferred until the tool
                    # result (or failure) has reached the event log.
                    continue
            if task.cancelled():
                raise RuntimeContractError(
                    "shielded tool operation was unexpectedly cancelled"
                ) from cancellation
            exception = task.exception()
            if exception is not None:
                raise exception from cancellation
            raise

    async def _call_model(
        self, state: RunProjection, session_started: float
    ) -> RunProjection:
        request, compiled = self._request(state, self.tools)
        request_sha256 = self._request_sha256(request)
        if state.pending_model is None:
            request_data: dict[str, object] = {
                "turn": request.turn,
                "message_count": len(request.messages),
                "message_roles": [message.role for message in request.messages],
                "request_sha256": request_sha256,
                "max_output_tokens": request.max_output_tokens,
            }
            if compiled is not None:
                request_data["context"] = compiled.receipt.to_dict()
            state = self._append(
                state,
                "model.requested",
                request_data,
            )
        else:
            if state.pending_model.turn != request.turn:
                raise RuntimeContractError("pending model turn changed during resume")
            if state.pending_model.request_sha256 != request_sha256:
                raise RuntimeContractError(
                    "pending model request changed during resume"
                )

        elapsed = monotonic() - session_started
        remaining_wall = state.config.limits.wall_timeout_seconds - elapsed
        if remaining_wall <= 0:
            raise TimeoutError
        response = await asyncio.wait_for(
            self.model.complete(request), timeout=remaining_wall
        )
        self._validate_response(response)
        state = self._append(
            state,
            "model.responded",
            {"turn": request.turn, **response.to_dict()},
        )
        return self._append_budget(state)

    @staticmethod
    def _validate_response(response: ModelResponse) -> None:
        try:
            ModelResponse.from_dict(response.to_dict())
        except ValueError as exc:
            raise RuntimeContractError(f"invalid model response: {exc}") from exc
        call_ids = [call.id for call in response.tool_calls]
        if len(call_ids) != len(set(call_ids)):
            raise RuntimeContractError(
                "one model response cannot contain duplicate tool call ids"
            )

    async def _start_pending_tool(self, state: RunProjection) -> RunProjection:
        pending = state.pending_tools[0]
        cached = state.completed_calls.get(pending.call.id)
        if cached is not None:
            if cached.fingerprint != pending.fingerprint:
                raise RuntimeContractError(
                    f"tool call id {pending.call.id!r} was reused with new arguments"
                )
            return self._append(
                state,
                "tool.reused",
                {
                    "operation_id": cached.operation_id,
                    "turn": pending.turn,
                    "call_index": pending.call_index,
                    "fingerprint": pending.fingerprint,
                    **cached.outcome.to_dict(),
                },
            )

        if state.tool_calls >= state.config.limits.max_tool_calls:
            raise _StopRun("tool_call_limit")

        operation_id = (
            f"{state.run_id}:tool:{pending.turn}:{pending.call_index}:{pending.call.id}"
        )
        plan = None
        try:
            plan = await self.tools.prepare(
                pending.call, operation_id, pending.fingerprint
            )
        except (OSError, PermissionError, ValueError):
            # The legacy execute path returns a structured observation. A null plan
            # makes an interrupted mutating call conservative rather than guessable.
            plan = None
        state = self._append(
            state,
            "tool.started",
            {
                "operation_id": operation_id,
                "turn": pending.turn,
                "call_index": pending.call_index,
                "call_id": pending.call.id,
                "tool_name": pending.call.name,
                "arguments_json": pending.call.arguments_json,
                "fingerprint": pending.fingerprint,
                "plan": None if plan is None else plan.to_dict(),
            },
        )
        outcome = (
            await self.tools.execute(pending.call)
            if plan is None
            else await self.tools.execute_prepared(plan)
        )
        return self._tool_result_event(
            state,
            operation_id=operation_id,
            turn=pending.turn,
            call_index=pending.call_index,
            fingerprint=pending.fingerprint,
            outcome=outcome,
        )

    async def _recover_pending_tool(
        self,
        state: RunProjection,
        resolution: PendingToolResolution | None,
    ) -> tuple[RunProjection, str | None]:
        pending = state.pending_tools[0]
        operation_id = pending.operation_id
        if not pending.started or operation_id is None:
            raise RuntimeContractError("pending recovery tool was not started")

        if resolution == "mark_failed":
            outcome = ToolOutcome(
                call_id=pending.call.id,
                tool_name=pending.call.name,
                ok=False,
                content=(
                    "A previous process ended while this tool was running. The "
                    "operator marked its result indeterminate; inspect state before "
                    "issuing another call."
                ),
                error_code="indeterminate_previous_execution",
                metadata={"operation_id": operation_id},
            )
            return (
                self._tool_result_event(
                    state,
                    operation_id=operation_id,
                    turn=pending.turn,
                    call_index=pending.call_index,
                    fingerprint=pending.fingerprint,
                    outcome=outcome,
                ),
                None,
            )

        if resolution == "retry":
            outcome = (
                await self.tools.execute(pending.call)
                if pending.plan is None
                else await self.tools.execute_prepared(pending.plan)
            )
            return (
                self._tool_result_event(
                    state,
                    operation_id=operation_id,
                    turn=pending.turn,
                    call_index=pending.call_index,
                    fingerprint=pending.fingerprint,
                    outcome=outcome,
                ),
                None,
            )

        if pending.plan is None:
            try:
                policy = tool_replay_policy(pending.call.name)
            except ValueError:
                policy = "safe"
            if policy != "safe":
                return state, "pending_tool_without_recovery_plan"
            outcome = await self.tools.execute(pending.call)
            return (
                self._tool_result_event(
                    state,
                    operation_id=operation_id,
                    turn=pending.turn,
                    call_index=pending.call_index,
                    fingerprint=pending.fingerprint,
                    outcome=outcome,
                ),
                None,
            )

        reconciliation = await self.tools.reconcile(pending.plan)
        if reconciliation.action == "completed":
            assert reconciliation.outcome is not None
            return (
                self._tool_result_event(
                    state,
                    operation_id=operation_id,
                    turn=pending.turn,
                    call_index=pending.call_index,
                    fingerprint=pending.fingerprint,
                    outcome=reconciliation.outcome,
                ),
                None,
            )
        if reconciliation.action == "retry":
            outcome = await self.tools.execute_prepared(pending.plan)
            return (
                self._tool_result_event(
                    state,
                    operation_id=operation_id,
                    turn=pending.turn,
                    call_index=pending.call_index,
                    fingerprint=pending.fingerprint,
                    outcome=outcome,
                ),
                None,
            )
        return state, f"pending_tool_{reconciliation.action}:{reconciliation.reason}"

    async def _drive(
        self,
        state: RunProjection,
        *,
        session_started: float,
        pending_tool_resolution: PendingToolResolution | None = None,
    ) -> AgentRunResult:
        resolution = pending_tool_resolution
        while True:
            if state.phase != "running":
                raise RuntimeContractError(
                    f"cannot drive a run in phase {state.phase!r}"
                )

            limits = state.config.limits
            if (
                limits.max_total_tokens is not None
                and state.usage.total_tokens > limits.max_total_tokens
            ):
                return self._finish(
                    state,
                    "stopped",
                    "token_limit_after_response",
                    session_started,
                )

            if state.awaiting_terminal:
                return self._finish(
                    state, "completed", "model_stopped", session_started
                )

            if state.pending_tools:
                pending = state.pending_tools[0]
                if pending.started:
                    state, pause_reason = await self._settle_tool_before_cancellation(
                        self._recover_pending_tool(state, resolution)
                    )
                    resolution = None
                    if pause_reason is not None:
                        return self._pause(state, pause_reason)
                else:
                    try:
                        state = await self._settle_tool_before_cancellation(
                            self._start_pending_tool(state)
                        )
                    except _StopRun as stop:
                        return self._finish(
                            state, "stopped", stop.reason, session_started
                        )
                continue

            if state.pending_model is not None:
                try:
                    state = await self._call_model(state, session_started)
                except TimeoutError:
                    return self._finish(
                        state, "stopped", "wall_timeout", session_started
                    )
                continue

            if state.turn >= limits.max_turns:
                return self._finish(state, "stopped", "turn_limit", session_started)
            if (
                limits.max_total_tokens is not None
                and state.usage.total_tokens >= limits.max_total_tokens
            ):
                return self._finish(state, "stopped", "token_limit", session_started)
            if monotonic() - session_started >= limits.wall_timeout_seconds:
                return self._finish(state, "stopped", "wall_timeout", session_started)

            try:
                state = await self._call_model(state, session_started)
            except TimeoutError:
                return self._finish(state, "stopped", "wall_timeout", session_started)

    async def run(self, task: str) -> AgentRunResult:
        """Start a new run whose event log must be empty."""

        if not task.strip():
            raise ValueError("task must not be empty")
        if self.event_log.event_count:
            raise ValueError("a new run requires an empty event log")
        initial_messages = (
            AgentMessage(role="system", content=self.system_prompt),
            AgentMessage(role="user", content=task),
        )
        config = RunConfigSnapshot.create(
            task=task,
            initial_messages=initial_messages,
            model_fingerprint=self.model.configuration_fingerprint,
            tool_fingerprint=self.tools.configuration_fingerprint,
            limits=self.limits,
            permissions=self.tools.permissions,
            context_config=(
                None
                if self.context_compiler is None
                else self.context_compiler.config.to_dict()
            ),
        )
        state = self._append(
            None,
            "run.started",
            {"status": "running", "config": config.to_dict()},
        )
        return await self._run_guarded(state)

    async def resume(
        self,
        *,
        pending_tool_resolution: PendingToolResolution | None = None,
    ) -> AgentRunResult:
        """Resume a non-terminal schema-v2 run after validating its configuration."""

        if not self.event_log.event_count:
            raise ValueError("cannot resume an empty event log")
        events = read_events(self.event_log.path)
        state = replay_events_with_checkpoint(events, self.checkpoint_path)
        if state.phase == "terminal":
            raise ValueError("terminal runs cannot be resumed")
        if self.model.configuration_fingerprint != state.config.model_fingerprint:
            raise ValueError("model configuration does not match the original run")
        if self.tools.configuration_fingerprint != state.config.tool_fingerprint:
            raise ValueError("tool configuration does not match the original run")
        if self.limits != state.config.limits:
            raise ValueError("run limits do not match the original run")
        if self.tools.permissions != state.config.permissions:
            raise ValueError("run permissions do not match the original run")
        context_fingerprint = (
            None
            if self.context_compiler is None
            else self.context_compiler.configuration_fingerprint
        )
        if context_fingerprint != state.config.context_fingerprint:
            raise ValueError(
                "context compiler configuration does not match the original run"
            )

        # Restore adapter-local cursors before writing any resume marker so a
        # rejected adapter cannot leave the durable run in a changed phase.
        self.model.resume_from_turn(state.turn)
        if self._budget_event_missing(events):
            state = self._append_budget(state)
        if state.phase == "running":
            state = self._append(
                state,
                "run.interrupted",
                {"status": "interrupted", "reason": "unclean_process_exit"},
            )
        if state.phase in {"paused", "interrupted"}:
            resume_data: dict[str, object] = {
                "config_sha256": state.config.config_sha256,
                "model_fingerprint": self.model.configuration_fingerprint,
                "tool_fingerprint": self.tools.configuration_fingerprint,
                "reason": "operator_resume",
            }
            if self.context_compiler is not None:
                resume_data["context_fingerprint"] = (
                    self.context_compiler.configuration_fingerprint
                )
            state = self._append(
                state,
                "run.resumed",
                resume_data,
            )
        return await self._run_guarded(
            state,
            pending_tool_resolution=pending_tool_resolution,
        )

    async def _run_guarded(
        self,
        state: RunProjection,
        *,
        pending_tool_resolution: PendingToolResolution | None = None,
    ) -> AgentRunResult:
        session_started = monotonic()
        try:
            return await self._drive(
                state,
                session_started=session_started,
                pending_tool_resolution=pending_tool_resolution,
            )
        except asyncio.CancelledError:
            current = self._projection()
            if current.phase == "running":
                self._append(
                    current,
                    "run.interrupted",
                    {"status": "interrupted", "reason": "cancelled"},
                )
            self.event_log.close()
            raise
        except ModelError as exc:
            current = self._projection()
            current = self._append(
                current,
                "model.failed",
                {
                    "code": exc.code,
                    "retryable": exc.retryable,
                    "message": str(exc),
                    **(
                        {"turn": current.pending_model.turn}
                        if current.pending_model is not None
                        else {}
                    ),
                },
            )
            return self._finish(
                current,
                "failed",
                f"model_error:{exc.code}",
                session_started,
            )
        except ContextBudgetError as exc:
            current = self._projection()
            current = self._append(
                current,
                "runtime.failed",
                {"error_type": type(exc).__name__, "message": str(exc)},
            )
            return self._finish(
                current,
                "stopped",
                "context_budget_exceeded",
                session_started,
            )
        except (
            RecoveryError,
            RuntimeContractError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            current = self._projection()
            current = self._append(
                current,
                "runtime.failed",
                {"error_type": type(exc).__name__, "message": str(exc)},
            )
            return self._finish(
                current, "failed", "runtime_contract_error", session_started
            )


class _StopRun(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def require_completed(result: AgentRunResult) -> AgentRunResult:
    """Narrow a result for callers that intentionally require normal model stop."""

    if result.status != "completed":
        raise RuntimeError(f"agent run did not complete: {result.reason}")
    return result
