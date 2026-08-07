from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from contextopt.runtime.protocol import (
    AgentMessage,
    RunLimits,
    RunPermissions,
    TokenUsage,
    ToolCall,
    ToolOutcome,
)
from contextopt.runtime.recovery import (
    GENESIS_EVENT_SHA256,
    RecoveryError,
    RunConfigSnapshot,
    canonical_sha256,
    load_projection_checkpoint,
    replay_events,
    replay_events_with_checkpoint,
    replay_from_checkpoint,
    tool_call_fingerprint,
    write_projection_checkpoint,
)
from contextopt.runtime.tool_state import ToolExecutionPlan


class _Trace:
    def __init__(self, run_id: str = "recovery-run") -> None:
        self.run_id = run_id
        self.events: list[dict[str, Any]] = []
        self.previous_sha256 = GENESIS_EVENT_SHA256

    def append(self, event_type: str, data: Mapping[str, Any]) -> dict[str, Any]:
        unsigned: dict[str, Any] = {
            "schema_version": "2",
            "run_id": self.run_id,
            "seq": len(self.events),
            "timestamp": "2026-08-08T00:00:00.000Z",
            "type": event_type,
            "data": dict(data),
            "prev_event_sha256": self.previous_sha256,
        }
        event = {**unsigned, "event_sha256": canonical_sha256(unsigned)}
        self.events.append(event)
        self.previous_sha256 = event["event_sha256"]
        return event


def _config() -> RunConfigSnapshot:
    return RunConfigSnapshot.create(
        task="Read one file and report the evidence.",
        initial_messages=(
            AgentMessage(role="system", content="Be careful."),
            AgentMessage(role="user", content="Read note.txt."),
        ),
        model_fingerprint=canonical_sha256({"model": "scripted:v2"}),
        tool_fingerprint=canonical_sha256({"tools": ["read_file"]}),
        limits=RunLimits(max_turns=4, max_tool_calls=4),
        permissions=RunPermissions(),
    )


def _start(trace: _Trace, config: RunConfigSnapshot) -> None:
    trace.append(
        "run.started",
        {"status": "running", "config": config.to_dict()},
    )


def _request(trace: _Trace, turn: int, message_count: int) -> None:
    trace.append(
        "model.requested",
        {
            "turn": turn,
            "request_sha256": canonical_sha256({"turn": turn}),
            "message_count": message_count,
            "max_output_tokens": 64,
        },
    )


def _response(
    trace: _Trace,
    turn: int,
    *,
    content: str,
    calls: tuple[ToolCall, ...] = (),
    usage: TokenUsage | None = None,
) -> None:
    trace.append(
        "model.responded",
        {
            "turn": turn,
            "content": content,
            "tool_calls": [call.to_dict() for call in calls],
            "finish_reason": "tool_calls" if calls else "stop",
            "usage": (usage or TokenUsage()).to_dict(),
            "response_id": f"response-{turn}",
        },
    )


def _plan(call: ToolCall, operation_id: str, outcome: ToolOutcome) -> ToolExecutionPlan:
    return ToolExecutionPlan(
        operation_id=operation_id,
        call=call,
        fingerprint=tool_call_fingerprint(call),
        replay_policy="safe",
        planned_outcome=outcome,
    )


def _tool_started(
    trace: _Trace,
    call: ToolCall,
    *,
    operation_id: str,
    turn: int,
    call_index: int,
    plan: ToolExecutionPlan | None,
) -> None:
    trace.append(
        "tool.started",
        {
            "operation_id": operation_id,
            "turn": turn,
            "call_index": call_index,
            "fingerprint": tool_call_fingerprint(call),
            "call_id": call.id,
            "tool_name": call.name,
            "arguments_json": call.arguments_json,
            "plan": None if plan is None else plan.to_dict(),
        },
    )


def _tool_result(
    trace: _Trace,
    event_type: str,
    call: ToolCall,
    outcome: ToolOutcome,
    *,
    operation_id: str,
    turn: int,
    call_index: int,
) -> None:
    trace.append(
        event_type,
        {
            "operation_id": operation_id,
            "turn": turn,
            "call_index": call_index,
            "fingerprint": tool_call_fingerprint(call),
            **outcome.to_dict(),
        },
    )


def _terminal(
    trace: _Trace,
    event_type: str,
    *,
    status: str,
    reason: str,
    turns: int,
    tool_calls: int,
    usage: TokenUsage,
    final_text: str,
) -> None:
    trace.append(
        event_type,
        {
            "status": status,
            "reason": reason,
            "turns": turns,
            "tool_calls": tool_calls,
            "usage": usage.to_dict(),
            "final_text": final_text,
        },
    )


class RecoveryProjectionTests(unittest.TestCase):
    def test_full_replay_reconstructs_messages_usage_cache_and_terminal(self) -> None:
        config = _config()
        trace = _Trace()
        _start(trace, config)
        _request(trace, 1, 2)
        call = ToolCall("read-note", "read_file", '{"path":"note.txt"}')
        first_usage = TokenUsage(input_tokens=7, output_tokens=3)
        _response(
            trace,
            1,
            content="I will read the note.",
            calls=(call,),
            usage=first_usage,
        )
        outcome = ToolOutcome(
            call_id=call.id,
            tool_name=call.name,
            ok=True,
            content="path=note.txt\n1: evidence",
            metadata={"sha256": "a" * 64},
        )
        plan = _plan(call, "operation-read", outcome)
        _tool_started(
            trace,
            call,
            operation_id=plan.operation_id,
            turn=1,
            call_index=0,
            plan=plan,
        )
        _tool_result(
            trace,
            "tool.completed",
            call,
            outcome,
            operation_id=plan.operation_id,
            turn=1,
            call_index=0,
        )
        trace.append("run.paused", {"status": "paused", "reason": "operator"})

        paused = replay_events(trace.events)

        self.assertEqual(paused.phase, "paused")
        self.assertEqual(paused.turn, 1)
        self.assertEqual(paused.tool_calls, 1)
        self.assertEqual(paused.usage, first_usage)
        self.assertEqual(
            [message.role for message in paused.messages],
            ["system", "user", "assistant", "tool"],
        )
        self.assertEqual(paused.pending_tools, ())
        self.assertIsNone(paused.pending_model)
        self.assertEqual(paused.completed_calls[call.id].outcome, outcome)

        trace.append(
            "run.resumed",
            {
                "config_sha256": config.config_sha256,
                "model_fingerprint": config.model_fingerprint,
                "tool_fingerprint": config.tool_fingerprint,
            },
        )
        _request(trace, 2, 4)
        second_usage = TokenUsage(input_tokens=5, output_tokens=2)
        _response(
            trace,
            2,
            content="The evidence was read.",
            usage=second_usage,
        )
        total_usage = first_usage + second_usage
        _terminal(
            trace,
            "run.completed",
            status="completed",
            reason="model_stopped",
            turns=2,
            tool_calls=1,
            usage=total_usage,
            final_text="The evidence was read.",
        )

        finished = replay_events(trace.events)

        self.assertEqual(finished.phase, "terminal")
        self.assertIsNotNone(finished.terminal)
        assert finished.terminal is not None
        self.assertEqual(finished.terminal.status, "completed")
        self.assertEqual(finished.terminal.final_text, "The evidence was read.")
        self.assertEqual(finished.usage, total_usage)
        self.assertEqual(finished.through_seq, len(trace.events) - 1)
        self.assertEqual(
            finished.through_event_sha256,
            trace.events[-1]["event_sha256"],
        )

    def test_pending_plan_survives_interrupt_pause_and_checkpoint(self) -> None:
        config = _config()
        trace = _Trace("pending-run")
        _start(trace, config)
        _request(trace, 1, 2)
        call = ToolCall("create", "create_file", '{"path":"x","content":"y"}')
        _response(trace, 1, content="Creating the file.", calls=(call,))
        outcome = ToolOutcome(call.id, call.name, True, "created x")
        plan = ToolExecutionPlan(
            operation_id="operation-create",
            call=call,
            fingerprint=tool_call_fingerprint(call),
            replay_policy="reconcile",
            preconditions={"exists": False},
            postconditions={"sha256": "b" * 64},
            planned_outcome=outcome,
        )
        _tool_started(
            trace,
            call,
            operation_id=plan.operation_id,
            turn=1,
            call_index=0,
            plan=plan,
        )
        trace.append("run.interrupted", {"reason": "process_lost"})
        trace.append("run.paused", {"reason": "manual_reconciliation"})

        projection = replay_events(trace.events)

        self.assertEqual(projection.phase, "paused")
        self.assertEqual(len(projection.pending_tools), 1)
        pending = projection.pending_tools[0]
        self.assertTrue(pending.started)
        self.assertEqual(pending.operation_id, plan.operation_id)
        self.assertEqual(pending.plan, plan)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "checkpoint.json"
            written = write_projection_checkpoint(checkpoint_path, projection)
            loaded = load_projection_checkpoint(
                checkpoint_path,
                through_event=trace.events[-1],
            )

            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.state, projection)
            self.assertEqual(loaded.state.pending_tools[0].plan, plan)
            self.assertEqual(written.state_sha256, loaded.state_sha256)

    def test_checkpoint_suffix_matches_full_replay_and_corruption_is_unavailable(
        self,
    ) -> None:
        config = _config()
        trace = _Trace("checkpoint-run")
        _start(trace, config)
        _request(trace, 1, 2)
        _response(trace, 1, content="Done.", usage=TokenUsage(3, 2))
        prefix = tuple(trace.events)
        prefix_state = replay_events(prefix)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "projection.json"
            checkpoint = write_projection_checkpoint(checkpoint_path, prefix_state)
            self.assertEqual(
                [
                    path
                    for path in checkpoint_path.parent.iterdir()
                    if path.suffix == ".tmp"
                ],
                [],
            )

            _terminal(
                trace,
                "run.completed",
                status="completed",
                reason="model_stopped",
                turns=1,
                tool_calls=0,
                usage=TokenUsage(3, 2),
                final_text="Done.",
            )
            full = replay_events(trace.events)
            suffix = replay_from_checkpoint(checkpoint, trace.events[len(prefix) :])
            cached = replay_events_with_checkpoint(trace.events, checkpoint_path)
            self.assertEqual(suffix, full)
            self.assertEqual(cached, full)

            decoded = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            decoded["state"]["turn"] = 99
            checkpoint_path.write_text(json.dumps(decoded), encoding="utf-8")
            self.assertIsNone(load_projection_checkpoint(checkpoint_path))
            self.assertEqual(
                replay_events_with_checkpoint(trace.events, checkpoint_path),
                full,
            )

            write_projection_checkpoint(checkpoint_path, prefix_state)
            forged = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            forged["state"]["usage"] = TokenUsage(0, 0).to_dict()
            forged["state_sha256"] = canonical_sha256(forged["state"])
            checkpoint_path.write_text(json.dumps(forged), encoding="utf-8")
            self.assertIsNotNone(
                load_projection_checkpoint(
                    checkpoint_path,
                    through_event=prefix[-1],
                )
            )
            self.assertEqual(
                replay_events_with_checkpoint(trace.events, checkpoint_path),
                full,
            )

    def test_illegal_transitions_and_out_of_order_tools_are_rejected(self) -> None:
        config = _config()
        trace = _Trace("strict-run")
        _start(trace, config)
        _request(trace, 1, 2)
        first = ToolCall("first", "read_file", '{"path":"a"}')
        second = ToolCall("second", "read_file", '{"path":"b"}')
        _response(trace, 1, content="Read both.", calls=(first, second))
        outcome = ToolOutcome(second.id, second.name, True, "b")
        second_plan = _plan(second, "operation-second", outcome)
        _tool_started(
            trace,
            second,
            operation_id=second_plan.operation_id,
            turn=1,
            call_index=1,
            plan=second_plan,
        )

        with self.assertRaisesRegex(RecoveryError, "model-declared order"):
            replay_events(trace.events)

        response_without_request = _Trace("response-first")
        _start(response_without_request, config)
        _response(response_without_request, 1, content="invalid")
        with self.assertRaisesRegex(RecoveryError, "pending model request"):
            replay_events(response_without_request.events)

        terminal = _Trace("after-terminal")
        _start(terminal, config)
        _request(terminal, 1, 2)
        _response(terminal, 1, content="Done.")
        _terminal(
            terminal,
            "run.completed",
            status="completed",
            reason="model_stopped",
            turns=1,
            tool_calls=0,
            usage=TokenUsage(),
            final_text="Done.",
        )
        terminal.append("run.resumed", {"config_sha256": config.config_sha256})
        with self.assertRaisesRegex(RecoveryError, "after a terminal"):
            replay_events(terminal.events)

    def test_failure_audit_events_can_precede_terminal_failure(self) -> None:
        config = _config()
        trace = _Trace("failure-run")
        _start(trace, config)
        _request(trace, 1, 2)
        trace.append(
            "model.failed",
            {
                "turn": 1,
                "code": "network_error",
                "retryable": True,
                "message": "connection lost",
            },
        )
        trace.append(
            "runtime.failed",
            {"error_type": "ModelError", "message": "retry budget exhausted"},
        )
        _terminal(
            trace,
            "run.failed",
            status="failed",
            reason="model_error:network_error",
            turns=0,
            tool_calls=0,
            usage=TokenUsage(),
            final_text="",
        )

        projection = replay_events(trace.events)

        self.assertEqual(projection.phase, "terminal")
        self.assertIsNone(projection.pending_model)
        assert projection.terminal is not None
        self.assertEqual(projection.terminal.status, "failed")

    def test_null_plan_preflight_failure_is_cached_after_started_event(self) -> None:
        config = _config()
        trace = _Trace("preflight-failure")
        _start(trace, config)
        _request(trace, 1, 2)
        call = ToolCall("invalid-read", "read_file", "not-json")
        _response(trace, 1, content="Trying the read.", calls=(call,))
        _tool_started(
            trace,
            call,
            operation_id="operation-invalid-read",
            turn=1,
            call_index=0,
            plan=None,
        )
        outcome = ToolOutcome(
            call_id=call.id,
            tool_name=call.name,
            ok=False,
            content="arguments are not valid JSON",
            error_code="invalid_arguments",
        )
        _tool_result(
            trace,
            "tool.failed",
            call,
            outcome,
            operation_id="operation-invalid-read",
            turn=1,
            call_index=0,
        )

        projection = replay_events(trace.events)

        self.assertEqual(projection.tool_calls, 1)
        self.assertEqual(projection.pending_tools, ())
        self.assertEqual(projection.completed_calls[call.id].outcome, outcome)

    def test_protocol_from_dict_is_strict(self) -> None:
        message = AgentMessage(role="assistant", content="ok")
        self.assertEqual(AgentMessage.from_dict(message.to_dict()), message)

        invalid_message = {**message.to_dict(), "unexpected": True}
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            AgentMessage.from_dict(invalid_message)

        invalid_usage = TokenUsage(2, 3).to_dict()
        invalid_usage["total_tokens"] = 99
        with self.assertRaisesRegex(ValueError, "does not match"):
            TokenUsage.from_dict(invalid_usage)


if __name__ == "__main__":
    unittest.main()
