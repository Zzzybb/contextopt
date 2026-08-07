from __future__ import annotations

import asyncio
import hashlib
import sys
import tempfile
import time
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

from contextopt.runtime import (
    AgentRunner,
    EventLog,
    RunLimits,
    RunPermissions,
    ScriptedModel,
    WorkspaceTools,
    read_events,
)
from contextopt.runtime.events import RunEvent, RunLeaseError
from contextopt.runtime.recovery import load_projection_checkpoint


class _InjectedCrash(BaseException):
    pass


class _FaultEventLog(EventLog):
    def __init__(
        self,
        path: str | Path,
        run_id: str,
        *,
        event_type: str,
        timing: Literal["before", "after"],
        occurrence: int = 1,
    ) -> None:
        super().__init__(path, run_id)
        self._crash_event_type = event_type
        self._timing = timing
        self._remaining = occurrence

    def append(self, event_type: str, data: Mapping[str, Any]) -> RunEvent:
        should_crash = event_type == self._crash_event_type
        if should_crash:
            self._remaining -= 1
            should_crash = self._remaining == 0
        if should_crash and self._timing == "before":
            raise _InjectedCrash(f"crash before {event_type}")
        event = super().append(event_type, data)
        if should_crash:
            raise _InjectedCrash(f"crash after {event_type}")
        return event


def _limits(**overrides: Any) -> RunLimits:
    values: dict[str, Any] = {
        "max_turns": 6,
        "max_tool_calls": 6,
        "max_total_tokens": 500,
        "max_output_tokens_per_call": 64,
        "wall_timeout_seconds": 30.0,
        "command_timeout_seconds": 10.0,
        "max_tool_output_bytes": 64 * 1024,
    }
    values.update(overrides)
    return RunLimits(**values)


def _scripted(steps: Sequence[Mapping[str, Any]]) -> ScriptedModel:
    return ScriptedModel(steps, name="resume-script:v1")


def _runner(
    *,
    model: ScriptedModel,
    tools: WorkspaceTools,
    event_log: EventLog,
    limits: RunLimits,
) -> AgentRunner:
    return AgentRunner(
        model=model,
        tools=tools,
        event_log=event_log,
        limits=limits,
    )


def _event_types(path: Path) -> list[str]:
    return [str(event["type"]) for event in read_events(path)]


def _assert_checkpoint_covers_log(test: unittest.TestCase, event_path: Path) -> None:
    events = read_events(event_path)
    checkpoint = load_projection_checkpoint(
        event_path.with_name(event_path.name + ".checkpoint.json"),
        through_event=events[-1],
    )
    test.assertIsNotNone(checkpoint)
    assert checkpoint is not None
    test.assertEqual(checkpoint.through_seq, len(events) - 1)


class RuntimeResumeTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancellation_holds_lease_until_tool_result_is_durable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            workspace.mkdir()
            event_path = root / "cancel-events.jsonl"
            limits = _limits(command_timeout_seconds=5.0)
            permissions = RunPermissions(allow_command=True)
            script = [
                {
                    "response": {
                        "tool_calls": [
                            {
                                "id": "slow-command",
                                "name": "run_tests",
                                "arguments": {"scope": "visible"},
                            }
                        ]
                    }
                },
                {"response": {"content": "must not be requested"}},
            ]
            event_log = EventLog(event_path, "cancel-settle")
            runner = _runner(
                model=_scripted(script),
                tools=WorkspaceTools(
                    workspace,
                    permissions=permissions,
                    limits=limits,
                    test_commands={
                        "visible": (
                            sys.executable,
                            "-c",
                            "import time; time.sleep(0.4); print('settled')",
                        )
                    },
                ),
                event_log=event_log,
                limits=limits,
            )
            running = asyncio.create_task(runner.run("Run the slow visible command."))
            for _ in range(100):
                if "tool.started" in _event_types(event_path):
                    break
                await asyncio.sleep(0.01)
            self.assertIn("tool.started", _event_types(event_path))

            cancelled_at = time.monotonic()
            running.cancel()
            await asyncio.sleep(0.05)
            with self.assertRaises(RunLeaseError):
                EventLog(event_path, "cancel-settle", repair_truncated=True)

            with self.assertRaises(asyncio.CancelledError):
                await running
            self.assertGreaterEqual(time.monotonic() - cancelled_at, 0.25)

            event_types = _event_types(event_path)
            self.assertLess(
                event_types.index("tool.completed"),
                event_types.index("run.interrupted"),
            )
            reopened = EventLog(
                event_path,
                "cancel-settle",
                repair_truncated=True,
            )
            reopened.close()

    async def test_checkpoint_io_failure_falls_back_to_event_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            workspace.mkdir()
            event_path = root / "checkpoint-failure-events.jsonl"
            limits = _limits()
            model = _scripted(
                [
                    {
                        "response": {
                            "content": "The event log is authoritative.",
                            "usage": {"input_tokens": 4, "output_tokens": 3},
                        }
                    }
                ]
            )
            event_log = EventLog(event_path, "checkpoint-failure")

            with patch(
                "contextopt.runtime.runner.write_projection_checkpoint",
                side_effect=OSError("simulated cache write failure"),
            ):
                result = await _runner(
                    model=model,
                    tools=WorkspaceTools(workspace, limits=limits),
                    event_log=event_log,
                    limits=limits,
                ).run("Complete without relying on a checkpoint cache.")

            self.assertEqual(result.status, "completed")
            self.assertEqual(result.usage.total_tokens, 7)
            self.assertFalse(
                event_path.with_name(event_path.name + ".checkpoint.json").exists()
            )
            events = read_events(event_path)
            self.assertEqual(events[-1]["type"], "run.completed")

    async def test_read_only_started_crash_retries_and_completes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "note.txt").write_text("durable evidence\n", encoding="utf-8")
            event_path = root / "read-events.jsonl"
            limits = _limits()
            script: list[dict[str, Any]] = [
                {
                    "response": {
                        "content": "I will read the file.",
                        "tool_calls": [
                            {
                                "id": "read-note",
                                "name": "read_file",
                                "arguments": {"path": "note.txt"},
                            }
                        ],
                        "usage": {"input_tokens": 7, "output_tokens": 3},
                    }
                },
                {
                    "expect": {
                        "turn": 2,
                        "last_tool": "read_file",
                        "tool_call_id": "read-note",
                        "observation_contains": "durable evidence",
                    },
                    "response": {
                        "content": "The durable evidence was recovered.",
                        "usage": {"input_tokens": 11, "output_tokens": 4},
                    },
                },
            ]
            first_model = _scripted(script)
            first_tools = WorkspaceTools(workspace, limits=limits)
            crashed_log = _FaultEventLog(
                event_path,
                "read-resume",
                event_type="tool.started",
                timing="after",
            )
            try:
                with self.assertRaises(_InjectedCrash):
                    await _runner(
                        model=first_model,
                        tools=first_tools,
                        event_log=crashed_log,
                        limits=limits,
                    ).run("Read note.txt and summarize it.")
            finally:
                crashed_log.close()

            stale_checkpoint = load_projection_checkpoint(
                event_path.with_name(event_path.name + ".checkpoint.json")
            )
            self.assertIsNotNone(stale_checkpoint)
            assert stale_checkpoint is not None
            self.assertLess(
                stale_checkpoint.through_seq,
                len(read_events(event_path)) - 1,
            )

            resumed_model = _scripted(script)
            resumed_log = EventLog(
                event_path,
                "read-resume",
                repair_truncated=True,
            )
            resumed_tools = WorkspaceTools(workspace, limits=limits)
            result = await _runner(
                model=resumed_model,
                tools=resumed_tools,
                event_log=resumed_log,
                limits=limits,
            ).resume()

            self.assertEqual(result.status, "completed")
            self.assertEqual(result.turns, 2)
            self.assertEqual(result.tool_calls, 1)
            self.assertEqual(result.usage.input_tokens, 18)
            self.assertEqual(result.usage.output_tokens, 7)
            self.assertEqual(len(first_model.requests), 1)
            self.assertEqual(len(resumed_model.requests), 1)
            self.assertEqual(resumed_model.requests[0].turn, 2)
            event_types = _event_types(event_path)
            self.assertEqual(event_types.count("tool.started"), 1)
            self.assertEqual(event_types.count("tool.completed"), 1)
            self.assertIn("run.interrupted", event_types)
            self.assertIn("run.resumed", event_types)
            self.assertLess(
                event_types.index("run.interrupted"),
                event_types.index("run.resumed"),
            )
            _assert_checkpoint_covers_log(self, event_path)

    async def test_replace_postcondition_is_reconciled_without_second_write(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            workspace.mkdir()
            target = workspace / "value.txt"
            target.write_text("before\n", encoding="utf-8")
            before_sha256 = hashlib.sha256(target.read_bytes()).hexdigest()
            event_path = root / "replace-events.jsonl"
            limits = _limits()
            script: list[dict[str, Any]] = [
                {
                    "response": {
                        "content": "I will update the value.",
                        "tool_calls": [
                            {
                                "id": "replace-value",
                                "name": "replace_text",
                                "arguments": {
                                    "path": "value.txt",
                                    "old_text": "before",
                                    "new_text": "after",
                                    "expected_sha256": before_sha256,
                                    "expected_occurrences": 1,
                                },
                            }
                        ],
                        "usage": {"input_tokens": 8, "output_tokens": 3},
                    }
                },
                {
                    "expect": {
                        "turn": 2,
                        "last_tool": "replace_text",
                        "tool_call_id": "replace-value",
                        "observation_contains": "replaced 1 occurrence(s)",
                    },
                    "response": {
                        "content": "The value was updated once.",
                        "usage": {"input_tokens": 10, "output_tokens": 4},
                    },
                },
            ]
            permissions = RunPermissions(allow_write=True)
            crashed_log = _FaultEventLog(
                event_path,
                "replace-resume",
                event_type="tool.completed",
                timing="before",
            )
            try:
                with self.assertRaises(_InjectedCrash):
                    await _runner(
                        model=_scripted(script),
                        tools=WorkspaceTools(
                            workspace,
                            permissions=permissions,
                            limits=limits,
                        ),
                        event_log=crashed_log,
                        limits=limits,
                    ).run("Replace before with after in value.txt.")
            finally:
                crashed_log.close()

            self.assertEqual(target.read_text(encoding="utf-8"), "after\n")
            modified_after_crash = target.stat().st_mtime_ns

            resumed_model = _scripted(script)
            resumed_log = EventLog(
                event_path,
                "replace-resume",
                repair_truncated=True,
            )
            result = await _runner(
                model=resumed_model,
                tools=WorkspaceTools(
                    workspace,
                    permissions=permissions,
                    limits=limits,
                ),
                event_log=resumed_log,
                limits=limits,
            ).resume()

            self.assertEqual(result.status, "completed")
            self.assertEqual(result.tool_calls, 1)
            self.assertEqual(result.usage.total_tokens, 25)
            self.assertEqual(target.read_text(encoding="utf-8"), "after\n")
            self.assertEqual(target.stat().st_mtime_ns, modified_after_crash)
            events = read_events(event_path)
            tool_results = [
                event
                for event in events
                if event["type"] in {"tool.completed", "tool.failed"}
            ]
            self.assertEqual(len(tool_results), 1)
            self.assertEqual(tool_results[0]["type"], "tool.completed")
            self.assertEqual(
                tool_results[0]["data"]["metadata"]["before_sha256"],
                before_sha256,
            )
            self.assertIn("run.interrupted", _event_types(event_path))
            self.assertIn("run.resumed", _event_types(event_path))
            _assert_checkpoint_covers_log(self, event_path)

    async def test_run_tests_crash_pauses_then_mark_failed_continues(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            workspace.mkdir()
            counter_script = workspace / "counter.py"
            counter_script.write_text(
                "from pathlib import Path\n"
                "path = Path('command-count.txt')\n"
                "count = int(path.read_text()) if path.exists() else 0\n"
                "path.write_text(str(count + 1))\n",
                encoding="utf-8",
            )
            counter = workspace / "command-count.txt"
            test_commands = {"visible": (sys.executable, counter_script.as_posix())}
            event_path = root / "command-events.jsonl"
            limits = _limits()
            permissions = RunPermissions(allow_command=True)
            script: list[dict[str, Any]] = [
                {
                    "response": {
                        "content": "I will run the visible tests.",
                        "tool_calls": [
                            {
                                "id": "run-visible",
                                "name": "run_tests",
                                "arguments": {"scope": "visible"},
                            }
                        ],
                        "usage": {"input_tokens": 6, "output_tokens": 2},
                    }
                },
                {
                    "expect": {
                        "turn": 2,
                        "last_tool": "run_tests",
                        "tool_call_id": "run-visible",
                        "observation_contains": "indeterminate_previous_execution",
                    },
                    "response": {
                        "content": "The interrupted command was marked failed.",
                        "usage": {"input_tokens": 9, "output_tokens": 3},
                    },
                },
            ]
            crashed_log = _FaultEventLog(
                event_path,
                "command-resume",
                event_type="tool.started",
                timing="after",
            )
            try:
                with self.assertRaises(_InjectedCrash):
                    await _runner(
                        model=_scripted(script),
                        tools=WorkspaceTools(
                            workspace,
                            permissions=permissions,
                            limits=limits,
                            test_commands=test_commands,
                        ),
                        event_log=crashed_log,
                        limits=limits,
                    ).run("Run the visible tests.")
            finally:
                crashed_log.close()

            self.assertFalse(counter.exists())

            paused_model = _scripted(script)
            paused_log = EventLog(
                event_path,
                "command-resume",
                repair_truncated=True,
            )
            paused = await _runner(
                model=paused_model,
                tools=WorkspaceTools(
                    workspace,
                    permissions=permissions,
                    limits=limits,
                    test_commands=test_commands,
                ),
                event_log=paused_log,
                limits=limits,
            ).resume()

            self.assertEqual(paused.status, "paused")
            self.assertIn("pending_tool_paused", paused.reason)
            self.assertEqual(paused.turns, 1)
            self.assertEqual(paused.tool_calls, 1)
            self.assertEqual(paused.usage.total_tokens, 8)
            self.assertEqual(paused_model.requests, [])
            self.assertFalse(counter.exists())

            final_model = _scripted(script)
            final_log = EventLog(
                event_path,
                "command-resume",
                repair_truncated=True,
            )
            completed = await _runner(
                model=final_model,
                tools=WorkspaceTools(
                    workspace,
                    permissions=permissions,
                    limits=limits,
                    test_commands=test_commands,
                ),
                event_log=final_log,
                limits=limits,
            ).resume(pending_tool_resolution="mark_failed")

            self.assertEqual(completed.status, "completed")
            self.assertEqual(completed.tool_calls, 1)
            self.assertEqual(completed.usage.total_tokens, 20)
            self.assertEqual(len(final_model.requests), 1)
            self.assertFalse(counter.exists())
            event_types = _event_types(event_path)
            self.assertEqual(event_types.count("tool.started"), 1)
            self.assertEqual(event_types.count("tool.failed"), 1)
            self.assertEqual(event_types.count("run.paused"), 1)
            self.assertEqual(event_types.count("run.resumed"), 2)
            self.assertIn("run.interrupted", event_types)
            _assert_checkpoint_covers_log(self, event_path)

    async def test_final_response_crash_only_appends_terminal_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            workspace.mkdir()
            event_path = root / "final-events.jsonl"
            limits = _limits()
            script: list[dict[str, Any]] = [
                {
                    "response": {
                        "content": "The task is already complete.",
                        "usage": {"input_tokens": 5, "output_tokens": 3},
                    }
                }
            ]
            crashed_log = _FaultEventLog(
                event_path,
                "final-resume",
                event_type="run.completed",
                timing="before",
            )
            first_model = _scripted(script)
            try:
                with self.assertRaises(_InjectedCrash):
                    await _runner(
                        model=first_model,
                        tools=WorkspaceTools(workspace, limits=limits),
                        event_log=crashed_log,
                        limits=limits,
                    ).run("Finish without tools.")
            finally:
                crashed_log.close()

            resumed_model = _scripted(script)
            resumed_log = EventLog(
                event_path,
                "final-resume",
                repair_truncated=True,
            )
            result = await _runner(
                model=resumed_model,
                tools=WorkspaceTools(workspace, limits=limits),
                event_log=resumed_log,
                limits=limits,
            ).resume()

            self.assertEqual(result.status, "completed")
            self.assertEqual(result.turns, 1)
            self.assertEqual(result.tool_calls, 0)
            self.assertEqual(result.usage.total_tokens, 8)
            self.assertEqual(len(first_model.requests), 1)
            self.assertEqual(resumed_model.requests, [])
            event_types = _event_types(event_path)
            self.assertEqual(event_types.count("model.requested"), 1)
            self.assertEqual(event_types.count("model.responded"), 1)
            self.assertEqual(event_types.count("run.completed"), 1)
            self.assertIn("run.interrupted", event_types)
            self.assertIn("run.resumed", event_types)
            _assert_checkpoint_covers_log(self, event_path)

    async def test_resume_rejects_model_tool_and_limit_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "note.txt").write_text("evidence", encoding="utf-8")
            event_path = root / "mismatch-events.jsonl"
            limits = _limits()
            script: list[dict[str, Any]] = [
                {
                    "response": {
                        "tool_calls": [
                            {
                                "id": "read-note",
                                "name": "read_file",
                                "arguments": {"path": "note.txt"},
                            }
                        ]
                    }
                },
                {"response": {"content": "done"}},
            ]
            crashed_log = _FaultEventLog(
                event_path,
                "mismatch-resume",
                event_type="tool.started",
                timing="after",
            )
            try:
                with self.assertRaises(_InjectedCrash):
                    await _runner(
                        model=_scripted(script),
                        tools=WorkspaceTools(workspace, limits=limits),
                        event_log=crashed_log,
                        limits=limits,
                    ).run("Read the note.")
            finally:
                crashed_log.close()

            mismatched_script = [*script, {"response": {"content": "extra"}}]
            model_log = EventLog(
                event_path,
                "mismatch-resume",
                repair_truncated=True,
            )
            try:
                with self.assertRaisesRegex(ValueError, "model configuration"):
                    await _runner(
                        model=_scripted(mismatched_script),
                        tools=WorkspaceTools(workspace, limits=limits),
                        event_log=model_log,
                        limits=limits,
                    ).resume()
            finally:
                model_log.close()

            other_workspace = root / "other-workspace"
            other_workspace.mkdir()
            tool_log = EventLog(
                event_path,
                "mismatch-resume",
                repair_truncated=True,
            )
            try:
                with self.assertRaisesRegex(ValueError, "tool configuration"):
                    await _runner(
                        model=_scripted(script),
                        tools=WorkspaceTools(other_workspace, limits=limits),
                        event_log=tool_log,
                        limits=limits,
                    ).resume()
            finally:
                tool_log.close()

            different_limits = _limits(max_turns=7)
            limit_log = EventLog(
                event_path,
                "mismatch-resume",
                repair_truncated=True,
            )
            try:
                with self.assertRaisesRegex(ValueError, "run limits"):
                    await _runner(
                        model=_scripted(script),
                        tools=WorkspaceTools(
                            workspace,
                            limits=different_limits,
                        ),
                        event_log=limit_log,
                        limits=different_limits,
                    ).resume()
            finally:
                limit_log.close()

            event_types = _event_types(event_path)
            self.assertNotIn("run.interrupted", event_types)
            self.assertNotIn("run.resumed", event_types)


if __name__ == "__main__":
    unittest.main()
