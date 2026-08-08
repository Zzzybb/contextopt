from __future__ import annotations

import asyncio
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from contextopt.cli import main
from contextopt.runtime import (
    AgentRunner,
    EventLog,
    ModelRequest,
    ModelResponse,
    ScriptedModel,
    WorkspaceTools,
    read_events,
)


class _BlockingAfterFirstScriptedModel:
    """Use the production script identity, but simulate a process interruption."""

    def __init__(self, steps: list[dict[str, Any]], *, name: str) -> None:
        self._delegate = ScriptedModel(steps, name=name)
        self._calls = 0
        self.blocked = asyncio.Event()
        self._release = asyncio.Event()

    @property
    def name(self) -> str:
        return self._delegate.name

    @property
    def configuration_fingerprint(self) -> str:
        return self._delegate.configuration_fingerprint

    def resume_from_turn(self, completed_turns: int) -> None:
        self._delegate.resume_from_turn(completed_turns)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self._calls += 1
        if self._calls == 1:
            return await self._delegate.complete(request)
        self.blocked.set()
        await self._release.wait()
        return await self._delegate.complete(request)


class RuntimeCliTests(unittest.TestCase):
    @staticmethod
    def _write_script(root: Path, steps: list[dict[str, Any]]) -> Path:
        path = root / "script.json"
        path.write_text(
            json.dumps({"name": "cli-script", "steps": steps}),
            encoding="utf-8",
        )
        return path

    def _invoke(self, arguments: list[str]) -> tuple[int, str]:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main(arguments)
        return exit_code, stdout.getvalue()

    def _invoke_json(self, arguments: list[str]) -> tuple[int, dict[str, Any]]:
        exit_code, output = self._invoke(arguments)
        payload, end = json.JSONDecoder().raw_decode(output)
        self.assertEqual(output[end:].strip(), "")
        self.assertIsInstance(payload, dict)
        return exit_code, payload

    def test_scripted_run_succeeds_and_trace_is_readable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "note.txt").write_text(
                "the offline evidence\n",
                encoding="utf-8",
            )
            script = self._write_script(
                root,
                [
                    {
                        "response": {
                            "content": "I will read the evidence.",
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
                            "observation_contains": "the offline evidence",
                        },
                        "response": {
                            "content": "The offline evidence was read successfully.",
                            "usage": {"input_tokens": 9, "output_tokens": 4},
                        },
                    },
                ],
            )
            event_log = root / "success-events.jsonl"

            exit_code, result = self._invoke_json(
                [
                    "run",
                    "Read the note and report its evidence.",
                    "--workspace",
                    str(workspace),
                    "--script",
                    str(script),
                    "--run-id",
                    "cli-success",
                    "--event-log",
                    str(event_log),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(result["run_id"], "cli-success")
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["reason"], "model_stopped")
            self.assertEqual(
                result["final_text"],
                "The offline evidence was read successfully.",
            )
            self.assertEqual(result["turns"], 2)
            self.assertEqual(result["tool_calls"], 1)
            self.assertEqual(result["usage"]["total_tokens"], 23)
            self.assertEqual(result["event_log"], event_log.as_posix())
            self.assertTrue(event_log.is_file())
            events = read_events(event_log)
            context_config = events[0]["data"]["config"]["context_config"]
            self.assertEqual(context_config["compiler_version"], 1)
            self.assertEqual(context_config["policy"], "submodular")
            self.assertEqual(context_config["memory_policy"], "versioned-v1")

            trace_code, trace = self._invoke(["trace", str(event_log)])
            trace_html = root / "trace.html"
            html_code, _ = self._invoke(
                ["trace", str(event_log), "--html", str(trace_html)]
            )

            self.assertEqual(trace_code, 0)
            self.assertEqual(html_code, 0)
            self.assertIn("0000 run.started status=running", trace)
            self.assertIn("model.responded turn=1 calls=1", trace)
            self.assertIn("model.requested turn=1 messages=2 policy=submodular", trace)
            self.assertIn(
                "tool.completed tool=read_file call=read-note ok=True",
                trace,
            )
            self.assertIn(
                "run.completed status=completed reason=model_stopped",
                trace,
            )
            html = trace_html.read_text(encoding="utf-8")
            self.assertIn("ContextOpt run trace", html)
            self.assertIn("context receipts", html)
            self.assertIn("model.requested", html)

    def test_write_permission_denial_is_observable_and_recoverable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            workspace.mkdir()
            script = self._write_script(
                root,
                [
                    {
                        "response": {
                            "content": "I will try to create the requested file.",
                            "tool_calls": [
                                {
                                    "id": "write-attempt",
                                    "name": "create_file",
                                    "arguments": {
                                        "path": "created.txt",
                                        "content": "must not be written",
                                    },
                                }
                            ],
                        }
                    },
                    {
                        "expect": {
                            "turn": 2,
                            "last_tool": "create_file",
                            "tool_call_id": "write-attempt",
                            "observation_contains": "write permission is disabled",
                        },
                        "response": {
                            "content": "The write was correctly denied.",
                        },
                    },
                ],
            )

            exit_code, result = self._invoke_json(
                [
                    "run",
                    "Try to create a file without write permission.",
                    "--workspace",
                    str(workspace),
                    "--script",
                    str(script),
                    "--run-id",
                    "cli-read-only",
                    "--event-log",
                    str(root / "permission-events.jsonl"),
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["tool_calls"], 1)
            self.assertEqual(result["final_text"], "The write was correctly denied.")
            self.assertFalse((workspace / "created.txt").exists())

    def test_token_budget_stop_returns_json_and_nonzero_exit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            workspace.mkdir()
            script = self._write_script(
                root,
                [
                    {
                        "response": {
                            "content": "I would create a file next.",
                            "tool_calls": [
                                {
                                    "id": "over-budget-write",
                                    "name": "create_file",
                                    "arguments": {
                                        "path": "blocked.txt",
                                        "content": "must not be written",
                                    },
                                }
                            ],
                            "usage": {"input_tokens": 8, "output_tokens": 3},
                        }
                    }
                ],
            )
            event_log = root / "budget-events.jsonl"

            exit_code, result = self._invoke_json(
                [
                    "run",
                    "Create a file only if the token budget permits it.",
                    "--workspace",
                    str(workspace),
                    "--script",
                    str(script),
                    "--allow-write",
                    "--max-total-tokens",
                    "10",
                    "--run-id",
                    "cli-budget-stop",
                    "--event-log",
                    str(event_log),
                ]
            )

            self.assertEqual(exit_code, 2)
            self.assertEqual(result["status"], "stopped")
            self.assertEqual(result["reason"], "token_limit_after_response")
            self.assertEqual(result["turns"], 1)
            self.assertEqual(result["tool_calls"], 0)
            self.assertEqual(result["usage"]["total_tokens"], 11)
            self.assertFalse((workspace / "blocked.txt").exists())

            trace_code, trace = self._invoke(["trace", str(event_log)])
            self.assertEqual(trace_code, 0)
            self.assertNotIn("tool.started", trace)
            self.assertIn(
                "run.stopped status=stopped reason=token_limit_after_response",
                trace,
            )

    def test_status_and_resume_return_existing_terminal_schema2_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            workspace.mkdir()
            script = self._write_script(
                root,
                [
                    {
                        "response": {
                            "content": "The terminal result is already durable.",
                            "usage": {"input_tokens": 6, "output_tokens": 4},
                        }
                    }
                ],
            )
            event_log = root / "terminal-events.jsonl"
            run_code, original = self._invoke_json(
                [
                    "run",
                    "Finish without tools.",
                    "--workspace",
                    str(workspace),
                    "--script",
                    str(script),
                    "--run-id",
                    "terminal-cli-run",
                    "--event-log",
                    str(event_log),
                ]
            )
            self.assertEqual(run_code, 0)
            event_count = len(read_events(event_log))

            status_code, status = self._invoke_json(["status", str(event_log)])

            self.assertEqual(status_code, 0)
            self.assertEqual(status["schema_version"], "2")
            self.assertEqual(status["run_id"], "terminal-cli-run")
            self.assertEqual(status["phase"], "terminal")
            self.assertFalse(status["resumable"])
            self.assertEqual(status["turns"], 1)
            self.assertEqual(status["tool_calls"], 0)
            self.assertEqual(status["usage"]["total_tokens"], 10)
            self.assertIsNone(status["pending_model"])
            self.assertEqual(status["pending_tools"], [])
            self.assertEqual(status["terminal"]["status"], "completed")
            self.assertEqual(status["terminal"]["reason"], "model_stopped")
            self.assertEqual(len(status["through_event_sha256"]), 64)
            self.assertEqual(status["context"]["config"]["policy"], "submodular")
            self.assertEqual(
                status["context"]["last_receipt"]["config_fingerprint"],
                status["context"]["fingerprint"],
            )

            resume_code, resumed = self._invoke_json(["resume", str(event_log)])

            self.assertEqual(resume_code, 0)
            self.assertEqual(resumed, original)
            self.assertEqual(len(read_events(event_log)), event_count)

    def test_resume_completes_an_interrupted_scripted_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "note.txt").write_text("durable evidence\n", encoding="utf-8")
            steps: list[dict[str, Any]] = [
                {
                    "response": {
                        "content": "I will read the evidence.",
                        "tool_calls": [
                            {
                                "id": "read-before-interrupt",
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
                        "tool_call_id": "read-before-interrupt",
                        "observation_contains": "durable evidence",
                    },
                    "response": {
                        "content": "Recovered the evidence after interruption.",
                        "usage": {"input_tokens": 9, "output_tokens": 4},
                    },
                },
            ]
            script = self._write_script(root, steps)
            event_log = root / "interrupted-events.jsonl"
            model = _BlockingAfterFirstScriptedModel(steps, name="cli-script")

            async def create_interrupted_run() -> None:
                durable_log = EventLog(event_log, "interrupted-cli-run")
                runner = AgentRunner(
                    model=model,
                    tools=WorkspaceTools(workspace),
                    event_log=durable_log,
                )
                task = asyncio.create_task(
                    runner.run("Read the note and report its durable evidence.")
                )
                await asyncio.wait_for(model.blocked.wait(), timeout=5)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task

            asyncio.run(create_interrupted_run())
            interrupted_events = read_events(event_log)
            self.assertEqual(interrupted_events[-1]["type"], "run.interrupted")
            status_code, status = self._invoke_json(["status", str(event_log)])
            self.assertEqual(status_code, 0)
            self.assertEqual(status["phase"], "interrupted")
            self.assertTrue(status["resumable"])
            self.assertEqual(status["pending_model"]["turn"], 2)

            resume_code, result = self._invoke_json(
                [
                    "resume",
                    str(event_log),
                    "--workspace",
                    str(workspace),
                    "--script",
                    str(script),
                ]
            )

            self.assertEqual(resume_code, 0)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["reason"], "model_stopped")
            self.assertEqual(
                result["final_text"],
                "Recovered the evidence after interruption.",
            )
            self.assertEqual(result["turns"], 2)
            self.assertEqual(result["tool_calls"], 1)
            self.assertEqual(result["usage"]["total_tokens"], 23)
            event_types = [event["type"] for event in read_events(event_log)]
            self.assertEqual(event_types.count("run.resumed"), 1)
            self.assertLess(
                event_types.index("run.interrupted"),
                event_types.index("run.resumed"),
            )
            self.assertEqual(event_types[-1], "run.completed")
            trace_code, trace = self._invoke(["trace", str(event_log)])
            self.assertEqual(trace_code, 0)
            self.assertIn("run.resumed", trace)
            self.assertIn("run.completed status=completed reason=model_stopped", trace)

    def test_status_marks_schema1_trace_audit_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "legacy-events.jsonl"
            legacy = [
                {
                    "schema_version": "1",
                    "run_id": "legacy-cli-run",
                    "seq": 0,
                    "timestamp": "2026-08-08T00:00:00.000Z",
                    "type": "run.started",
                    "data": {"status": "running"},
                },
                {
                    "schema_version": "1",
                    "run_id": "legacy-cli-run",
                    "seq": 1,
                    "timestamp": "2026-08-08T00:00:01.000Z",
                    "type": "run.completed",
                    "data": {"status": "completed", "reason": "legacy"},
                },
            ]
            path.write_text(
                "".join(
                    json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n"
                    for event in legacy
                ),
                encoding="utf-8",
            )

            status_code, status = self._invoke_json(["status", str(path)])

            self.assertEqual(status_code, 0)
            self.assertEqual(status["schema_version"], "1")
            self.assertFalse(status["resumable"])
            self.assertEqual(status["reason"], "schema 1 traces are audit-only")
            self.assertEqual(status["events"], 2)
            self.assertEqual(status["last_event_type"], "run.completed")


if __name__ == "__main__":
    unittest.main()
