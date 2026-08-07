from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from contextopt.cli import main


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

            trace_code, trace = self._invoke(["trace", str(event_log)])

            self.assertEqual(trace_code, 0)
            self.assertIn("0000 run.started status=running", trace)
            self.assertIn("model.responded turn=1 calls=1", trace)
            self.assertIn(
                "tool.completed tool=read_file call=read-note ok=True",
                trace,
            )
            self.assertIn(
                "run.completed status=completed reason=model_stopped",
                trace,
            )

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


if __name__ == "__main__":
    unittest.main()
