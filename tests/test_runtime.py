from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from contextopt.runtime import (
    AgentRunner,
    EventLog,
    RunLimits,
    RunPermissions,
    ScriptedModel,
    WorkspaceTools,
    read_events,
)

BUGGY_SOURCE = """from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def merge_settings(
    defaults: Mapping[str, Any], overrides: Mapping[str, Any]
) -> dict[str, Any]:
    \"\"\"Merge settings; ``None`` means that the default should be inherited.\"\"\"

    merged = dict(defaults)
    merged.update({key: value for key, value in overrides.items() if value})
    return merged
"""

VISIBLE_TESTS = """from __future__ import annotations

import unittest

from settings import merge_settings


class MergeSettingsTests(unittest.TestCase):
    def test_false_override_is_explicit(self) -> None:
        self.assertEqual(
            merge_settings({\"enabled\": True}, {\"enabled\": False}),
            {\"enabled\": False},
        )

    def test_none_inherits_default(self) -> None:
        self.assertEqual(
            merge_settings({\"region\": \"cn\"}, {\"region\": None}),
            {\"region\": \"cn\"},
        )


if __name__ == \"__main__\":
    unittest.main()
"""

HIDDEN_ORACLE = """from copy import deepcopy
from settings import merge_settings

defaults = {\"enabled\": True, \"retries\": 3, \"label\": \"prod\"}
overrides = {
    \"enabled\": False,
    \"retries\": 0,
    \"label\": \"\",
    \"ignored\": None,
}
defaults_before = deepcopy(defaults)
overrides_before = deepcopy(overrides)
actual = merge_settings(defaults, overrides)
expected = {\"enabled\": False, \"retries\": 0, \"label\": \"\"}
assert actual == expected, (actual, expected)
assert defaults == defaults_before
assert overrides == overrides_before
print(\"hidden oracle passed\")
"""


def _write_fixture(root: Path) -> tuple[Path, str]:
    workspace = root / "workspace"
    tests = workspace / "tests"
    tests.mkdir(parents=True)
    source = workspace / "settings.py"
    source.write_bytes(BUGGY_SOURCE.encode("utf-8"))
    (tests / "test_settings.py").write_bytes(VISIBLE_TESTS.encode("utf-8"))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return workspace, digest


def _limits(**overrides: Any) -> RunLimits:
    values: dict[str, Any] = {
        "max_turns": 8,
        "max_tool_calls": 8,
        "max_total_tokens": 500,
        "max_output_tokens_per_call": 64,
        "wall_timeout_seconds": 30.0,
        "command_timeout_seconds": 10.0,
        "max_tool_output_bytes": 64 * 1024,
    }
    values.update(overrides)
    return RunLimits(**values)


def _tool_names(events: tuple[dict[str, object], ...]) -> list[str]:
    names: list[str] = []
    for event in events:
        if event["type"] == "tool.started":
            data = event["data"]
            assert isinstance(data, dict)
            names.append(str(data["tool_name"]))
    return names


class RunLimitsTests(unittest.TestCase):
    def test_limits_reject_non_finite_and_wrong_type_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "wall_timeout_seconds"):
            RunLimits(wall_timeout_seconds=float("nan"))
        with self.assertRaisesRegex(ValueError, "command_timeout_seconds"):
            RunLimits(command_timeout_seconds=float("inf"))
        with self.assertRaisesRegex(ValueError, "max_turns"):
            RunLimits(max_turns=1.5)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "max_total_tokens"):
            RunLimits(max_total_tokens=True)  # type: ignore[arg-type]

    def test_from_dict_rejects_non_finite_timeout(self) -> None:
        payload = _limits().to_dict()
        payload["wall_timeout_seconds"] = float("nan")
        with self.assertRaisesRegex(ValueError, "wall_timeout_seconds"):
            RunLimits.from_dict(payload)


class RuntimeIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_scripted_model_fixes_real_python_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workspace, source_sha256 = _write_fixture(root)
            trace_path = root / "events.jsonl"
            limits = _limits()
            tools = WorkspaceTools(
                workspace,
                permissions=RunPermissions(allow_write=True, allow_command=True),
                limits=limits,
                test_commands={
                    "visible": (
                        sys.executable,
                        "-m",
                        "unittest",
                        "discover",
                        "-s",
                        "tests",
                        "-v",
                    )
                },
            )
            script: list[dict[str, Any]] = [
                {
                    "response": {
                        "content": "I will inspect the implementation first.",
                        "tool_calls": [
                            {
                                "id": "read-source",
                                "name": "read_file",
                                "arguments": {"path": "settings.py"},
                            }
                        ],
                        "usage": {"input_tokens": 20, "output_tokens": 5},
                    },
                },
                {
                    "expect": {
                        "turn": 2,
                        "last_tool": "read_file",
                        "tool_call_id": "read-source",
                        "observation_contains": [
                            f"sha256={source_sha256}",
                            "if value",
                        ],
                    },
                    "response": {
                        "content": "Now I will inspect the visible contract.",
                        "tool_calls": [
                            {
                                "id": "read-tests",
                                "name": "read_file",
                                "arguments": {"path": "tests/test_settings.py"},
                            }
                        ],
                        "usage": {"input_tokens": 30, "output_tokens": 5},
                    },
                },
                {
                    "expect": {
                        "turn": 3,
                        "last_tool": "read_file",
                        "tool_call_id": "read-tests",
                        "observation_contains": "test_false_override_is_explicit",
                    },
                    "response": {
                        "content": "I will reproduce the failure before editing.",
                        "tool_calls": [
                            {
                                "id": "test-before",
                                "name": "run_tests",
                                "arguments": {"scope": "visible"},
                            }
                        ],
                        "usage": {"input_tokens": 40, "output_tokens": 5},
                    },
                },
                {
                    "expect": {
                        "turn": 4,
                        "last_tool": "run_tests",
                        "tool_call_id": "test-before",
                        "observation_contains": [
                            "exit_code=1",
                            "test_false_override_is_explicit",
                        ],
                    },
                    "response": {
                        "content": (
                            "Only None should inherit, so I will make one guarded edit."
                        ),
                        "tool_calls": [
                            {
                                "id": "fix-condition",
                                "name": "replace_text",
                                "arguments": {
                                    "path": "settings.py",
                                    "old_text": "if value",
                                    "new_text": "if value is not None",
                                    "expected_sha256": source_sha256,
                                    "expected_occurrences": 1,
                                },
                            }
                        ],
                        "usage": {"input_tokens": 60, "output_tokens": 15},
                    },
                },
                {
                    "expect": {
                        "turn": 5,
                        "last_tool": "replace_text",
                        "tool_call_id": "fix-condition",
                        "observation_contains": "replaced 1 occurrence(s)",
                    },
                    "response": {
                        "content": "I will rerun the registered tests.",
                        "tool_calls": [
                            {
                                "id": "test-after",
                                "name": "run_tests",
                                "arguments": {"scope": "visible"},
                            }
                        ],
                        "usage": {"input_tokens": 50, "output_tokens": 5},
                    },
                },
                {
                    "expect": {
                        "turn": 6,
                        "last_tool": "run_tests",
                        "tool_call_id": "test-after",
                        "observation_contains": ["exit_code=0", "OK"],
                    },
                    "response": {
                        "content": "Fixed falsey overrides; the visible tests pass.",
                        "usage": {"input_tokens": 40, "output_tokens": 10},
                    },
                },
            ]
            model = ScriptedModel(script)
            runner = AgentRunner(
                model=model,
                tools=tools,
                event_log=EventLog(trace_path, "fixture-run"),
                limits=limits,
            )

            result = await runner.run(
                "Fix merge_settings so only None inherits a default. Preserve inputs."
            )

            self.assertEqual(
                result.status,
                "completed",
                msg={"result": result.to_dict(), "events": read_events(trace_path)},
            )
            self.assertEqual(result.reason, "model_stopped")
            self.assertEqual(result.turns, 6)
            self.assertEqual(result.tool_calls, 5)
            self.assertEqual(result.usage.input_tokens, 240)
            self.assertEqual(result.usage.output_tokens, 45)
            self.assertEqual(result.usage.total_tokens, 285)
            self.assertEqual(len(model.requests), 6)
            self.assertIn(
                "if value is not None", (workspace / "settings.py").read_text()
            )

            hidden = subprocess.run(
                [sys.executable, "-c", HIDDEN_ORACLE],
                cwd=workspace,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                check=False,
            )
            self.assertEqual(
                hidden.returncode,
                0,
                msg=f"stdout:\n{hidden.stdout}\nstderr:\n{hidden.stderr}",
            )
            self.assertIn("hidden oracle passed", hidden.stdout)

            events = read_events(trace_path)
            self.assertEqual(
                [event["seq"] for event in events], list(range(len(events)))
            )
            expected_types = ["run.started"]
            for _ in range(5):
                expected_types.extend(
                    [
                        "model.requested",
                        "model.responded",
                        "budget.updated",
                        "tool.started",
                        "tool.completed",
                    ]
                )
            expected_types.extend(
                [
                    "model.requested",
                    "model.responded",
                    "budget.updated",
                    "run.completed",
                ]
            )
            self.assertEqual([event["type"] for event in events], expected_types)
            self.assertEqual(
                _tool_names(events),
                [
                    "read_file",
                    "read_file",
                    "run_tests",
                    "replace_text",
                    "run_tests",
                ],
            )

            test_events = [
                event
                for event in events
                if event["type"] == "tool.completed"
                and isinstance(event["data"], dict)
                and event["data"].get("tool_name") == "run_tests"
            ]
            self.assertEqual(
                [event["data"]["metadata"]["exit_code"] for event in test_events],
                [1, 0],
            )
            self.assertTrue(all(event["data"]["ok"] for event in test_events))
            failed_test_index = events.index(test_events[0])
            later_types = [event["type"] for event in events[failed_test_index + 1 :]]
            self.assertIn("model.requested", later_types)
            self.assertIn("tool.started", later_types)
            self.assertEqual(events[-1]["type"], "run.completed")

    async def test_tool_call_limit_stops_before_second_tool(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "note.txt").write_text("evidence", encoding="utf-8")
            limits = _limits(max_tool_calls=1)
            model = ScriptedModel(
                [
                    {
                        "response": {
                            "tool_calls": [
                                {
                                    "id": "allowed",
                                    "name": "read_file",
                                    "arguments": {"path": "note.txt"},
                                },
                                {
                                    "id": "blocked",
                                    "name": "read_file",
                                    "arguments": {"path": "note.txt"},
                                },
                            ],
                            "usage": {"input_tokens": 5, "output_tokens": 2},
                        }
                    }
                ]
            )
            trace_path = root / "events.jsonl"
            result = await AgentRunner(
                model=model,
                tools=WorkspaceTools(workspace, limits=limits),
                event_log=EventLog(trace_path, "tool-limit-run"),
                limits=limits,
            ).run("Read the evidence twice.")

            self.assertEqual(result.status, "stopped")
            self.assertEqual(result.reason, "tool_call_limit")
            self.assertEqual(result.tool_calls, 1)
            events = read_events(trace_path)
            started = [event for event in events if event["type"] == "tool.started"]
            self.assertEqual(len(started), 1)
            self.assertEqual(started[0]["data"]["call_id"], "allowed")
            self.assertEqual(events[-1]["type"], "run.stopped")

    async def test_token_limit_prevents_response_tool_from_running(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            workspace.mkdir()
            target = workspace / "value.txt"
            target.write_bytes(b"old")
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            limits = _limits(
                max_total_tokens=10,
                max_output_tokens_per_call=10,
            )
            model = ScriptedModel(
                [
                    {
                        "response": {
                            "tool_calls": [
                                {
                                    "id": "must-not-run",
                                    "name": "replace_text",
                                    "arguments": {
                                        "path": "value.txt",
                                        "old_text": "old",
                                        "new_text": "new",
                                        "expected_sha256": digest,
                                    },
                                }
                            ],
                            "usage": {"input_tokens": 8, "output_tokens": 3},
                        }
                    }
                ]
            )
            trace_path = root / "events.jsonl"
            result = await AgentRunner(
                model=model,
                tools=WorkspaceTools(
                    workspace,
                    permissions=RunPermissions(allow_write=True),
                    limits=limits,
                ),
                event_log=EventLog(trace_path, "token-limit-run"),
                limits=limits,
            ).run("Change old to new.")

            self.assertEqual(result.status, "stopped")
            self.assertEqual(result.reason, "token_limit_after_response")
            self.assertEqual(result.usage.total_tokens, 11)
            self.assertEqual(result.tool_calls, 0)
            self.assertEqual(target.read_text(encoding="utf-8"), "old")
            events = read_events(trace_path)
            self.assertNotIn("tool.started", [event["type"] for event in events])
            budget = next(
                event for event in events if event["type"] == "budget.updated"
            )
            self.assertEqual(budget["data"]["overage"], 1)
            self.assertEqual(events[-1]["type"], "run.stopped")

    async def test_model_failure_preserves_completed_tool_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "evidence.txt").write_text("important", encoding="utf-8")
            limits = _limits()
            model = ScriptedModel(
                [
                    {
                        "response": {
                            "tool_calls": [
                                {
                                    "id": "read-before-failure",
                                    "name": "read_file",
                                    "arguments": {"path": "evidence.txt"},
                                }
                            ],
                            "usage": {"input_tokens": 5, "output_tokens": 2},
                        }
                    }
                ]
            )
            trace_path = root / "events.jsonl"
            result = await AgentRunner(
                model=model,
                tools=WorkspaceTools(workspace, limits=limits),
                event_log=EventLog(trace_path, "model-failure-run"),
                limits=limits,
            ).run("Read evidence, then continue.")

            self.assertEqual(result.status, "failed")
            self.assertEqual(result.reason, "model_error:script_exhausted")
            self.assertEqual(result.turns, 1)
            self.assertEqual(result.tool_calls, 1)
            events = read_events(trace_path)
            self.assertEqual(
                [event["type"] for event in events],
                [
                    "run.started",
                    "model.requested",
                    "model.responded",
                    "budget.updated",
                    "tool.started",
                    "tool.completed",
                    "model.requested",
                    "model.failed",
                    "run.failed",
                ],
            )
            self.assertEqual(events[5]["data"]["call_id"], "read-before-failure")
            self.assertIn("important", events[5]["data"]["content"])
            self.assertEqual(
                events[-1]["data"]["reason"], "model_error:script_exhausted"
            )


if __name__ == "__main__":
    unittest.main()
