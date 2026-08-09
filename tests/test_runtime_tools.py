from __future__ import annotations

import errno
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from contextopt.runtime import (
    RunLimits,
    RunPermissions,
    ToolCall,
    ToolOutcome,
    WorkspaceTools,
)


class WorkspaceToolsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.temp_root = Path(self._temporary_directory.name)
        self.workspace = self.temp_root / "workspace"
        self.workspace.mkdir()
        self.source = self.workspace / "sample.txt"
        self.source.write_text("alpha target omega\n", encoding="utf-8")

    async def execute(
        self,
        tools: WorkspaceTools,
        name: str,
        arguments: object,
        *,
        call_id: str = "call-1",
    ) -> ToolOutcome:
        return await tools.execute(
            ToolCall(
                id=call_id,
                name=name,
                arguments_json=json.dumps(arguments),
            )
        )

    async def test_denies_parent_escape_absolute_backslash_and_internal_paths(
        self,
    ) -> None:
        tools = WorkspaceTools(self.workspace)
        denied_paths = (
            "../outside.txt",
            "nested/../../outside.txt",
            self.source.resolve().as_posix(),
            "nested\\sample.txt",
            ".git/config",
            ".GIT/config",
            ".contextopt/state.json",
            ".CONTEXTOPT/state.json",
        )

        for index, denied_path in enumerate(denied_paths):
            with self.subTest(path=denied_path):
                outcome = await self.execute(
                    tools,
                    "read_file",
                    {"path": denied_path},
                    call_id=f"path-{index}",
                )
                self.assertFalse(outcome.ok)
                self.assertEqual(outcome.error_code, "path_denied")
                self.assertEqual(outcome.call_id, f"path-{index}")

    async def test_denies_symlink_that_escapes_workspace_when_supported(self) -> None:
        outside = self.temp_root / "outside.txt"
        outside.write_text("outside secret\n", encoding="utf-8")
        link = self.workspace / "escape.txt"
        try:
            link.symlink_to(outside)
        except (NotImplementedError, OSError) as exc:
            self.skipTest(f"symbolic links are unavailable: {exc}")

        outcome = await self.execute(
            WorkspaceTools(self.workspace),
            "read_file",
            {"path": "escape.txt"},
        )

        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.error_code, "path_denied")
        self.assertNotIn("outside secret", outcome.content)

    async def test_read_only_permissions_reject_writes_and_commands(self) -> None:
        original = self.source.read_bytes()
        digest = hashlib.sha256(original).hexdigest()
        tools = WorkspaceTools(
            self.workspace,
            test_commands={"unit": (sys.executable, "-c", "print('not run')")},
        )

        create = await self.execute(
            tools,
            "create_file",
            {"path": "created.txt", "content": "new"},
            call_id="create",
        )
        replace = await self.execute(
            tools,
            "replace_text",
            {
                "path": "sample.txt",
                "old_text": "alpha",
                "new_text": "changed",
                "expected_sha256": digest,
            },
            call_id="replace",
        )
        command = await self.execute(
            tools,
            "run_tests",
            {"scope": "unit"},
            call_id="command",
        )

        self.assertEqual((create.ok, create.error_code), (False, "write_denied"))
        self.assertEqual((replace.ok, replace.error_code), (False, "write_denied"))
        self.assertEqual((command.ok, command.error_code), (False, "command_denied"))
        self.assertFalse((self.workspace / "created.txt").exists())
        self.assertEqual(self.source.read_bytes(), original)

    async def test_replace_text_rejects_stale_sha_compare_and_swap(self) -> None:
        tools = WorkspaceTools(
            self.workspace,
            permissions=RunPermissions(allow_write=True),
        )
        read = await self.execute(tools, "read_file", {"path": "sample.txt"})
        self.assertTrue(read.ok)
        stale_digest = read.metadata["sha256"]
        externally_changed = b"externally changed target\n"
        self.source.write_bytes(externally_changed)

        outcome = await self.execute(
            tools,
            "replace_text",
            {
                "path": "sample.txt",
                "old_text": "target",
                "new_text": "replacement",
                "expected_sha256": stale_digest,
            },
        )

        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.error_code, "content_conflict")
        self.assertEqual(self.source.read_bytes(), externally_changed)

    async def test_exact_replace_is_all_or_nothing_and_leaves_no_temp_file(
        self,
    ) -> None:
        original = b"first target; second target\n"
        self.source.write_bytes(original)
        digest = hashlib.sha256(original).hexdigest()
        tools = WorkspaceTools(
            self.workspace,
            permissions=RunPermissions(allow_write=True),
        )

        mismatch = await self.execute(
            tools,
            "replace_text",
            {
                "path": "sample.txt",
                "old_text": "target",
                "new_text": "done",
                "expected_sha256": digest,
                "expected_occurrences": 1,
            },
            call_id="mismatch",
        )
        self.assertFalse(mismatch.ok)
        self.assertEqual(mismatch.error_code, "content_conflict")
        self.assertEqual(self.source.read_bytes(), original)

        replaced = await self.execute(
            tools,
            "replace_text",
            {
                "path": "sample.txt",
                "old_text": "target",
                "new_text": "done",
                "expected_sha256": digest,
                "expected_occurrences": 2,
            },
            call_id="replace",
        )
        expected = b"first done; second done\n"
        self.assertTrue(replaced.ok)
        self.assertEqual(self.source.read_bytes(), expected)
        self.assertEqual(replaced.metadata["replacements"], 2)
        self.assertEqual(
            replaced.metadata["after_sha256"], hashlib.sha256(expected).hexdigest()
        )
        self.assertEqual(list(self.workspace.glob(".sample.txt.*")), [])

    async def test_failed_atomic_replace_preserves_original_and_cleans_temp_file(
        self,
    ) -> None:
        original = self.source.read_bytes()
        digest = hashlib.sha256(original).hexdigest()
        tools = WorkspaceTools(
            self.workspace,
            permissions=RunPermissions(allow_write=True),
        )

        with patch(
            "contextopt.runtime.tools.os.replace",
            side_effect=OSError(errno.EIO, "simulated replace failure"),
        ):
            outcome = await self.execute(
                tools,
                "replace_text",
                {
                    "path": "sample.txt",
                    "old_text": "target",
                    "new_text": "replacement",
                    "expected_sha256": digest,
                },
            )

        self.assertFalse(outcome.ok)
        self.assertEqual(outcome.error_code, "io_error")
        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual(list(self.workspace.glob(".sample.txt.*")), [])

    async def test_run_tests_reports_nonzero_exit_as_successful_tool_execution(
        self,
    ) -> None:
        command = (
            sys.executable,
            "-c",
            "import sys; print('ran tests'); print('failed', file=sys.stderr); "
            "raise SystemExit(7)",
        )
        tools = WorkspaceTools(
            self.workspace,
            permissions=RunPermissions(allow_command=True),
            test_commands={"failing": command},
        )

        outcome = await self.execute(
            tools,
            "run_tests",
            {"scope": "failing"},
        )

        self.assertTrue(outcome.ok)
        self.assertIsNone(outcome.error_code)
        self.assertEqual(outcome.metadata["exit_code"], 7)
        self.assertIn("exit_code=7", outcome.content)
        self.assertIn("ran tests", outcome.content)
        self.assertIn("failed", outcome.content)

    async def test_unknown_tool_and_invalid_json_return_structured_errors(self) -> None:
        tools = WorkspaceTools(self.workspace)
        unknown = await tools.execute(
            ToolCall(id="unknown", name="missing_tool", arguments_json="{}")
        )
        malformed = await tools.execute(
            ToolCall(id="malformed", name="read_file", arguments_json="{")
        )
        wrong_shape = await tools.execute(
            ToolCall(id="shape", name="read_file", arguments_json="[]")
        )

        self.assertEqual((unknown.ok, unknown.error_code), (False, "unknown_tool"))
        self.assertEqual(unknown.call_id, "unknown")
        self.assertEqual(unknown.tool_name, "missing_tool")
        self.assertEqual(
            (malformed.ok, malformed.error_code), (False, "invalid_arguments")
        )
        self.assertEqual(
            (wrong_shape.ok, wrong_shape.error_code), (False, "invalid_arguments")
        )

    async def test_run_tests_truncates_large_output_with_capture_metadata(self) -> None:
        limits = RunLimits(
            command_timeout_seconds=10,
            max_tool_output_bytes=128,
        )
        command = (
            sys.executable,
            "-c",
            "import sys; sys.stdout.write('A' * 600); sys.stderr.write('Z' * 600)",
        )
        tools = WorkspaceTools(
            self.workspace,
            permissions=RunPermissions(allow_command=True),
            limits=limits,
            test_commands={"verbose": command},
        )

        outcome = await self.execute(
            tools,
            "run_tests",
            {"scope": "verbose"},
        )

        capture = outcome.metadata["capture"]
        self.assertTrue(outcome.ok)
        self.assertTrue(outcome.truncated)
        self.assertTrue(capture["truncated"])
        self.assertGreater(capture["total_bytes"], capture["retained_bytes"])
        self.assertGreater(capture["omitted_bytes"], 0)
        self.assertEqual(len(capture["sha256"]), 64)
        self.assertIn("... output truncated ...", outcome.content)
        self.assertIn("AAAA", outcome.content)
        self.assertIn("ZZZZ", outcome.content)


if __name__ == "__main__":
    unittest.main()
