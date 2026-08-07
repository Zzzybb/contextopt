from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

from contextopt.runtime import RunPermissions, ToolCall, WorkspaceTools
from contextopt.runtime.tool_state import (
    ToolExecutionPlan,
    tool_call_fingerprint,
)


def _call(call_id: str, name: str, arguments: object) -> ToolCall:
    return ToolCall(
        id=call_id,
        name=name,
        arguments_json=json.dumps(
            arguments,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
    )


class WorkspaceToolRecoveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.workspace = Path(self._temporary_directory.name) / "workspace"
        self.workspace.mkdir()
        self.permissions = RunPermissions(allow_write=True)

    async def _prepare(
        self, tools: WorkspaceTools, call: ToolCall, operation_id: str
    ) -> ToolExecutionPlan:
        return await tools.prepare(
            call,
            operation_id=operation_id,
            fingerprint=tool_call_fingerprint(call),
        )

    async def test_plan_round_trip_and_tamper_detection(self) -> None:
        tools = WorkspaceTools(self.workspace, permissions=self.permissions)
        call = _call(
            "create-round-trip",
            "create_file",
            {"path": "round-trip.txt", "content": "sealed content\n"},
        )
        plan = await self._prepare(tools, call, "operation-round-trip")

        restored = ToolExecutionPlan.from_json(plan.to_json())
        self.assertEqual(restored, plan)
        self.assertEqual(restored.to_dict(), plan.to_dict())

        tampered = json.loads(plan.to_json())
        tampered["postconditions"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "checksum"):
            ToolExecutionPlan.from_dict(tampered)

        tampered_fingerprint = json.loads(plan.to_json())
        tampered_fingerprint["fingerprint"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            ToolExecutionPlan.from_dict(tampered_fingerprint)

        with self.assertRaisesRegex(ValueError, "fingerprint"):
            await tools.prepare(call, "operation-bad-fingerprint", "0" * 64)

    async def test_create_reconciles_pre_post_and_divergent_states(self) -> None:
        tools = WorkspaceTools(self.workspace, permissions=self.permissions)
        call = _call(
            "create-main",
            "create_file",
            {"path": "created.txt", "content": "intended\n"},
        )
        plan = await self._prepare(tools, call, "operation-create-main")

        before = await tools.reconcile(plan)
        self.assertEqual(before.action, "retry")
        self.assertIsNone(before.outcome)

        serialized = plan.to_json()
        outcome = await tools.execute_prepared(plan)
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.to_dict(), plan.planned_outcome.to_dict())

        restarted = WorkspaceTools(self.workspace, permissions=self.permissions)
        restored = ToolExecutionPlan.from_json(serialized)
        after = await restarted.reconcile(restored)
        self.assertEqual(after.action, "completed")
        self.assertIsNotNone(after.outcome)
        assert after.outcome is not None
        self.assertEqual(after.outcome.to_dict(), plan.planned_outcome.to_dict())
        self.assertEqual(
            (self.workspace / "created.txt").read_text(encoding="utf-8"),
            "intended\n",
        )

        divergent_call = _call(
            "create-divergent",
            "create_file",
            {"path": "divergent.txt", "content": "planned\n"},
        )
        divergent_plan = await self._prepare(
            restarted, divergent_call, "operation-create-divergent"
        )
        (self.workspace / "divergent.txt").write_text(
            "external change\n", encoding="utf-8"
        )

        divergent = await restarted.reconcile(divergent_plan)
        self.assertEqual(divergent.action, "divergence")
        self.assertIn("neither", divergent.reason)

    async def test_replace_reconciles_pre_post_and_divergent_states(self) -> None:
        source = self.workspace / "source.txt"
        original = b"alpha target omega\n"
        source.write_bytes(original)
        tools = WorkspaceTools(self.workspace, permissions=self.permissions)
        call = _call(
            "replace-main",
            "replace_text",
            {
                "path": "source.txt",
                "old_text": "target",
                "new_text": "replacement",
                "expected_sha256": hashlib.sha256(original).hexdigest(),
            },
        )
        plan = await self._prepare(tools, call, "operation-replace-main")

        before = await tools.reconcile(plan)
        self.assertEqual(before.action, "retry")

        serialized = plan.to_json()
        outcome = await tools.execute_prepared(plan)
        self.assertTrue(outcome.ok)
        self.assertEqual(outcome.to_dict(), plan.planned_outcome.to_dict())

        restarted = WorkspaceTools(self.workspace, permissions=self.permissions)
        restored = ToolExecutionPlan.from_json(serialized)
        after = await restarted.reconcile(restored)
        self.assertEqual(after.action, "completed")
        self.assertIsNotNone(after.outcome)
        assert after.outcome is not None
        self.assertEqual(after.outcome.to_dict(), plan.planned_outcome.to_dict())
        self.assertEqual(source.read_bytes(), b"alpha replacement omega\n")

        divergent_source = self.workspace / "replace-divergent.txt"
        divergent_source.write_bytes(original)
        divergent_call = _call(
            "replace-divergent",
            "replace_text",
            {
                "path": "replace-divergent.txt",
                "old_text": "target",
                "new_text": "replacement",
                "expected_sha256": hashlib.sha256(original).hexdigest(),
            },
        )
        divergent_plan = await self._prepare(
            restarted, divergent_call, "operation-replace-divergent"
        )
        divergent_source.write_text("third state\n", encoding="utf-8")

        divergent = await restarted.reconcile(divergent_plan)
        self.assertEqual(divergent.action, "divergence")
        self.assertIn("neither", divergent.reason)

    async def test_read_only_retries_but_commands_pause_without_execution(
        self,
    ) -> None:
        read_tools = WorkspaceTools(self.workspace)
        read_call = _call("read", "list_files", {})
        read_plan = await self._prepare(read_tools, read_call, "operation-read")

        self.assertEqual(read_plan.replay_policy, "safe")
        self.assertEqual((await read_tools.reconcile(read_plan)).action, "retry")

        marker = self.workspace / "command-ran.txt"
        command = (
            sys.executable,
            "-c",
            "from pathlib import Path; Path('command-ran.txt').write_text('ran')",
        )
        command_tools = WorkspaceTools(
            self.workspace,
            permissions=RunPermissions(allow_command=True),
            test_commands={"marker": command},
        )
        command_call = _call("command", "run_tests", {"scope": "marker"})
        command_plan = await self._prepare(
            command_tools, command_call, "operation-command"
        )

        self.assertEqual(command_plan.replay_policy, "never")
        restarted = WorkspaceTools(
            self.workspace,
            permissions=RunPermissions(allow_command=True),
            test_commands={"marker": command},
        )
        decision = await restarted.reconcile(
            ToolExecutionPlan.from_json(command_plan.to_json())
        )
        self.assertEqual(decision.action, "paused")
        self.assertIn("automatic replay", decision.reason)
        self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
