from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from contextopt.search import (
    WorkspaceConflict,
    apply_snapshot,
    read_apply_receipt,
    snapshot_fingerprint,
    write_apply_receipt,
)


class ApplySnapshotTests(unittest.TestCase):
    def test_apply_and_rollback_handle_create_modify_delete(self) -> None:
        baseline = {
            "src/main.py": "return 0\n",
            "obsolete.txt": "remove me\n",
        }
        target = {
            "src/main.py": "return 1\n",
            "new/feature.txt": "new file\n",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "src").mkdir()
            (workspace / "src/main.py").write_text(
                baseline["src/main.py"], encoding="utf-8"
            )
            (workspace / "obsolete.txt").write_text(
                baseline["obsolete.txt"], encoding="utf-8"
            )
            receipt = apply_snapshot(
                workspace,
                baseline,
                target,
                run_id="apply-test",
                candidate_id="candidate-1",
                session_fingerprint="session-fingerprint",
                allow_write=True,
                allow_delete=True,
            )
            self.assertEqual(receipt.created_paths, ("new/feature.txt",))
            self.assertEqual(receipt.modified_paths, ("src/main.py",))
            self.assertEqual(receipt.deleted_paths, ("obsolete.txt",))
            self.assertEqual(receipt.target_fingerprint, snapshot_fingerprint(target))
            self.assertEqual(
                (workspace / "src/main.py").read_text(encoding="utf-8"),
                target["src/main.py"],
            )
            self.assertFalse((workspace / "obsolete.txt").exists())
            self.assertEqual(
                (workspace / "new/feature.txt").read_text(encoding="utf-8"),
                target["new/feature.txt"],
            )

            rollback = apply_snapshot(
                workspace,
                target,
                baseline,
                run_id="apply-test",
                candidate_id="candidate-1",
                session_fingerprint="session-fingerprint",
                allow_write=True,
                allow_delete=True,
                operation="rollback",
                rollback_of=receipt.operation_id,
            )
            self.assertEqual(rollback.operation, "rollback")
            self.assertEqual(
                (workspace / "src/main.py").read_text(encoding="utf-8"),
                baseline["src/main.py"],
            )
            self.assertTrue((workspace / "obsolete.txt").exists())
            self.assertFalse((workspace / "new/feature.txt").exists())

    def test_stale_baseline_is_refused_before_any_write(self) -> None:
        baseline = {"main.py": "return 0\n"}
        target = {"main.py": "return 1\n"}
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            path = workspace / "main.py"
            path.write_text("out-of-band edit\n", encoding="utf-8")
            with self.assertRaises(WorkspaceConflict):
                apply_snapshot(
                    workspace,
                    baseline,
                    target,
                    run_id="apply-test",
                    candidate_id="candidate-1",
                    allow_write=True,
                )
            self.assertEqual(path.read_text(encoding="utf-8"), "out-of-band edit\n")

    def test_receipt_round_trip_and_tamper_detection(self) -> None:
        baseline = {"main.py": "return 0\n"}
        target = {"main.py": "return 1\n"}
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "main.py").write_text(baseline["main.py"], encoding="utf-8")
            receipt = apply_snapshot(
                workspace,
                baseline,
                target,
                run_id="apply-test",
                candidate_id="candidate-1",
                allow_write=True,
            )
            receipt_path = workspace / "receipt.json"
            write_apply_receipt(receipt, receipt_path)
            self.assertEqual(read_apply_receipt(receipt_path), receipt)
            forged = json.loads(receipt_path.read_text(encoding="utf-8"))
            forged["target_fingerprint"] = "forged"
            receipt_path.write_text(json.dumps(forged), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "operation_id is inconsistent"):
                read_apply_receipt(receipt_path)

    def test_paths_and_permissions_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "main.py").write_text("return 0\n", encoding="utf-8")
            with self.assertRaises(PermissionError):
                apply_snapshot(
                    workspace,
                    {"main.py": "return 0\n"},
                    {"main.py": "return 1\n"},
                    run_id="apply-test",
                    candidate_id="candidate-1",
                )
            with self.assertRaises(ValueError):
                apply_snapshot(
                    workspace,
                    {"main.py": "return 0\n"},
                    {".git/config": "bad\n"},
                    run_id="apply-test",
                    candidate_id="candidate-1",
                    allow_write=True,
                )


if __name__ == "__main__":
    unittest.main()
