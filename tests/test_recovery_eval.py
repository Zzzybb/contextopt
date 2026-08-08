from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from contextopt.cli import main
from contextopt.evaluation import (
    RecoveryEvalConfig,
    RecoveryMatrixReport,
    render_recovery_html,
    render_recovery_markdown,
    run_recovery_evaluation,
)


class RecoveryEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = run_recovery_evaluation()

    def test_all_durable_boundaries_recover(self) -> None:
        report = self.report
        self.assertEqual(len(report.results), 5)
        self.assertEqual(report.passed_count, 5)
        self.assertEqual(report.failed_count, 0)
        by_id = {result.scenario_id: result for result in report.results}
        self.assertEqual(
            by_id["pending-model-request"].pending_before_resume["model"],
            {"turn": 1},
        )
        self.assertEqual(
            by_id["durable-final-response"].model_calls_after_resume,
            0,
        )
        self.assertEqual(
            by_id["nonreplayable-test-pause"].first_resume_status,
            "paused",
        )
        self.assertEqual(
            by_id["nonreplayable-test-pause"].tool_event_counts["tool.failed"],
            1,
        )

    def test_report_roundtrip_and_rendered_claim_boundary(self) -> None:
        payload = self.report.to_dict()
        self.assertEqual(RecoveryMatrixReport.from_dict(payload).to_dict(), payload)
        markdown = render_recovery_markdown(self.report)
        html = render_recovery_html(self.report)
        self.assertIn("Durable event evidence", markdown)
        self.assertIn("Claim boundary", markdown)
        self.assertIn("recovery-report", html)
        self.assertIn("reconcile-write-before-effect", html)
        self.assertIn("PASS", html)

    def test_configuration_rejects_unknown_or_duplicate_scenarios(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            RecoveryEvalConfig(("pending-model-request", "pending-model-request"))
        with self.assertRaisesRegex(ValueError, "unknown"):
            RecoveryEvalConfig(("not-a-scenario",))

    def test_cli_writes_json_markdown_and_html_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "recovery.json"
            markdown = root / "recovery.md"
            html = root / "recovery.html"
            manifest = root / "recovery.manifest.json"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "recovery-eval",
                        "--scenarios",
                        "pending-model-request,nonreplayable-test-pause",
                        "--output",
                        str(output),
                        "--markdown",
                        str(markdown),
                        "--html",
                        str(html),
                        "--manifest",
                        str(manifest),
                    ]
                )
            self.assertEqual(exit_code, 0)
            self.assertIn("passed=2/2", stdout.getvalue())
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["summary"]["passed_count"], 2)
            self.assertIn("Claim boundary", markdown.read_text(encoding="utf-8"))
            self.assertIn("<table>", html.read_text(encoding="utf-8"))
            manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest_payload["fault_injection"]["model_adapter"], "scripted"
            )


if __name__ == "__main__":
    unittest.main()
