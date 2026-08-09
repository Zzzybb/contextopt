from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from contextopt.cli import main
from contextopt.evaluation import (
    RobustnessEvalConfig,
    RobustnessMatrixReport,
    render_robustness_html,
    render_robustness_markdown,
    run_robustness_evaluation,
)


class RobustnessEvaluationTests(unittest.TestCase):
    def test_default_fault_injection_matrix_passes(self) -> None:
        report = run_robustness_evaluation()

        self.assertEqual(report.passed_count, 4)
        self.assertEqual(report.failed_count, 0)
        self.assertEqual(
            report.results[0].observations["compacted_block_ids"], ["block-000001"]
        )
        self.assertEqual(
            report.results[1].observations["persisted_status"], "invalidated"
        )
        self.assertEqual(
            report.results[2].observations["error"],
            "duplicate tool result at message index 0",
        )
        self.assertEqual(
            report.results[3].observations["error_code"], "content_conflict"
        )

    def test_report_round_trip_and_renderers(self) -> None:
        report = run_robustness_evaluation(
            RobustnessEvalConfig(scenarios=("tool-output-compaction",))
        )
        restored = RobustnessMatrixReport.from_dict(report.to_dict())

        self.assertEqual(restored.to_dict(), report.to_dict())
        self.assertIn(
            "runtime robustness evaluation", render_robustness_markdown(report)
        )
        self.assertIn("robustness-report", render_robustness_html(report))

        tampered = json.loads(json.dumps(report.to_dict()))
        tampered["summary"]["passed_count"] = 0
        with self.assertRaisesRegex(ValueError, "passed count"):
            RobustnessMatrixReport.from_dict(tampered)

    def test_configuration_rejects_duplicate_and_unknown_scenarios(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            RobustnessEvalConfig(
                scenarios=("cas-write-conflict", "cas-write-conflict")  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(ValueError, "unknown"):
            RobustnessEvalConfig(scenarios=("not-a-scenario",))  # type: ignore[arg-type]

    def test_cli_writes_json_markdown_html_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "report.json"
            markdown = root / "report.md"
            html = root / "report.html"
            manifest = root / "manifest.json"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "robustness-eval",
                        "--scenarios",
                        "duplicate-tool-result,cas-write-conflict",
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
            self.assertIn("duplicate-tool-result", stdout.getvalue())
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["scenario_count"], 2)
            self.assertEqual(len(payload["results"]), 2)
            self.assertIn("ContextOpt Level 4 robustness", markdown.read_text())
            self.assertIn("<table>", html.read_text())
            manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest_payload["kind"], "contextopt.robustness-eval.manifest"
            )
            self.assertFalse(manifest_payload["fault_injection"]["external_network"])


if __name__ == "__main__":
    unittest.main()
