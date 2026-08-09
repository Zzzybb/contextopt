from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from contextopt.cli import main
from contextopt.evaluation import (
    SemanticContextEvalConfig,
    render_semantic_context_console,
    render_semantic_context_html,
    render_semantic_context_markdown,
    run_semantic_context_evaluation,
)


class SemanticContextEvaluationTests(unittest.TestCase):
    def test_fixed_matrix_is_model_free_and_replayable(self) -> None:
        config = SemanticContextEvalConfig(
            policies=("recent", "submodular"),
            budgets=(128, 512),
            repetitions=2,
        )
        report = run_semantic_context_evaluation(config)

        self.assertEqual(report["schema_version"], "1")
        self.assertEqual(report["model_calls"], 0)
        self.assertEqual(len(report["runs"]), 16)
        summary = report["summary"]
        self.assertEqual(summary["failed_count"], 0)
        self.assertEqual(summary["retrieved_recall_rate"], 1.0)
        self.assertEqual(summary["budget_compliant_rate"], 1.0)
        self.assertEqual(summary["deterministic_rate"], 1.0)
        self.assertEqual(summary["replayable_rate"], 1.0)
        self.assertGreater(summary["mean_evicted_candidates"], 0.0)
        self.assertIn("does not measure", report["claim_boundary"])
        self.assertTrue(all(run["error"] is None for run in report["runs"]))

        console = render_semantic_context_console(report)
        markdown = render_semantic_context_markdown(report)
        html = render_semantic_context_html(report)
        self.assertIn("summary:", console)
        self.assertIn("Selected candidate tokens", markdown)
        self.assertIn("Full JSON ledger", html)
        self.assertIn("semantic-context-report", html)

    def test_report_is_byte_deterministic(self) -> None:
        config = SemanticContextEvalConfig(
            policies=("density",), budgets=(256,), repetitions=2
        )
        first = run_semantic_context_evaluation(config)
        second = run_semantic_context_evaluation(config)
        self.assertEqual(
            json.dumps(first, ensure_ascii=False, sort_keys=True),
            json.dumps(second, ensure_ascii=False, sort_keys=True),
        )

    def test_cli_writes_json_markdown_and_html_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "semantic-context-eval.json"
            markdown = root / "semantic-context-eval.md"
            html = root / "semantic-context-eval.html"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "semantic-context-eval",
                        "--policies",
                        "recent",
                        "--budgets",
                        "128,512",
                        "--repetitions",
                        "2",
                        "--output",
                        str(output),
                        "--markdown",
                        str(markdown),
                        "--html",
                        str(html),
                    ]
                )
            self.assertEqual(exit_code, 0)
            self.assertIn("summary:", stdout.getvalue())
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["config"]["repetitions"], 2)
            self.assertTrue(markdown.read_text(encoding="utf-8").startswith("# "))
            self.assertIn("<html", html.read_text(encoding="utf-8"))

    def test_config_rejects_one_shot_or_unknown_policy(self) -> None:
        with self.assertRaisesRegex(ValueError, "repetitions"):
            SemanticContextEvalConfig(repetitions=1)
        with self.assertRaisesRegex(ValueError, "unknown"):
            SemanticContextEvalConfig(policies=("embedding",))  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
