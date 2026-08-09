from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from contextopt.cli import main
from contextopt.evaluation import (
    SemanticMemoryEvalConfig,
    render_semantic_memory_console,
    render_semantic_memory_html,
    render_semantic_memory_markdown,
    run_semantic_memory_evaluation,
)


class SemanticMemoryEvaluationTests(unittest.TestCase):
    def test_fixed_fixture_reports_retrieval_and_safety_metrics(self) -> None:
        report = run_semantic_memory_evaluation(
            SemanticMemoryEvalConfig(repetitions=3, limit=3)
        )
        self.assertEqual(report["model_calls"], 0)
        self.assertEqual(report["store_revision"], 6)
        summary = report["summary"]
        self.assertEqual(summary["hit_at_1_rate"], 1.0)
        self.assertEqual(summary["hit_at_k_rate"], 1.0)
        self.assertEqual(summary["mean_reciprocal_rank"], 1.0)
        self.assertEqual(summary["negative_pass_rate"], 1.0)
        self.assertEqual(summary["scope_isolation_rate"], 1.0)
        self.assertEqual(summary["invalidated_exclusion_rate"], 1.0)
        self.assertEqual(summary["deterministic_rate"], 1.0)
        self.assertTrue(all(run["deterministic"] for run in report["runs"]))
        self.assertIn("does not measure", report["claim_boundary"])

        console = render_semantic_memory_console(report)
        markdown = render_semantic_memory_markdown(report)
        html = render_semantic_memory_html(report)
        self.assertIn("global-cas-procedure", console)
        self.assertIn("hit@1", markdown)
        self.assertIn("Full JSON ledger", html)
        self.assertIn("<details>", html)

    def test_cli_writes_json_markdown_and_html_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "memory-eval.json"
            markdown = root / "memory-eval.md"
            html = root / "memory-eval.html"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "memory-eval",
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

    def test_config_rejects_one_shot_determinism(self) -> None:
        with self.assertRaises(ValueError):
            SemanticMemoryEvalConfig(repetitions=1)


if __name__ == "__main__":
    unittest.main()
