from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from contextopt.benchmark import BenchmarkConfig, render_markdown, run_benchmark
from contextopt.cli import main
from contextopt.synthetic import generate_case


class SyntheticTests(unittest.TestCase):
    def test_generation_is_reproducible(self) -> None:
        first = generate_case(seed=123).to_dict()
        second = generate_case(seed=123).to_dict()
        self.assertEqual(first, second)

    def test_generated_graph_is_feasible_for_mandatory_context(self) -> None:
        case = generate_case(seed=9, graph_rate=0.5, conflict_rate=0.05)
        self.assertTrue(case.problem.is_feasible(case.problem.mandatory_closure()))


class BenchmarkTests(unittest.TestCase):
    def test_paired_report_contains_all_policies(self) -> None:
        report = run_benchmark(
            BenchmarkConfig(instances=3, item_count=10, critical_count=3, budget=800),
            ("topk", "submodular", "oracle"),
        )
        self.assertEqual(len(report["runs"]), 9)
        self.assertEqual(
            [row["policy"] for row in report["summary"]],
            ["topk", "submodular", "oracle"],
        )
        self.assertIn("Critical recall", render_markdown(report))

    def test_graph_benchmark_marks_knapsack_unsupported(self) -> None:
        report = run_benchmark(
            BenchmarkConfig(
                instances=2,
                item_count=10,
                critical_count=3,
                budget=800,
                graph_rate=1.0,
            ),
            ("knapsack", "oracle"),
        )
        knapsack = report["summary"][0]
        self.assertEqual(knapsack["supported_runs"], 0)
        self.assertEqual(knapsack["unsupported_runs"], 2)

    def test_cli_writes_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "report.json"
            markdown_path = Path(temp_dir) / "report.md"
            exit_code = main(
                [
                    "benchmark",
                    "--instances",
                    "2",
                    "--items",
                    "10",
                    "--budget",
                    "800",
                    "--policies",
                    "topk,oracle",
                    "--output",
                    str(json_path),
                    "--markdown",
                    str(markdown_path),
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertEqual(json.loads(json_path.read_text())["schema_version"], "1")
            self.assertIn("ContextOpt", markdown_path.read_text())


if __name__ == "__main__":
    unittest.main()
