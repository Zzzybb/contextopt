from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from contextopt.cli import main
from contextopt.evaluation import (
    AgentEvalConfig,
    AgentEvalReport,
    build_algorithm_fixtures,
    build_openai_model_factory,
    run_agent_evaluation,
)


class AgentEvaluationFixtureTests(unittest.TestCase):
    def test_acm_and_math_fixtures_are_complete_snapshots(self) -> None:
        fixtures = build_algorithm_fixtures()

        self.assertEqual(
            [fixture.fixture_id for fixture in fixtures], ["two-sum", "extended-gcd"]
        )
        self.assertEqual(
            {fixture.category for fixture in fixtures}, {"acm-algorithm", "mathematics"}
        )
        for fixture in fixtures:
            self.assertEqual(set(fixture.root_files), set(fixture.bad_files))
            self.assertEqual(set(fixture.root_files), set(fixture.good_files))
            self.assertNotEqual(fixture.bad_files, fixture.good_files)
            self.assertIn("test_", " ".join(fixture.root_files))


class AgentEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = run_agent_evaluation(
            AgentEvalConfig(fixtures=("two-sum",), repetitions=1)
        )

    def test_strategy_matrix_has_visible_oracle_signal(self) -> None:
        report = self.report
        self.assertEqual(len(report.runs), 3)
        by_strategy = {summary.strategy: summary for summary in report.summaries}
        self.assertEqual(by_strategy["single_pass"].success_count, 0)
        self.assertEqual(by_strategy["best_of_n"].success_count, 1)
        self.assertEqual(by_strategy["orchestrated"].success_count, 1)
        self.assertEqual(by_strategy["best_of_n"].hidden_success_rate, 1.0)
        self.assertEqual(by_strategy["orchestrated"].hidden_success_rate, 1.0)
        self.assertEqual(by_strategy["single_pass"].hidden_run_count, 0)
        self.assertGreater(
            by_strategy["orchestrated"].mean_model_calls,
            by_strategy["best_of_n"].mean_model_calls,
        )
        self.assertIn(
            "does not measure general model capability", report.claim_boundary
        )

    def test_hidden_tests_can_be_disabled_without_leaking_the_grader(self) -> None:
        report = run_agent_evaluation(
            AgentEvalConfig(
                fixtures=("two-sum",),
                strategies=("best_of_n",),
                include_hidden_tests=False,
            )
        )
        self.assertEqual(report.summaries[0].hidden_run_count, 0)
        self.assertEqual(report.runs[0].hidden_test_calls, 0)
        self.assertNotIn("hidden_files", report.fixtures[0].to_dict())

    def test_openai_factory_is_fresh_and_role_specific_without_calling_network(
        self,
    ) -> None:
        factory = build_openai_model_factory(
            base_url="https://example.invalid/v1",
            api_key="test-key",
            model="solver-model",
            planner_model="planner-model",
            reviewer_model="reviewer-model",
        )
        models = factory(build_algorithm_fixtures()[0], "orchestrated")
        self.assertEqual(
            [model.name for model in models],
            [
                "openai-compatible:planner-model",
                "openai-compatible:solver-model",
                "openai-compatible:reviewer-model",
            ],
        )
        self.assertIsNot(
            models[0], factory(build_algorithm_fixtures()[0], "orchestrated")[0]
        )

    def test_report_roundtrip_and_tamper_detection(self) -> None:
        payload = self.report.to_dict()
        self.assertEqual(AgentEvalReport.from_dict(payload).to_dict(), payload)

        tampered = json.loads(json.dumps(payload))
        tampered["summaries"][1]["success_count"] = 0
        with self.assertRaisesRegex(ValueError, "success_rate|summary metrics"):
            AgentEvalReport.from_dict(tampered)

    def test_configuration_rejects_duplicate_or_unknown_strategy(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            AgentEvalConfig(strategies=("single_pass", "single_pass"))
        with self.assertRaisesRegex(ValueError, "unknown"):
            AgentEvalConfig(strategies=("made_up",))  # type: ignore[arg-type]

    def test_cli_writes_json_and_markdown_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "agent-eval.json"
            markdown = root / "agent-eval.md"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "agent-eval",
                        "--fixtures",
                        "two-sum",
                        "--output",
                        str(output),
                        "--markdown",
                        str(markdown),
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertIn("orchestrated", stdout.getvalue())
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["config"]["fixtures"], ["two-sum"])
            self.assertEqual(len(report["runs"]), 3)
            rendered = markdown.read_text(encoding="utf-8")
            self.assertIn("Claim boundary", rendered)
            self.assertIn("best_of_n", rendered)


if __name__ == "__main__":
    unittest.main()
