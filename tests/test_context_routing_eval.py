from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import patch

from contextopt.cli import main
from contextopt.evaluation.context_routing import (
    ContextRoutingEvalConfig,
    build_long_coding_traces,
    run_context_routing_evaluation,
    tool_protocol_issues,
)
from contextopt.runtime.context import estimate_messages_tokens
from contextopt.runtime.protocol import AgentMessage, ToolCall


class ContextRoutingFixtureTests(unittest.TestCase):
    def test_fixtures_are_long_protocol_valid_and_probe_complete(self) -> None:
        cases = build_long_coding_traces()

        self.assertEqual(len(cases), 3)
        for case in cases:
            self.assertGreaterEqual(len(case.messages), 40)
            self.assertGreater(estimate_messages_tokens(case.messages), 10_000)
            self.assertEqual(tool_protocol_issues(case.messages), ())
            source = "\n".join(message.content for message in case.messages)
            for probe in case.evidence:
                self.assertIn(probe.exact_text, source)

    def test_protocol_checker_rejects_orphan_out_of_order_and_incomplete(self) -> None:
        first = ToolCall("one", "read_file", '{"path":"a.py"}')
        second = ToolCall("two", "search_text", '{"query":"needle"}')
        assistant = AgentMessage(
            role="assistant",
            content="inspect",
            tool_calls=(first, second),
        )

        orphan = AgentMessage(
            role="tool",
            content="orphan",
            tool_call_id="none",
            tool_name="read_file",
        )
        self.assertIn("orphan", tool_protocol_issues((orphan,))[0])

        reversed_results = (
            assistant,
            AgentMessage(
                role="tool",
                content="second",
                tool_call_id="two",
                tool_name="search_text",
            ),
            AgentMessage(
                role="tool",
                content="first",
                tool_call_id="one",
                tool_name="read_file",
            ),
        )
        issues = tool_protocol_issues(reversed_results)
        self.assertEqual(len(issues), 4)
        self.assertIn("expected 'one'", issues[0])

        incomplete = (
            assistant,
            AgentMessage(
                role="tool",
                content="first",
                tool_call_id="one",
                tool_name="read_file",
            ),
        )
        self.assertIn("incomplete", tool_protocol_issues(incomplete)[0])


class ContextRoutingEvaluationTests(unittest.TestCase):
    config: ClassVar[ContextRoutingEvalConfig]
    report: ClassVar[dict[str, Any]]

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = ContextRoutingEvalConfig(repetitions=2)
        cls.report = run_context_routing_evaluation(cls.config)

    def test_report_is_paired_and_model_free(self) -> None:
        report = self.report
        expected_runs = (
            len(self.config.policies) * len(self.config.budgets) * len(report["cases"])
        )

        self.assertEqual(report["schema_version"], "1")
        self.assertEqual(report["context_compiler_version"], 1)
        self.assertEqual(report["model_calls"], 0)
        self.assertEqual(len(report["runs"]), expected_runs)
        cells = {
            (run["case_id"], run["policy"], run["budget_tokens"])
            for run in report["runs"]
        }
        self.assertEqual(len(cells), expected_runs)
        self.assertIn("does not measure model quality", report["interpretation"])

    def test_configuration_rejects_ambiguous_or_trivial_matrices(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            ContextRoutingEvalConfig(policies=("recent", "recent"))
        with self.assertRaisesRegex(ValueError, "positive"):
            ContextRoutingEvalConfig(budgets=(True,))
        with self.assertRaisesRegex(ValueError, "two repetitions"):
            ContextRoutingEvalConfig(repetitions=1)
        with self.assertRaisesRegex(ValueError, "at least"):
            ContextRoutingEvalConfig(max_tool_output_tokens=47)

    def test_bounded_policies_conform_at_fixed_budgets(self) -> None:
        for run in self.report["runs"]:
            self.assertTrue(run["supported"], run["error"])
            self.assertTrue(run["protocol_valid"], run["protocol_issues"])
            self.assertTrue(run["budget_compliant"])
            self.assertEqual(run["budget_overage_tokens"], 0)
            self.assertTrue(run["deterministic"])
            self.assertLessEqual(run["compiled_estimated_tokens"], run["budget_tokens"])

    def test_metrics_match_receipts_and_definitions(self) -> None:
        for run in self.report["runs"]:
            receipt = run["receipt"]
            self.assertIsNotNone(receipt)
            assert receipt is not None
            self.assertEqual(
                run["compiled_estimated_tokens"],
                receipt["estimated_selected_tokens"],
            )
            expected_compression = 1.0 - (
                run["compiled_estimated_tokens"] / run["original_estimated_tokens"]
            )
            self.assertAlmostEqual(run["compression_ratio"], expected_compression)
            self.assertAlmostEqual(
                run["evidence_recall"],
                run["evidence_retained"] / run["evidence_total"],
            )

    def test_comparison_has_a_nontrivial_recall_signal(self) -> None:
        high_budget = max(self.config.budgets)
        recall_by_policy = {
            row["policy"]: row["mean_evidence_recall"]
            for row in self.report["summary"]
            if row["budget_tokens"] == high_budget
        }

        self.assertGreater(max(recall_by_policy.values()), 0.0)
        self.assertGreater(len(set(recall_by_policy.values())), 1)

    def test_complete_report_is_byte_deterministic(self) -> None:
        repeated = run_context_routing_evaluation(self.config)

        canonical = json.dumps(
            self.report,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        repeated_canonical = json.dumps(
            repeated,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        self.assertEqual(repeated_canonical, canonical)

    def test_full_policy_reports_budget_incompatibility_without_fake_scores(
        self,
    ) -> None:
        config = ContextRoutingEvalConfig(
            policies=("full",),
            budgets=(512,),
            repetitions=2,
        )

        report = run_context_routing_evaluation(config)

        for run in report["runs"]:
            self.assertFalse(run["supported"])
            self.assertIsNone(run["evidence_recall"])
            self.assertIsNone(run["compression_ratio"])
            self.assertIn("ContextBudgetError", run["error"])

    def test_unexpected_compiler_failures_are_not_reported_as_unsupported(self) -> None:
        config = ContextRoutingEvalConfig(
            policies=("recent",),
            budgets=(1_024,),
            repetitions=2,
        )

        with (
            patch(
                "contextopt.evaluation.context_routing.ContextCompiler.compile",
                side_effect=RuntimeError("compiler regression"),
            ),
            self.assertRaisesRegex(RuntimeError, "compiler regression"),
        ):
            run_context_routing_evaluation(config)

    def test_cli_writes_machine_readable_and_markdown_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "report.json"
            markdown = root / "report.md"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "context-eval",
                        "--policies",
                        "recent,submodular",
                        "--budgets",
                        "1024",
                        "--repetitions",
                        "2",
                        "--output",
                        str(output),
                        "--markdown",
                        str(markdown),
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertIn("submodular", stdout.getvalue())
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["model_calls"], 0)
            self.assertEqual(len(report["summary"]), 2)
            rendered = markdown.read_text(encoding="utf-8")
            self.assertIn("Evidence recall", rendered)
            self.assertIn("not coding success", rendered)


if __name__ == "__main__":
    unittest.main()
