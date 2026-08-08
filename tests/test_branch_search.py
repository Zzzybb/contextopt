from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from contextopt.cli import main
from contextopt.search import (
    BranchCase,
    BranchSearch,
    BranchSearchConfig,
    BranchSearchReport,
    CandidatePatch,
    TestResult,
    demo_case,
    render_branch_html,
    validate_search_report,
    verify_search_events,
)


def _mcts_case() -> BranchCase:
    def files(candidate_id: str) -> dict[str, str]:
        return {"src/solver.py": f"def solve(): return {candidate_id!r}\n"}

    candidates = tuple(
        CandidatePatch(
            id=candidate_id,
            parent_id=parent_id,
            hypothesis=f"try {candidate_id}",
            files=files(candidate_id),
        )
        for candidate_id, parent_id in (
            ("a-low", "root"),
            ("b-mid", "root"),
            ("c-high", "root"),
            ("a-low-fix", "a-low"),
            ("b-mid-fix", "b-mid"),
            ("c-high-fix", "c-high"),
        )
    )
    tests = {
        "a-low": TestResult(
            suite="mcts",
            passed_tests=("t1",),
            failed_tests=("t2", "t3", "t4"),
        ),
        "b-mid": TestResult(
            suite="mcts", passed_tests=("t1", "t2"), failed_tests=("t3", "t4")
        ),
        "c-high": TestResult(
            suite="mcts", passed_tests=("t1", "t2", "t3"), failed_tests=("t4",)
        ),
        "a-low-fix": TestResult(suite="mcts", error="not reached"),
        "b-mid-fix": TestResult(suite="mcts", error="not reached"),
        "c-high-fix": TestResult(suite="mcts", passed_tests=("t1", "t2")),
    }
    return BranchCase(
        task="follow the highest-quality observed parent",
        root_files={"src/solver.py": "def solve(): return 'root'\n"},
        candidates=candidates,
        tests=tests,
    )


class BranchSearchTests(unittest.TestCase):
    def test_demo_is_deterministic_and_exposes_search_accounting(self) -> None:
        first = BranchSearch().run(demo_case())
        second = BranchSearch().run(demo_case())
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.status, "accepted")
        self.assertEqual(first.best_node_id, "iterative-fix")
        self.assertEqual(dict(first.metrics)["duplicates"], 1)
        self.assertEqual(dict(first.metrics)["pruned"], 1)
        self.assertEqual(dict(first.metrics)["test_calls"], 5)

    def test_beam_pruning_and_workspace_deduplication_are_auditable(self) -> None:
        report = BranchSearch().run(demo_case())
        nodes = {node.id: node for node in report.nodes}
        self.assertEqual(nodes["candidate-greedy"].status, "pruned")
        self.assertEqual(nodes["recursive-fix"].status, "duplicate")
        self.assertEqual(nodes["recursive-fix"].duplicate_of, "iterative-fix")
        result = nodes["iterative-fix"].test_result
        if result is None:
            self.fail("accepted candidate must have a test result")
        self.assertEqual(result.total_tests, 5)
        self.assertEqual(result.passed_tests, tuple(sorted(result.passed_tests)))
        verify_search_events(report.events)
        self.assertEqual(report.events[0].type, "search.started")
        self.assertEqual(report.events[-1].type, "search.completed")

    def test_mcts_propagates_observed_quality_to_the_next_patch(self) -> None:
        config = BranchSearchConfig(search_policy="mcts", max_candidates=4, max_depth=2)
        first = BranchSearch(config).run(_mcts_case())
        second = BranchSearch(config).run(_mcts_case())
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(
            first.to_dict(), BranchSearchReport.from_dict(first.to_dict()).to_dict()
        )
        self.assertEqual(first.status, "accepted")
        self.assertEqual(first.best_node_id, "c-high-fix")
        selected = [
            event.data["candidate_id"]
            for event in first.events
            if event.type == "candidate.selected"
        ]
        self.assertEqual(selected, ["a-low", "b-mid", "c-high", "c-high-fix"])
        choice = next(
            event
            for event in first.events
            if event.type == "candidate.selected"
            and event.data["candidate_id"] == "c-high-fix"
        )
        self.assertEqual(choice.data["parent_id"], "c-high")
        self.assertEqual(choice.data["parent_visits"], 1)
        self.assertGreater(choice.data["parent_mean_quality"], 0.5)
        self.assertEqual(choice.data["selection_policy"], "mcts")
        verify_search_events(first.events)

    def test_report_round_trip_and_html_are_reproducible(self) -> None:
        report = BranchSearch().run(demo_case())
        payload = report.to_dict()
        restored = BranchSearchReport.from_dict(payload)
        validate_search_report(restored)
        self.assertEqual(payload, restored.to_dict())
        self.assertEqual(render_branch_html(report), render_branch_html(restored))
        self.assertIn("<svg", render_branch_html(report))
        self.assertNotIn("https://", render_branch_html(report))

    def test_event_tampering_is_rejected(self) -> None:
        report = BranchSearch().run(demo_case())
        tampered = list(report.events)
        tampered[1] = replace(tampered[1], data={"candidate_id": "forged"})
        with self.assertRaises(ValueError):
            verify_search_events(tampered)

    def test_candidate_budget_stops_without_fake_success(self) -> None:
        report = BranchSearch(
            BranchSearchConfig(max_candidates=1, max_depth=4, beam_width=2)
        ).run(demo_case())
        self.assertEqual(report.status, "budget_exhausted")
        self.assertIsNone(report.best_node_id)
        self.assertEqual(dict(report.metrics)["proposed"], 1)
        self.assertEqual(dict(report.metrics)["test_calls"], 1)

    def test_invalid_candidate_paths_and_empty_results_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CandidatePatch(
                id="bad",
                parent_id="root",
                hypothesis="escape",
                files={"../outside.py": "x"},
            )
        with self.assertRaises(ValueError):
            TestResult(suite="empty")

    def test_cli_writes_json_markdown_and_html_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            json_path = root / "branch.json"
            markdown_path = root / "branch.md"
            html_path = root / "branch.html"
            exit_code = main(
                [
                    "branch-search",
                    "--search-policy",
                    "mcts",
                    "--output",
                    str(json_path),
                    "--markdown",
                    str(markdown_path),
                    "--html",
                    str(html_path),
                ]
            )
            self.assertEqual(exit_code, 0)
            report = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "accepted")
            self.assertIn("deduplicated", markdown_path.read_text(encoding="utf-8"))
            self.assertIn("<svg", html_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
