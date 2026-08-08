from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from contextopt.cli import main
from contextopt.search import (
    BranchCase,
    BranchSearchConfig,
    CandidatePatch,
    ExecutableSearchConfig,
    TestResult,
    evaluate_candidate,
    evaluate_case,
    run_executable_search,
)


def _case() -> BranchCase:
    tests = {
        "bad": TestResult(suite="placeholder", error="not executed"),
        "good": TestResult(suite="placeholder", error="not executed"),
        "good-dup": TestResult(suite="placeholder", error="not executed"),
    }
    test_file = """import unittest
from solver import solve


class SolverTests(unittest.TestCase):
    def test_duplicates_are_preserved(self):
        self.assertEqual(solve([2, 1, 2]), [1, 2, 2])


if __name__ == "__main__":
    unittest.main()
"""
    bad_files = {
        "solver.py": "def solve(values):\n    return sorted(set(values))\n",
        "test_solver.py": test_file,
    }
    good_files = {
        "solver.py": "def solve(values):\n    return sorted(values)\n",
        "test_solver.py": test_file,
    }
    candidates = (
        CandidatePatch(
            id="bad",
            parent_id="root",
            hypothesis="remove duplicates for speed",
            files=bad_files,
        ),
        CandidatePatch(
            id="good",
            parent_id="root",
            hypothesis="sort without changing multiplicity",
            files=good_files,
        ),
        CandidatePatch(
            id="good-dup",
            parent_id="root",
            hypothesis="equivalent implementation from another proposal",
            files=good_files,
        ),
    )
    return BranchCase(
        task="preserve duplicates while sorting",
        root_files={
            "solver.py": "def solve(values):\n    return values\n",
            "test_solver.py": test_file,
        },
        candidates=candidates,
        tests=tests,
    )


class BranchExecutorTests(unittest.TestCase):
    def test_real_unittest_observations_drive_search_and_are_bounded(self) -> None:
        case = _case()
        config = ExecutableSearchConfig(
            command=(sys.executable, "-m", "unittest", "discover", "-s", "."),
            timeout_seconds=10,
            max_report_bytes=2_048,
        )
        observed = evaluate_case(case, config)
        self.assertFalse(observed.tests["bad"].is_success)
        self.assertTrue(observed.tests["good"].is_success)
        self.assertIs(observed.tests["good"], observed.tests["good-dup"])
        self.assertIsNotNone(observed.tests["good"].output_sha256)
        self.assertLessEqual(len(observed.tests["good"].output_excerpt or ""), 4096)
        self.assertEqual(
            TestResult.from_dict(observed.tests["good"].to_dict()),
            observed.tests["good"],
        )

        report = run_executable_search(
            case,
            BranchSearchConfig(beam_width=2, max_depth=1),
            config,
        )
        self.assertEqual(report.status, "accepted")
        self.assertEqual(report.best_node_id, "good")
        self.assertEqual(report.metrics["duplicates"], 1)

    def test_timeout_is_an_observable_error(self) -> None:
        candidate = _case().by_id["good"]
        result = evaluate_candidate(
            candidate,
            ExecutableSearchConfig(
                command=(sys.executable, "-c", "import time; time.sleep(0.2)"),
                timeout_seconds=0.01,
            ),
        )
        self.assertFalse(result.is_success)
        self.assertIsNotNone(result.error)
        self.assertIn("timed out", result.error or "")

    def test_cli_executes_a_serialized_case_only_with_explicit_permission(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case_path = root / "case.json"
            report_path = root / "report.json"
            case_path.write_text(json.dumps(_case().to_dict()), encoding="utf-8")
            with self.assertRaises(SystemExit):
                main(
                    [
                        "branch-search",
                        str(case_path),
                        "--test-command",
                        f"{sys.executable} -m unittest discover -s .",
                    ]
                )
            exit_code = main(
                [
                    "branch-search",
                    str(case_path),
                    "--test-command",
                    f"{sys.executable} -m unittest discover -s .",
                    "--allow-command",
                    "--output",
                    str(report_path),
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertEqual(json.loads(report_path.read_text())["status"], "accepted")


if __name__ == "__main__":
    unittest.main()
