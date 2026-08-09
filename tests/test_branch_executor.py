from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

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

    def test_candidate_commands_receive_sanitized_environment(self) -> None:
        candidate = _case().by_id["good"]
        with patch.dict(
            os.environ,
            {
                "CONTEXTOPT_API_KEY": "must-not-leak",
                "OPENAI_API_KEY": "must-not-leak-either",
                "CONTEXTOPT_TEST_MARKER": "keep-me",
            },
            clear=False,
        ):
            result = evaluate_candidate(
                candidate,
                ExecutableSearchConfig(
                    command=(
                        sys.executable,
                        "-c",
                        (
                            "import os; "
                            "print(os.getenv('CONTEXTOPT_API_KEY', 'missing')); "
                            "print(os.getenv('OPENAI_API_KEY', 'missing')); "
                            "print(os.getenv('CONTEXTOPT_TEST_MARKER', 'missing'))"
                        ),
                    ),
                    timeout_seconds=10,
                ),
            )
        self.assertTrue(result.is_success)
        output = result.output_excerpt or ""
        self.assertNotIn("must-not-leak", output)
        self.assertIn("missing", output)
        self.assertIn("keep-me", output)

    def test_timeout_terminates_descendant_processes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "descendant-survived.txt"
            child = (
                "import pathlib,sys,time; time.sleep(2.0); "
                "pathlib.Path(sys.argv[1]).write_text('survived', encoding='utf-8')"
            )
            parent = (
                "import subprocess,sys,time; "
                f"subprocess.Popen([sys.executable, '-c', {child!r}, "
                f"{str(marker)!r}]); "
                "time.sleep(5)"
            )
            result = evaluate_candidate(
                _case().by_id["good"],
                ExecutableSearchConfig(
                    command=(sys.executable, "-c", parent),
                    timeout_seconds=0.5,
                ),
            )
            self.assertIn("process group terminated", result.error or "")
            time.sleep(2.4)
            self.assertFalse(marker.exists())

    def test_docker_sandbox_is_explicit_and_reports_missing_runtime(self) -> None:
        config = ExecutableSearchConfig(
            command=(sys.executable, "-c", "print('sandbox')"),
            sandbox="docker",
            container_image="python:3.12-slim@sha256:example",
        )
        self.assertEqual(ExecutableSearchConfig.from_dict(config.to_dict()), config)
        with patch("contextopt.search.executor.shutil.which", return_value=None):
            result = evaluate_candidate(_case().by_id["good"], config)
        self.assertFalse(result.is_success)
        self.assertIn("docker executable was not found", result.error or "")

    @unittest.skipUnless(shutil.which("docker"), "Docker is not installed")
    def test_docker_sandbox_executes_candidate_on_enabled_runner(self) -> None:
        result = evaluate_candidate(
            _case().by_id["good"],
            ExecutableSearchConfig(
                command=(
                    "python",
                    "-c",
                    "from pathlib import Path; print(Path('solver.py').exists())",
                ),
                sandbox="docker",
                container_image="python:3.12-slim",
                timeout_seconds=30,
            ),
        )
        self.assertTrue(result.is_success, result.error)
        self.assertIn("True", result.output_excerpt or "")

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
