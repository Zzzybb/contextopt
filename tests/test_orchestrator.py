from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from contextopt.cli import main
from contextopt.runtime.model import ScriptedModel
from contextopt.runtime.protocol import ModelResponse, ToolCall
from contextopt.search import (
    BranchSearchConfig,
    ExecutableSearchConfig,
    OrchestrationConfig,
    PlannerConfig,
    ProposalConfig,
    ReviewerConfig,
    parse_planner_response,
    parse_reviewer_response,
    read_orchestration_checkpoint,
    run_orchestration,
)

ROOT_FILES = {
    "solver.py": "def solve(values):\n    return list(values)\n",
    "test_solver.py": (
        "import unittest\n"
        "from solver import solve\n\n\n"
        "class SolverTests(unittest.TestCase):\n"
        "    def test_orders_values(self):\n"
        "        self.assertEqual(solve([3, 1, 2]), [1, 2, 3])\n\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n"
    ),
}


def _plan() -> str:
    return json.dumps(
        {
            "goal": "return values in ascending order",
            "constraints": ["preserve the public solve signature"],
            "hypotheses": ["use the standard sorted primitive"],
            "test_focus": ["visible ordering test"],
            "risks": ["do not mutate caller-owned input"],
        }
    )


def _proposal(candidate_id: str, implementation: str) -> str:
    return json.dumps(
        {
            "candidates": [
                {
                    "id": candidate_id,
                    "parent_id": "root",
                    "hypothesis": f"try {candidate_id}",
                    "files": {
                        "solver.py": implementation,
                        "test_solver.py": ROOT_FILES["test_solver.py"],
                    },
                    "evidence": ["visible test is the oracle"],
                }
            ]
        }
    )


def _review(decision: str, candidate_id: str | None) -> str:
    return json.dumps(
        {
            "decision": decision,
            "candidate_id": candidate_id,
            "confidence": 0.9 if decision == "accept" else 0.4,
            "blocking_issues": []
            if decision == "accept"
            else ["visible test evidence is incomplete"],
            "required_checks": []
            if decision == "accept"
            else ["run the visible test again"],
            "summary": "the visible oracle supports this decision",
        }
    )


def _execution() -> ExecutableSearchConfig:
    return ExecutableSearchConfig(
        command=(sys.executable, "-m", "unittest", "discover", "-s", "."),
        suite="orchestrator-visible-tests",
        test_name="solver-order",
    )


class OrchestratorTests(unittest.IsolatedAsyncioTestCase):
    async def test_roles_share_budget_and_reviewer_acceptance_is_oracle_gated(
        self,
    ) -> None:
        planner = ScriptedModel(
            [
                {"expect": {"turn": 1}, "response": {"content": _plan()}},
                {
                    "expect": {"turn": 2},
                    "response": {"content": _plan()},
                },
            ],
            name="planner-script:v1",
        )
        solver = ScriptedModel(
            [
                {
                    "expect": {"turn": 1},
                    "response": {
                        "content": _proposal(
                            "bad",
                            "def solve(values):\n    return list(reversed(values))\n",
                        )
                    },
                },
                {
                    "expect": {"turn": 2},
                    "response": {
                        "content": _proposal(
                            "good",
                            "def solve(values):\n    return sorted(values)\n",
                        )
                    },
                },
            ],
            name="solver-script:v1",
        )
        reviewer = ScriptedModel(
            [
                {
                    "expect": {"turn": 1},
                    "response": {"content": _review("retry", None)},
                },
                {
                    "expect": {"turn": 2},
                    "response": {"content": _review("accept", "round-1-good")},
                },
            ],
            name="reviewer-script:v1",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "orchestration.json"
            report = await run_orchestration(
                planner,
                solver,
                reviewer,
                task="make solve return ascending values",
                root_files=ROOT_FILES,
                execution_config=_execution(),
                config=OrchestrationConfig(
                    max_rounds=2,
                    max_model_calls=6,
                    max_planner_calls=2,
                    max_solver_calls=2,
                    max_reviewer_calls=2,
                    max_candidates=2,
                    max_test_calls=2,
                ),
                planner_config=PlannerConfig(max_items=2),
                solver_config=ProposalConfig(max_candidates=1),
                reviewer_config=ReviewerConfig(),
                search_config=BranchSearchConfig(max_depth=1, beam_width=1),
                run_id="orchestrator-test",
                checkpoint_path=checkpoint,
            )
            self.assertEqual(report.status, "accepted")
            self.assertEqual(report.best_candidate_id, "round-1-good")
            self.assertEqual(report.model_calls, 6)
            self.assertEqual(report.test_calls, 2)
            self.assertEqual(len(report.rounds), 2)
            self.assertIn("review", planner.requests[1].messages[-1].content)
            self.assertEqual(
                read_orchestration_checkpoint(checkpoint).to_dict(), report.to_dict()
            )

    async def test_reviewer_cannot_accept_failing_candidate(self) -> None:
        planner = ScriptedModel([{"response": {"content": _plan()}}], name="p:v1")
        solver = ScriptedModel(
            [
                {
                    "response": {
                        "content": _proposal(
                            "bad",
                            "def solve(values):\n    return list(reversed(values))\n",
                        )
                    }
                }
            ],
            name="s:v1",
        )
        reviewer = ScriptedModel(
            [{"response": {"content": _review("accept", "round-0-bad")}}],
            name="r:v1",
        )
        report = await run_orchestration(
            planner,
            solver,
            reviewer,
            task="make solve return ascending values",
            root_files=ROOT_FILES,
            execution_config=_execution(),
            config=OrchestrationConfig(
                max_rounds=1,
                max_model_calls=3,
                max_planner_calls=1,
                max_solver_calls=1,
                max_reviewer_calls=1,
                max_candidates=1,
                max_test_calls=1,
            ),
            solver_config=ProposalConfig(max_candidates=1),
            search_config=BranchSearchConfig(max_depth=1, beam_width=1),
        )
        self.assertEqual(report.status, "budget_exhausted")
        self.assertEqual(report.best_candidate_id, "round-0-bad")
        self.assertEqual(report.rounds[0].review.decision, "accept")
        self.assertTrue(any(event.type == "reviewer.gated" for event in report.events))

    def test_role_parsers_reject_tools_and_unknown_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not contain tool calls"):
            parse_planner_response(
                ModelResponse(
                    content=_plan(),
                    tool_calls=(
                        ToolCall(id="call-1", name="run_tests", arguments_json="{}"),
                    ),
                )
            )
        invalid = json.loads(_review("retry", None))
        invalid["extra"] = True
        with self.assertRaisesRegex(ValueError, "only the review fields"):
            parse_reviewer_response(ModelResponse(content=json.dumps(invalid)))

    def test_cli_orchestrate_supports_three_offline_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            root_files = root / "root.json"
            planner_script = root / "planner.json"
            solver_script = root / "solver.json"
            reviewer_script = root / "reviewer.json"
            output = root / "report.json"
            root_files.write_text(json.dumps(ROOT_FILES), encoding="utf-8")
            planner_script.write_text(
                json.dumps([{"response": {"content": _plan()}}]),
                encoding="utf-8",
            )
            solver_script.write_text(
                json.dumps(
                    [
                        {
                            "response": {
                                "content": _proposal(
                                    "good",
                                    "def solve(values):\n    return sorted(values)\n",
                                )
                            }
                        }
                    ]
                ),
                encoding="utf-8",
            )
            reviewer_script.write_text(
                json.dumps(
                    [{"response": {"content": _review("accept", "round-0-good")}}]
                ),
                encoding="utf-8",
            )
            exit_code = main(
                [
                    "orchestrate",
                    "--task",
                    "make solve return ascending values",
                    "--root-files",
                    str(root_files),
                    "--checkpoint",
                    str(root / "checkpoint.json"),
                    "--planner-script",
                    str(planner_script),
                    "--solver-script",
                    str(solver_script),
                    "--reviewer-script",
                    str(reviewer_script),
                    "--test-command",
                    f"{sys.executable} -m unittest discover -s .",
                    "--allow-command",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8"))["status"], "accepted"
            )


if __name__ == "__main__":
    unittest.main()
