from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

from contextopt.cli import main
from contextopt.runtime.identity import stable_hash
from contextopt.runtime.model import ScriptedModel
from contextopt.runtime.protocol import ModelRequest, ModelResponse, ToolCall
from contextopt.search import (
    BranchSearchConfig,
    CandidatePatch,
    ExecutableSearchConfig,
    OrchestrationConfig,
    PlannerConfig,
    ProposalConfig,
    ReviewerConfig,
    parse_planner_response,
    parse_reviewer_response,
    read_orchestration_checkpoint,
    run_orchestration,
    three_way_merge,
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


def _multi_proposal(count: int = 4) -> str:
    return json.dumps(
        {
            "candidates": [
                {
                    "id": f"candidate-{index}",
                    "parent_id": "root",
                    "hypothesis": f"parallel candidate {index}",
                    "files": {
                        "solver.py": (
                            "def solve(values):\n"
                            "    return sorted(values)\n"
                            f"# candidate {index}\n"
                        ),
                        "test_solver.py": ROOT_FILES["test_solver.py"],
                    },
                    "evidence": ["candidate is independently testable"],
                }
                for index in range(count)
            ]
        }
    )


def _execution() -> ExecutableSearchConfig:
    return ExecutableSearchConfig(
        command=(sys.executable, "-m", "unittest", "discover", "-s", "."),
        suite="orchestrator-visible-tests",
        test_name="solver-order",
    )


def _parallel_execution() -> ExecutableSearchConfig:
    return ExecutableSearchConfig(
        command=(sys.executable, "-c", "import time; time.sleep(0.15)"),
        suite="orchestrator-parallel-tests",
        test_name="parallel-smoke",
    )


class DelayedScriptedModel(ScriptedModel):
    """Make overlapping provider awaits observable without network access."""

    async def complete(self, request):  # type: ignore[no-untyped-def]
        await asyncio.sleep(0.05)
        return await super().complete(request)


class FirstValidSpeculativeModel:
    """Return lane one immediately and make lane two cancellation-observable."""

    def __init__(self, content: str) -> None:
        self._name = "first-valid-speculative-solver:v1"
        self._content = content
        self._never = asyncio.Event()
        self.requests: list[ModelRequest] = []
        self.cancelled = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def configuration_fingerprint(self) -> str:
        return stable_hash({"adapter": "first-valid-test", "name": self._name})

    def resume_from_turn(self, completed_turns: int) -> None:
        if completed_turns < 0:
            raise ValueError("completed_turns must be non-negative")

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if "You are lane 2 of 2" in request.messages[-1].content:
            try:
                await self._never.wait()
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
        await asyncio.sleep(0)
        return ModelResponse(content=self._content)


class PartialSpeculativeModel:
    """Return one lane, then block another so checkpoint reuse can be tested."""

    def __init__(self, content: str, *, block_after: int | None) -> None:
        self._name = "partial-speculative-solver:v1"
        self._content = content
        self._block_after = block_after
        self._never = asyncio.Event()
        self.requests: list[ModelRequest] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def configuration_fingerprint(self) -> str:
        return stable_hash({"adapter": "partial-test", "name": self._name})

    def resume_from_turn(self, completed_turns: int) -> None:
        if completed_turns < 0:
            raise ValueError("completed_turns must be non-negative")

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if self._block_after is not None and len(self.requests) > self._block_after:
            await self._never.wait()
        else:
            await asyncio.sleep(0.01)
        return ModelResponse(content=self._content)


MERGE_ROOT_FILES = {
    "one.py": "def one():\n    return 0\n",
    "two.py": "def two():\n    return 0\n",
    "test_merge.py": (
        "import unittest\n"
        "from one import one\n"
        "from two import two\n\n\n"
        "class MergeTests(unittest.TestCase):\n"
        "    def test_both_branches(self):\n"
        "        self.assertEqual(one(), 1)\n"
        "        self.assertEqual(two(), 2)\n\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n"
    ),
}


def _merge_proposal() -> str:
    candidates = []
    for candidate_id, one, two in (
        ("left", "def one():\n    return 1\n", MERGE_ROOT_FILES["two.py"]),
        ("right", MERGE_ROOT_FILES["one.py"], "def two():\n    return 2\n"),
    ):
        candidates.append(
            {
                "id": candidate_id,
                "parent_id": "root",
                "hypothesis": f"change {candidate_id} file",
                "files": {
                    "one.py": one,
                    "two.py": two,
                    "test_merge.py": MERGE_ROOT_FILES["test_merge.py"],
                },
                "evidence": ["independent branch"],
            }
        )
    return json.dumps({"candidates": candidates})


class OrchestratorTests(unittest.IsolatedAsyncioTestCase):
    def test_three_way_merge_is_conflict_safe(self) -> None:
        left = CandidatePatch(
            id="left",
            parent_id="root",
            hypothesis="left",
            files={"one.py": "left", "two.py": "base"},
        )
        right = CandidatePatch(
            id="right",
            parent_id="root",
            hypothesis="right",
            files={"one.py": "right", "two.py": "base"},
        )
        merged, conflicts = three_way_merge(
            {"one.py": "base", "two.py": "base"}, left, right
        )
        self.assertIsNone(merged)
        self.assertEqual([conflict.path for conflict in conflicts], ["one.py"])

    async def test_disjoint_merge_adds_candidate_and_records_evidence(self) -> None:
        report = await run_orchestration(
            ScriptedModel([{"response": {"content": _plan()}}], name="merge-p:v1"),
            ScriptedModel(
                [{"response": {"content": _merge_proposal()}}], name="merge-s:v1"
            ),
            ScriptedModel(
                [
                    {
                        "response": {
                            "content": _review("accept", "round-0-merge-left--right")
                        }
                    }
                ],
                name="merge-r:v1",
            ),
            task="fix both independent modules",
            root_files=MERGE_ROOT_FILES,
            execution_config=_execution(),
            config=OrchestrationConfig(
                max_rounds=1,
                max_model_calls=3,
                max_planner_calls=1,
                max_solver_calls=1,
                max_reviewer_calls=1,
                max_candidates=3,
                max_test_calls=3,
                max_parallel_tests=3,
                merge_policy="disjoint",
            ),
            solver_config=ProposalConfig(max_candidates=2),
            search_config=BranchSearchConfig(max_depth=1, beam_width=3),
            run_id="merge-orchestration-test",
        )
        self.assertEqual(report.status, "accepted")
        self.assertEqual(report.best_candidate_id, "round-0-merge-left--right")
        self.assertEqual(report.test_calls, 3)
        merge_events = [
            event for event in report.events if event.type == "solver.merge"
        ]
        self.assertEqual(len(merge_events), 1)
        self.assertEqual(
            merge_events[0].data["merged_candidate_ids"], ["merge-left--right"]
        )
        self.assertEqual(
            merge_events[0].data["namespaced_candidate_ids"],
            ["round-0-merge-left--right"],
        )

    async def test_parallel_candidate_scheduler_uses_isolated_workspaces(self) -> None:
        planner = ScriptedModel(
            [{"response": {"content": _plan()}}], name="parallel-planner:v1"
        )
        solver = ScriptedModel(
            [{"response": {"content": _multi_proposal()}}],
            name="parallel-solver:v1",
        )
        reviewer = ScriptedModel(
            [{"response": {"content": _review("accept", "round-0-candidate-0")}}],
            name="parallel-reviewer:v1",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "parallel-orchestration.json"
            report = await run_orchestration(
                planner,
                solver,
                reviewer,
                task="find a correct sorting implementation",
                root_files=ROOT_FILES,
                execution_config=_parallel_execution(),
                config=OrchestrationConfig(
                    max_rounds=1,
                    max_model_calls=3,
                    max_planner_calls=1,
                    max_solver_calls=1,
                    max_reviewer_calls=1,
                    max_candidates=4,
                    max_test_calls=4,
                    max_parallel_tests=2,
                ),
                solver_config=ProposalConfig(max_candidates=4),
                search_config=BranchSearchConfig(max_depth=1, beam_width=4),
                run_id="parallel-orchestration-test",
                checkpoint_path=checkpoint,
            )
            self.assertEqual(report.status, "accepted")
            self.assertEqual(report.test_calls, 4)
            self.assertEqual(report.max_in_flight, 2)
            self.assertEqual(
                sum(event.type == "candidate.requested" for event in report.events),
                4,
            )
            self.assertEqual(
                sum(event.type == "candidate.completed" for event in report.events),
                4,
            )
            self.assertEqual(
                read_orchestration_checkpoint(checkpoint).to_dict(), report.to_dict()
            )

    async def test_speculative_solver_lanes_run_concurrently_and_are_audited(
        self,
    ) -> None:
        planner = DelayedScriptedModel(
            [{"response": {"content": _plan()}}], name="speculative-planner:v1"
        )
        solver = DelayedScriptedModel(
            [
                {
                    "response": {
                        "content": _proposal(
                            "same", "def solve(values):\n    return sorted(values)\n"
                        )
                    }
                },
                {
                    "response": {
                        "content": _proposal(
                            "same", "def solve(values):\n    return sorted(values)\n"
                        )
                    }
                },
            ],
            name="speculative-solver:v1",
        )
        reviewer = DelayedScriptedModel(
            [{"response": {"content": _review("accept", "round-0-spec-0-same")}}],
            name="speculative-reviewer:v1",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "speculative-orchestration.json"
            report = await run_orchestration(
                planner,
                solver,
                reviewer,
                task="find a correct sorting implementation",
                root_files=ROOT_FILES,
                execution_config=_execution(),
                config=OrchestrationConfig(
                    max_rounds=1,
                    max_model_calls=4,
                    max_planner_calls=1,
                    max_solver_calls=2,
                    max_reviewer_calls=1,
                    max_candidates=2,
                    max_test_calls=2,
                    speculative_solver_width=2,
                ),
                solver_config=ProposalConfig(max_candidates=1),
                search_config=BranchSearchConfig(max_depth=1, beam_width=2),
                run_id="speculative-orchestration-test",
                checkpoint_path=checkpoint,
            )
            checkpoint_report = read_orchestration_checkpoint(checkpoint)
        self.assertEqual(report.status, "accepted")
        self.assertEqual(report.solver_calls, 2)
        self.assertEqual(report.model_calls, 4)
        self.assertEqual(report.max_provider_in_flight, 2)
        self.assertEqual(len(report.rounds[0].solver_variants), 2)
        self.assertEqual(
            [call.response_id for call in report.rounds[0].solver_variants],
            [None, None],
        )
        self.assertEqual(
            sum(event.type == "solver.speculative.received" for event in report.events),
            2,
        )
        received = [event for event in report.events if event.type == "solver.received"]
        self.assertEqual(received[0].data["speculative_width"], 2)
        self.assertEqual(received[0].data["valid_lanes"], [0, 1])
        self.assertEqual(checkpoint_report.to_dict(), report.to_dict())

    async def test_speculative_solver_stops_after_first_parseable_candidate(
        self,
    ) -> None:
        solver = FirstValidSpeculativeModel(
            _proposal("good", "def solve(values):\n    return sorted(values)\n")
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "first-valid-speculative.json"
            report = await run_orchestration(
                ScriptedModel([{"response": {"content": _plan()}}], name="stop-p:v1"),
                solver,
                ScriptedModel(
                    [
                        {
                            "response": {
                                "content": _review("accept", "round-0-spec-0-good")
                            }
                        }
                    ],
                    name="stop-r:v1",
                ),
                task="find a correct sorting implementation",
                root_files=ROOT_FILES,
                execution_config=_execution(),
                config=OrchestrationConfig(
                    max_rounds=1,
                    max_model_calls=4,
                    max_planner_calls=1,
                    max_solver_calls=2,
                    max_reviewer_calls=1,
                    max_candidates=2,
                    max_test_calls=2,
                    speculative_solver_width=2,
                    speculative_solver_stop_on_valid=True,
                ),
                solver_config=ProposalConfig(max_candidates=1),
                search_config=BranchSearchConfig(max_depth=1, beam_width=1),
                run_id="first-valid-speculative-test",
                checkpoint_path=checkpoint,
            )
            checkpoint_report = read_orchestration_checkpoint(checkpoint)
        self.assertEqual(report.status, "accepted")
        self.assertEqual(report.best_candidate_id, "round-0-spec-0-good")
        self.assertEqual(report.solver_calls, 2)
        self.assertEqual(report.model_calls, 4)
        self.assertEqual(len(report.rounds[0].solver_variants), 1)
        self.assertEqual(report.metrics["speculative_winners"], 1)
        self.assertEqual(report.metrics["cancelled_solver_lanes"], 1)
        self.assertEqual(
            sum(event.type == "solver.speculative.winner" for event in report.events),
            1,
        )
        self.assertEqual(
            sum(
                event.type == "solver.speculative.cancelled" for event in report.events
            ),
            1,
        )
        self.assertEqual(solver.cancelled, 1)
        self.assertEqual(checkpoint_report.to_dict(), report.to_dict())

    async def test_speculative_resume_reuses_durable_lane_response(self) -> None:
        planner_steps = [{"response": {"content": _plan()}}]
        reviewer_steps = [
            {"response": {"content": _review("accept", "round-0-spec-0-good")}}
        ]
        planner = ScriptedModel(planner_steps, name="partial-planner:v1")
        reviewer = ScriptedModel(reviewer_steps, name="partial-reviewer:v1")
        solver = PartialSpeculativeModel(
            _proposal("good", "def solve(values):\n    return sorted(values)\n"),
            block_after=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "partial-speculative.json"
            runner_task = asyncio.create_task(
                run_orchestration(
                    planner,
                    solver,
                    reviewer,
                    task="find a correct sorting implementation",
                    root_files=ROOT_FILES,
                    execution_config=_execution(),
                    config=OrchestrationConfig(
                        max_rounds=1,
                        max_model_calls=4,
                        max_planner_calls=1,
                        max_solver_calls=2,
                        max_reviewer_calls=1,
                        max_candidates=2,
                        max_test_calls=2,
                        speculative_solver_width=2,
                    ),
                    solver_config=ProposalConfig(max_candidates=1),
                    search_config=BranchSearchConfig(max_depth=1, beam_width=2),
                    run_id="partial-speculative-test",
                    checkpoint_path=checkpoint,
                )
            )
            for _ in range(100):
                if checkpoint.exists():
                    partial = read_orchestration_checkpoint(checkpoint)
                    if partial.pending_solver_responses:
                        break
                await asyncio.sleep(0.01)
            else:
                self.fail("solver lane response was not checkpointed")
            runner_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await runner_task
            partial = read_orchestration_checkpoint(checkpoint)
            self.assertEqual(
                sorted(partial.pending_solver_responses or {}),
                [0],
            )

            resumed_solver = PartialSpeculativeModel(
                _proposal("good", "def solve(values):\n    return sorted(values)\n"),
                block_after=None,
            )
            resumed = await run_orchestration(
                ScriptedModel(planner_steps, name="partial-planner:v1"),
                resumed_solver,
                ScriptedModel(reviewer_steps, name="partial-reviewer:v1"),
                checkpoint_path=checkpoint,
                resume=True,
            )
        self.assertEqual(resumed.status, "accepted")
        self.assertEqual(resumed.solver_calls, 2)
        self.assertEqual(len(resumed_solver.requests), 1)
        self.assertIsNone(resumed.pending_solver_responses)
        self.assertTrue(
            any(event.type == "solver.speculative.reused" for event in resumed.events)
        )

    async def test_adaptive_scheduler_stops_after_first_passing_batch(self) -> None:
        report = await run_orchestration(
            ScriptedModel([{"response": {"content": _plan()}}], name="adaptive-p:v1"),
            ScriptedModel(
                [{"response": {"content": _multi_proposal()}}],
                name="adaptive-s:v1",
            ),
            ScriptedModel(
                [{"response": {"content": _review("accept", "round-0-candidate-0")}}],
                name="adaptive-r:v1",
            ),
            task="find a correct sorting implementation",
            root_files=ROOT_FILES,
            execution_config=_parallel_execution(),
            config=OrchestrationConfig(
                max_rounds=1,
                max_model_calls=3,
                max_planner_calls=1,
                max_solver_calls=1,
                max_reviewer_calls=1,
                max_candidates=4,
                max_test_calls=4,
                max_parallel_tests=2,
                scheduler_policy="adaptive",
            ),
            solver_config=ProposalConfig(max_candidates=4),
            search_config=BranchSearchConfig(max_depth=1, beam_width=4),
            run_id="adaptive-orchestration-test",
        )
        self.assertEqual(report.status, "accepted")
        self.assertEqual(report.test_calls, 2)
        self.assertEqual(report.max_in_flight, 2)

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
            self.assertTrue(
                all(
                    call.context_receipt is not None
                    for item in report.rounds
                    for call in (
                        item.planner_call,
                        item.solver_call,
                        item.reviewer_call,
                    )
                )
            )
            self.assertTrue(
                all(
                    any(message.role == "assistant" for message in request.messages)
                    for request in (
                        planner.requests[1],
                        solver.requests[1],
                        reviewer.requests[1],
                    )
                )
            )
            self.assertEqual(
                {
                    role: len(messages)
                    for role, messages in report.role_histories.items()
                },
                {"planner": 2, "solver": 2, "reviewer": 2},
            )
            self.assertTrue(
                all(
                    call.context_receipt.memory_fingerprint is not None
                    for item in report.rounds
                    for call in (
                        item.planner_call,
                        item.solver_call,
                        item.reviewer_call,
                    )
                )
            )
            self.assertEqual(
                read_orchestration_checkpoint(checkpoint).to_dict(), report.to_dict()
            )
            legacy = report.to_dict()
            legacy["config"].pop("context_config")
            legacy["config"].pop("max_parallel_tests")
            legacy.pop("max_in_flight")
            legacy.pop("role_histories")
            for round_payload in legacy["rounds"]:
                for role in ("planner_call", "solver_call", "reviewer_call"):
                    round_payload[role].pop("context_receipt")
            restored_legacy = type(report).from_dict(legacy)
            self.assertEqual(
                restored_legacy.role_histories,
                {"planner": (), "solver": (), "reviewer": ()},
            )
            self.assertIsNone(restored_legacy.rounds[0].planner_call.context_receipt)

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
