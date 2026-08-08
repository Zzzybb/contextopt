from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from contextopt.runtime.model import ScriptedModel
from contextopt.runtime.protocol import ModelRequest, ModelResponse
from contextopt.search import (
    BranchSearchConfig,
    ExecutableSearchConfig,
    ProposalConfig,
    SearchSessionConfig,
    SearchSessionReport,
    read_session_checkpoint,
    run_search_session,
    verify_search_events,
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


def _proposal(candidate_id: str, implementation: str) -> str:
    return json.dumps(
        {
            "candidates": [
                {
                    "id": candidate_id,
                    "parent_id": "root",
                    "hypothesis": f"use {candidate_id}",
                    "files": {
                        "solver.py": implementation,
                        "test_solver.py": ROOT_FILES["test_solver.py"],
                    },
                    "evidence": ["visible test is the oracle"],
                }
            ]
        }
    )


def _scripted_model() -> ScriptedModel:
    return ScriptedModel(
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
        name="session-script:v1",
    )


def _repeating_model() -> ScriptedModel:
    implementation = "def solve(values):\n    return list(reversed(values))\n"
    return ScriptedModel(
        [
            {
                "expect": {"turn": 1},
                "response": {"content": _proposal("bad", implementation)},
            },
            {
                "expect": {"turn": 2},
                "response": {"content": _proposal("same-state", implementation)},
            },
        ],
        name="session-repeating-script:v1",
    )


def _multi_candidate_model(count: int = 4) -> ScriptedModel:
    candidates = []
    for index in range(count):
        candidates.append(
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
        )
    return ScriptedModel(
        [
            {
                "expect": {"turn": 1},
                "response": {"content": json.dumps({"candidates": candidates})},
            }
        ],
        name="parallel-session-script:v1",
    )


def _execution() -> ExecutableSearchConfig:
    return ExecutableSearchConfig(
        command=(sys.executable, "-m", "unittest", "discover", "-s", "."),
        suite="session-visible-tests",
        test_name="solver-order",
    )


def _parallel_execution() -> ExecutableSearchConfig:
    return ExecutableSearchConfig(
        command=(sys.executable, "-c", "import time; time.sleep(0.15)"),
        suite="parallel-visible-tests",
        test_name="parallel-smoke",
    )


class _BlockingModel:
    def __init__(self, delegate: ScriptedModel) -> None:
        self.delegate = delegate
        self.released = asyncio.Event()

    @property
    def name(self) -> str:
        return self.delegate.name

    @property
    def configuration_fingerprint(self) -> str:
        return self.delegate.configuration_fingerprint

    def resume_from_turn(self, completed_turns: int) -> None:
        self.delegate.resume_from_turn(completed_turns)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        await self.released.wait()
        return await self.delegate.complete(request)


class SearchSessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_parallel_resume_reuses_durable_observations(self) -> None:
        class StopAfterFirstCheckpoint(Exception):
            pass

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "parallel-resume.json"
            original_writer = __import__(
                "contextopt.search.session", fromlist=["write_session_checkpoint"]
            ).write_session_checkpoint
            writes = 0

            def interrupt_after_first_result(
                report: SearchSessionReport, path: Path
            ) -> None:
                nonlocal writes
                original_writer(report, path)
                if any(event.type == "candidate.completed" for event in report.events):
                    writes += 1
                    if writes == 1:
                        raise StopAfterFirstCheckpoint()

            with (
                patch(
                    "contextopt.search.session.write_session_checkpoint",
                    side_effect=interrupt_after_first_result,
                ),
                self.assertRaises(StopAfterFirstCheckpoint),
            ):
                await run_search_session(
                    _multi_candidate_model(),
                    task="find a correct sorting implementation",
                    root_files=ROOT_FILES,
                    execution_config=_parallel_execution(),
                    config=SearchSessionConfig(
                        max_rounds=1,
                        max_model_calls=1,
                        max_candidates=4,
                        max_test_calls=4,
                        max_parallel_tests=2,
                    ),
                    proposal_config=ProposalConfig(max_candidates=4),
                    search_config=BranchSearchConfig(max_depth=1, beam_width=4),
                    run_id="parallel-resume-test",
                    checkpoint_path=checkpoint,
                )
            partial = read_session_checkpoint(checkpoint)
            self.assertEqual(partial.phase, "evaluating")
            self.assertEqual(partial.test_calls, 1)
            self.assertEqual(len(partial.observations), 1)

            resumed = await run_search_session(
                _multi_candidate_model(),
                execution_config=None,
                checkpoint_path=checkpoint,
                resume=True,
            )
            self.assertEqual(resumed.status, "accepted")
            self.assertEqual(resumed.test_calls, 4)
            self.assertEqual(len(resumed.observations), 4)

    async def test_parallel_isolated_candidate_scheduler_is_durable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "parallel.json"
            report = await run_search_session(
                _multi_candidate_model(),
                task="find a correct sorting implementation",
                root_files=ROOT_FILES,
                execution_config=_parallel_execution(),
                config=SearchSessionConfig(
                    max_rounds=1,
                    max_model_calls=1,
                    max_candidates=4,
                    max_test_calls=4,
                    max_parallel_tests=2,
                ),
                proposal_config=ProposalConfig(max_candidates=4),
                search_config=BranchSearchConfig(max_depth=1, beam_width=4),
                run_id="parallel-session-test",
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
                read_session_checkpoint(checkpoint).to_dict(), report.to_dict()
            )

    async def test_iterative_session_feeds_failures_into_next_round(self) -> None:
        model = _scripted_model()
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "session.json"
            report = await run_search_session(
                model,
                task="Implement solve so it returns ascending values",
                root_files=ROOT_FILES,
                execution_config=_execution(),
                config=SearchSessionConfig(
                    max_rounds=2,
                    max_model_calls=2,
                    max_candidates=2,
                    max_test_calls=2,
                ),
                proposal_config=ProposalConfig(max_candidates=1),
                search_config=BranchSearchConfig(max_depth=1, beam_width=1),
                run_id="session-test",
                checkpoint_path=checkpoint,
            )
            self.assertEqual(report.status, "accepted")
            self.assertEqual(len(report.rounds), 2)
            self.assertEqual(report.model_calls, 2)
            self.assertEqual(report.test_calls, 2)
            self.assertEqual(report.test_reuses, 0)
            self.assertEqual(report.best_candidate_id, "round-1-good")
            self.assertTrue(
                report.rounds[0].report.case.tests["round-0-bad"].failed_tests
            )
            self.assertTrue(
                any(
                    "failed_tests" in str(item)
                    for item in model.requests[1].messages[-1].content.splitlines()
                )
            )
            verify_search_events(report.events)
            restored = read_session_checkpoint(checkpoint)
            self.assertEqual(restored.to_dict(), report.to_dict())
            self.assertEqual(
                SearchSessionReport.from_dict(report.to_dict()).to_dict(),
                report.to_dict(),
            )

    async def test_cross_round_duplicate_reuses_the_visible_test_observation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = await run_search_session(
                _repeating_model(),
                task="Implement solve so it returns ascending values",
                root_files=ROOT_FILES,
                execution_config=_execution(),
                config=SearchSessionConfig(
                    max_rounds=2,
                    max_model_calls=2,
                    max_candidates=2,
                    max_test_calls=2,
                ),
                proposal_config=ProposalConfig(max_candidates=1),
                search_config=BranchSearchConfig(max_depth=1, beam_width=1),
                run_id="reuse-test",
                checkpoint_path=Path(temp_dir) / "session.json",
            )
            self.assertEqual(report.status, "budget_exhausted")
            self.assertEqual(len(report.rounds), 2)
            self.assertEqual(report.test_calls, 1)
            self.assertEqual(report.test_reuses, 1)
            self.assertEqual(len(report.observations), 1)

    async def test_pending_model_request_can_pause_and_resume(self) -> None:
        delegate = _scripted_model()
        blocking = _BlockingModel(delegate)
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "pending.json"
            task = asyncio.create_task(
                run_search_session(
                    blocking,
                    task="Implement solve so it returns ascending values",
                    root_files=ROOT_FILES,
                    execution_config=_execution(),
                    config=SearchSessionConfig(
                        max_rounds=2,
                        max_model_calls=2,
                        max_candidates=2,
                        max_test_calls=2,
                    ),
                    proposal_config=ProposalConfig(max_candidates=1),
                    search_config=BranchSearchConfig(max_depth=1, beam_width=1),
                    run_id="pending-test",
                    checkpoint_path=checkpoint,
                )
            )
            for _ in range(100):
                if (
                    checkpoint.exists()
                    and read_session_checkpoint(checkpoint).phase == "proposing"
                ):
                    break
                await asyncio.sleep(0.005)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            pending = read_session_checkpoint(checkpoint)
            self.assertEqual(pending.phase, "proposing")
            self.assertEqual(pending.status, "running")

            resumed = await run_search_session(
                _scripted_model(),
                execution_config=None,
                checkpoint_path=checkpoint,
                resume=True,
                retry_pending=True,
            )
            self.assertEqual(resumed.status, "accepted")
            self.assertEqual(resumed.model_calls, 2)
            self.assertEqual(len(resumed.rounds), 2)

    async def test_checkpoint_rejects_event_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "tampered.json"
            report = await run_search_session(
                _scripted_model(),
                task="Implement solve so it returns ascending values",
                root_files=ROOT_FILES,
                execution_config=_execution(),
                config=SearchSessionConfig(
                    max_rounds=1,
                    max_model_calls=1,
                    max_candidates=1,
                    max_test_calls=1,
                ),
                proposal_config=ProposalConfig(max_candidates=1),
                search_config=BranchSearchConfig(max_depth=1, beam_width=1),
                run_id="tamper-test",
                checkpoint_path=checkpoint,
            )
            payload = report.to_dict()
            payload["events"][1]["data"] = {"forged": True}
            checkpoint.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                read_session_checkpoint(checkpoint)

            payload = report.to_dict()
            payload["metrics"]["model_calls"] = 99
            checkpoint.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "metric 'model_calls' is inconsistent"
            ):
                read_session_checkpoint(checkpoint)


if __name__ == "__main__":
    unittest.main()
