"""Generate the deterministic first-valid speculative solver artifact."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path

from contextopt.runtime.identity import stable_hash
from contextopt.runtime.model import ScriptedModel
from contextopt.runtime.protocol import ModelRequest, ModelResponse
from contextopt.search import (
    BranchSearchConfig,
    ExecutableSearchConfig,
    OrchestrationConfig,
    ProposalConfig,
    read_orchestration_checkpoint,
    render_orchestration_html,
    render_orchestration_markdown,
    run_orchestration,
)

OUTPUT_DIR = Path("experiments/v0.9-speculative-cancellation")
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


def _proposal() -> str:
    return json.dumps(
        {
            "candidates": [
                {
                    "id": "good",
                    "parent_id": "root",
                    "hypothesis": "use sorted for a stable ascending snapshot",
                    "files": {
                        "solver.py": "def solve(values):\n    return sorted(values)\n",
                        "test_solver.py": ROOT_FILES["test_solver.py"],
                    },
                    "evidence": ["visible test is the oracle"],
                }
            ]
        }
    )


def _review() -> str:
    return json.dumps(
        {
            "decision": "accept",
            "candidate_id": "round-0-spec-0-good",
            "confidence": 0.9,
            "blocking_issues": [],
            "required_checks": [],
            "summary": "the visible oracle supports this decision",
        }
    )


class FirstValidSolver:
    """A provider-neutral fixture: lane one wins, lane two waits for cancellation."""

    def __init__(self) -> None:
        self._name = "artifact-first-valid-solver:v1"
        self._never = asyncio.Event()
        self.cancelled = 0
        self.cancellation_requests = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def configuration_fingerprint(self) -> str:
        return stable_hash({"adapter": "artifact-first-valid", "name": self._name})

    def resume_from_turn(self, completed_turns: int) -> None:
        if completed_turns < 0:
            raise ValueError("completed_turns must be non-negative")

    async def request_cancellation(self, request: ModelRequest) -> str:
        _ = request
        self.cancellation_requests += 1
        return "acknowledged"

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if "You are lane 2 of 2" in request.messages[-1].content:
            try:
                await self._never.wait()
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
        await asyncio.sleep(0)
        return ModelResponse(content=_proposal())


async def _run() -> tuple[dict[str, object], FirstValidSolver]:
    solver = FirstValidSolver()
    report = await run_orchestration(
        ScriptedModel([{"response": {"content": _plan()}}], name="artifact-planner:v1"),
        solver,
        ScriptedModel(
            [{"response": {"content": _review()}}], name="artifact-reviewer:v1"
        ),
        task="find a correct sorting implementation",
        root_files=ROOT_FILES,
        execution_config=ExecutableSearchConfig(
            command=(sys.executable, "-m", "unittest", "discover", "-s", "."),
            suite="artifact-visible-tests",
            test_name="solver-order",
        ),
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
        run_id="v0.9-speculative-cancellation-artifact",
        checkpoint_path=OUTPUT_DIR / "checkpoint.json",
    )
    return report.to_dict(), solver


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def main() -> None:
    payload, solver = asyncio.run(_run())
    checkpoint = read_orchestration_checkpoint(OUTPUT_DIR / "checkpoint.json")
    if checkpoint.to_dict() != payload:
        raise RuntimeError("artifact checkpoint does not match the final report")
    _write(
        OUTPUT_DIR / "report.json",
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )
    # Rehydrate through the public report parser so the rendered artifacts
    # use the same schema.
    from contextopt.search import OrchestrationReport

    report = OrchestrationReport.from_dict(payload)
    _write(OUTPUT_DIR / "report.md", render_orchestration_markdown(report))
    _write(OUTPUT_DIR / "report.html", render_orchestration_html(report))
    files = {}
    for name in ("report.json", "report.md", "report.html", "checkpoint.json"):
        data = (OUTPUT_DIR / name).read_bytes()
        files[name] = hashlib.sha256(data).hexdigest()
    manifest = {
        "schema_version": "1",
        "artifact": "v0.9-speculative-cancellation",
        "model_adapter": "scripted-deterministic-cancellation-fixture",
        "run_id": "v0.9-speculative-cancellation-artifact",
        "winner_lane": 0,
        "cancelled_lane_count": solver.cancelled,
        "provider_cancel_status": "acknowledged"
        if solver.cancellation_requests
        else "unsupported",
        "claim_boundary": (
            "The winner is the first protocol-parseable candidate. Visible tests and "
            "reviewer "
            "approval remain authoritative; provider cancellation is best-effort."
        ),
        "files": files,
    }
    _write(
        OUTPUT_DIR / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )


if __name__ == "__main__":
    main()
