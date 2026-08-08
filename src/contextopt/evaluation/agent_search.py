"""Reproducible strategy evaluation for coding-agent search.

The evaluator compares three *control policies* on small ACM/math repair tasks:

* ``single_pass``: one solver call produces one candidate;
* ``best_of_n``: one solver call produces several candidates and the visible oracle
  chooses the first passing snapshot;
* ``orchestrated``: planner, solver, and reviewer run a bounded retry loop.

The built-in models are scripted on purpose.  This module measures the state
machine, budgets, oracle gates, and accounting of a coding-agent implementation;
it does not claim that a scripted response is a model-quality benchmark.  The
fixture snapshots and test commands are complete and executable, so the result is
still useful as a deterministic contract test before plugging in a real provider.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from html import escape
from math import isfinite, sqrt
from pathlib import Path, PurePosixPath
from statistics import fmean, pstdev
from typing import Any, Literal, cast

from contextopt.runtime.errors import ModelError, RuntimeContractError
from contextopt.runtime.model import OpenAICompatibleModel, ScriptedModel
from contextopt.runtime.protocol import ModelClient
from contextopt.search import (
    BranchSearchConfig,
    CandidatePatch,
    ExecutableSearchConfig,
    OrchestrationConfig,
    PlannerConfig,
    ProposalConfig,
    ReviewerConfig,
    SearchPolicy,
    SearchSessionConfig,
    evaluate_candidate,
    run_orchestration,
    run_search_session,
)

AgentStrategy = Literal["single_pass", "best_of_n", "orchestrated"]
AgentModelFactory = Callable[
    ["AgentEvalFixture", AgentStrategy], tuple[ModelClient, ...]
]
_STRATEGIES: tuple[AgentStrategy, ...] = (
    "single_pass",
    "best_of_n",
    "orchestrated",
)
_CLAIM_BOUNDARY = (
    "This is a deterministic control-policy and protocol evaluation with scripted "
    "model responses. It measures visible-test success, role/model-call budgets, "
    "independent hidden-test success when enabled, candidate accounting, and token "
    "accounting on the bundled fixtures; it does not measure general model capability, "
    "latency, provider reliability, or production safety."
)
_REAL_MODEL_CLAIM_BOUNDARY = (
    "This is an exploratory fixed-fixture provider evaluation. It records visible-test "
    "and independent hidden-test outcomes, role/model-call budgets, candidate "
    "accounting, and provider-reported token usage for the selected model; it is not "
    "a statistically powered benchmark and does not establish general coding ability, "
    "latency, provider "
    "reliability, security isolation, or production safety."
)


def _claim_boundary(adapter: str) -> str:
    return _CLAIM_BOUNDARY if adapter == "scripted" else _REAL_MODEL_CLAIM_BOUNDARY


def _non_empty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _non_negative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _snapshot(value: Mapping[str, str], label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} must be a non-empty file mapping")
    normalized: dict[str, str] = {}
    for raw_path, content in value.items():
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError(f"{label} paths must be non-empty strings")
        if "\\" in raw_path or raw_path.startswith("/"):
            raise ValueError(f"{label} paths must be relative POSIX paths")
        path = PurePosixPath(raw_path)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError(f"{label} contains an invalid path: {raw_path!r}")
        if not isinstance(content, str):
            raise ValueError(f"{label} contents must be strings")
        normalized[path.as_posix()] = content
    return dict(sorted(normalized.items()))


def _json_text(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _proposal(candidate_id: str, files: Mapping[str, str], hypothesis: str) -> str:
    return _json_text(
        {
            "candidates": [
                {
                    "id": candidate_id,
                    "parent_id": "root",
                    "hypothesis": hypothesis,
                    "files": dict(files),
                    "evidence": ["bundled visible tests are the authoritative oracle"],
                }
            ]
        }
    )


def _proposals(
    candidates: Sequence[tuple[str, Mapping[str, str], str]],
) -> str:
    return _json_text(
        {
            "candidates": [
                {
                    "id": candidate_id,
                    "parent_id": "root",
                    "hypothesis": hypothesis,
                    "files": dict(files),
                    "evidence": ["bundled visible tests are the authoritative oracle"],
                }
                for candidate_id, files, hypothesis in candidates
            ]
        }
    )


def _plan(task: str) -> str:
    return _json_text(
        {
            "goal": task,
            "constraints": ["preserve the public function signature"],
            "hypotheses": ["use a standard algorithm with explicit edge cases"],
            "test_focus": ["run every bundled visible test before acceptance"],
            "risks": ["do not mutate caller-owned inputs or weaken the contract"],
        }
    )


def _review(decision: str, candidate_id: str | None) -> str:
    return _json_text(
        {
            "decision": decision,
            "candidate_id": candidate_id,
            "confidence": 0.9 if decision == "accept" else 0.4,
            "blocking_issues": []
            if decision == "accept"
            else ["the visible-test evidence is not yet sufficient"],
            "required_checks": []
            if decision == "accept"
            else ["use the oracle feedback to produce a corrected candidate"],
            "summary": "the scripted reviewer only summarizes the oracle evidence",
        }
    )


@dataclass(frozen=True, slots=True)
class AgentEvalFixture:
    """One complete repair task used by every strategy."""

    fixture_id: str
    category: str
    title: str
    task: str
    root_files: Mapping[str, str]
    bad_files: Mapping[str, str]
    good_files: Mapping[str, str]
    test_name: str
    hidden_files: Mapping[str, str] = field(default_factory=dict)
    hidden_test_name: str = "hidden-oracle"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "fixture_id", _non_empty(self.fixture_id, "fixture_id")
        )
        object.__setattr__(self, "category", _non_empty(self.category, "category"))
        object.__setattr__(self, "title", _non_empty(self.title, "title"))
        object.__setattr__(self, "task", _non_empty(self.task, "task"))
        object.__setattr__(self, "test_name", _non_empty(self.test_name, "test_name"))
        object.__setattr__(
            self,
            "hidden_test_name",
            _non_empty(self.hidden_test_name, "hidden_test_name"),
        )
        for name in ("root_files", "bad_files", "good_files"):
            value = getattr(self, name)
            object.__setattr__(self, name, _snapshot(value, name))
        object.__setattr__(
            self,
            "hidden_files",
            _snapshot(self.hidden_files, "hidden_files") if self.hidden_files else {},
        )
        if set(self.bad_files) != set(self.root_files) or set(self.good_files) != set(
            self.root_files
        ):
            raise ValueError("bad_files and good_files must be complete root snapshots")

    @property
    def execution_config(self) -> ExecutableSearchConfig:
        return ExecutableSearchConfig(
            command=(sys.executable, "-m", "unittest", "discover", "-s", "."),
            suite=f"agent-eval:{self.fixture_id}",
            test_name=self.test_name,
        )

    @property
    def hidden_execution_config(self) -> ExecutableSearchConfig:
        """Return a separate command that never enters the model-visible root."""

        module_name = f"grader.oracle_{self.fixture_id.replace('-', '_')}"
        return ExecutableSearchConfig(
            command=(sys.executable, "-m", "unittest", module_name),
            suite=f"agent-eval-hidden:{self.fixture_id}",
            test_name=self.hidden_test_name,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return fixture metadata without embedding every scripted response."""

        return {
            "fixture_id": self.fixture_id,
            "category": self.category,
            "title": self.title,
            "task": self.task,
            "root_files": dict(self.root_files),
            "bad_files": dict(self.bad_files),
            "good_files": dict(self.good_files),
            "test_name": self.test_name,
            # Hidden grader source is intentionally not serialized into a public
            # report.  The fixture digest and result fields provide the audit trail
            # without leaking the independent oracle before a run is reviewed.
            "hidden_test_name": self.hidden_test_name,
        }


def build_algorithm_fixtures() -> tuple[AgentEvalFixture, ...]:
    """Return fixed ACM and mathematics tasks with a deliberately failing root."""

    two_sum_tests = (
        "import unittest\n"
        "from two_sum import two_sum\n\n\n"
        "class TwoSumTests(unittest.TestCase):\n"
        "    def test_pair_not_at_prefix(self):\n"
        "        self.assertEqual(two_sum([3, 2, 4], 6), [1, 2])\n\n"
        "    def test_duplicate_values(self):\n"
        "        self.assertEqual(two_sum([3, 3], 6), [0, 1])\n\n"
        "    def test_missing_pair(self):\n"
        "        self.assertEqual(two_sum([1, 2], 10), [])\n\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n"
    )
    two_sum_root = """def two_sum(numbers, target):
    # TODO: return the indices of two values that add to target.
    return []
"""
    two_sum_bad = """def two_sum(numbers, target):
    # This shortcut only works when the answer happens to be the first pair.
    return [0, 1] if len(numbers) >= 2 else []
"""
    two_sum_good = """def two_sum(numbers, target):
    seen = {}
    for index, value in enumerate(numbers):
        complement = target - value
        if complement in seen:
            return [seen[complement], index]
        seen[value] = index
    return []
"""
    two_sum_hidden = (
        "import unittest\n"
        "from two_sum import two_sum\n\n\n"
        "class HiddenTwoSumTests(unittest.TestCase):\n"
        "    def test_negative_values_and_late_pair(self):\n"
        "        values = [10, -4, 8, 6, 1]\n"
        "        self.assertEqual(two_sum(values, 2), [1, 3])\n"
        "        self.assertEqual(values, [10, -4, 8, 6, 1])\n\n"
        "    def test_empty_and_singleton_inputs(self):\n"
        "        self.assertEqual(two_sum([], 0), [])\n"
        "        self.assertEqual(two_sum([4], 8), [])\n\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n"
    )

    gcd_tests = (
        "import unittest\n"
        "from extended_gcd import extended_gcd\n\n\n"
        "class ExtendedGcdTests(unittest.TestCase):\n"
        "    def assert_bezout(self, a, b):\n"
        "        gcd, x, y = extended_gcd(a, b)\n"
        "        self.assertEqual(gcd, __import__('math').gcd(a, b))\n"
        "        self.assertEqual(a * x + b * y, gcd)\n\n"
        "    def test_classic_pair(self):\n"
        "        self.assert_bezout(30, 12)\n\n"
        "    def test_coprime_pair(self):\n"
        "        self.assert_bezout(17, 5)\n\n"
        "    def test_zero_rhs(self):\n"
        "        self.assert_bezout(21, 0)\n\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n"
    )
    gcd_root = """def extended_gcd(a, b):
    # TODO: return (gcd, x, y) with a*x + b*y == gcd.
    return abs(a), 1, 0
"""
    gcd_bad = """def extended_gcd(a, b):
    # Returning gcd alone loses the Bezout coefficients.
    return abs(a), 1, 0
"""
    gcd_good = """def extended_gcd(a, b):
    if b == 0:
        sign = -1 if a < 0 else 1
        return abs(a), sign, 0
    gcd, x1, y1 = extended_gcd(b, a % b)
    return gcd, y1, x1 - (a // b) * y1
"""
    gcd_hidden = (
        "import unittest\n"
        "from extended_gcd import extended_gcd\n\n\n"
        "class HiddenExtendedGcdTests(unittest.TestCase):\n"
        "    def assert_bezout(self, a, b):\n"
        "        gcd, x, y = extended_gcd(a, b)\n"
        "        self.assertEqual(gcd, __import__('math').gcd(a, b))\n"
        "        self.assertEqual(a * x + b * y, gcd)\n\n"
        "    def test_negative_inputs(self):\n"
        "        self.assert_bezout(-30, 12)\n"
        "        self.assert_bezout(30, -12)\n\n"
        "    def test_large_coprime_inputs(self):\n"
        "        self.assert_bezout(1234567, 76543)\n\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n"
    )

    return (
        AgentEvalFixture(
            fixture_id="two-sum",
            category="acm-algorithm",
            title="Two Sum with duplicate values",
            task=(
                "Implement two_sum(numbers, target) and return the two indices in "
                "input order, or [] when no pair exists."
            ),
            root_files={"two_sum.py": two_sum_root, "test_two_sum.py": two_sum_tests},
            bad_files={"two_sum.py": two_sum_bad, "test_two_sum.py": two_sum_tests},
            good_files={"two_sum.py": two_sum_good, "test_two_sum.py": two_sum_tests},
            test_name="two-sum-visible-tests",
            hidden_files={
                "grader/__init__.py": "",
                "grader/oracle_two_sum.py": two_sum_hidden,
            },
            hidden_test_name="two-sum-hidden-tests",
        ),
        AgentEvalFixture(
            fixture_id="extended-gcd",
            category="mathematics",
            title="Extended Euclidean algorithm",
            task=(
                "Implement extended_gcd(a, b) returning (gcd, x, y) such that "
                "a*x + b*y == gcd, including b == 0."
            ),
            root_files={
                "extended_gcd.py": gcd_root,
                "test_extended_gcd.py": gcd_tests,
            },
            bad_files={
                "extended_gcd.py": gcd_bad,
                "test_extended_gcd.py": gcd_tests,
            },
            good_files={
                "extended_gcd.py": gcd_good,
                "test_extended_gcd.py": gcd_tests,
            },
            test_name="extended-gcd-visible-tests",
            hidden_files={
                "grader/__init__.py": "",
                "grader/oracle_extended_gcd.py": gcd_hidden,
            },
            hidden_test_name="extended-gcd-hidden-tests",
        ),
    )


@dataclass(frozen=True, slots=True)
class AgentEvalConfig:
    """Paired budgets and matrix selection for the coding-agent evaluation."""

    strategies: tuple[AgentStrategy, ...] = _STRATEGIES
    fixtures: tuple[str, ...] = ("two-sum", "extended-gcd")
    repetitions: int = 1
    max_rounds: int = 2
    max_model_calls: int = 6
    max_candidates: int = 2
    max_test_calls: int = 2
    search_policy: SearchPolicy = "beam"
    exploration_constant: float = 1.0
    include_hidden_tests: bool = True
    model_adapter: str = "scripted"

    def __post_init__(self) -> None:
        if not self.strategies:
            raise ValueError("at least one agent strategy is required")
        if any(not isinstance(strategy, str) for strategy in self.strategies):
            raise ValueError("agent strategies must be strings")
        if len(self.strategies) != len(set(self.strategies)):
            raise ValueError("agent strategies must be unique")
        unknown = set(self.strategies) - set(_STRATEGIES)
        if unknown:
            raise ValueError(f"unknown agent strategies: {sorted(unknown)!r}")
        if not self.fixtures:
            raise ValueError("at least one agent fixture is required")
        if any(
            not isinstance(fixture, str) or not fixture.strip()
            for fixture in self.fixtures
        ):
            raise ValueError("agent fixtures must be non-empty strings")
        if len(self.fixtures) != len(set(self.fixtures)):
            raise ValueError("agent fixtures must be unique")
        for name in (
            "repetitions",
            "max_rounds",
            "max_model_calls",
            "max_candidates",
            "max_test_calls",
        ):
            _positive_int(getattr(self, name), name)
        if not isinstance(self.search_policy, str) or self.search_policy not in {
            "beam",
            "mcts",
        }:
            raise ValueError(f"unsupported search policy: {self.search_policy!r}")
        if not isinstance(self.exploration_constant, (int, float)) or isinstance(
            self.exploration_constant, bool
        ):
            raise ValueError("exploration_constant must be a number")
        if self.exploration_constant <= 0:
            raise ValueError("exploration_constant must be positive")
        if not isinstance(self.include_hidden_tests, bool):
            raise ValueError("include_hidden_tests must be a boolean")
        if self.model_adapter not in {"scripted", "openai-compatible", "custom"}:
            raise ValueError(f"unsupported agent model adapter: {self.model_adapter!r}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AgentEvalConfig:
        value = _mapping(data, "agent evaluation config")
        allowed = {
            "strategies",
            "fixtures",
            "repetitions",
            "max_rounds",
            "max_model_calls",
            "max_candidates",
            "max_test_calls",
            "search_policy",
            "exploration_constant",
            "include_hidden_tests",
            "model_adapter",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                f"agent evaluation config has unknown fields: {sorted(unknown)!r}"
            )
        strategies = value.get("strategies", list(_STRATEGIES))
        fixtures = value.get("fixtures", ["two-sum", "extended-gcd"])
        if not isinstance(strategies, list) or not isinstance(fixtures, list):
            raise ValueError("strategies and fixtures must be arrays")
        return cls(
            strategies=tuple(cast(AgentStrategy, item) for item in strategies),
            fixtures=tuple(_non_empty(item, "fixture id") for item in fixtures),
            repetitions=_positive_int(value.get("repetitions", 1), "repetitions"),
            max_rounds=_positive_int(value.get("max_rounds", 2), "max_rounds"),
            max_model_calls=_positive_int(
                value.get("max_model_calls", 6), "max_model_calls"
            ),
            max_candidates=_positive_int(
                value.get("max_candidates", 2), "max_candidates"
            ),
            max_test_calls=_positive_int(
                value.get("max_test_calls", 2), "max_test_calls"
            ),
            search_policy=cast(SearchPolicy, value.get("search_policy", "beam")),
            exploration_constant=float(value.get("exploration_constant", 1.0)),
            include_hidden_tests=_boolean(
                value.get("include_hidden_tests", True), "include_hidden_tests"
            ),
            model_adapter=_non_empty(
                value.get("model_adapter", "scripted"), "model_adapter"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategies": list(self.strategies),
            "fixtures": list(self.fixtures),
            "repetitions": self.repetitions,
            "max_rounds": self.max_rounds,
            "max_model_calls": self.max_model_calls,
            "max_candidates": self.max_candidates,
            "max_test_calls": self.max_test_calls,
            "search_policy": self.search_policy,
            "exploration_constant": self.exploration_constant,
            "include_hidden_tests": self.include_hidden_tests,
            "model_adapter": self.model_adapter,
        }


@dataclass(frozen=True, slots=True)
class AgentEvalCheckpoint:
    """Durable partial matrix state for long-running provider evaluations."""

    config: AgentEvalConfig
    fixture_ids: tuple[str, ...]
    runs: tuple[AgentEvalRun, ...] = ()
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if self.schema_version != "1":
            raise ValueError(
                "unsupported agent evaluation checkpoint schema: "
                f"{self.schema_version!r}"
            )
        if self.fixture_ids != self.config.fixtures:
            raise ValueError("checkpoint fixture ids must match config fixtures")
        expected = {
            (fixture_id, strategy, repetition)
            for fixture_id in self.fixture_ids
            for strategy in self.config.strategies
            for repetition in range(self.config.repetitions)
        }
        actual = {(run.fixture_id, run.strategy, run.repetition) for run in self.runs}
        if not actual <= expected:
            raise ValueError("checkpoint contains a run outside the configured matrix")
        if len(actual) != len(self.runs):
            raise ValueError("checkpoint runs must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "config": self.config.to_dict(),
            "fixture_ids": list(self.fixture_ids),
            "runs": [run.to_dict() for run in self.runs],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AgentEvalCheckpoint:
        value = _mapping(data, "agent evaluation checkpoint")
        allowed = {"schema_version", "config", "fixture_ids", "runs"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                f"agent evaluation checkpoint has unknown fields: {sorted(unknown)!r}"
            )
        raw_fixture_ids = value.get("fixture_ids")
        raw_runs = value.get("runs")
        if not isinstance(raw_fixture_ids, list) or not isinstance(raw_runs, list):
            raise ValueError("checkpoint fixture_ids and runs must be arrays")
        fixture_ids = tuple(_non_empty(item, "fixture id") for item in raw_fixture_ids)
        if not isinstance(value.get("config"), Mapping):
            raise ValueError("checkpoint config must be an object")
        return cls(
            schema_version=str(value.get("schema_version", "")),
            config=AgentEvalConfig.from_dict(value["config"]),
            fixture_ids=fixture_ids,
            runs=tuple(AgentEvalRun.from_dict(item) for item in raw_runs),
        )


@dataclass(frozen=True, slots=True)
class AgentEvalRun:
    """One strategy/fixture/repetition cell with auditable accounting."""

    fixture_id: str
    strategy: AgentStrategy
    repetition: int
    status: str
    success: bool
    rounds: int
    model_calls: int
    planner_calls: int
    solver_calls: int
    reviewer_calls: int
    candidate_proposals: int
    test_calls: int
    test_reuses: int
    total_tokens: int
    best_candidate_id: str | None
    error: str | None = None
    hidden_test_calls: int = 0
    hidden_success: bool | None = None
    hidden_error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "fixture_id", _non_empty(self.fixture_id, "fixture_id")
        )
        if self.strategy not in _STRATEGIES:
            raise ValueError(f"unknown agent strategy: {self.strategy!r}")
        if (
            not isinstance(self.repetition, int)
            or isinstance(self.repetition, bool)
            or self.repetition < 0
        ):
            raise ValueError("repetition must be a non-negative integer")
        if self.status not in {"accepted", "exhausted", "budget_exhausted", "failed"}:
            raise ValueError(f"unsupported agent evaluation status: {self.status!r}")
        if not isinstance(self.success, bool):
            raise ValueError("success must be a boolean")
        if self.success != (self.status == "accepted"):
            raise ValueError("success must match accepted status")
        for name in (
            "rounds",
            "model_calls",
            "planner_calls",
            "solver_calls",
            "reviewer_calls",
            "candidate_proposals",
            "test_calls",
            "test_reuses",
            "total_tokens",
        ):
            if (
                not isinstance(getattr(self, name), int)
                or isinstance(getattr(self, name), bool)
                or getattr(self, name) < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer")
        if self.best_candidate_id is not None:
            object.__setattr__(
                self,
                "best_candidate_id",
                _non_empty(self.best_candidate_id, "best_candidate_id"),
            )
        if self.error is not None:
            object.__setattr__(self, "error", _non_empty(self.error, "error"))
        if (
            not isinstance(self.hidden_test_calls, int)
            or isinstance(self.hidden_test_calls, bool)
            or self.hidden_test_calls < 0
        ):
            raise ValueError("hidden_test_calls must be a non-negative integer")
        if self.hidden_success is not None and not isinstance(
            self.hidden_success, bool
        ):
            raise ValueError("hidden_success must be a boolean or null")
        if self.hidden_success is None and self.hidden_test_calls != 0:
            raise ValueError("hidden test calls require a hidden result")
        if self.hidden_success is not None and self.hidden_test_calls <= 0:
            raise ValueError("hidden result requires at least one hidden test call")
        if self.hidden_error is not None:
            object.__setattr__(
                self, "hidden_error", _non_empty(self.hidden_error, "hidden_error")
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AgentEvalRun:
        value = _mapping(data, "agent evaluation run")
        allowed = {
            "fixture_id",
            "strategy",
            "repetition",
            "status",
            "success",
            "rounds",
            "model_calls",
            "planner_calls",
            "solver_calls",
            "reviewer_calls",
            "candidate_proposals",
            "test_calls",
            "test_reuses",
            "total_tokens",
            "best_candidate_id",
            "error",
            "hidden_test_calls",
            "hidden_success",
            "hidden_error",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                f"agent evaluation run has unknown fields: {sorted(unknown)!r}"
            )
        return cls(
            fixture_id=_non_empty(value.get("fixture_id"), "fixture_id"),
            strategy=cast(AgentStrategy, value.get("strategy")),
            repetition=int(value.get("repetition", -1)),
            status=str(value.get("status")),
            success=_boolean(value.get("success"), "success"),
            rounds=_non_negative_int(value.get("rounds"), "rounds"),
            model_calls=_non_negative_int(value.get("model_calls"), "model_calls"),
            planner_calls=_non_negative_int(
                value.get("planner_calls"), "planner_calls"
            ),
            solver_calls=_non_negative_int(value.get("solver_calls"), "solver_calls"),
            reviewer_calls=_non_negative_int(
                value.get("reviewer_calls"), "reviewer_calls"
            ),
            candidate_proposals=_non_negative_int(
                value.get("candidate_proposals"), "candidate_proposals"
            ),
            test_calls=_non_negative_int(value.get("test_calls"), "test_calls"),
            test_reuses=_non_negative_int(value.get("test_reuses"), "test_reuses"),
            total_tokens=_non_negative_int(value.get("total_tokens"), "total_tokens"),
            best_candidate_id=(
                None
                if value.get("best_candidate_id") is None
                else str(value["best_candidate_id"])
            ),
            error=None if value.get("error") is None else str(value["error"]),
            hidden_test_calls=_non_negative_int(
                value.get("hidden_test_calls", 0), "hidden_test_calls"
            ),
            hidden_success=(
                None
                if value.get("hidden_success") is None
                else _boolean(value["hidden_success"], "hidden_success")
            ),
            hidden_error=(
                None
                if value.get("hidden_error") is None
                else str(value["hidden_error"])
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "strategy": self.strategy,
            "repetition": self.repetition,
            "status": self.status,
            "success": self.success,
            "rounds": self.rounds,
            "model_calls": self.model_calls,
            "planner_calls": self.planner_calls,
            "solver_calls": self.solver_calls,
            "reviewer_calls": self.reviewer_calls,
            "candidate_proposals": self.candidate_proposals,
            "test_calls": self.test_calls,
            "test_reuses": self.test_reuses,
            "total_tokens": self.total_tokens,
            "best_candidate_id": self.best_candidate_id,
            "error": self.error,
            "hidden_test_calls": self.hidden_test_calls,
            "hidden_success": self.hidden_success,
            "hidden_error": self.hidden_error,
        }


@dataclass(frozen=True, slots=True)
class AgentEvalSummary:
    """Aggregated metrics for one strategy across paired fixture cells."""

    strategy: AgentStrategy
    run_count: int
    success_count: int
    success_rate: float
    mean_rounds: float
    mean_model_calls: float
    mean_test_calls: float
    mean_test_reuses: float
    mean_candidate_proposals: float
    mean_total_tokens: float
    hidden_run_count: int
    hidden_success_count: int
    hidden_success_rate: float | None
    mean_hidden_test_calls: float

    def __post_init__(self) -> None:
        if self.strategy not in _STRATEGIES:
            raise ValueError(f"unknown agent strategy: {self.strategy!r}")
        if not isinstance(self.run_count, int) or self.run_count <= 0:
            raise ValueError("run_count must be positive")
        if (
            not isinstance(self.success_count, int)
            or not 0 <= self.success_count <= self.run_count
        ):
            raise ValueError("success_count must be within run_count")
        if self.success_rate != self.success_count / self.run_count:
            raise ValueError("success_rate is inconsistent with success_count")
        for name in (
            "mean_rounds",
            "mean_model_calls",
            "mean_test_calls",
            "mean_test_reuses",
            "mean_candidate_proposals",
            "mean_total_tokens",
            "mean_hidden_test_calls",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative number")
        if (
            not isinstance(self.hidden_run_count, int)
            or isinstance(self.hidden_run_count, bool)
            or self.hidden_run_count < 0
            or self.hidden_run_count > self.run_count
        ):
            raise ValueError("hidden_run_count must be within run_count")
        if (
            not isinstance(self.hidden_success_count, int)
            or isinstance(self.hidden_success_count, bool)
            or not 0 <= self.hidden_success_count <= self.hidden_run_count
        ):
            raise ValueError("hidden_success_count must be within hidden_run_count")
        expected_hidden_rate = (
            None
            if self.hidden_run_count == 0
            else self.hidden_success_count / self.hidden_run_count
        )
        if self.hidden_success_rate != expected_hidden_rate:
            raise ValueError("hidden_success_rate is inconsistent with hidden counts")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AgentEvalSummary:
        value = _mapping(data, "agent evaluation summary")
        allowed = {
            "strategy",
            "run_count",
            "success_count",
            "success_rate",
            "mean_rounds",
            "mean_model_calls",
            "mean_test_calls",
            "mean_test_reuses",
            "mean_candidate_proposals",
            "mean_total_tokens",
            "hidden_run_count",
            "hidden_success_count",
            "hidden_success_rate",
            "mean_hidden_test_calls",
        }

        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                f"agent evaluation summary has unknown fields: {sorted(unknown)!r}"
            )
        return cls(
            strategy=cast(AgentStrategy, value.get("strategy")),
            run_count=int(value.get("run_count", -1)),
            success_count=int(value.get("success_count", -1)),
            success_rate=float(value.get("success_rate", -1.0)),
            mean_rounds=float(value.get("mean_rounds", -1.0)),
            mean_model_calls=float(value.get("mean_model_calls", -1.0)),
            mean_test_calls=float(value.get("mean_test_calls", -1.0)),
            mean_test_reuses=float(value.get("mean_test_reuses", -1.0)),
            mean_candidate_proposals=float(value.get("mean_candidate_proposals", -1.0)),
            mean_total_tokens=float(value.get("mean_total_tokens", -1.0)),
            hidden_run_count=_non_negative_int(
                value.get("hidden_run_count", 0), "hidden_run_count"
            ),
            hidden_success_count=_non_negative_int(
                value.get("hidden_success_count", 0), "hidden_success_count"
            ),
            hidden_success_rate=(
                None
                if value.get("hidden_success_rate") is None
                else float(value["hidden_success_rate"])
            ),
            mean_hidden_test_calls=float(value.get("mean_hidden_test_calls", 0.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "run_count": self.run_count,
            "success_count": self.success_count,
            "success_rate": self.success_rate,
            "mean_rounds": self.mean_rounds,
            "mean_model_calls": self.mean_model_calls,
            "mean_test_calls": self.mean_test_calls,
            "mean_test_reuses": self.mean_test_reuses,
            "mean_candidate_proposals": self.mean_candidate_proposals,
            "mean_total_tokens": self.mean_total_tokens,
            "hidden_run_count": self.hidden_run_count,
            "hidden_success_count": self.hidden_success_count,
            "hidden_success_rate": self.hidden_success_rate,
            "mean_hidden_test_calls": self.mean_hidden_test_calls,
        }


@dataclass(frozen=True, slots=True)
class AgentEvalComparison:
    """Paired outcome and cost deltas against one baseline strategy."""

    baseline_strategy: AgentStrategy
    strategy: AgentStrategy
    paired_count: int
    wins: int
    losses: int
    ties: int
    visible_delta: float
    hidden_paired_count: int
    hidden_wins: int
    hidden_losses: int
    hidden_ties: int
    hidden_delta: float | None
    mean_test_call_delta: float
    stddev_test_call_delta: float
    mean_token_delta: float
    stddev_token_delta: float

    def __post_init__(self) -> None:
        if self.baseline_strategy not in _STRATEGIES:
            raise ValueError(f"unknown baseline strategy: {self.baseline_strategy!r}")
        if self.strategy not in _STRATEGIES:
            raise ValueError(f"unknown comparison strategy: {self.strategy!r}")
        if self.strategy == self.baseline_strategy:
            raise ValueError("comparison strategy must differ from baseline")
        if not isinstance(self.paired_count, int) or self.paired_count <= 0:
            raise ValueError("paired_count must be positive")
        if self.wins + self.losses + self.ties != self.paired_count:
            raise ValueError("visible paired outcome counts are inconsistent")
        if self.visible_delta != (self.wins - self.losses) / self.paired_count:
            raise ValueError("visible_delta is inconsistent with paired outcomes")
        if (
            not isinstance(self.hidden_paired_count, int)
            or not 0 <= (self.hidden_paired_count) <= self.paired_count
        ):
            raise ValueError("hidden_paired_count must be within paired_count")
        if (
            self.hidden_wins + self.hidden_losses + self.hidden_ties
            != self.hidden_paired_count
        ):
            raise ValueError("hidden paired outcome counts are inconsistent")
        expected_hidden_delta = (
            None
            if self.hidden_paired_count == 0
            else (self.hidden_wins - self.hidden_losses) / self.hidden_paired_count
        )
        if self.hidden_delta != expected_hidden_delta:
            raise ValueError("hidden_delta is inconsistent with paired outcomes")
        for name in (
            "mean_test_call_delta",
            "stddev_test_call_delta",
            "mean_token_delta",
            "stddev_token_delta",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{name} must be a number")
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_strategy": self.baseline_strategy,
            "strategy": self.strategy,
            "paired_count": self.paired_count,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "visible_delta": self.visible_delta,
            "hidden_paired_count": self.hidden_paired_count,
            "hidden_wins": self.hidden_wins,
            "hidden_losses": self.hidden_losses,
            "hidden_ties": self.hidden_ties,
            "hidden_delta": self.hidden_delta,
            "mean_test_call_delta": self.mean_test_call_delta,
            "stddev_test_call_delta": self.stddev_test_call_delta,
            "mean_token_delta": self.mean_token_delta,
            "stddev_token_delta": self.stddev_token_delta,
        }


def wilson_interval(
    successes: int, total: int, *, z: float = 1.96
) -> tuple[float, float]:
    """Return a bounded Wilson interval for a Bernoulli success rate."""

    if not isinstance(successes, int) or not isinstance(total, int):
        raise ValueError("successes and total must be integers")
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("successes must be within a positive total")
    if not isfinite(z) or z <= 0:
        raise ValueError("z must be finite and positive")
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def build_agent_eval_comparisons(
    report: AgentEvalReport, *, baseline_strategy: AgentStrategy = "single_pass"
) -> tuple[AgentEvalComparison, ...]:
    """Build paired deltas without changing the raw report ledger."""

    if baseline_strategy not in report.config.strategies:
        raise ValueError("baseline strategy is not configured in this report")
    by_key = {
        (run.fixture_id, run.repetition, run.strategy): run for run in report.runs
    }
    comparisons: list[AgentEvalComparison] = []
    for strategy in report.config.strategies:
        if strategy == baseline_strategy:
            continue
        pairs = [
            (
                by_key[(fixture_id, repetition, baseline_strategy)],
                by_key[(fixture_id, repetition, strategy)],
            )
            for fixture_id in report.config.fixtures
            for repetition in range(report.config.repetitions)
        ]
        wins = sum(
            candidate.success and not baseline.success for baseline, candidate in pairs
        )
        losses = sum(
            baseline.success and not candidate.success for baseline, candidate in pairs
        )
        hidden_pairs = [
            (baseline, candidate)
            for baseline, candidate in pairs
            if baseline.hidden_success is not None
            and candidate.hidden_success is not None
        ]
        hidden_wins = sum(
            candidate.hidden_success is True and baseline.hidden_success is False
            for baseline, candidate in hidden_pairs
        )
        hidden_losses = sum(
            baseline.hidden_success is True and candidate.hidden_success is False
            for baseline, candidate in hidden_pairs
        )
        test_call_deltas = tuple(
            candidate.test_calls - baseline.test_calls for baseline, candidate in pairs
        )
        token_deltas = tuple(
            candidate.total_tokens - baseline.total_tokens
            for baseline, candidate in pairs
        )
        comparisons.append(
            AgentEvalComparison(
                baseline_strategy=baseline_strategy,
                strategy=strategy,
                paired_count=len(pairs),
                wins=wins,
                losses=losses,
                ties=len(pairs) - wins - losses,
                visible_delta=(wins - losses) / len(pairs),
                hidden_paired_count=len(hidden_pairs),
                hidden_wins=hidden_wins,
                hidden_losses=hidden_losses,
                hidden_ties=len(hidden_pairs) - hidden_wins - hidden_losses,
                hidden_delta=(
                    None
                    if not hidden_pairs
                    else (hidden_wins - hidden_losses) / len(hidden_pairs)
                ),
                mean_test_call_delta=fmean(test_call_deltas),
                stddev_test_call_delta=pstdev(test_call_deltas),
                mean_token_delta=fmean(token_deltas),
                stddev_token_delta=pstdev(token_deltas),
            )
        )
    return tuple(comparisons)


def _summary(strategy: AgentStrategy, runs: Sequence[AgentEvalRun]) -> AgentEvalSummary:
    if not runs:
        raise ValueError("cannot summarize an empty strategy run set")

    def mean(name: str) -> float:
        return fmean(getattr(run, name) for run in runs)

    successes = sum(run.success for run in runs)
    hidden_runs = tuple(run for run in runs if run.hidden_success is not None)
    hidden_successes = sum(run.hidden_success is True for run in hidden_runs)
    return AgentEvalSummary(
        strategy=strategy,
        run_count=len(runs),
        success_count=successes,
        success_rate=successes / len(runs),
        mean_rounds=mean("rounds"),
        mean_model_calls=mean("model_calls"),
        mean_test_calls=mean("test_calls"),
        mean_test_reuses=mean("test_reuses"),
        mean_candidate_proposals=mean("candidate_proposals"),
        mean_total_tokens=mean("total_tokens"),
        hidden_run_count=len(hidden_runs),
        hidden_success_count=hidden_successes,
        hidden_success_rate=(
            None if not hidden_runs else hidden_successes / len(hidden_runs)
        ),
        mean_hidden_test_calls=mean("hidden_test_calls"),
    )


@dataclass(frozen=True, slots=True)
class AgentEvalReport:
    """Machine-readable paired report with self-consistent summary metrics."""

    config: AgentEvalConfig
    fixtures: tuple[AgentEvalFixture, ...]
    runs: tuple[AgentEvalRun, ...]
    summaries: tuple[AgentEvalSummary, ...]
    claim_boundary: str = _CLAIM_BOUNDARY
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if self.schema_version != "1":
            raise ValueError(
                f"unsupported agent evaluation schema: {self.schema_version!r}"
            )
        if self.claim_boundary != _claim_boundary(self.config.model_adapter):
            raise ValueError("agent evaluation claim_boundary is not canonical")
        fixture_ids = tuple(fixture.fixture_id for fixture in self.fixtures)
        if len(fixture_ids) != len(set(fixture_ids)):
            raise ValueError("agent evaluation fixture ids must be unique")
        if self.config.fixtures != fixture_ids:
            raise ValueError("config fixtures must match report fixtures")
        expected_cells = (
            len(fixture_ids) * len(self.config.strategies) * self.config.repetitions
        )
        if len(self.runs) != expected_cells:
            raise ValueError("agent evaluation runs do not cover the configured matrix")
        if len(self.summaries) != len(self.config.strategies):
            raise ValueError("agent evaluation summaries do not cover strategies")
        if {summary.strategy for summary in self.summaries} != set(
            self.config.strategies
        ):
            raise ValueError("agent evaluation summaries have the wrong strategies")
        if any(run.fixture_id not in fixture_ids for run in self.runs):
            raise ValueError("agent evaluation run references an unknown fixture")
        if any(run.strategy not in self.config.strategies for run in self.runs):
            raise ValueError("agent evaluation run references an unconfigured strategy")
        expected_keys = {
            (fixture_id, strategy, repetition)
            for fixture_id in fixture_ids
            for strategy in self.config.strategies
            for repetition in range(self.config.repetitions)
        }
        actual_keys = {
            (run.fixture_id, run.strategy, run.repetition) for run in self.runs
        }
        if actual_keys != expected_keys:
            raise ValueError("agent evaluation runs do not cover each matrix cell once")
        for strategy in self.config.strategies:
            expected = _summary(
                strategy, tuple(run for run in self.runs if run.strategy == strategy)
            )
            actual = next(
                summary for summary in self.summaries if summary.strategy == strategy
            )
            if actual.to_dict() != expected.to_dict():
                raise ValueError(f"summary metrics are inconsistent for {strategy}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AgentEvalReport:
        value = _mapping(data, "agent evaluation report")
        allowed = {
            "schema_version",
            "config",
            "fixtures",
            "runs",
            "summaries",
            "claim_boundary",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                f"agent evaluation report has unknown fields: {sorted(unknown)!r}"
            )
        raw_fixtures = value.get("fixtures")
        raw_runs = value.get("runs")
        raw_summaries = value.get("summaries")
        if (
            not isinstance(raw_fixtures, list)
            or not isinstance(raw_runs, list)
            or not isinstance(raw_summaries, list)
        ):
            raise ValueError(
                "agent evaluation fixtures, runs, and summaries must be arrays"
            )
        fixtures: list[AgentEvalFixture] = []
        for item in raw_fixtures:
            fixture = _mapping(item, "agent evaluation fixture")
            allowed_fixture = {
                "fixture_id",
                "category",
                "title",
                "task",
                "root_files",
                "bad_files",
                "good_files",
                "test_name",
                "hidden_test_name",
            }
            unknown_fixture = set(fixture) - allowed_fixture
            if unknown_fixture:
                raise ValueError(
                    "agent evaluation fixture has unknown fields: "
                    f"{sorted(unknown_fixture)!r}"
                )
            fixtures.append(
                AgentEvalFixture(
                    fixture_id=_non_empty(fixture.get("fixture_id"), "fixture_id"),
                    category=_non_empty(fixture.get("category"), "category"),
                    title=_non_empty(fixture.get("title"), "title"),
                    task=_non_empty(fixture.get("task"), "task"),
                    root_files=_mapping(fixture.get("root_files"), "root_files"),
                    bad_files=_mapping(fixture.get("bad_files"), "bad_files"),
                    good_files=_mapping(fixture.get("good_files"), "good_files"),
                    test_name=_non_empty(fixture.get("test_name"), "test_name"),
                    hidden_test_name=_non_empty(
                        fixture.get("hidden_test_name", "hidden-oracle"),
                        "hidden_test_name",
                    ),
                )
            )
        if not isinstance(value.get("config"), Mapping):
            raise ValueError("agent evaluation config must be an object")
        return cls(
            schema_version=str(value.get("schema_version", "")),
            config=AgentEvalConfig.from_dict(value["config"]),
            fixtures=tuple(fixtures),
            runs=tuple(AgentEvalRun.from_dict(item) for item in raw_runs),
            summaries=tuple(AgentEvalSummary.from_dict(item) for item in raw_summaries),
            claim_boundary=_non_empty(value.get("claim_boundary"), "claim_boundary"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "config": self.config.to_dict(),
            "fixtures": [fixture.to_dict() for fixture in self.fixtures],
            "runs": [run.to_dict() for run in self.runs],
            "summaries": [summary.to_dict() for summary in self.summaries],
            "claim_boundary": self.claim_boundary,
        }


def write_agent_evaluation_checkpoint(
    checkpoint: AgentEvalCheckpoint, path: str | Path
) -> None:
    """Atomically persist completed evaluation cells after each matrix step."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        checkpoint.to_dict(), ensure_ascii=False, sort_keys=True, indent=2
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload + "\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        if temporary_path is None:
            raise OSError("temporary evaluation checkpoint path was not created")
        os.replace(temporary_path, target)
    except OSError:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def read_agent_evaluation_checkpoint(path: str | Path) -> AgentEvalCheckpoint:
    """Read and strictly validate a partial evaluation matrix checkpoint."""

    target = Path(path)
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid agent evaluation checkpoint: {target}") from exc
    if not isinstance(data, Mapping):
        raise ValueError("agent evaluation checkpoint must be an object")
    return AgentEvalCheckpoint.from_dict(data)


def _model_usage(input_tokens: int, output_tokens: int) -> dict[str, int]:
    return {"prompt_tokens": input_tokens, "completion_tokens": output_tokens}


def _script_step(
    content: str, *, turn: int | None = None, usage: tuple[int, int]
) -> dict[str, Any]:
    step: dict[str, Any] = {
        "response": {"content": content, "usage": _model_usage(*usage)}
    }
    if turn is not None:
        step["expect"] = {"turn": turn}
    return step


def _strategy_models(
    fixture: AgentEvalFixture, strategy: AgentStrategy
) -> tuple[ModelClient, ...]:
    bad = fixture.bad_files
    good = fixture.good_files
    if strategy == "single_pass":
        return (
            ScriptedModel(
                [
                    _script_step(
                        _proposal("single-bad", bad, "try the simplest repair"),
                        turn=1,
                        usage=(100, 60),
                    )
                ],
                name=f"{fixture.fixture_id}:single-pass:v1",
            ),
        )
    if strategy == "best_of_n":
        return (
            ScriptedModel(
                [
                    _script_step(
                        _proposals(
                            (
                                (
                                    "best-bad",
                                    bad,
                                    "keep a short constant-time shortcut",
                                ),
                                (
                                    "best-good",
                                    good,
                                    "use the standard complete algorithm",
                                ),
                            )
                        ),
                        turn=1,
                        usage=(140, 100),
                    )
                ],
                name=f"{fixture.fixture_id}:best-of-n:v1",
            ),
        )
    return (
        ScriptedModel(
            [
                _script_step(_plan(fixture.task), turn=1, usage=(120, 70)),
                _script_step(_plan(fixture.task), turn=2, usage=(120, 70)),
            ],
            name=f"{fixture.fixture_id}:planner:v1",
        ),
        ScriptedModel(
            [
                _script_step(
                    _proposal("orchestrated-bad", bad, "follow the first hypothesis"),
                    turn=1,
                    usage=(180, 100),
                ),
                _script_step(
                    _proposal(
                        "orchestrated-good", good, "repair the failing oracle case"
                    ),
                    turn=2,
                    usage=(180, 100),
                ),
            ],
            name=f"{fixture.fixture_id}:solver:v1",
        ),
        ScriptedModel(
            [
                _script_step(_review("retry", None), turn=1, usage=(70, 40)),
                _script_step(
                    _review("accept", "round-1-orchestrated-good"),
                    turn=2,
                    usage=(70, 40),
                ),
            ],
            name=f"{fixture.fixture_id}:reviewer:v1",
        ),
    )


def build_openai_model_factory(
    *,
    base_url: str,
    api_key: str,
    model: str,
    planner_model: str | None = None,
    solver_model: str | None = None,
    reviewer_model: str | None = None,
    timeout_seconds: float = 90.0,
    max_retries: int = 2,
    temperature: float | None = 0.0,
) -> AgentModelFactory:
    """Build a fresh OpenAI-compatible model tuple for every matrix cell.

    The factory keeps provider credentials outside the persisted report.  A new adapter
    is created for every fixture/strategy/repetition so a failed trajectory cannot leak
    cursor or conversation state into the next paired cell.
    """

    if not model:
        raise ValueError("model must not be empty")
    names = {
        "planner": planner_model or model,
        "solver": solver_model or model,
        "reviewer": reviewer_model or model,
    }

    def create(role: str) -> OpenAICompatibleModel:
        return OpenAICompatibleModel(
            base_url=base_url,
            api_key=api_key,
            model=names[role],
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            temperature=temperature,
        )

    def factory(
        _fixture: AgentEvalFixture, strategy: AgentStrategy
    ) -> tuple[ModelClient, ...]:
        if strategy == "orchestrated":
            return (create("planner"), create("solver"), create("reviewer"))
        return (create("solver"),)

    return factory


def _hidden_outcome(
    fixture: AgentEvalFixture,
    files: Mapping[str, str] | None,
    config: AgentEvalConfig,
) -> tuple[int, bool | None, str | None]:
    """Run the independent oracle only after a visible candidate is accepted."""

    if not config.include_hidden_tests or files is None:
        return 0, None, None
    try:
        combined = dict(files)
        overlap = set(combined) & set(fixture.hidden_files)
        if overlap:
            raise ValueError(
                f"hidden grader path collides with candidate: {sorted(overlap)!r}"
            )
        combined.update(fixture.hidden_files)
        candidate = CandidatePatch(
            id="hidden-oracle-candidate",
            parent_id="root",
            hypothesis=(
                "evaluate the accepted visible-test snapshot against hidden tests"
            ),
            files=combined,
        )
        result = evaluate_candidate(candidate, fixture.hidden_execution_config)
        if result.is_success:
            return 1, True, None
        return 1, False, result.output_excerpt or result.error or "hidden oracle failed"
    except (ValueError, OSError) as exc:
        return 1, False, f"{type(exc).__name__}: {str(exc)[:800]}"


def _run_strategy(
    fixture: AgentEvalFixture,
    strategy: AgentStrategy,
    config: AgentEvalConfig,
    repetition: int,
    model_factory: AgentModelFactory | None = None,
) -> AgentEvalRun:
    try:
        models = (
            _strategy_models(fixture, strategy)
            if model_factory is None
            else model_factory(fixture, strategy)
        )
        expected_models = 3 if strategy == "orchestrated" else 1
        if len(models) != expected_models:
            raise ValueError(
                f"model factory returned {len(models)} models for {strategy}; "
                f"expected {expected_models}"
            )
        execution = fixture.execution_config
        branch = BranchSearchConfig(
            beam_width=2,
            max_depth=1,
            max_candidates=config.max_candidates,
            search_policy=config.search_policy,
            exploration_constant=config.exploration_constant,
        )
        proposal = ProposalConfig(max_candidates=config.max_candidates)
        if strategy == "single_pass":
            session_report = asyncio.run(
                run_search_session(
                    models[0],
                    task=fixture.task,
                    root_files=fixture.root_files,
                    execution_config=execution,
                    config=SearchSessionConfig(
                        max_rounds=1,
                        max_model_calls=1,
                        max_candidates=1,
                        max_test_calls=1,
                    ),
                    proposal_config=ProposalConfig(max_candidates=1),
                    search_config=BranchSearchConfig(
                        beam_width=1,
                        max_depth=1,
                        max_candidates=1,
                        search_policy=config.search_policy,
                        exploration_constant=config.exploration_constant,
                    ),
                    run_id=f"agent-eval-{fixture.fixture_id}-single-{repetition}",
                )
            )
            hidden_calls, hidden_success, hidden_error = _hidden_outcome(
                fixture,
                session_report.best_files
                if session_report.status == "accepted"
                else None,
                config,
            )
            return AgentEvalRun(
                fixture_id=fixture.fixture_id,
                strategy=strategy,
                repetition=repetition,
                status=session_report.status,
                success=session_report.status == "accepted",
                rounds=len(session_report.rounds),
                model_calls=session_report.model_calls,
                planner_calls=0,
                solver_calls=session_report.model_calls,
                reviewer_calls=0,
                candidate_proposals=session_report.candidate_proposals,
                test_calls=session_report.test_calls,
                test_reuses=session_report.test_reuses,
                total_tokens=session_report.usage.total_tokens,
                best_candidate_id=session_report.best_candidate_id,
                error=(
                    session_report.reason if session_report.status == "failed" else None
                ),
                hidden_test_calls=hidden_calls,
                hidden_success=hidden_success,
                hidden_error=hidden_error,
            )
        if strategy == "best_of_n":
            session_report = asyncio.run(
                run_search_session(
                    models[0],
                    task=fixture.task,
                    root_files=fixture.root_files,
                    execution_config=execution,
                    config=SearchSessionConfig(
                        max_rounds=1,
                        max_model_calls=1,
                        max_candidates=config.max_candidates,
                        max_test_calls=config.max_test_calls,
                    ),
                    proposal_config=proposal,
                    search_config=branch,
                    run_id=f"agent-eval-{fixture.fixture_id}-best-{repetition}",
                )
            )
            hidden_calls, hidden_success, hidden_error = _hidden_outcome(
                fixture,
                session_report.best_files
                if session_report.status == "accepted"
                else None,
                config,
            )
            return AgentEvalRun(
                fixture_id=fixture.fixture_id,
                strategy=strategy,
                repetition=repetition,
                status=session_report.status,
                success=session_report.status == "accepted",
                rounds=len(session_report.rounds),
                model_calls=session_report.model_calls,
                planner_calls=0,
                solver_calls=session_report.model_calls,
                reviewer_calls=0,
                candidate_proposals=session_report.candidate_proposals,
                test_calls=session_report.test_calls,
                test_reuses=session_report.test_reuses,
                total_tokens=session_report.usage.total_tokens,
                best_candidate_id=session_report.best_candidate_id,
                error=(
                    session_report.reason if session_report.status == "failed" else None
                ),
                hidden_test_calls=hidden_calls,
                hidden_success=hidden_success,
                hidden_error=hidden_error,
            )
        planner, solver, reviewer = models
        orchestration_report = asyncio.run(
            run_orchestration(
                planner,
                solver,
                reviewer,
                task=fixture.task,
                root_files=fixture.root_files,
                execution_config=execution,
                config=OrchestrationConfig(
                    max_rounds=config.max_rounds,
                    max_model_calls=config.max_model_calls,
                    max_planner_calls=config.max_rounds,
                    max_solver_calls=config.max_rounds,
                    max_reviewer_calls=config.max_rounds,
                    max_candidates=config.max_candidates,
                    max_test_calls=config.max_test_calls,
                ),
                planner_config=PlannerConfig(max_items=2),
                solver_config=ProposalConfig(max_candidates=1),
                reviewer_config=ReviewerConfig(),
                search_config=branch,
                run_id=f"agent-eval-{fixture.fixture_id}-orchestrated-{repetition}",
            )
        )
        hidden_calls, hidden_success, hidden_error = _hidden_outcome(
            fixture,
            orchestration_report.best_files
            if orchestration_report.status == "accepted"
            else None,
            config,
        )
        return AgentEvalRun(
            fixture_id=fixture.fixture_id,
            strategy=strategy,
            repetition=repetition,
            status=orchestration_report.status,
            success=orchestration_report.status == "accepted",
            rounds=len(orchestration_report.rounds),
            model_calls=orchestration_report.model_calls,
            planner_calls=orchestration_report.planner_calls,
            solver_calls=orchestration_report.solver_calls,
            reviewer_calls=orchestration_report.reviewer_calls,
            candidate_proposals=orchestration_report.candidate_proposals,
            test_calls=orchestration_report.test_calls,
            test_reuses=orchestration_report.test_reuses,
            total_tokens=orchestration_report.usage.total_tokens,
            best_candidate_id=orchestration_report.best_candidate_id,
            error=(
                orchestration_report.reason
                if orchestration_report.status == "failed"
                else None
            ),
            hidden_test_calls=hidden_calls,
            hidden_success=hidden_success,
            hidden_error=hidden_error,
        )
    except (ModelError, RuntimeContractError, ValueError, OSError) as exc:
        return AgentEvalRun(
            fixture_id=fixture.fixture_id,
            strategy=strategy,
            repetition=repetition,
            status="failed",
            success=False,
            rounds=0,
            model_calls=0,
            planner_calls=0,
            solver_calls=0,
            reviewer_calls=0,
            candidate_proposals=0,
            test_calls=0,
            test_reuses=0,
            total_tokens=0,
            best_candidate_id=None,
            error=f"{type(exc).__name__}: {str(exc)[:800]}",
        )


def run_agent_evaluation(
    config: AgentEvalConfig | None = None,
    *,
    model_factory: AgentModelFactory | None = None,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
) -> AgentEvalReport:
    """Run the paired matrix, optionally checkpointing and resuming each cell."""

    cfg = config or AgentEvalConfig()
    if resume and checkpoint_path is None:
        raise ValueError("checkpoint_path is required when resuming an evaluation")
    if model_factory is not None and cfg.model_adapter == "scripted":
        cfg = replace(cfg, model_adapter="custom")
    available = {fixture.fixture_id: fixture for fixture in build_algorithm_fixtures()}
    unknown = set(cfg.fixtures) - set(available)
    if unknown:
        raise ValueError(f"unknown agent evaluation fixtures: {sorted(unknown)!r}")
    fixtures = tuple(available[name] for name in cfg.fixtures)
    checkpoint = None
    if resume:
        checkpoint = read_agent_evaluation_checkpoint(checkpoint_path)  # type: ignore[arg-type]
        if checkpoint.config.to_dict() != cfg.to_dict():
            raise ValueError("evaluation checkpoint configuration does not match")
        if checkpoint.fixture_ids != tuple(fixture.fixture_id for fixture in fixtures):
            raise ValueError("evaluation checkpoint fixtures do not match")
    completed = (
        {(run.fixture_id, run.strategy, run.repetition): run for run in checkpoint.runs}
        if checkpoint is not None
        else {}
    )
    if checkpoint_path is not None and not resume:
        write_agent_evaluation_checkpoint(
            AgentEvalCheckpoint(
                config=cfg,
                fixture_ids=tuple(fixture.fixture_id for fixture in fixtures),
            ),
            checkpoint_path,
        )
    for fixture in fixtures:
        for strategy in cfg.strategies:
            for repetition in range(cfg.repetitions):
                key = (fixture.fixture_id, strategy, repetition)
                if key in completed:
                    continue
                completed[key] = _run_strategy(
                    fixture, strategy, cfg, repetition, model_factory=model_factory
                )
                if checkpoint_path is not None:
                    ordered_runs = tuple(
                        completed[item]
                        for item in (
                            (fixture_item.fixture_id, strategy_item, repetition_item)
                            for fixture_item in fixtures
                            for strategy_item in cfg.strategies
                            for repetition_item in range(cfg.repetitions)
                        )
                        if item in completed
                    )
                    write_agent_evaluation_checkpoint(
                        AgentEvalCheckpoint(
                            config=cfg,
                            fixture_ids=tuple(
                                fixture_item.fixture_id for fixture_item in fixtures
                            ),
                            runs=ordered_runs,
                        ),
                        checkpoint_path,
                    )
    runs = [
        completed[(fixture.fixture_id, strategy, repetition)]
        for fixture in fixtures
        for strategy in cfg.strategies
        for repetition in range(cfg.repetitions)
    ]
    summaries = tuple(
        _summary(strategy, tuple(run for run in runs if run.strategy == strategy))
        for strategy in cfg.strategies
    )
    return AgentEvalReport(
        config=cfg,
        fixtures=fixtures,
        runs=tuple(runs),
        summaries=summaries,
        claim_boundary=_claim_boundary(cfg.model_adapter),
    )


def _format_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def _format_interval(successes: int, total: int) -> str:
    if total <= 0:
        return "n/a"
    low, high = wilson_interval(successes, total)
    return f"{low:.0%}-{high:.0%}"


def _format_percent_delta(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.0%}"


def _comparison_baseline(report: AgentEvalReport) -> AgentStrategy | None:
    if len(report.config.strategies) < 2:
        return None
    if "single_pass" in report.config.strategies:
        return "single_pass"
    return report.config.strategies[0]


def _hidden_label(value: bool | None) -> str:
    return "n/a" if value is None else ("pass" if value else "fail")


def render_agent_evaluation_console(report: AgentEvalReport) -> str:
    lines = [
        "strategy visible | visible 95% CI | hidden | mean model calls | "
        "mean visible tests | mean tokens",
        "--- | ---: | ---: | ---: | ---: | ---:",
    ]
    lines.extend(
        f"{summary.strategy} {summary.success_count}/{summary.run_count} "
        f"({summary.success_rate:.0%}) | "
        f"{_format_interval(summary.success_count, summary.run_count)} | "
        f"{summary.hidden_success_count}/{summary.hidden_run_count} "
        f"({_format_rate(summary.hidden_success_rate)}) | "
        f"{summary.mean_model_calls:.1f} | {summary.mean_test_calls:.1f} | "
        f"{summary.mean_total_tokens:.0f}"
        for summary in report.summaries
    )
    failures = sum(not run.success for run in report.runs)
    lines.append(
        f"runs={len(report.runs)} failures={failures} fixtures={len(report.fixtures)}"
    )
    return "\n".join(lines) + "\n"


def render_agent_evaluation_markdown(report: AgentEvalReport) -> str:
    lines = [
        "# ContextOpt coding-agent strategy evaluation",
        "",
        "This report uses executable ACM/math fixtures, independent hidden tests, "
        "and deterministic scripted model responses.",
        "",
        "| Strategy | Visible | Visible 95% CI | Hidden | Mean model calls | "
        "Mean visible tests | "
        "Mean reuses | Mean candidates | Mean tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| `{summary.strategy}` | {summary.success_count}/{summary.run_count} "
        f"({summary.success_rate:.0%}) | "
        f"{_format_interval(summary.success_count, summary.run_count)} | "
        f"{summary.hidden_success_count}/{summary.hidden_run_count} "
        f"({_format_rate(summary.hidden_success_rate)}) | "
        f"{summary.mean_model_calls:.1f} | "
        f"{summary.mean_test_calls:.1f} | {summary.mean_test_reuses:.1f} | "
        f"{summary.mean_candidate_proposals:.1f} | {summary.mean_total_tokens:.0f} |"
        for summary in report.summaries
    )
    baseline = _comparison_baseline(report)
    if baseline is not None:
        comparisons = build_agent_eval_comparisons(report, baseline_strategy=baseline)
        lines.extend(
            (
                "",
                "## Paired comparisons",
                "",
                "Each delta is candidate minus the baseline on the same fixture and "
                "repetition; positive visible delta means more paired wins.",
                "",
                "| Strategy | Baseline | Paired | Wins | Losses | Ties | Visible Δ | "
                "Hidden Δ | Mean tests Δ (stdev) | Mean tokens Δ (stdev) |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            )
        )
        lines.extend(
            f"| `{comparison.strategy}` | `{comparison.baseline_strategy}` | "
            f"{comparison.paired_count} | {comparison.wins} | {comparison.losses} | "
            f"{comparison.ties} | {_format_percent_delta(comparison.visible_delta)} | "
            f"{_format_percent_delta(comparison.hidden_delta)} | "
            f"{comparison.mean_test_call_delta:+.1f} "
            f"(stdev {comparison.stddev_test_call_delta:.1f}) | "
            f"{comparison.mean_token_delta:+.0f} "
            f"(stdev {comparison.stddev_token_delta:.0f}) |"
            for comparison in comparisons
        )
    lines.extend(("", "## Fixtures", "", "| ID | Category | Task |", "|---|---|---|"))
    lines.extend(
        f"| `{fixture.fixture_id}` | {fixture.category} | {fixture.title} |"
        for fixture in report.fixtures
    )
    lines.extend(
        (
            "",
            "## Run ledger",
            "",
            "| Fixture | Strategy | Status | Visible tests | Hidden | Best candidate | "
            "Error |",
            "|---|---|---|---:|---:|---|---|",
        )
    )
    lines.extend(
        f"| `{run.fixture_id}` | `{run.strategy}` | {run.status} | {run.test_calls} | "
        f"{_hidden_label(run.hidden_success)} "
        "| "
        f"{run.best_candidate_id or 'none'} | {run.error or ''} |"
        for run in report.runs
    )
    lines.extend(("", "## Claim boundary", "", f"> {report.claim_boundary}", ""))
    return "\n".join(lines)


def _render_agent_eval_comparison_html(report: AgentEvalReport) -> str:
    baseline = _comparison_baseline(report)
    if baseline is None:
        return ""
    comparisons = build_agent_eval_comparisons(report, baseline_strategy=baseline)
    rows = "".join(
        "<tr>"
        f"<td><code>{escape(comparison.strategy)}</code></td>"
        f"<td><code>{escape(comparison.baseline_strategy)}</code></td>"
        f"<td>{comparison.paired_count}</td>"
        f"<td>{comparison.wins}/{comparison.losses}/{comparison.ties}</td>"
        f"<td>{_format_percent_delta(comparison.visible_delta)}</td>"
        f"<td>{_format_percent_delta(comparison.hidden_delta)}</td>"
        f"<td>{comparison.mean_test_call_delta:+.1f} "
        f"(stdev {comparison.stddev_test_call_delta:.1f})</td>"
        f"<td>{comparison.mean_token_delta:+.0f} "
        f"(stdev {comparison.stddev_token_delta:.0f})</td>"
        "</tr>"
        for comparison in comparisons
    )
    return (
        "<h2>Paired comparisons</h2>"
        "<p class='muted'>Deltas are candidate minus baseline on the same "
        "fixture and repetition. Win/loss/tie is visible outcome count.</p>"
        "<table><thead><tr><th>strategy</th><th>baseline</th><th>paired</th>"
        "<th>wins/losses/ties</th><th>visible Δ</th><th>hidden Δ</th>"
        "<th>mean tests Δ (stdev)</th><th>mean tokens Δ (stdev)</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )


def render_agent_evaluation_html(report: AgentEvalReport) -> str:
    """Render a dependency-free portfolio dashboard for repeated evaluations."""

    summary_rows: list[str] = []
    chart_rows: list[str] = []
    for summary in report.summaries:
        visible = summary.success_rate * 100
        hidden = (
            0.0
            if summary.hidden_success_rate is None
            else summary.hidden_success_rate * 100
        )
        summary_rows.append(
            "<tr>"
            f"<td><code>{escape(summary.strategy)}</code></td>"
            f"<td>{summary.success_count}/{summary.run_count} "
            f"({summary.success_rate:.0%})</td>"
            f"<td>{_format_interval(summary.success_count, summary.run_count)}</td>"
            f"<td>{summary.hidden_success_count}/{summary.hidden_run_count} "
            f"({_format_rate(summary.hidden_success_rate)})</td>"
            f"<td>{summary.mean_model_calls:.1f}</td>"
            f"<td>{summary.mean_test_calls:.1f}</td>"
            f"<td>{summary.mean_total_tokens:.0f}</td>"
            "</tr>"
        )
        chart_rows.append(
            "<div class='chart-row'>"
            f"<span class='label'>{escape(summary.strategy)}</span>"
            "<div class='track'><span class='visible' "
            f"style='width:{visible:.2f}%'></span></div>"
            "<div class='track'><span class='hidden' "
            f"style='width:{hidden:.2f}%'></span></div>"
            f"<span class='rate'>{summary.success_rate:.0%} / "
            f"{_format_rate(summary.hidden_success_rate)}</span>"
            "</div>"
        )
    payload = escape(json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True))
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>ContextOpt agent evaluation</title>"
        "<style>body{font:14px system-ui,sans-serif;margin:2rem;color:#172033;"
        "max-width:1100px}h1{margin-bottom:.25rem}.muted{color:#5d6b82}"
        ".legend{display:flex;gap:1rem;margin:.8rem 0}.swatch{display:inline-block;"
        "width:12px;height:12px;border-radius:3px;margin-right:.3rem}"
        ".visible-swatch{background:#2563eb}.hidden-swatch{background:#f97316}"
        ".chart{border:1px solid #d7dce8;border-radius:10px;padding:1rem;"
        "background:#fbfcff}.chart-row{display:grid;grid-template-columns:130px 1fr "
        "1fr 100px;gap:.6rem;align-items:center;margin:.55rem 0}.track{height:12px;"
        "background:#e8edf5;border-radius:999px;overflow:hidden}.track span{"
        "display:block;"
        "height:100%}.visible{background:#2563eb}.hidden{background:#f97316}.rate{"
        "text-align:right;font-variant-numeric:tabular-nums}table{border-collapse:collapse;"
        "width:100%;margin-top:1.5rem}th,td{border:1px solid #d7dce8;padding:.55rem;"
        "text-align:left}th{background:#f5f7fb}details{margin-top:1.5rem}"
        "code{white-space:pre-wrap}</style></head><body>"
        "<h1>ContextOpt coding-agent evaluation</h1>"
        f"<p class='muted'>{escape(report.claim_boundary)}</p>"
        "<div class='legend'><span><i class='swatch visible-swatch'></i>"
        "visible success</span>"
        "<span><i class='swatch hidden-swatch'></i>hidden success</span></div>"
        f"<section class='chart'>{''.join(chart_rows)}</section>"
        "<table><thead><tr><th>strategy</th><th>visible</th><th>visible 95% CI</th>"
        "<th>hidden</th>"
        "<th>mean model calls</th><th>mean tests</th><th>mean tokens</th>"
        f"</tr></thead><tbody>{''.join(summary_rows)}</tbody></table>"
        f"{_render_agent_eval_comparison_html(report)}"
        "<details><summary>durable evaluation JSON</summary><code>"
        f"{payload}</code></details></body></html>\n"
    )


__all__ = [
    "AgentEvalCheckpoint",
    "AgentEvalComparison",
    "AgentEvalConfig",
    "AgentEvalFixture",
    "AgentEvalReport",
    "AgentEvalRun",
    "AgentEvalSummary",
    "AgentModelFactory",
    "AgentStrategy",
    "build_agent_eval_comparisons",
    "build_algorithm_fixtures",
    "build_openai_model_factory",
    "read_agent_evaluation_checkpoint",
    "render_agent_evaluation_console",
    "render_agent_evaluation_html",
    "render_agent_evaluation_markdown",
    "run_agent_evaluation",
    "wilson_interval",
    "write_agent_evaluation_checkpoint",
]
