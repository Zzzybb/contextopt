"""Durable iterative proposal/test/search sessions for coding agents.

``BranchSearch`` is intentionally pure and ``propose_case`` is intentionally one-shot.
This module composes them into the long-running seam that a coding-agent product needs:
the model proposes bounded snapshots, an executable oracle evaluates them in disposable
workspaces, failures become the next proposal's evidence, and every completed phase is
checkpointed before the next model call.  A pending model request is treated as
retryable rather than silently claimed to be exactly-once.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

from contextopt.runtime.errors import ModelError, RuntimeContractError
from contextopt.runtime.identity import stable_hash
from contextopt.runtime.protocol import ModelClient, TokenUsage
from contextopt.search.branching import (
    BranchCase,
    BranchSearch,
    BranchSearchConfig,
    BranchSearchReport,
    CandidatePatch,
    SearchEvent,
    TestResult,
    verify_search_events,
)
from contextopt.search.executor import ExecutableSearchConfig, evaluate_candidate
from contextopt.search.proposer import (
    ProposalConfig,
    propose_case,
)
from contextopt.search.scheduler import SchedulerPolicy, select_candidate_batch

SessionStatus = Literal[
    "running",
    "accepted",
    "exhausted",
    "budget_exhausted",
    "paused",
    "failed",
]
SessionPhase = Literal["idle", "proposing", "evaluating"]


def _non_empty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _files(value: Mapping[str, str], label: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object mapping paths to text")
    # BranchCase owns the canonical relative-POSIX path validation.  Reusing it here
    # keeps checkpoints and model inputs on exactly the same path contract.
    normalized = BranchCase(
        task="session-file-validation",
        root_files=value,
        candidates=(),
        tests={},
    ).root_files
    return dict(normalized)


def _json_safe(value: Any, label: str) -> None:
    try:
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be canonical JSON") from exc


@dataclass(frozen=True, slots=True)
class SearchSessionConfig:
    """Global budgets shared by proposal, oracle, and retry phases."""

    max_rounds: int = 3
    max_model_calls: int = 4
    max_candidates: int = 16
    max_test_calls: int = 16
    max_parallel_tests: int = 1
    scheduler_policy: SchedulerPolicy = "fixed"

    def __post_init__(self) -> None:
        for name in (
            "max_rounds",
            "max_model_calls",
            "max_candidates",
            "max_test_calls",
            "max_parallel_tests",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.scheduler_policy not in {"fixed", "adaptive"}:
            raise ValueError(f"unsupported scheduler policy: {self.scheduler_policy!r}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SearchSessionConfig:
        value = _mapping(data, "session config")
        allowed = {
            "max_rounds",
            "max_model_calls",
            "max_candidates",
            "max_test_calls",
            "max_parallel_tests",
            "scheduler_policy",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"session config has unknown fields: {sorted(unknown)!r}")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_rounds": self.max_rounds,
            "max_model_calls": self.max_model_calls,
            "max_candidates": self.max_candidates,
            "max_test_calls": self.max_test_calls,
            "max_parallel_tests": self.max_parallel_tests,
            "scheduler_policy": self.scheduler_policy,
        }


@dataclass(frozen=True, slots=True)
class SearchSessionRound:
    """One completed proposal and oracle/search decision."""

    index: int
    base_fingerprint: str
    report: BranchSearchReport
    model_turn: int
    model_name: str
    response_sha256: str
    response_id: str | None
    usage: TokenUsage

    def __post_init__(self) -> None:
        if (
            not isinstance(self.index, int)
            or isinstance(self.index, bool)
            or self.index < 0
        ):
            raise ValueError("round index must be a non-negative integer")
        object.__setattr__(
            self,
            "base_fingerprint",
            _non_empty(self.base_fingerprint, "base_fingerprint"),
        )
        if not isinstance(self.model_turn, int) or self.model_turn <= 0:
            raise ValueError("model_turn must be a positive integer")
        object.__setattr__(
            self, "model_name", _non_empty(self.model_name, "model_name")
        )
        object.__setattr__(
            self,
            "response_sha256",
            _non_empty(self.response_sha256, "response_sha256"),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SearchSessionRound:
        value = _mapping(data, "session round")
        allowed = {
            "index",
            "base_fingerprint",
            "report",
            "model_turn",
            "model_name",
            "response_sha256",
            "response_id",
            "usage",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"session round has unknown fields: {sorted(unknown)!r}")
        report = value.get("report")
        usage = value.get("usage")
        if not isinstance(report, Mapping) or not isinstance(usage, Mapping):
            raise ValueError("session round report and usage must be objects")
        response_id = value.get("response_id")
        if response_id is not None and not isinstance(response_id, str):
            raise ValueError("session round response_id must be a string or null")
        return cls(
            index=int(value.get("index", -1)),
            base_fingerprint=_non_empty(
                value.get("base_fingerprint"), "base_fingerprint"
            ),
            report=BranchSearchReport.from_dict(report),
            model_turn=int(value.get("model_turn", -1)),
            model_name=_non_empty(value.get("model_name"), "model_name"),
            response_sha256=_non_empty(value.get("response_sha256"), "response_sha256"),
            response_id=response_id,
            usage=TokenUsage.from_dict(usage),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "base_fingerprint": self.base_fingerprint,
            "report": self.report.to_dict(),
            "model_turn": self.model_turn,
            "model_name": self.model_name,
            "response_sha256": self.response_sha256,
            "response_id": self.response_id,
            "usage": self.usage.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class SearchSessionReport:
    """Checkpointed, hash-chained state for one iterative coding search."""

    run_id: str
    task: str
    model_fingerprint: str
    status: SessionStatus
    phase: SessionPhase
    config: SearchSessionConfig
    proposal_config: ProposalConfig
    search_config: BranchSearchConfig
    execution_config: ExecutableSearchConfig
    root_files: Mapping[str, str]
    base_files: Mapping[str, str]
    rounds: tuple[SearchSessionRound, ...]
    observations: Mapping[str, TestResult]
    events: tuple[SearchEvent, ...]
    model_calls: int
    test_calls: int
    candidate_proposals: int
    test_reuses: int
    usage: TokenUsage
    best_candidate_id: str | None
    best_files: Mapping[str, str] | None
    max_in_flight: int = 1
    reason: str | None = None
    pending_case: BranchCase | None = None
    pending_model_turn: int | None = None
    pending_model_name: str | None = None
    pending_response_sha256: str | None = None
    pending_response_id: str | None = None
    pending_response_usage: TokenUsage | None = None
    feedback: tuple[Mapping[str, Any], ...] = ()
    schema_version: str = "1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _non_empty(self.run_id, "run_id"))
        object.__setattr__(self, "task", _non_empty(self.task, "task"))
        object.__setattr__(
            self,
            "model_fingerprint",
            _non_empty(self.model_fingerprint, "model_fingerprint"),
        )
        if self.schema_version != "1":
            raise ValueError(f"unsupported session schema: {self.schema_version!r}")
        if self.status not in {
            "running",
            "accepted",
            "exhausted",
            "budget_exhausted",
            "paused",
            "failed",
        }:
            raise ValueError(f"unsupported session status: {self.status!r}")
        if self.phase not in {"idle", "proposing", "evaluating"}:
            raise ValueError(f"unsupported session phase: {self.phase!r}")
        if self.status != "running" and self.phase != "idle":
            raise ValueError("terminal or paused session must be idle")
        for name in (
            "model_calls",
            "test_calls",
            "candidate_proposals",
            "test_reuses",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if (
            not isinstance(self.max_in_flight, int)
            or isinstance(self.max_in_flight, bool)
            or self.max_in_flight <= 0
        ):
            raise ValueError("max_in_flight must be a positive integer")
        if self.best_candidate_id is not None:
            object.__setattr__(
                self,
                "best_candidate_id",
                _non_empty(self.best_candidate_id, "best_candidate_id"),
            )
        if self.reason is not None:
            object.__setattr__(self, "reason", _non_empty(self.reason, "reason"))
        object.__setattr__(self, "root_files", _files(self.root_files, "root_files"))
        object.__setattr__(self, "base_files", _files(self.base_files, "base_files"))
        if self.best_files is not None:
            object.__setattr__(
                self, "best_files", _files(self.best_files, "best_files")
            )
        rounds = tuple(sorted(self.rounds, key=lambda item: item.index))
        if [item.index for item in rounds] != list(range(len(rounds))):
            raise ValueError("session round indexes must be contiguous from zero")
        if rounds and any(item.report.task != self.task for item in rounds):
            raise ValueError("session round task must match session task")
        object.__setattr__(self, "rounds", rounds)
        object.__setattr__(self, "observations", dict(self.observations))
        for fingerprint, result in self.observations.items():
            _non_empty(fingerprint, "observation fingerprint")
            if not isinstance(result, TestResult):
                raise ValueError("observations must map fingerprints to TestResult")
        verify_search_events(self.events)
        feedback = tuple(dict(item) for item in self.feedback)
        _json_safe(feedback, "feedback")
        object.__setattr__(self, "feedback", feedback)
        if self.phase == "evaluating":
            if self.pending_case is None:
                raise ValueError("evaluating session must have a pending case")
            if self.pending_model_turn is None or self.pending_model_turn <= 0:
                raise ValueError("evaluating session must have a pending model turn")
            if self.pending_response_usage is None:
                raise ValueError("evaluating session must have pending response usage")
        elif any(
            value is not None
            for value in (
                self.pending_case,
                self.pending_model_turn,
                self.pending_model_name,
                self.pending_response_sha256,
                self.pending_response_id,
                self.pending_response_usage,
            )
        ):
            raise ValueError("pending proposal fields require evaluating phase")
        if self.status == "accepted" and (
            self.best_candidate_id is None or self.best_files is None
        ):
            raise ValueError("accepted session must identify its best candidate")

    @property
    def metrics(self) -> Mapping[str, int | float]:
        return {
            "rounds": len(self.rounds),
            "model_calls": self.model_calls,
            "candidate_proposals": self.candidate_proposals,
            "test_calls": self.test_calls,
            "test_reuses": self.test_reuses,
            "cached_observations": len(self.observations),
            "total_tokens": self.usage.total_tokens,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SearchSessionReport:
        value = _mapping(data, "session report")
        allowed = {
            "schema_version",
            "run_id",
            "task",
            "model_fingerprint",
            "status",
            "phase",
            "config",
            "proposal_config",
            "search_config",
            "execution_config",
            "root_files",
            "base_files",
            "rounds",
            "observations",
            "events",
            "model_calls",
            "test_calls",
            "candidate_proposals",
            "test_reuses",
            "usage",
            "metrics",
            "best_candidate_id",
            "best_files",
            "max_in_flight",
            "reason",
            "pending_case",
            "pending_model_turn",
            "pending_model_name",
            "pending_response_sha256",
            "pending_response_id",
            "pending_response_usage",
            "feedback",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"session report has unknown fields: {sorted(unknown)!r}")
        raw_rounds = value.get("rounds")
        raw_observations = value.get("observations")
        raw_events = value.get("events")
        raw_feedback = value.get("feedback", [])
        if (
            not isinstance(raw_rounds, list)
            or not isinstance(raw_observations, Mapping)
            or not isinstance(raw_events, list)
            or not isinstance(raw_feedback, list)
        ):
            raise ValueError(
                "session rounds, observations, events, and feedback have invalid shapes"
            )
        config = value.get("config")
        proposal_config = value.get("proposal_config")
        search_config = value.get("search_config")
        execution_config = value.get("execution_config")
        usage = value.get("usage")
        if not all(
            isinstance(item, Mapping)
            for item in (
                config,
                proposal_config,
                search_config,
                execution_config,
                usage,
            )
        ):
            raise ValueError("session configs and usage must be objects")
        raw_metrics = value.get("metrics")
        if raw_metrics is not None:
            if not isinstance(raw_metrics, Mapping):
                raise ValueError("session metrics must be an object")
            for key, metric in raw_metrics.items():
                if not isinstance(key, str):
                    raise ValueError("session metric names must be strings")
                if not isinstance(metric, (int, float)) or isinstance(metric, bool):
                    raise ValueError(f"session metric {key!r} must be numeric")
        root_files = value.get("root_files")
        base_files = value.get("base_files")
        if not isinstance(root_files, Mapping) or not isinstance(base_files, Mapping):
            raise ValueError("session root_files and base_files must be objects")
        best_files = value.get("best_files")
        if best_files is not None and not isinstance(best_files, Mapping):
            raise ValueError("session best_files must be an object or null")
        pending_case = value.get("pending_case")
        if pending_case is not None and not isinstance(pending_case, Mapping):
            raise ValueError("session pending_case must be an object or null")
        pending_usage = value.get("pending_response_usage")
        if pending_usage is not None and not isinstance(pending_usage, Mapping):
            raise ValueError("session pending_response_usage must be an object or null")
        observations: dict[str, TestResult] = {}
        for fingerprint, raw_result in raw_observations.items():
            if not isinstance(fingerprint, str) or not isinstance(raw_result, Mapping):
                raise ValueError("session observations must map strings to objects")
            observations[fingerprint] = TestResult.from_dict(raw_result)
        feedback: list[Mapping[str, Any]] = []
        for index, item in enumerate(raw_feedback):
            if not isinstance(item, Mapping):
                raise ValueError(f"session feedback[{index}] must be an object")
            feedback.append(item)
        status = value.get("status")
        phase = value.get("phase")
        if status not in {
            "running",
            "accepted",
            "exhausted",
            "budget_exhausted",
            "paused",
            "failed",
        }:
            raise ValueError("invalid session status")
        if phase not in {"idle", "proposing", "evaluating"}:
            raise ValueError("invalid session phase")
        config_value = _mapping(config, "session config")
        proposal_config_value = _mapping(proposal_config, "proposal config")
        search_config_value = _mapping(search_config, "search config")
        execution_config_value = _mapping(execution_config, "execution config")
        usage_value = _mapping(usage, "usage")
        report = cls(
            schema_version=str(value.get("schema_version", "")),
            run_id=_non_empty(value.get("run_id"), "run_id"),
            task=_non_empty(value.get("task"), "task"),
            model_fingerprint=_non_empty(
                value.get("model_fingerprint"), "model_fingerprint"
            ),
            status=cast(SessionStatus, status),
            phase=cast(SessionPhase, phase),
            config=SearchSessionConfig.from_dict(config_value),
            proposal_config=ProposalConfig.from_dict(proposal_config_value),
            search_config=BranchSearchConfig.from_dict(search_config_value),
            execution_config=ExecutableSearchConfig.from_dict(execution_config_value),
            root_files=root_files,
            base_files=base_files,
            rounds=tuple(SearchSessionRound.from_dict(item) for item in raw_rounds),
            observations=observations,
            events=tuple(SearchEvent.from_dict(item) for item in raw_events),
            model_calls=int(value.get("model_calls", -1)),
            test_calls=int(value.get("test_calls", -1)),
            candidate_proposals=int(value.get("candidate_proposals", -1)),
            test_reuses=int(value.get("test_reuses", -1)),
            usage=TokenUsage.from_dict(usage_value),
            best_candidate_id=(
                None
                if value.get("best_candidate_id") is None
                else str(value["best_candidate_id"])
            ),
            best_files=best_files,
            max_in_flight=int(value.get("max_in_flight", 1)),
            reason=(None if value.get("reason") is None else str(value["reason"])),
            pending_case=(
                None if pending_case is None else BranchCase.from_dict(pending_case)
            ),
            pending_model_turn=(
                None
                if value.get("pending_model_turn") is None
                else int(value["pending_model_turn"])
            ),
            pending_model_name=(
                None
                if value.get("pending_model_name") is None
                else str(value["pending_model_name"])
            ),
            pending_response_sha256=(
                None
                if value.get("pending_response_sha256") is None
                else str(value["pending_response_sha256"])
            ),
            pending_response_id=(
                None
                if value.get("pending_response_id") is None
                else str(value["pending_response_id"])
            ),
            pending_response_usage=(
                None if pending_usage is None else TokenUsage.from_dict(pending_usage)
            ),
            feedback=tuple(feedback),
        )
        if raw_metrics is not None:
            expected_metrics = report.metrics
            if set(raw_metrics) != set(expected_metrics):
                raise ValueError("session metrics are inconsistent")
            for key, expected in expected_metrics.items():
                if raw_metrics[key] != expected:
                    raise ValueError(f"session metric {key!r} is inconsistent")
        return report

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "task": self.task,
            "model_fingerprint": self.model_fingerprint,
            "status": self.status,
            "phase": self.phase,
            "config": self.config.to_dict(),
            "proposal_config": self.proposal_config.to_dict(),
            "search_config": self.search_config.to_dict(),
            "execution_config": self.execution_config.to_dict(),
            "root_files": dict(self.root_files),
            "base_files": dict(self.base_files),
            "rounds": [item.to_dict() for item in self.rounds],
            "observations": {
                fingerprint: result.to_dict()
                for fingerprint, result in sorted(self.observations.items())
            },
            "events": [event.to_dict() for event in self.events],
            "model_calls": self.model_calls,
            "test_calls": self.test_calls,
            "candidate_proposals": self.candidate_proposals,
            "test_reuses": self.test_reuses,
            "usage": self.usage.to_dict(),
            "metrics": dict(self.metrics),
            "best_candidate_id": self.best_candidate_id,
            "best_files": None if self.best_files is None else dict(self.best_files),
            "max_in_flight": self.max_in_flight,
            "reason": self.reason,
            "pending_case": None
            if self.pending_case is None
            else self.pending_case.to_dict(),
            "pending_model_turn": self.pending_model_turn,
            "pending_model_name": self.pending_model_name,
            "pending_response_sha256": self.pending_response_sha256,
            "pending_response_id": self.pending_response_id,
            "pending_response_usage": None
            if self.pending_response_usage is None
            else self.pending_response_usage.to_dict(),
            "feedback": [dict(item) for item in self.feedback],
        }


def _append_event(
    report: SearchSessionReport,
    event_type: str,
    data: Mapping[str, Any],
) -> SearchSessionReport:
    event = SearchEvent.create(
        len(report.events),
        event_type,
        data,
        report.events[-1].sha256 if report.events else None,
    )
    return replace(report, events=(*report.events, event))


def write_session_checkpoint(report: SearchSessionReport, path: str | Path) -> None:
    """Atomically write a JSON checkpoint after a durable session phase."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True, indent=2)
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
            raise OSError("temporary checkpoint path was not created")
        os.replace(temporary_path, target)
    except OSError:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def read_session_checkpoint(path: str | Path) -> SearchSessionReport:
    """Read and strictly validate one session checkpoint."""

    target = Path(path)
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read session checkpoint {target}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ValueError("session checkpoint must be a JSON object")
    return SearchSessionReport.from_dict(data)


def _namespace_case(case: BranchCase, round_index: int) -> BranchCase:
    prefix = f"round-{round_index}-"
    ids = {candidate.id: prefix + candidate.id for candidate in case.candidates}
    candidates = tuple(
        CandidatePatch(
            id=ids[candidate.id],
            parent_id=(
                "root" if candidate.parent_id == "root" else ids[candidate.parent_id]
            ),
            hypothesis=candidate.hypothesis,
            files=candidate.files,
            evidence=candidate.evidence,
        )
        for candidate in case.candidates
    )
    tests = {ids[candidate_id]: result for candidate_id, result in case.tests.items()}
    return BranchCase(
        task=case.task,
        root_files=case.root_files,
        candidates=candidates,
        tests=tests,
    )


def _feedback_from_report(report: BranchSearchReport) -> list[dict[str, Any]]:
    nodes = [node for node in report.nodes if node.id != "root" and node.test_result]
    nodes.sort(key=lambda node: (-node.score, node.depth, node.id))
    feedback: list[dict[str, Any]] = []
    for node in nodes[:8]:
        result = node.test_result
        if result is None:
            continue
        feedback.append(
            {
                "candidate_id": node.id,
                "hypothesis": node.hypothesis,
                "score": round(node.score, 6),
                "passed_tests": list(result.passed_tests),
                "failed_tests": list(result.failed_tests),
                "error": result.error,
                "output_excerpt": (
                    None
                    if result.output_excerpt is None
                    else result.output_excerpt[:1200]
                ),
            }
        )
    return feedback


def _best_node(report: BranchSearchReport) -> Any | None:
    nodes = [node for node in report.nodes if node.id != "root"]
    if not nodes:
        return None
    return sorted(nodes, key=lambda node: (-node.score, node.depth, node.id))[0]


def _next_base_files(
    report: BranchSearchReport, fallback: Mapping[str, str]
) -> dict[str, str]:
    node = _best_node(report)
    if node is None:
        return dict(fallback)
    candidate = report.case.by_id.get(node.id)
    if candidate is None:
        return dict(fallback)
    return dict(candidate.files)


def _terminal_status(
    state: SearchSessionReport,
    *,
    rounds: int | None = None,
    candidates: int | None = None,
) -> SessionStatus:
    round_count = len(state.rounds) if rounds is None else rounds
    candidate_count = state.candidate_proposals if candidates is None else candidates
    if round_count >= state.config.max_rounds:
        return "budget_exhausted"
    if state.model_calls >= state.config.max_model_calls:
        return "budget_exhausted"
    if candidate_count >= state.config.max_candidates:
        return "budget_exhausted"
    if state.test_calls >= state.config.max_test_calls:
        return "budget_exhausted"
    return "exhausted"


def _finish(
    state: SearchSessionReport,
    status: SessionStatus,
    reason: str,
) -> SearchSessionReport:
    updated = replace(state, status=status, phase="idle", reason=reason)
    return _append_event(
        updated,
        "session.completed",
        {
            "status": status,
            "reason": reason,
            "metrics": dict(updated.metrics),
            "best_candidate_id": updated.best_candidate_id,
        },
    )


class SearchSessionRunner:
    """Run or resume an iterative model proposal and test-guided search session."""

    def __init__(
        self,
        model: ModelClient,
        execution_config: ExecutableSearchConfig | None = None,
        *,
        config: SearchSessionConfig | None = None,
        proposal_config: ProposalConfig | None = None,
        search_config: BranchSearchConfig | None = None,
    ) -> None:
        self.model = model
        self.execution_config = execution_config
        self.config = config
        self.proposal_config = proposal_config
        self.search_config = search_config

    def _adopt_checkpoint_config(self, state: SearchSessionReport) -> None:
        if self.model.configuration_fingerprint != state.model_fingerprint:
            raise ValueError("model configuration does not match session checkpoint")
        for supplied, stored, label in (
            (self.config, state.config, "session"),
            (self.proposal_config, state.proposal_config, "proposal"),
            (self.search_config, state.search_config, "search"),
            (self.execution_config, state.execution_config, "execution"),
        ):
            if supplied is None:
                continue
            if supplied.to_dict() != stored.to_dict():
                raise ValueError(f"{label} configuration does not match checkpoint")
        self.config = state.config
        self.proposal_config = state.proposal_config
        self.search_config = state.search_config
        self.execution_config = state.execution_config

    def _initial(
        self,
        task: str,
        root_files: Mapping[str, str],
        run_id: str,
    ) -> SearchSessionReport:
        if self.execution_config is None:
            raise ValueError("execution_config is required for a new session")
        config = self.config or SearchSessionConfig()
        proposal_config = self.proposal_config or ProposalConfig()
        search_config = self.search_config or BranchSearchConfig()
        normalized_files = _files(root_files, "root_files")
        state = SearchSessionReport(
            run_id=run_id,
            task=task,
            model_fingerprint=self.model.configuration_fingerprint,
            status="running",
            phase="idle",
            config=config,
            proposal_config=proposal_config,
            search_config=search_config,
            execution_config=self.execution_config,
            root_files=normalized_files,
            base_files=normalized_files,
            rounds=(),
            observations={},
            events=(),
            model_calls=0,
            test_calls=0,
            candidate_proposals=0,
            test_reuses=0,
            usage=TokenUsage(),
            best_candidate_id=None,
            best_files=None,
        )
        return _append_event(
            state,
            "session.started",
            {
                "run_id": run_id,
                "task": task,
                "model_fingerprint": state.model_fingerprint,
                "config": config.to_dict(),
                "proposal_config": proposal_config.to_dict(),
                "search_config": search_config.to_dict(),
                "execution_config": self.execution_config.to_dict(),
                "root_fingerprint": stable_hash(normalized_files),
            },
        )

    async def _evaluate_pending(
        self,
        state: SearchSessionReport,
        checkpoint: Callable[[SearchSessionReport], None] | None = None,
    ) -> SearchSessionReport:
        if (
            state.pending_case is None
            or state.pending_model_turn is None
            or state.pending_model_name is None
            or state.pending_response_sha256 is None
            or state.pending_response_usage is None
            or self.execution_config is None
            or self.search_config is None
            or self.config is None
        ):
            raise ValueError("session has an incomplete evaluating checkpoint")
        execution_config = self.execution_config
        config = self.config
        results: dict[str, TestResult] = {}
        observations = dict(state.observations)
        reuses = 0
        candidates = tuple(state.pending_case.candidates)
        remaining_budget = max(0, config.max_test_calls - state.test_calls)
        working_state = state
        max_in_flight = working_state.max_in_flight
        actual_calls = 0

        async def evaluate_batch(batch: Sequence[CandidatePatch]) -> bool:
            nonlocal actual_calls, max_in_flight, working_state
            for candidate in batch:
                working_state = _append_event(
                    working_state,
                    "candidate.requested",
                    {
                        "candidate_id": candidate.id,
                        "workspace_fingerprint": candidate.workspace_fingerprint,
                        "max_parallel_tests": config.max_parallel_tests,
                        "scheduler_policy": config.scheduler_policy,
                    },
                )
            if checkpoint is not None:
                checkpoint(working_state)
            semaphore = asyncio.Semaphore(config.max_parallel_tests)
            in_flight = 0

            async def evaluate_one(candidate: CandidatePatch) -> TestResult:
                nonlocal in_flight, max_in_flight
                async with semaphore:
                    in_flight += 1
                    max_in_flight = max(max_in_flight, in_flight)
                    try:
                        return await asyncio.to_thread(
                            evaluate_candidate,
                            candidate,
                            execution_config,
                        )
                    finally:
                        in_flight -= 1

            tasks = [
                asyncio.create_task(evaluate_one(candidate)) for candidate in batch
            ]
            found_success = False
            try:
                for candidate, task in zip(batch, tasks, strict=True):
                    result = await task
                    fingerprint = candidate.workspace_fingerprint
                    observations[fingerprint] = result
                    results[candidate.id] = result
                    actual_calls += 1
                    found_success = found_success or result.is_success
                    working_state = _append_event(
                        replace(
                            working_state,
                            observations=observations,
                            test_calls=state.test_calls + actual_calls,
                            max_in_flight=max_in_flight,
                        ),
                        "candidate.completed",
                        {
                            "candidate_id": candidate.id,
                            "workspace_fingerprint": fingerprint,
                            "behavior_fingerprint": result.behavior_fingerprint,
                            "test_calls": state.test_calls + actual_calls,
                            "max_in_flight": max_in_flight,
                            "scheduler_policy": config.scheduler_policy,
                        },
                    )
                    if checkpoint is not None:
                        checkpoint(working_state)
            except BaseException:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            return found_success

        pending = [
            candidate
            for candidate in candidates
            if candidate.workspace_fingerprint not in observations
        ]
        while pending and actual_calls < remaining_budget:
            batch = select_candidate_batch(
                pending,
                observations,
                limit=(
                    remaining_budget - actual_calls
                    if config.scheduler_policy == "fixed"
                    else min(
                        config.max_parallel_tests,
                        remaining_budget - actual_calls,
                    )
                ),
                policy=config.scheduler_policy,
            )
            if not batch:
                break
            found_success = await evaluate_batch(batch)
            pending = [
                candidate
                for candidate in pending
                if candidate.workspace_fingerprint not in observations
            ]
            if config.scheduler_policy == "adaptive" and found_success:
                break

        for candidate in candidates:
            fingerprint = candidate.workspace_fingerprint
            if fingerprint in observations and candidate.id not in results:
                results[candidate.id] = observations[fingerprint]
                reuses += 1

        budget_result = TestResult(
            suite=execution_config.suite,
            error="session test budget exhausted before this candidate ran",
        )
        for candidate in candidates:
            if candidate.id in results:
                continue
            fingerprint = candidate.workspace_fingerprint
            if fingerprint in observations:
                results[candidate.id] = observations[fingerprint]
                reuses += 1
            else:
                results[candidate.id] = budget_result
        observed_case = BranchCase(
            task=state.pending_case.task,
            root_files=state.pending_case.root_files,
            candidates=state.pending_case.candidates,
            tests=results,
        )
        branch_report = BranchSearch(self.search_config).run(observed_case)
        session_round = SearchSessionRound(
            index=len(state.rounds),
            base_fingerprint=stable_hash(dict(state.pending_case.root_files)),
            report=branch_report,
            model_turn=state.pending_model_turn,
            model_name=state.pending_model_name,
            response_sha256=state.pending_response_sha256,
            response_id=state.pending_response_id,
            usage=state.pending_response_usage,
        )
        all_rounds = (*state.rounds, session_round)
        best = _best_node(branch_report)
        best_id = state.best_candidate_id
        best_files = state.best_files
        if best is not None:
            candidate = branch_report.case.by_id[best.id]
            if best_files is None or best.score >= max(
                (
                    node.score
                    for item in state.rounds
                    for node in item.report.nodes
                    if node.id != "root"
                ),
                default=-1.0,
            ):
                best_id = best.id
                best_files = dict(candidate.files)
        candidate_total = state.candidate_proposals + len(observed_case.candidates)
        test_total = working_state.test_calls
        feedback = tuple(
            [*state.feedback[-8:], *_feedback_from_report(branch_report)][-16:]
        )
        status: SessionStatus = "running"
        reason: str | None = None
        if branch_report.status == "accepted":
            status = "accepted"
            reason = "a visible-test-passing candidate was found"
            if branch_report.best_node_id is not None:
                best_id = branch_report.best_node_id
                best_files = dict(
                    branch_report.case.by_id[branch_report.best_node_id].files
                )
        elif (
            len(all_rounds) >= state.config.max_rounds
            or state.model_calls >= state.config.max_model_calls
            or candidate_total >= state.config.max_candidates
            or test_total >= state.config.max_test_calls
        ):
            provisional = replace(
                state,
                rounds=all_rounds,
                candidate_proposals=candidate_total,
                test_calls=test_total,
            )
            status = _terminal_status(
                provisional,
                rounds=len(all_rounds),
                candidates=candidate_total,
            )
            reason = "session budget exhausted before a passing candidate was found"
        updated = replace(
            working_state,
            status=status,
            phase="idle",
            rounds=all_rounds,
            observations=observations,
            test_calls=test_total,
            candidate_proposals=candidate_total,
            test_reuses=working_state.test_reuses + reuses,
            max_in_flight=max_in_flight,
            best_candidate_id=best_id,
            best_files=best_files,
            base_files=_next_base_files(branch_report, state.base_files),
            reason=reason,
            feedback=feedback,
            pending_case=None,
            pending_model_turn=None,
            pending_model_name=None,
            pending_response_sha256=None,
            pending_response_id=None,
            pending_response_usage=None,
        )
        updated = _append_event(
            updated,
            "round.completed",
            {
                "round": session_round.index,
                "case_fingerprint": observed_case.fingerprint,
                "report_status": branch_report.status,
                "best_node_id": branch_report.best_node_id,
                "actual_test_calls": actual_calls,
                "test_reuses": reuses,
                "max_in_flight": max_in_flight,
                "metrics": dict(branch_report.metrics),
            },
        )
        if status != "running":
            updated = _append_event(
                updated,
                "session.completed",
                {
                    "status": status,
                    "reason": reason,
                    "metrics": dict(updated.metrics),
                    "best_candidate_id": updated.best_candidate_id,
                },
            )
        return updated

    async def _drive(
        self,
        state: SearchSessionReport,
        checkpoint_path: Path | None,
        retry_pending: bool,
    ) -> SearchSessionReport:
        if (
            self.config is None
            or self.proposal_config is None
            or self.search_config is None
        ):
            raise ValueError("session configuration is incomplete")

        def checkpoint(current: SearchSessionReport) -> None:
            if checkpoint_path is not None:
                write_session_checkpoint(current, checkpoint_path)

        while state.status == "running":
            if state.phase == "evaluating":
                state = await self._evaluate_pending(state, checkpoint)
                checkpoint(state)
                continue
            if state.phase == "proposing" and not retry_pending:
                state = _append_event(
                    replace(
                        state,
                        status="paused",
                        phase="idle",
                        reason=(
                            "a model proposal was pending; resume with retry_pending"
                        ),
                    ),
                    "session.paused",
                    {"reason": "pending model proposal requires explicit retry"},
                )
                checkpoint(state)
                return state
            if len(state.rounds) >= self.config.max_rounds:
                state = _finish(
                    state,
                    "budget_exhausted",
                    "maximum proposal rounds reached",
                )
                checkpoint(state)
                return state
            if state.model_calls >= self.config.max_model_calls:
                state = _finish(
                    state, "budget_exhausted", "maximum model calls reached"
                )
                checkpoint(state)
                return state
            remaining_candidates = (
                self.config.max_candidates - state.candidate_proposals
            )
            if remaining_candidates <= 0:
                state = _finish(
                    state,
                    "budget_exhausted",
                    "maximum candidate proposals reached",
                )
                checkpoint(state)
                return state
            proposal_config = replace(
                self.proposal_config,
                max_candidates=min(
                    self.proposal_config.max_candidates,
                    remaining_candidates,
                ),
            )
            model_turn = state.model_calls + 1
            state = _append_event(
                replace(state, phase="proposing", reason=None),
                "proposal.requested",
                {
                    "round": len(state.rounds),
                    "model_turn": model_turn,
                    "base_fingerprint": stable_hash(state.base_files),
                    "proposal_config": proposal_config.to_dict(),
                    "feedback_count": len(state.feedback),
                },
            )
            checkpoint(state)
            try:
                case, response = await propose_case(
                    self.model,
                    state.task,
                    state.base_files,
                    proposal_config,
                    run_id=state.run_id,
                    turn=model_turn,
                    feedback=state.feedback,
                )
                namespaced = _namespace_case(case, len(state.rounds))
                response_sha256 = stable_hash(response.to_dict())
                state = replace(
                    state,
                    phase="evaluating",
                    model_calls=state.model_calls + 1,
                    usage=state.usage + response.usage,
                    pending_case=namespaced,
                    pending_model_turn=model_turn,
                    pending_model_name=self.model.name,
                    pending_response_sha256=response_sha256,
                    pending_response_id=response.response_id,
                    pending_response_usage=response.usage,
                )
                state = _append_event(
                    state,
                    "proposal.received",
                    {
                        "round": len(state.rounds),
                        "model_turn": model_turn,
                        "model_name": self.model.name,
                        "response_sha256": response_sha256,
                        "response_id": response.response_id,
                        "usage": response.usage.to_dict(),
                        "case_fingerprint": namespaced.fingerprint,
                        "candidate_count": len(namespaced.candidates),
                    },
                )
                checkpoint(state)
            except (ModelError, RuntimeContractError, ValueError) as exc:
                message = f"{type(exc).__name__}: {str(exc)[:800]}"
                state = replace(
                    state,
                    phase="idle",
                    model_calls=state.model_calls + 1,
                    feedback=tuple(
                        [
                            *state.feedback,
                            {
                                "proposal_rejected": message,
                                "round": len(state.rounds),
                            },
                        ][-16:]
                    ),
                )
                state = _append_event(
                    state,
                    "proposal.rejected",
                    {
                        "round": len(state.rounds),
                        "model_turn": model_turn,
                        "error": message,
                    },
                )
                if state.model_calls >= self.config.max_model_calls:
                    state = _finish(
                        state,
                        "failed",
                        "model proposal contract failed until the model-call "
                        "budget was exhausted",
                    )
                checkpoint(state)
        return state

    async def run(
        self,
        *,
        task: str | None = None,
        root_files: Mapping[str, str] | None = None,
        run_id: str = "search-session",
        checkpoint_path: str | Path | None = None,
        resume: bool = False,
        retry_pending: bool = False,
    ) -> SearchSessionReport:
        """Start or resume a session, checkpointing every model/evaluation boundary."""

        checkpoint = None if checkpoint_path is None else Path(checkpoint_path)
        if resume:
            if checkpoint is None:
                raise ValueError("checkpoint_path is required when resuming")
            state = read_session_checkpoint(checkpoint)
            self._adopt_checkpoint_config(state)
            self.model.resume_from_turn(state.model_calls)
            if state.status != "running":
                return state
        else:
            if task is None or root_files is None:
                raise ValueError("task and root_files are required for a new session")
            state = self._initial(task, root_files, run_id)
            if checkpoint is not None:
                write_session_checkpoint(state, checkpoint)
        return await self._drive(state, checkpoint, retry_pending)


async def run_search_session(
    model: ModelClient,
    *,
    task: str | None = None,
    root_files: Mapping[str, str] | None = None,
    execution_config: ExecutableSearchConfig | None = None,
    config: SearchSessionConfig | None = None,
    proposal_config: ProposalConfig | None = None,
    search_config: BranchSearchConfig | None = None,
    run_id: str = "search-session",
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
    retry_pending: bool = False,
) -> SearchSessionReport:
    """Convenience wrapper around :class:`SearchSessionRunner`."""

    runner = SearchSessionRunner(
        model,
        execution_config,
        config=config,
        proposal_config=proposal_config,
        search_config=search_config,
    )
    return await runner.run(
        task=task,
        root_files=root_files,
        run_id=run_id,
        checkpoint_path=checkpoint_path,
        resume=resume,
        retry_pending=retry_pending,
    )


def render_session_console(report: SearchSessionReport) -> str:
    """Render concise progress without hiding per-round branch reports."""

    lines = [
        f"status={report.status} phase={report.phase} rounds={len(report.rounds)} "
        f"model_calls={report.model_calls} test_calls={report.test_calls} "
        f"reuses={report.test_reuses} max_in_flight={report.max_in_flight}",
    ]
    if report.best_candidate_id:
        lines.append(f"best_candidate={report.best_candidate_id}")
    if report.reason:
        lines.append(f"reason={report.reason}")
    for item in report.rounds:
        lines.append(
            f"round {item.index}: {item.report.status} "
            f"best={item.report.best_node_id or '—'} "
            f"tests={item.report.metrics.get('test_calls', 0)} "
            f"duplicates={item.report.metrics.get('duplicates', 0)}"
        )
    return "\n".join(lines) + "\n"


def render_session_markdown(report: SearchSessionReport) -> str:
    lines = [
        "# ContextOpt iterative search session",
        "",
        f"- Status: `{report.status}`",
        f"- Task: `{report.task}`",
        f"- Rounds: `{len(report.rounds)}`",
        f"- Model calls: `{report.model_calls}`",
        f"- Actual test calls: `{report.test_calls}`",
        f"- Cached test reuses: `{report.test_reuses}`",
        f"- Scheduler: max configured parallel tests "
        f"`{report.config.max_parallel_tests}`; "
        f"observed max in-flight `{report.max_in_flight}`",
        f"- Best candidate: `{report.best_candidate_id or 'none'}`",
        "",
        "| Round | Search status | Best branch | Proposals | Test calls | Duplicates |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for item in report.rounds:
        metrics = item.report.metrics
        lines.append(
            f"| {item.index} | {item.report.status} | "
            f"`{item.report.best_node_id or 'none'}` | "
            f"{metrics.get('proposed', 0)} | {metrics.get('test_calls', 0)} | "
            f"{metrics.get('duplicates', 0)} |"
        )
    if report.reason:
        lines.extend(("", f"> {report.reason}"))
    return "\n".join(lines) + "\n"


__all__ = [
    "SearchSessionConfig",
    "SearchSessionReport",
    "SearchSessionRound",
    "SearchSessionRunner",
    "read_session_checkpoint",
    "render_session_console",
    "render_session_markdown",
    "run_search_session",
    "write_session_checkpoint",
]
