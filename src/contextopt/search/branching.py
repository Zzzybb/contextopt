"""Auditable test-guided branch search for coding-agent candidates.

The runtime in :mod:`contextopt.runtime` executes one durable agent trajectory.  This
module adds the complementary outer loop used when a coding model proposes several
alternative patches for the same failing task:

* candidates are immutable workspace snapshots, so a branch has a reproducible state;
* a fixed test environment turns identical snapshots into cacheable duplicate states;
* test outcomes drive beam ordering and stopping instead of model confidence alone;
* every proposal, evaluation, duplicate, prune, and terminal decision is hash chained.

The module deliberately stops at the adapter boundary.  A model or workspace runner
can create ``BranchCase`` from its generated patches and test observations; this core
does not execute arbitrary code or silently claim that a synthetic test result proves a
real repair.  That boundary keeps the offline demo deterministic and makes the search
algorithm independently testable.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from html import escape
from math import isfinite
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Literal, cast

from contextopt.runtime.identity import stable_hash

CandidateStatus = Literal[
    "root", "frontier", "accepted", "duplicate", "pruned", "failed"
]
SearchStatus = Literal["accepted", "exhausted", "budget_exhausted"]
SearchPolicy = Literal["beam", "mcts"]
SEARCH_POLICIES: tuple[SearchPolicy, ...] = ("beam", "mcts")


def _non_empty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _string_tuple(value: Iterable[Any], label: str) -> tuple[str, ...]:
    result = tuple(_non_empty(item, label) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must not contain duplicates")
    return tuple(sorted(result))


def _normal_path(raw: Any, label: str) -> str:
    value = _non_empty(raw, label)
    if "\\" in value or value.startswith("/"):
        raise ValueError(f"{label} must be a relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} must not contain traversal or empty components")
    return path.as_posix()


def _normalize_files(value: Mapping[Any, Any], label: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object mapping paths to text")
    normalized: dict[str, str] = {}
    for raw_path, raw_content in value.items():
        path = _normal_path(raw_path, f"{label} path")
        if not isinstance(raw_content, str):
            raise ValueError(f"{label}[{path!r}] must be a string")
        if path in normalized:
            raise ValueError(f"{label} contains duplicate path {path!r}")
        normalized[path] = raw_content
    return MappingProxyType(dict(sorted(normalized.items())))


@dataclass(frozen=True, slots=True)
class TestResult:
    """One deterministic visible-test observation for a candidate branch."""

    suite: str
    passed_tests: tuple[str, ...] = ()
    failed_tests: tuple[str, ...] = ()
    error: str | None = None
    duration_ms: float = 0.0
    output_sha256: str | None = None
    output_excerpt: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "suite", _non_empty(self.suite, "suite"))
        passed = _string_tuple(self.passed_tests, "passed_tests")
        failed = _string_tuple(self.failed_tests, "failed_tests")
        if set(passed) & set(failed):
            raise ValueError("passed_tests and failed_tests must be disjoint")
        if self.error is not None:
            object.__setattr__(self, "error", _non_empty(self.error, "error"))
        if not isfinite(self.duration_ms) or self.duration_ms < 0:
            raise ValueError("duration_ms must be finite and non-negative")
        if self.output_sha256 is not None:
            digest = _non_empty(self.output_sha256, "output_sha256")
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError("output_sha256 must be a lowercase SHA-256 hex digest")
            object.__setattr__(self, "output_sha256", digest)
        if self.output_excerpt is not None:
            if not isinstance(self.output_excerpt, str):
                raise ValueError("output_excerpt must be a string or null")
            if len(self.output_excerpt) > 4096:
                raise ValueError("output_excerpt must be at most 4096 characters")
        if not passed and not failed and self.error is None:
            raise ValueError("a test result must contain tests or an error")
        object.__setattr__(self, "passed_tests", passed)
        object.__setattr__(self, "failed_tests", failed)

    @property
    def total_tests(self) -> int:
        return (
            len(self.passed_tests)
            + len(self.failed_tests)
            + (1 if self.error is not None else 0)
        )

    @property
    def passed_ratio(self) -> float:
        return len(self.passed_tests) / self.total_tests

    @property
    def is_success(self) -> bool:
        return bool(self.passed_tests) and not self.failed_tests and self.error is None

    @property
    def quality_score(self) -> float:
        """A bounded, transparent score used only to order search candidates."""

        denominator = self.total_tests
        score = len(self.passed_tests) / denominator
        score -= 0.20 * len(self.failed_tests) / denominator
        if self.error is not None:
            score -= 0.50
        return max(0.0, min(1.0, score))

    @property
    def behavior_fingerprint(self) -> str:
        return stable_hash(
            {
                "suite": self.suite,
                "passed_tests": list(self.passed_tests),
                "failed_tests": list(self.failed_tests),
                "error": self.error,
                "output_sha256": self.output_sha256,
            }
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TestResult:
        if not isinstance(data, Mapping):
            raise ValueError("test result must be an object")
        allowed = {
            "suite",
            "passed_tests",
            "failed_tests",
            "error",
            "duration_ms",
            "output_sha256",
            "output_excerpt",
            "total_tests",
            "passed_ratio",
            "quality_score",
            "behavior_fingerprint",
            "is_success",
        }
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"test result has unknown fields: {sorted(unknown)!r}")
        passed = data.get("passed_tests", [])
        failed = data.get("failed_tests", [])
        if not isinstance(passed, list) or not isinstance(failed, list):
            raise ValueError("passed_tests and failed_tests must be arrays")
        error = data.get("error")
        if error is not None and not isinstance(error, str):
            raise ValueError("test result error must be a string or null")
        duration = data.get("duration_ms", 0.0)
        if not isinstance(duration, (int, float)) or isinstance(duration, bool):
            raise ValueError("duration_ms must be a number")
        output_sha256 = data.get("output_sha256")
        if output_sha256 is not None and not isinstance(output_sha256, str):
            raise ValueError("output_sha256 must be a string or null")
        output_excerpt = data.get("output_excerpt")
        if output_excerpt is not None and not isinstance(output_excerpt, str):
            raise ValueError("output_excerpt must be a string or null")
        result = cls(
            suite=_non_empty(data.get("suite"), "suite"),
            passed_tests=tuple(passed),
            failed_tests=tuple(failed),
            error=error,
            duration_ms=float(duration),
            output_sha256=output_sha256,
            output_excerpt=output_excerpt,
        )
        computed: dict[str, Any] = {
            "total_tests": result.total_tests,
            "passed_ratio": result.passed_ratio,
            "quality_score": result.quality_score,
            "behavior_fingerprint": result.behavior_fingerprint,
            "is_success": result.is_success,
        }
        for key, expected in computed.items():
            if key not in data:
                continue
            actual = data[key]
            if key in {"passed_ratio", "quality_score"}:
                if not isinstance(actual, (int, float)) or isinstance(actual, bool):
                    raise ValueError(f"test result {key} must be a number")
                if float(actual) != expected:
                    raise ValueError(f"test result {key} is inconsistent")
            elif actual != expected:
                raise ValueError(f"test result {key} is inconsistent")
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "passed_tests": list(self.passed_tests),
            "failed_tests": list(self.failed_tests),
            "error": self.error,
            "duration_ms": self.duration_ms,
            "output_sha256": self.output_sha256,
            "output_excerpt": self.output_excerpt,
            "total_tests": self.total_tests,
            "passed_ratio": self.passed_ratio,
            "quality_score": self.quality_score,
            "behavior_fingerprint": self.behavior_fingerprint,
            "is_success": self.is_success,
        }


@dataclass(frozen=True, slots=True)
class CandidatePatch:
    """A model-generated candidate represented by a complete workspace snapshot."""

    id: str
    parent_id: str
    hypothesis: str
    files: Mapping[str, str]
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _non_empty(self.id, "candidate id"))
        object.__setattr__(
            self, "parent_id", _non_empty(self.parent_id, "candidate parent_id")
        )
        if self.id == "root":
            raise ValueError("candidate id 'root' is reserved for the initial state")
        if self.id == self.parent_id:
            raise ValueError("candidate cannot be its own parent")
        object.__setattr__(
            self, "hypothesis", _non_empty(self.hypothesis, "hypothesis")
        )
        object.__setattr__(self, "files", _normalize_files(self.files, "files"))
        object.__setattr__(self, "evidence", _string_tuple(self.evidence, "evidence"))

    @property
    def workspace_fingerprint(self) -> str:
        return stable_hash({"files": [[path, self.files[path]] for path in self.files]})

    @property
    def patch_fingerprint(self) -> str:
        return stable_hash(
            {
                "parent_id": self.parent_id,
                "hypothesis": self.hypothesis,
                "files": [[path, self.files[path]] for path in self.files],
            }
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CandidatePatch:
        if not isinstance(data, Mapping):
            raise ValueError("candidate must be an object")
        allowed = {
            "id",
            "parent_id",
            "hypothesis",
            "files",
            "evidence",
            "workspace_fingerprint",
            "patch_fingerprint",
        }
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"candidate has unknown fields: {sorted(unknown)!r}")
        evidence = data.get("evidence", [])
        if not isinstance(evidence, list):
            raise ValueError("candidate evidence must be an array")
        files = data.get("files")
        if not isinstance(files, Mapping):
            raise ValueError("candidate files must be an object")
        candidate = cls(
            id=_non_empty(data.get("id"), "candidate id"),
            parent_id=_non_empty(data.get("parent_id"), "candidate parent_id"),
            hypothesis=_non_empty(data.get("hypothesis"), "hypothesis"),
            files=files,
            evidence=tuple(evidence),
        )
        for key, expected in {
            "workspace_fingerprint": candidate.workspace_fingerprint,
            "patch_fingerprint": candidate.patch_fingerprint,
        }.items():
            if key in data and data[key] != expected:
                raise ValueError(f"candidate {key} is inconsistent")
        return candidate

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "hypothesis": self.hypothesis,
            "files": dict(self.files),
            "evidence": list(self.evidence),
            "workspace_fingerprint": self.workspace_fingerprint,
            "patch_fingerprint": self.patch_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class BranchCase:
    """A fixed set of candidate snapshots and their observed test outcomes."""

    task: str
    root_files: Mapping[str, str]
    candidates: tuple[CandidatePatch, ...]
    tests: Mapping[str, TestResult]

    def __post_init__(self) -> None:
        object.__setattr__(self, "task", _non_empty(self.task, "task"))
        object.__setattr__(
            self, "root_files", _normalize_files(self.root_files, "root_files")
        )
        candidate_ids = [candidate.id for candidate in self.candidates]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate ids must be unique")
        if "root" in candidate_ids:
            raise ValueError("candidate id 'root' is reserved")
        by_id = set(candidate_ids)
        for candidate in self.candidates:
            if candidate.parent_id != "root" and candidate.parent_id not in by_id:
                raise ValueError(
                    f"candidate {candidate.id!r} references unknown parent "
                    f"{candidate.parent_id!r}"
                )
        self._assert_acyclic()
        tests = dict(self.tests)
        if set(tests) != by_id:
            missing = sorted(by_id - set(tests))
            extra = sorted(set(tests) - by_id)
            raise ValueError(
                f"tests must match candidates; missing={missing}, extra={extra}"
            )
        object.__setattr__(
            self, "candidates", tuple(sorted(self.candidates, key=lambda x: x.id))
        )
        object.__setattr__(self, "tests", MappingProxyType(dict(sorted(tests.items()))))

    def _assert_acyclic(self) -> None:
        parents = {candidate.id: candidate.parent_id for candidate in self.candidates}
        for candidate_id in parents:
            seen: set[str] = set()
            current = candidate_id
            while current != "root":
                if current in seen:
                    raise ValueError("candidate parent graph must be acyclic")
                seen.add(current)
                current = parents[current]

    @property
    def by_id(self) -> Mapping[str, CandidatePatch]:
        return MappingProxyType(
            {candidate.id: candidate for candidate in self.candidates}
        )

    @property
    def root_fingerprint(self) -> str:
        return stable_hash(
            {"files": [[path, self.root_files[path]] for path in self.root_files]}
        )

    @property
    def fingerprint(self) -> str:
        return stable_hash(
            {
                "task": self.task,
                "root_files": dict(self.root_files),
                "candidates": [candidate.to_dict() for candidate in self.candidates],
                "tests": {key: self.tests[key].to_dict() for key in sorted(self.tests)},
            }
        )

    def children(self, parent_id: str) -> tuple[CandidatePatch, ...]:
        return tuple(
            candidate
            for candidate in self.candidates
            if candidate.parent_id == parent_id
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BranchCase:
        if not isinstance(data, Mapping):
            raise ValueError("branch case must be an object")
        allowed = {"task", "root_files", "candidates", "tests", "fingerprint"}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"branch case has unknown fields: {sorted(unknown)!r}")
        raw_candidates = data.get("candidates")
        raw_tests = data.get("tests")
        if not isinstance(raw_candidates, list) or not isinstance(raw_tests, Mapping):
            raise ValueError(
                "branch case candidates must be an array and tests an object"
            )
        root_files = data.get("root_files")
        if not isinstance(root_files, Mapping):
            raise ValueError("branch case root_files must be an object")
        candidates = tuple(
            CandidatePatch.from_dict(item)
            for item in raw_candidates
            if isinstance(item, Mapping)
        )
        if len(candidates) != len(raw_candidates):
            raise ValueError("every candidate must be an object")
        tests = {
            _non_empty(candidate_id, "test candidate id"): TestResult.from_dict(value)
            for candidate_id, value in raw_tests.items()
        }
        case = cls(
            task=_non_empty(data.get("task"), "task"),
            root_files=root_files,
            candidates=candidates,
            tests=tests,
        )
        if "fingerprint" in data and data["fingerprint"] != case.fingerprint:
            raise ValueError("branch case fingerprint is inconsistent")
        return case

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "root_files": dict(self.root_files),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "tests": {key: self.tests[key].to_dict() for key in sorted(self.tests)},
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True, slots=True)
class BranchSearchConfig:
    """Limits and deterministic tie-breaking rules for branch search.

    ``beam`` is the historical breadth-first baseline. ``mcts`` uses observed
    test quality as a bounded UCT signal while traversing the same immutable
    candidate tree; it never invents an unobserved reward.
    """

    beam_width: int = 2
    max_depth: int = 4
    max_candidates: int = 32
    test_environment_fingerprint: str = "visible-tests-v1"
    stop_on_pass: bool = True
    search_policy: SearchPolicy = "beam"
    exploration_constant: float = 1.0

    def __post_init__(self) -> None:
        for name in ("beam_width", "max_depth", "max_candidates"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        object.__setattr__(
            self,
            "test_environment_fingerprint",
            _non_empty(
                self.test_environment_fingerprint, "test_environment_fingerprint"
            ),
        )
        if not isinstance(self.stop_on_pass, bool):
            raise ValueError("stop_on_pass must be a boolean")
        if not isinstance(self.search_policy, str) or (
            self.search_policy not in SEARCH_POLICIES
        ):
            raise ValueError(f"unsupported search policy: {self.search_policy!r}")
        if not isfinite(self.exploration_constant) or self.exploration_constant <= 0:
            raise ValueError("exploration_constant must be finite and positive")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BranchSearchConfig:
        if not isinstance(data, Mapping):
            raise ValueError("search config must be an object")
        allowed = {
            "beam_width",
            "max_depth",
            "max_candidates",
            "test_environment_fingerprint",
            "stop_on_pass",
            "search_policy",
            "exploration_constant",
        }
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"search config has unknown fields: {sorted(unknown)!r}")
        return cls(**dict(data))

    def to_dict(self) -> dict[str, Any]:
        return {
            "beam_width": self.beam_width,
            "max_depth": self.max_depth,
            "max_candidates": self.max_candidates,
            "test_environment_fingerprint": self.test_environment_fingerprint,
            "stop_on_pass": self.stop_on_pass,
            "search_policy": self.search_policy,
            "exploration_constant": self.exploration_constant,
        }


@dataclass(frozen=True, slots=True)
class BranchNode:
    """Auditable state and decision attached to one candidate id."""

    id: str
    parent_id: str | None
    depth: int
    status: CandidateStatus
    hypothesis: str
    workspace_fingerprint: str
    dedup_key: str
    score: float
    test_result: TestResult | None = None
    duplicate_of: str | None = None
    reason: str | None = None
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _non_empty(self.id, "node id"))
        if self.parent_id is not None:
            object.__setattr__(
                self, "parent_id", _non_empty(self.parent_id, "parent_id")
            )
        if self.depth < 0:
            raise ValueError("depth must be non-negative")
        object.__setattr__(
            self, "hypothesis", _non_empty(self.hypothesis, "hypothesis")
        )
        object.__setattr__(
            self,
            "workspace_fingerprint",
            _non_empty(self.workspace_fingerprint, "workspace_fingerprint"),
        )
        object.__setattr__(self, "dedup_key", _non_empty(self.dedup_key, "dedup_key"))
        if not isfinite(self.score):
            raise ValueError("score must be finite")
        object.__setattr__(self, "evidence", _string_tuple(self.evidence, "evidence"))
        if self.status not in {
            "root",
            "frontier",
            "accepted",
            "duplicate",
            "pruned",
            "failed",
        }:
            raise ValueError(f"unsupported node status: {self.status!r}")
        if self.status == "duplicate" and not self.duplicate_of:
            raise ValueError("duplicate nodes must identify duplicate_of")
        if self.status == "accepted" and (
            self.test_result is None or not self.test_result.is_success
        ):
            raise ValueError("accepted nodes require a successful test result")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BranchNode:
        if not isinstance(data, Mapping):
            raise ValueError("branch node must be an object")
        allowed = {
            "id",
            "parent_id",
            "depth",
            "status",
            "hypothesis",
            "workspace_fingerprint",
            "dedup_key",
            "score",
            "test_result",
            "duplicate_of",
            "reason",
            "evidence",
        }
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"branch node has unknown fields: {sorted(unknown)!r}")
        evidence = data.get("evidence", [])
        if not isinstance(evidence, list):
            raise ValueError("node evidence must be an array")
        result = data.get("test_result")
        if result is not None and not isinstance(result, Mapping):
            raise ValueError("node test_result must be an object or null")
        raw_status = data.get("status")
        if raw_status not in {
            "root",
            "frontier",
            "accepted",
            "duplicate",
            "pruned",
            "failed",
        }:
            raise ValueError(f"unsupported node status: {raw_status!r}")
        return cls(
            id=_non_empty(data.get("id"), "node id"),
            parent_id=(
                None if data.get("parent_id") is None else str(data["parent_id"])
            ),
            depth=int(data.get("depth", -1)),
            status=cast(CandidateStatus, raw_status),
            hypothesis=_non_empty(data.get("hypothesis"), "hypothesis"),
            workspace_fingerprint=_non_empty(
                data.get("workspace_fingerprint"), "workspace_fingerprint"
            ),
            dedup_key=_non_empty(data.get("dedup_key"), "dedup_key"),
            score=float(data.get("score", 0.0)),
            test_result=(None if result is None else TestResult.from_dict(result)),
            duplicate_of=(
                None if data.get("duplicate_of") is None else str(data["duplicate_of"])
            ),
            reason=(None if data.get("reason") is None else str(data["reason"])),
            evidence=tuple(evidence),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "status": self.status,
            "hypothesis": self.hypothesis,
            "workspace_fingerprint": self.workspace_fingerprint,
            "dedup_key": self.dedup_key,
            "score": self.score,
            "test_result": None
            if self.test_result is None
            else self.test_result.to_dict(),
            "duplicate_of": self.duplicate_of,
            "reason": self.reason,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True, slots=True)
class SearchEvent:
    """One hash-chained branch-search event."""

    seq: int
    type: str
    data: Mapping[str, Any]
    prev_sha256: str | None
    sha256: str

    @classmethod
    def create(
        cls, seq: int, event_type: str, data: Mapping[str, Any], previous: str | None
    ) -> SearchEvent:
        payload = {
            "seq": seq,
            "type": event_type,
            "data": dict(data),
            "prev_sha256": previous,
        }
        return cls(
            seq=seq,
            type=_non_empty(event_type, "event type"),
            data=dict(data),
            prev_sha256=previous,
            sha256=stable_hash(payload),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SearchEvent:
        if not isinstance(data, Mapping):
            raise ValueError("search event must be an object")
        allowed = {"seq", "type", "data", "prev_sha256", "sha256"}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"search event has unknown fields: {sorted(unknown)!r}")
        event_data = data.get("data")
        if not isinstance(event_data, Mapping):
            raise ValueError("search event data must be an object")
        previous = data.get("prev_sha256")
        if previous is not None and not isinstance(previous, str):
            raise ValueError("prev_sha256 must be a string or null")
        return cls(
            seq=int(data.get("seq", -1)),
            type=_non_empty(data.get("type"), "event type"),
            data=dict(event_data),
            prev_sha256=previous,
            sha256=_non_empty(data.get("sha256"), "event sha256"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "type": self.type,
            "data": dict(self.data),
            "prev_sha256": self.prev_sha256,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class BranchSearchReport:
    """Complete deterministic result suitable for JSON storage and visualization."""

    task: str
    case_fingerprint: str
    config: BranchSearchConfig
    status: SearchStatus
    best_node_id: str | None
    nodes: tuple[BranchNode, ...]
    events: tuple[SearchEvent, ...]
    metrics: Mapping[str, int | float]
    case: BranchCase
    schema_version: str = "1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "task", _non_empty(self.task, "task"))
        if self.schema_version != "1":
            raise ValueError(
                f"unsupported search report schema: {self.schema_version!r}"
            )
        if self.task != self.case.task:
            raise ValueError("report task must match case task")
        if self.case_fingerprint != self.case.fingerprint:
            raise ValueError("report case fingerprint does not match case")
        if self.status not in {"accepted", "exhausted", "budget_exhausted"}:
            raise ValueError(f"unsupported search status: {self.status!r}")
        if len({node.id for node in self.nodes}) != len(self.nodes):
            raise ValueError("report node ids must be unique")
        if self.best_node_id is not None and self.best_node_id not in {
            node.id for node in self.nodes
        }:
            raise ValueError("best_node_id must reference a report node")
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BranchSearchReport:
        if not isinstance(data, Mapping):
            raise ValueError("search report must be an object")
        allowed = {
            "schema_version",
            "task",
            "case_fingerprint",
            "config",
            "status",
            "best_node_id",
            "nodes",
            "events",
            "metrics",
            "case",
        }
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"search report has unknown fields: {sorted(unknown)!r}")
        nodes = data.get("nodes")
        events = data.get("events")
        metrics = data.get("metrics")
        case_data = data.get("case")
        config_data = data.get("config")
        if not isinstance(nodes, list) or not isinstance(events, list):
            raise ValueError("report nodes and events must be arrays")
        if not isinstance(metrics, Mapping) or not isinstance(case_data, Mapping):
            raise ValueError("report metrics and case must be objects")
        if not isinstance(config_data, Mapping):
            raise ValueError("report config must be an object")
        for key, value in metrics.items():
            if not isinstance(key, str):
                raise ValueError("report metric names must be strings")
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"report metric {key!r} must be numeric")
        raw_status = data.get("status")
        if raw_status not in {"accepted", "exhausted", "budget_exhausted"}:
            raise ValueError(f"unsupported search status: {raw_status!r}")
        report = cls(
            schema_version=str(data.get("schema_version", "")),
            task=_non_empty(data.get("task"), "task"),
            case_fingerprint=_non_empty(
                data.get("case_fingerprint"), "case_fingerprint"
            ),
            config=BranchSearchConfig.from_dict(config_data),
            status=cast(SearchStatus, raw_status),
            best_node_id=(
                None if data.get("best_node_id") is None else str(data["best_node_id"])
            ),
            nodes=tuple(BranchNode.from_dict(item) for item in nodes),
            events=tuple(SearchEvent.from_dict(item) for item in events),
            metrics={str(key): value for key, value in metrics.items()},
            case=BranchCase.from_dict(case_data),
        )
        validate_search_report(report)
        return report

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task": self.task,
            "case_fingerprint": self.case_fingerprint,
            "config": self.config.to_dict(),
            "status": self.status,
            "best_node_id": self.best_node_id,
            "nodes": [node.to_dict() for node in self.nodes],
            "events": [event.to_dict() for event in self.events],
            "metrics": dict(self.metrics),
            "case": self.case.to_dict(),
        }


class _EventBuilder:
    def __init__(self) -> None:
        self.events: list[SearchEvent] = []

    def emit(self, event_type: str, data: Mapping[str, Any]) -> SearchEvent:
        previous = self.events[-1].sha256 if self.events else None
        event = SearchEvent.create(len(self.events), event_type, data, previous)
        self.events.append(event)
        return event


class BranchSearch:
    """Deterministic beam search with fixed-test deduplication and audit events."""

    def __init__(self, config: BranchSearchConfig | None = None) -> None:
        self.config = config or BranchSearchConfig()

    def _dedup_key(self, workspace_fingerprint: str) -> str:
        return stable_hash(
            {
                "workspace_fingerprint": workspace_fingerprint,
                "test_environment_fingerprint": (
                    self.config.test_environment_fingerprint
                ),
                "dedup_abi": "workspace-v1",
            }
        )

    @staticmethod
    def _score(result: TestResult, depth: int) -> float:
        # A tiny depth penalty makes shorter repairs win exact ties, while the test
        # result remains the dominant signal.  This is an ordering heuristic, not a
        # claim about semantic correctness.
        return max(0.0, result.quality_score - 0.01 * depth)

    @staticmethod
    def _priority(node: BranchNode) -> tuple[float, float, int, str]:
        quality = 0.0 if node.test_result is None else node.test_result.quality_score
        return (-node.score, -quality, node.depth, node.id)

    def run(self, case: BranchCase) -> BranchSearchReport:
        if self.config.search_policy == "mcts":
            from contextopt.search.mcts import run_mcts_search

            return run_mcts_search(case, self.config)
        builder = _EventBuilder()
        root_dedup = self._dedup_key(case.root_fingerprint)
        root = BranchNode(
            id="root",
            parent_id=None,
            depth=0,
            status="root",
            hypothesis="initial workspace state",
            workspace_fingerprint=case.root_fingerprint,
            dedup_key=root_dedup,
            score=0.0,
            reason="search root",
        )
        nodes: dict[str, BranchNode] = {root.id: root}
        seen: dict[str, str] = {root_dedup: root.id}
        frontier: list[BranchNode] = [root]
        proposed = evaluated = duplicates = pruned = passed = 0
        test_calls = 0
        max_depth = 0
        builder.emit(
            "search.started",
            {
                "task": case.task,
                "case_fingerprint": case.fingerprint,
                "config": self.config.to_dict(),
            },
        )

        accepted: list[BranchNode] = []
        budget_exhausted = False
        for depth in range(1, self.config.max_depth + 1):
            if not frontier:
                break
            next_frontier: list[BranchNode] = []
            max_depth = depth
            for parent in sorted(frontier, key=self._priority):
                for candidate in case.children(parent.id):
                    if proposed >= self.config.max_candidates:
                        budget_exhausted = True
                        break
                    proposed += 1
                    workspace = candidate.workspace_fingerprint
                    dedup_key = self._dedup_key(workspace)
                    builder.emit(
                        "candidate.proposed",
                        {
                            "candidate_id": candidate.id,
                            "parent_id": candidate.parent_id,
                            "depth": depth,
                            "patch_fingerprint": candidate.patch_fingerprint,
                            "workspace_fingerprint": workspace,
                            "dedup_key": dedup_key,
                        },
                    )
                    if dedup_key in seen:
                        duplicate_of = seen[dedup_key]
                        duplicate = BranchNode(
                            id=candidate.id,
                            parent_id=candidate.parent_id,
                            depth=depth,
                            status="duplicate",
                            hypothesis=candidate.hypothesis,
                            workspace_fingerprint=workspace,
                            dedup_key=dedup_key,
                            score=0.0,
                            duplicate_of=duplicate_of,
                            reason=(
                                "same workspace state under the fixed test environment"
                            ),
                            evidence=candidate.evidence,
                        )
                        nodes[candidate.id] = duplicate
                        duplicates += 1
                        builder.emit(
                            "candidate.duplicate",
                            {
                                "candidate_id": candidate.id,
                                "duplicate_of": duplicate_of,
                                "dedup_key": dedup_key,
                            },
                        )
                        continue

                    seen[dedup_key] = candidate.id
                    result = case.tests[candidate.id]
                    test_calls += 1
                    evaluated += 1
                    score = self._score(result, depth)
                    candidate_status: CandidateStatus = (
                        "accepted" if result.is_success else "frontier"
                    )
                    node = BranchNode(
                        id=candidate.id,
                        parent_id=candidate.parent_id,
                        depth=depth,
                        status=candidate_status,
                        hypothesis=candidate.hypothesis,
                        workspace_fingerprint=workspace,
                        dedup_key=dedup_key,
                        score=score,
                        test_result=result,
                        reason=(
                            "all visible tests passed"
                            if result.is_success
                            else "tests remain failing"
                        ),
                        evidence=candidate.evidence,
                    )
                    nodes[candidate.id] = node
                    builder.emit(
                        "candidate.evaluated",
                        {
                            "candidate_id": candidate.id,
                            "test_result": result.to_dict(),
                            "test_fingerprint": result.behavior_fingerprint,
                            "score": score,
                        },
                    )
                    if result.is_success:
                        accepted.append(node)
                        passed += 1
                    else:
                        next_frontier.append(node)
                if budget_exhausted:
                    break

            if accepted and self.config.stop_on_pass:
                break
            if budget_exhausted:
                break
            next_frontier.sort(key=self._priority)
            kept = next_frontier[: self.config.beam_width]
            for discarded in next_frontier[self.config.beam_width :]:
                pruned_node = BranchNode(
                    id=discarded.id,
                    parent_id=discarded.parent_id,
                    depth=discarded.depth,
                    status="pruned",
                    hypothesis=discarded.hypothesis,
                    workspace_fingerprint=discarded.workspace_fingerprint,
                    dedup_key=discarded.dedup_key,
                    score=discarded.score,
                    test_result=discarded.test_result,
                    reason=f"beam width {self.config.beam_width} exceeded",
                    evidence=discarded.evidence,
                )
                nodes[discarded.id] = pruned_node
                pruned += 1
                builder.emit(
                    "candidate.pruned",
                    {
                        "candidate_id": discarded.id,
                        "reason": pruned_node.reason,
                        "score": discarded.score,
                    },
                )
            frontier = kept

        if accepted:
            best = sorted(accepted, key=self._priority)[0]
            status: SearchStatus = "accepted"
            best_node_id: str | None = best.id
        elif budget_exhausted:
            status = "budget_exhausted"
            best_node_id = None
        else:
            status = "exhausted"
            best_node_id = None
            for remaining in frontier:
                if (
                    remaining.status == "frontier"
                    and remaining.depth >= self.config.max_depth
                ):
                    nodes[remaining.id] = BranchNode(
                        id=remaining.id,
                        parent_id=remaining.parent_id,
                        depth=remaining.depth,
                        status="pruned",
                        hypothesis=remaining.hypothesis,
                        workspace_fingerprint=remaining.workspace_fingerprint,
                        dedup_key=remaining.dedup_key,
                        score=remaining.score,
                        test_result=remaining.test_result,
                        reason="maximum search depth reached",
                        evidence=remaining.evidence,
                    )
                    pruned += 1

        builder.emit(
            "search.completed",
            {
                "status": status,
                "best_node_id": best_node_id,
                "proposed": proposed,
                "evaluated": evaluated,
                "test_calls": test_calls,
                "duplicates": duplicates,
                "pruned": pruned,
                "passed": passed,
                "max_depth": max_depth,
            },
        )
        report = BranchSearchReport(
            task=case.task,
            case_fingerprint=case.fingerprint,
            config=self.config,
            status=status,
            best_node_id=best_node_id,
            nodes=tuple(nodes[node_id] for node_id in sorted(nodes)),
            events=tuple(builder.events),
            metrics={
                "proposed": proposed,
                "evaluated": evaluated,
                "test_calls": test_calls,
                "duplicates": duplicates,
                "pruned": pruned,
                "passed": passed,
                "max_depth": max_depth,
            },
            case=case,
        )
        validate_search_report(report)
        return report


def verify_search_events(events: Iterable[SearchEvent]) -> None:
    """Validate sequence numbers, previous links, and event hashes."""

    previous: str | None = None
    for expected_seq, event in enumerate(events):
        if event.seq != expected_seq:
            raise ValueError(f"search event sequence mismatch at {expected_seq}")
        if event.prev_sha256 != previous:
            raise ValueError(f"search event {event.seq} has an invalid previous hash")
        expected_hash = stable_hash(
            {
                "seq": event.seq,
                "type": event.type,
                "data": dict(event.data),
                "prev_sha256": event.prev_sha256,
            }
        )
        if event.sha256 != expected_hash:
            raise ValueError(f"search event {event.seq} hash mismatch")
        previous = event.sha256


def validate_search_report(report: BranchSearchReport) -> None:
    """Check report-level invariants before a report is persisted or rendered."""

    verify_search_events(report.events)
    by_id = {node.id: node for node in report.nodes}
    root = by_id.get("root")
    if root is None or root.status != "root" or root.parent_id is not None:
        raise ValueError("search report must contain a parentless root node")
    for node in report.nodes:
        if node.id == "root":
            continue
        if node.parent_id not in by_id:
            raise ValueError(f"node {node.id!r} references an absent parent")
        if node.status == "duplicate" and node.duplicate_of not in by_id:
            raise ValueError(
                f"duplicate node {node.id!r} references an absent original"
            )
    completed = [event for event in report.events if event.type == "search.completed"]
    if len(completed) != 1:
        raise ValueError(
            "search report must contain exactly one search.completed event"
        )
    payload = completed[0].data
    if (
        payload.get("status") != report.status
        or payload.get("best_node_id") != report.best_node_id
    ):
        raise ValueError("search.completed does not match report terminal state")


def _short(value: str, length: int = 10) -> str:
    return value[:length]


def render_branch_console(report: BranchSearchReport) -> str:
    headers = ("branch", "parent", "depth", "status", "score", "tests", "reason")
    rows: list[tuple[str, ...]] = [headers]
    for node in report.nodes:
        if node.test_result is None:
            tests = "—"
        else:
            tests = (
                f"{len(node.test_result.passed_tests)}/{node.test_result.total_tests}"
            )
        rows.append(
            (
                node.id,
                node.parent_id or "—",
                str(node.depth),
                node.status,
                f"{node.score:.3f}",
                tests,
                node.reason or "",
            )
        )
    widths = [max(len(row[index]) for row in rows) for index in range(len(headers))]
    rendered: list[str] = []
    for index, row in enumerate(rows):
        rendered.append(
            "  ".join(value.ljust(widths[col]) for col, value in enumerate(row))
        )
        if index == 0:
            rendered.append("  ".join("-" * width for width in widths))
    rendered.append("")
    rendered.append(
        f"status={report.status} best={report.best_node_id or '—'} "
        f"policy={report.config.search_policy} "
        f"proposed={report.metrics['proposed']} "
        f"evaluated={report.metrics['evaluated']} "
        f"test_calls={report.metrics['test_calls']} "
        f"duplicates={report.metrics['duplicates']} "
        f"pruned={report.metrics['pruned']}"
    )
    return "\n".join(rendered)


def render_branch_markdown(report: BranchSearchReport) -> str:
    lines = [
        "# ContextOpt test-guided branch search",
        "",
        f"Task: **{report.task}**",
        "",
        (
            f"Status: **{report.status}**; best candidate: "
            f"**{report.best_node_id or 'none'}**. "
            f"Selection policy: **{report.config.search_policy}**. "
            "The fixed test environment is "
            f"`{report.config.test_environment_fingerprint}`."
        ),
        "",
        "| Branch | Parent | Depth | Status | Score | Tests | Duplicate/prune reason |",
        "|---|---|---:|---|---:|---:|---|",
    ]
    for node in report.nodes:
        tests = "—"
        if node.test_result is not None:
            tests = (
                f"{len(node.test_result.passed_tests)}/{node.test_result.total_tests}"
            )
        reason = node.reason or ""
        if node.duplicate_of:
            reason = f"duplicate of `{node.duplicate_of}`"
        lines.append(
            f"| `{node.id}` | `{node.parent_id or '—'}` | {node.depth} | "
            f"{node.status} | {node.score:.3f} | {tests} | {reason} |"
        )
    lines.extend(
        [
            "",
            "## Search accounting",
            "",
            (
                f"Proposed **{report.metrics['proposed']}**, evaluated with tests "
                f"**{report.metrics['test_calls']}**, "
                f"deduplicated **{report.metrics['duplicates']}**, "
                f"pruned **{report.metrics['pruned']}**."
            ),
            "",
            "A duplicate is the same complete workspace snapshot under the declared "
            "test-environment fingerprint. Test outcomes are evidence for ordering and "
            "stopping; they are not a hidden correctness proof.",
            "",
        ]
    )
    return "\n".join(lines)


def _status_color(status: str) -> str:
    return {
        "root": "#64748b",
        "frontier": "#2563eb",
        "accepted": "#16a34a",
        "duplicate": "#9333ea",
        "pruned": "#94a3b8",
        "failed": "#dc2626",
    }.get(status, "#475569")


def render_branch_html(report: BranchSearchReport) -> str:
    """Render a self-contained SVG branch graph with no external dependencies."""

    by_depth: dict[int, list[BranchNode]] = defaultdict(list)
    for node in report.nodes:
        by_depth[node.depth].append(node)
    positions: dict[str, tuple[int, int]] = {}
    for depth in sorted(by_depth):
        for index, node in enumerate(sorted(by_depth[depth], key=lambda item: item.id)):
            positions[node.id] = (180 + index * 220, 90 + depth * 130)
    max_x = max((position[0] for position in positions.values()), default=180) + 180
    max_y = max((position[1] for position in positions.values()), default=90) + 100
    svg: list[str] = [
        f'<svg viewBox="0 0 {max_x} {max_y}" role="img" '
        'aria-label="branch search graph">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
    ]
    for node in sorted(report.nodes, key=lambda item: item.id):
        if node.parent_id is None or node.parent_id not in positions:
            continue
        x1, y1 = positions[node.parent_id]
        x2, y2 = positions[node.id]
        svg.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            'stroke="#cbd5e1" stroke-width="2"/>'
        )
    for node in sorted(report.nodes, key=lambda item: item.id):
        x, y = positions[node.id]
        color = _status_color(node.status)
        label = escape(node.id)
        score = f"{node.score:.3f}"
        svg.extend(
            [
                f'<g><circle cx="{x}" cy="{y}" r="34" fill="{color}"/>',
                f'<text x="{x}" y="{y - 3}" text-anchor="middle" fill="white" '
                f'font-family="ui-monospace,monospace" font-size="12">{label}</text>',
                f'<text x="{x}" y="{y + 14}" text-anchor="middle" fill="white" '
                f'font-family="ui-monospace,monospace" font-size="11">'
                f"{score}</text></g>",
            ]
        )
    svg.append("</svg>")
    rows = []
    for node in report.nodes:
        rows.append(
            f"<tr><td><code>{escape(node.id)}</code></td><td>{escape(node.status)}</td>"
            f"<td>{node.score:.3f}</td><td>{escape(node.reason or '')}</td></tr>"
        )
    return (
        """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>ContextOpt branch search</title>
<style>
body{font-family:system-ui,sans-serif;margin:2rem;color:#0f172a;background:#fff}
main{max-width:1100px;margin:auto}h1{margin-bottom:.3rem}
.metrics{display:flex;flex-wrap:wrap;gap:.6rem;margin:1rem 0}
.metric{background:#f1f5f9;border-radius:.6rem;padding:.6rem .8rem}
.metric b{display:block;font-size:1.25rem}
.graph{border:1px solid #e2e8f0;border-radius:.6rem;overflow:auto}
svg{min-width:600px;min-height:260px}
table{border-collapse:collapse;width:100%;margin-top:1rem}
th,td{text-align:left;border-bottom:1px solid #e2e8f0;padding:.45rem}
.accepted{color:#15803d}
</style></head><body><main>
<h1>ContextOpt test-guided branch search</h1>
<p><b>Task:</b> __TASK__</p>
<p><b>Status:</b> <span class="__STATUS_CLASS__">__STATUS__</span>
· <b>Best:</b> __BEST__</p>
<p><b>Selection policy:</b> __POLICY__</p>
<div class="metrics">__METRICS__</div><div class="graph">__SVG__</div>
<table><thead><tr><th>Branch</th><th>Status</th><th>Score</th><th>Decision</th></tr></thead>
<tbody>__ROWS__</tbody></table>
</main></body></html>
""".replace("__TASK__", escape(report.task))
        .replace("__STATUS_CLASS__", "accepted" if report.status == "accepted" else "")
        .replace("__STATUS__", escape(report.status))
        .replace("__BEST__", escape(report.best_node_id or "none"))
        .replace("__POLICY__", escape(report.config.search_policy))
        .replace(
            "__METRICS__",
            "".join(
                f'<div class="metric"><b>{report.metrics[key]}</b>{escape(key)}</div>'
                for key in ("proposed", "test_calls", "duplicates", "pruned")
            ),
        )
        .replace("__SVG__", "".join(svg))
        .replace("__ROWS__", "".join(rows))
    )


def _demo_result(
    passed: Iterable[str], failed: Iterable[str], *, duration_ms: float = 12.0
) -> TestResult:
    return TestResult(
        suite="solver-visible",
        passed_tests=tuple(passed),
        failed_tests=tuple(failed),
        duration_ms=duration_ms,
    )


def demo_case() -> BranchCase:
    """Return a small deterministic case that exhibits pruning and deduplication."""

    tests = ("empty", "single", "duplicates", "negative", "large")
    base = "def solve(values):\n    return values\n"
    greedy = "def solve(values):\n    return sorted(set(values))\n"
    recursive = "def solve(values):\n    return sorted(values)  # handles duplicates\n"
    iterative = (
        "def solve(values):\n    return sorted(list(values))  # stable iterative path\n"
    )
    final = "def solve(values):\n    return sorted(values)\n"
    fallback = (
        "def solve(values):\n    return list(sorted(values))  # extra conversion\n"
    )
    candidates = (
        CandidatePatch(
            id="candidate-greedy",
            parent_id="root",
            hypothesis="Use a fast set-based implementation.",
            files={"src/solver.py": greedy},
            evidence=("failure:duplicates", "complexity:O(n)"),
        ),
        CandidatePatch(
            id="candidate-iterative",
            parent_id="root",
            hypothesis="Sort through a stable iterative path.",
            files={"src/solver.py": iterative},
            evidence=("failure:negative", "invariant:ordering"),
        ),
        CandidatePatch(
            id="candidate-recursive",
            parent_id="root",
            hypothesis="Preserve duplicates with a recursive-style rewrite.",
            files={"src/solver.py": recursive},
            evidence=("failure:duplicates",),
        ),
        CandidatePatch(
            id="iterative-fallback",
            parent_id="candidate-iterative",
            hypothesis="Keep the iterative approach and normalize the return value.",
            files={"src/solver.py": fallback},
            evidence=("test:large",),
        ),
        CandidatePatch(
            id="iterative-fix",
            parent_id="candidate-iterative",
            hypothesis="Remove the speculative conversion and preserve exact ordering.",
            files={"src/solver.py": final},
            evidence=("test:negative", "test:duplicates"),
        ),
        CandidatePatch(
            id="recursive-fix",
            parent_id="candidate-recursive",
            hypothesis=(
                "Apply the same minimal sorted implementation after test feedback."
            ),
            files={"src/solver.py": final},
            evidence=("test:large",),
        ),
    )
    return BranchCase(
        task="Implement solve(values) so ordering and duplicates are preserved.",
        root_files={"src/solver.py": base},
        candidates=candidates,
        tests={
            "candidate-greedy": _demo_result(
                ("empty", "single"), ("duplicates", "negative", "large")
            ),
            "candidate-iterative": _demo_result(
                ("empty", "single", "duplicates", "negative"), ("large",)
            ),
            "candidate-recursive": _demo_result(
                ("empty", "single", "duplicates"), ("negative", "large")
            ),
            "iterative-fallback": _demo_result(
                ("empty", "single", "duplicates", "negative"), ("large",)
            ),
            "iterative-fix": _demo_result(tests, ()),
            "recursive-fix": _demo_result(tests, ()),
        },
    )
