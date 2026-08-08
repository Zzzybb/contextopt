"""Deterministic, evidence-first three-way merging of candidate snapshots."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Literal

from contextopt.runtime.identity import stable_hash
from contextopt.search.branching import BranchCase, CandidatePatch, TestResult

MergePolicy = Literal["disabled", "disjoint"]
MERGE_POLICIES: tuple[MergePolicy, ...] = ("disabled", "disjoint")
_MISSING = object()


def _hash_optional(value: object) -> str | None:
    return None if value is _MISSING else stable_hash(value)


@dataclass(frozen=True, slots=True)
class MergeConflict:
    """One path where two speculative candidates changed the same base differently."""

    left_id: str
    right_id: str
    path: str
    base_sha256: str | None
    left_sha256: str | None
    right_sha256: str | None

    def __post_init__(self) -> None:
        for name in ("left_id", "right_id", "path"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"merge conflict {name} must be non-empty")
        for name in ("base_sha256", "left_sha256", "right_sha256"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or len(value) != 64):
                raise ValueError(f"merge conflict {name} must be a hash or null")

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_id": self.left_id,
            "right_id": self.right_id,
            "path": self.path,
            "base_sha256": self.base_sha256,
            "left_sha256": self.left_sha256,
            "right_sha256": self.right_sha256,
        }


@dataclass(frozen=True, slots=True)
class MergeReport:
    """Auditable summary of bounded speculative pair reconciliation."""

    policy: MergePolicy
    base_fingerprint: str
    pairs_considered: int
    merged_candidate_ids: tuple[str, ...]
    conflicts: tuple[MergeConflict, ...]

    def __post_init__(self) -> None:
        if self.policy not in MERGE_POLICIES:
            raise ValueError(f"unsupported merge policy: {self.policy!r}")
        if (
            not isinstance(self.base_fingerprint, str)
            or len(self.base_fingerprint) != 64
        ):
            raise ValueError("merge base_fingerprint must be a hash")
        if not isinstance(self.pairs_considered, int) or self.pairs_considered < 0:
            raise ValueError("merge pairs_considered must be non-negative")
        if len(set(self.merged_candidate_ids)) != len(self.merged_candidate_ids):
            raise ValueError("merged candidate ids must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "base_fingerprint": self.base_fingerprint,
            "pairs_considered": self.pairs_considered,
            "merged_candidate_ids": list(self.merged_candidate_ids),
            "conflicts": [conflict.to_dict() for conflict in self.conflicts],
        }


def three_way_merge(
    root_files: Mapping[str, str],
    left: CandidatePatch,
    right: CandidatePatch,
) -> tuple[dict[str, str] | None, tuple[MergeConflict, ...]]:
    """Merge two complete snapshots against ``root_files``.

    A path is safe when one side is unchanged, both sides made the same change, or both
    sides deleted it. Any divergent edit is surfaced as a conflict and produces no
    partial merged workspace.
    """

    merged: dict[str, str] = {}
    conflicts: list[MergeConflict] = []
    paths = sorted(set(root_files) | set(left.files) | set(right.files))
    for path in paths:
        base = root_files.get(path, _MISSING)
        left_value = left.files.get(path, _MISSING)
        right_value = right.files.get(path, _MISSING)
        if left_value == right_value:
            chosen = left_value
        elif left_value == base:
            chosen = right_value
        elif right_value == base:
            chosen = left_value
        else:
            conflicts.append(
                MergeConflict(
                    left_id=left.id,
                    right_id=right.id,
                    path=path,
                    base_sha256=_hash_optional(base),
                    left_sha256=_hash_optional(left_value),
                    right_sha256=_hash_optional(right_value),
                )
            )
            continue
        if chosen is not _MISSING:
            if not isinstance(chosen, str):
                raise ValueError(f"merged file {path!r} is not text")
            merged[path] = chosen
    return (None if conflicts else merged), tuple(conflicts)


def merge_candidate_pairs(
    case: BranchCase,
    *,
    policy: MergePolicy = "disabled",
    max_pairs: int | None = None,
) -> tuple[BranchCase, MergeReport]:
    """Append bounded, conflict-free merged candidates to a proposal case."""

    if policy not in MERGE_POLICIES:
        raise ValueError(f"unsupported merge policy: {policy!r}")
    if max_pairs is not None and (
        not isinstance(max_pairs, int) or isinstance(max_pairs, bool) or max_pairs < 0
    ):
        raise ValueError("max_pairs must be a non-negative integer or null")
    root_candidates = tuple(
        sorted(
            (
                candidate
                for candidate in case.candidates
                if candidate.parent_id == "root"
            ),
            key=lambda candidate: candidate.id,
        )
    )
    pair_candidates = (
        () if policy == "disabled" else tuple(combinations(root_candidates, 2))
    )
    if max_pairs is not None:
        pair_candidates = pair_candidates[:max_pairs]
    merged_candidates: list[CandidatePatch] = []
    conflicts: list[MergeConflict] = []
    existing_ids = {candidate.id for candidate in case.candidates}
    existing_fingerprints = {
        candidate.workspace_fingerprint for candidate in case.candidates
    }
    for left, right in pair_candidates:
        files, pair_conflicts = three_way_merge(case.root_files, left, right)
        if pair_conflicts:
            conflicts.extend(pair_conflicts)
            continue
        if files is None:
            continue
        candidate_id = f"merge-{left.id}--{right.id}"
        if candidate_id in existing_ids:
            pair_hash = stable_hash({"left": left.id, "right": right.id})[:10]
            candidate_id = f"{candidate_id}-{pair_hash}"
        candidate = CandidatePatch(
            id=candidate_id,
            parent_id="root",
            hypothesis=f"merge independent branches {left.id} and {right.id}",
            files=files,
            evidence=(
                f"three-way merge against {stable_hash(dict(case.root_files))}",
                f"merged candidates: {left.id}, {right.id}",
            ),
        )
        if candidate.workspace_fingerprint in existing_fingerprints:
            continue
        existing_ids.add(candidate.id)
        existing_fingerprints.add(candidate.workspace_fingerprint)
        merged_candidates.append(candidate)
    candidates = (*case.candidates, *merged_candidates)
    tests: dict[str, TestResult] = dict(case.tests)
    tests.update(
        {
            candidate.id: TestResult(
                suite="not-executed",
                error="candidate has not been evaluated by the visible-test oracle",
            )
            for candidate in merged_candidates
        }
    )
    merged_case = (
        case
        if not merged_candidates
        else BranchCase(
            task=case.task,
            root_files=case.root_files,
            candidates=candidates,
            tests=tests,
        )
    )
    report = MergeReport(
        policy=policy,
        base_fingerprint=stable_hash(dict(case.root_files)),
        pairs_considered=len(pair_candidates),
        merged_candidate_ids=tuple(candidate.id for candidate in merged_candidates),
        conflicts=tuple(conflicts),
    )
    return merged_case, report


__all__ = [
    "MERGE_POLICIES",
    "MergeConflict",
    "MergePolicy",
    "MergeReport",
    "merge_candidate_pairs",
    "three_way_merge",
]
