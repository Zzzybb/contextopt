"""Deterministic candidate scheduling policies for bounded coding search.

The executor remains the side-effect boundary; this module only decides which complete
candidate snapshots should be offered to the next oracle batch. ``fixed`` is the stable
baseline used by existing reports. ``adaptive`` uses observed parent quality to expand
the most promising part of the candidate graph first. Later batches therefore depend
on real oracle evidence rather than on model confidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from contextopt.search.branching import CandidatePatch, TestResult

SchedulerPolicy = Literal["fixed", "adaptive"]
SCHEDULER_POLICIES: tuple[SchedulerPolicy, ...] = ("fixed", "adaptive")


def _depths(candidates: Sequence[CandidatePatch]) -> Mapping[str, int]:
    parents = {candidate.id: candidate.parent_id for candidate in candidates}
    depths: dict[str, int] = {}
    for candidate in candidates:
        depth = 1
        parent = candidate.parent_id
        seen: set[str] = set()
        while parent != "root" and parent in parents:
            if parent in seen:
                raise ValueError("candidate parent graph must be acyclic")
            seen.add(parent)
            depth += 1
            parent = parents[parent]
        depths[candidate.id] = depth
    return depths


def select_candidate_batch(
    candidates: Sequence[CandidatePatch],
    observations: Mapping[str, TestResult],
    *,
    limit: int,
    policy: SchedulerPolicy = "fixed",
) -> tuple[CandidatePatch, ...]:
    """Select one deterministic batch of not-yet-observed workspace states.

    ``observations`` is keyed by workspace fingerprint, matching both durable session
    reports and the branch oracle cache. Duplicate workspace states are emitted once.
    In adaptive mode, a candidate whose parent has a stronger observed quality score is
    promoted; its children become the next hypotheses after a useful partial repair.
    """

    if policy not in SCHEDULER_POLICIES:
        raise ValueError(f"unsupported scheduler policy: {policy!r}")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError("scheduler batch limit must be a positive integer")
    depths = _depths(candidates)
    unique: dict[str, CandidatePatch] = {}
    for candidate in candidates:
        unique.setdefault(candidate.workspace_fingerprint, candidate)
    pending = [
        candidate
        for candidate in unique.values()
        if candidate.workspace_fingerprint not in observations
    ]
    if policy == "fixed":
        return tuple(sorted(pending, key=lambda item: item.id)[:limit])

    by_id = {candidate.id: candidate for candidate in candidates}
    quality_by_id = {
        candidate.id: observations[candidate.workspace_fingerprint].quality_score
        for candidate in candidates
        if candidate.workspace_fingerprint in observations
    }

    def adaptive_key(candidate: CandidatePatch) -> tuple[float, int, str]:
        parent = by_id.get(candidate.parent_id)
        parent_quality = (
            quality_by_id.get(parent.id, 0.0) if parent is not None else 0.0
        )
        return (-parent_quality, depths[candidate.id], candidate.id)

    return tuple(sorted(pending, key=adaptive_key)[:limit])


__all__ = ["SCHEDULER_POLICIES", "SchedulerPolicy", "select_candidate_batch"]
