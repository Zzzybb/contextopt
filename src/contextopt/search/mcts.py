"""Bounded, auditable UCT traversal for immutable coding-candidate trees.

The implementation deliberately uses only observations already present in a
``BranchCase``. It is therefore a deterministic control-flow primitive for
comparing schedulers, not a claim that a model has discovered a correct patch.
Each rollout evaluates at most one new workspace snapshot, propagates its
visible-test quality to the parent path, and records the selection decision in
the existing hash-chained search ledger.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import replace
from math import isfinite, log, sqrt

from contextopt.search.branching import (
    BranchCase,
    BranchNode,
    BranchSearch,
    BranchSearchConfig,
    BranchSearchReport,
    CandidatePatch,
    CandidateStatus,
    SearchStatus,
    _EventBuilder,
    validate_search_report,
)


def _candidate_depths(case: BranchCase) -> Mapping[str, int]:
    parents = {candidate.id: candidate.parent_id for candidate in case.candidates}
    depths: dict[str, int] = {}
    for candidate in case.candidates:
        depth = 1
        parent = candidate.parent_id
        seen: set[str] = set()
        while parent != "root":
            if parent in seen:
                raise ValueError("candidate parent graph must be acyclic")
            seen.add(parent)
            depth += 1
            parent = parents[parent]
        depths[candidate.id] = depth
    return depths


def run_mcts_search(case: BranchCase, config: BranchSearchConfig) -> BranchSearchReport:
    """Run a deterministic UCT-style search over the supplied candidate tree."""

    if config.search_policy != "mcts":
        raise ValueError("run_mcts_search requires search_policy='mcts'")
    depths = _candidate_depths(case)
    by_id = case.by_id
    branch = BranchSearch(config)
    builder = _EventBuilder()
    root_dedup = branch._dedup_key(case.root_fingerprint)
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
    nodes: dict[str, BranchNode] = {"root": root}
    seen: dict[str, str] = {root_dedup: "root"}
    proposed_ids: set[str] = set()
    expandable_ids: set[str] = {"root"}
    visits: defaultdict[str, int] = defaultdict(int)
    value_sums: defaultdict[str, float] = defaultdict(float)
    accepted: list[BranchNode] = []
    proposed = evaluated = duplicates = pruned = passed = 0
    test_calls = 0
    max_depth = 0

    builder.emit(
        "search.started",
        {
            "task": case.task,
            "case_fingerprint": case.fingerprint,
            "config": config.to_dict(),
            "selection_policy": "mcts",
        },
    )

    def has_pending(parent_id: str) -> bool:
        for child in case.children(parent_id):
            if depths[child.id] > config.max_depth:
                continue
            if child.id not in proposed_ids:
                return True
            if child.id in expandable_ids and has_pending(child.id):
                return True
        return False

    def ucb_score(candidate: CandidatePatch, parent_id: str) -> float | None:
        if candidate.id not in proposed_ids:
            return None
        count = visits[candidate.id]
        if count <= 0:
            return None
        parent_count = max(1, visits[parent_id])
        exploitation = value_sums[candidate.id] / count
        exploration = config.exploration_constant * sqrt(
            log(parent_count + 1.0) / count
        )
        score = exploitation + exploration
        return score if isfinite(score) else None

    def select_child(parent_id: str) -> tuple[CandidatePatch, float | None] | None:
        viable: list[CandidatePatch] = []
        for candidate in case.children(parent_id):
            if depths[candidate.id] > config.max_depth:
                continue
            if candidate.id not in proposed_ids or (
                candidate.id in expandable_ids and has_pending(candidate.id)
            ):
                viable.append(candidate)
        if not viable:
            return None
        scored = [(ucb_score(candidate, parent_id), candidate) for candidate in viable]
        selected_score, selected = sorted(
            scored,
            key=lambda item: (
                item[0] is not None,
                0.0 if item[0] is None else -item[0],
                depths[item[1].id],
                item[1].id,
            ),
        )[0]
        return selected, selected_score

    def select_next(
        parent_id: str,
    ) -> tuple[CandidatePatch, str, float | None] | None:
        choice = select_child(parent_id)
        if choice is None:
            return None
        candidate, score = choice
        if candidate.id not in proposed_ids:
            return candidate, parent_id, score
        return select_next(candidate.id)

    def propagate(candidate_id: str, reward: float) -> None:
        current = candidate_id
        while current != "root":
            visits[current] += 1
            value_sums[current] += reward
            current = by_id[current].parent_id
        visits["root"] += 1
        value_sums["root"] += reward

    while proposed < config.max_candidates:
        selected = select_next("root")
        if selected is None:
            break
        candidate, parent_id, selection_score = selected
        proposed += 1
        proposed_ids.add(candidate.id)
        max_depth = max(max_depth, depths[candidate.id])
        builder.emit(
            "candidate.selected",
            {
                "candidate_id": candidate.id,
                "parent_id": parent_id,
                "selection_policy": "mcts",
                "selection_score": selection_score,
                "candidate_visits": visits[candidate.id],
                "parent_visits": visits[parent_id],
                "candidate_value": value_sums[candidate.id],
                "parent_mean_quality": (
                    None
                    if visits[parent_id] == 0
                    else value_sums[parent_id] / visits[parent_id]
                ),
            },
        )
        workspace = candidate.workspace_fingerprint
        dedup_key = branch._dedup_key(workspace)
        builder.emit(
            "candidate.proposed",
            {
                "candidate_id": candidate.id,
                "parent_id": candidate.parent_id,
                "depth": depths[candidate.id],
                "patch_fingerprint": candidate.patch_fingerprint,
                "workspace_fingerprint": workspace,
                "dedup_key": dedup_key,
                "selection_policy": "mcts",
            },
        )
        if dedup_key in seen:
            duplicate_of = seen[dedup_key]
            nodes[candidate.id] = BranchNode(
                id=candidate.id,
                parent_id=candidate.parent_id,
                depth=depths[candidate.id],
                status="duplicate",
                hypothesis=candidate.hypothesis,
                workspace_fingerprint=workspace,
                dedup_key=dedup_key,
                score=0.0,
                duplicate_of=duplicate_of,
                reason="same workspace state under the fixed test environment",
                evidence=candidate.evidence,
            )
            duplicates += 1
            builder.emit(
                "candidate.duplicate",
                {
                    "candidate_id": candidate.id,
                    "duplicate_of": duplicate_of,
                    "dedup_key": dedup_key,
                    "selection_policy": "mcts",
                },
            )
            continue

        seen[dedup_key] = candidate.id
        result = case.tests[candidate.id]
        evaluated += 1
        test_calls += 1
        score = branch._score(result, depths[candidate.id])
        node_status: CandidateStatus = "accepted" if result.is_success else "frontier"
        node = BranchNode(
            id=candidate.id,
            parent_id=candidate.parent_id,
            depth=depths[candidate.id],
            status=node_status,
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
        propagate(candidate.id, result.quality_score)
        if result.is_success:
            accepted.append(node)
            passed += 1
        else:
            expandable_ids.add(candidate.id)
        builder.emit(
            "candidate.evaluated",
            {
                "candidate_id": candidate.id,
                "test_result": result.to_dict(),
                "test_fingerprint": result.behavior_fingerprint,
                "score": score,
                "mcts_reward": result.quality_score,
                "mcts_visits": visits[candidate.id],
                "selection_policy": "mcts",
            },
        )
        if accepted and config.stop_on_pass:
            break

    pending = has_pending("root")
    status: SearchStatus
    if accepted:
        status = "accepted"
        best = sorted(accepted, key=branch._priority)[0]
        best_node_id: str | None = best.id
    elif proposed >= config.max_candidates and pending:
        status = "budget_exhausted"
        best_node_id = None
    else:
        status = "exhausted"
        best_node_id = None

    if not accepted:
        for node_id, node in list(nodes.items()):
            if node_id == "root" or node.status != "frontier":
                continue
            if node.depth < config.max_depth and has_pending(node.id):
                continue
            nodes[node_id] = replace(
                node,
                status="pruned",
                reason="maximum search depth reached"
                if node.depth >= config.max_depth
                else "no unobserved descendants remain",
            )
            pruned += 1
            builder.emit(
                "candidate.pruned",
                {
                    "candidate_id": node_id,
                    "reason": nodes[node_id].reason,
                    "selection_policy": "mcts",
                },
            )

    metrics: dict[str, int | float] = {
        "proposed": proposed,
        "evaluated": evaluated,
        "test_calls": test_calls,
        "duplicates": duplicates,
        "pruned": pruned,
        "passed": passed,
        "max_depth": max_depth,
        "mcts_rollouts": evaluated,
        "mcts_tree_visits": sum(visits.values()),
    }
    builder.emit(
        "search.completed",
        {
            "status": status,
            "best_node_id": best_node_id,
            **metrics,
            "selection_policy": "mcts",
        },
    )
    report = BranchSearchReport(
        task=case.task,
        case_fingerprint=case.fingerprint,
        config=config,
        status=status,
        best_node_id=best_node_id,
        nodes=tuple(nodes[node_id] for node_id in sorted(nodes)),
        events=tuple(builder.events),
        metrics=metrics,
        case=case,
    )
    validate_search_report(report)
    return report


__all__ = ["run_mcts_search"]
