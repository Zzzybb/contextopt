"""Deterministic synthetic instances with known critical facts and distractors."""

from __future__ import annotations

from dataclasses import dataclass, replace
from random import Random
from typing import Any

from contextopt.models import ContextItem, ObjectiveWeights, SelectionProblem


@dataclass(frozen=True, slots=True)
class SyntheticCase:
    case_id: str
    problem: SelectionProblem
    critical_ids: frozenset[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "critical_ids": sorted(self.critical_ids),
            "budget": self.problem.budget,
            "weights": self.problem.weights.to_dict(),
            "items": [item.to_dict() for item in self.problem.items],
        }


def generate_case(
    *,
    seed: int,
    item_count: int = 14,
    critical_count: int = 4,
    budget: int = 1_200,
    graph_rate: float = 0.0,
    conflict_rate: float = 0.0,
    weights: ObjectiveWeights | None = None,
) -> SyntheticCase:
    """Build one reproducible selection problem.

    Critical facts have high task importance and unique coverage but only medium lexical
    relevance. Distractors look highly relevant while repeating a small set of topics.
    This mirrors the failure mode the benchmark is intended to expose without hiding the
    objective or baking a particular policy into the generator.
    """

    if item_count < critical_count + 3:
        raise ValueError("item_count must leave room for a goal and distractors")
    if not 0.0 <= graph_rate <= 1.0:
        raise ValueError("graph_rate must be between 0 and 1")
    if not 0.0 <= conflict_rate <= 1.0:
        raise ValueError("conflict_rate must be between 0 and 1")

    rng = Random(seed)
    items: list[ContextItem] = [
        ContextItem(
            id="goal",
            tokens=64,
            relevance=1.0,
            importance=1.0,
            freshness=1.0,
            kind="state",
            source="run://goal",
            content="Current task goal and hard constraints.",
            topics=frozenset({"goal"}),
            mandatory=True,
        )
    ]

    critical_ids: set[str] = set()
    for index in range(critical_count):
        item_id = f"critical-{index:02d}"
        critical_ids.add(item_id)
        items.append(
            ContextItem(
                id=item_id,
                tokens=rng.randint(80, 220),
                relevance=rng.uniform(0.45, 0.74),
                importance=rng.uniform(0.85, 1.0),
                freshness=rng.uniform(0.82, 1.0),
                kind="constraint",
                source=f"spec://requirement/{index}",
                content=f"Task-critical requirement {index}.",
                topics=frozenset({f"requirement:{index}"}),
            )
        )

    remaining = item_count - len(items)
    distractor_count = max(2, remaining // 2)
    for index in range(distractor_count):
        group = f"popular-topic-{index % 2}"
        items.append(
            ContextItem(
                id=f"distractor-{index:02d}",
                tokens=rng.randint(120, 360),
                relevance=rng.uniform(0.76, 0.99),
                importance=rng.uniform(0.1, 0.45),
                freshness=rng.uniform(0.3, 0.95),
                kind="tool_output",
                source=f"event://search/{index}",
                content=f"Highly similar but repetitive observation {index}.",
                topics=frozenset({group}),
                duplicate_group=group,
            )
        )

    while len(items) < item_count:
        index = len(items)
        items.append(
            ContextItem(
                id=f"background-{index:02d}",
                tokens=rng.randint(60, 300),
                relevance=rng.uniform(0.05, 0.72),
                importance=rng.uniform(0.05, 0.75),
                freshness=rng.uniform(0.05, 1.0),
                kind=rng.choice(("code", "memory", "documentation", "tool_output")),
                source=f"synthetic://background/{index}",
                content=f"Background context item {index}.",
                topics=frozenset({f"background:{rng.randint(0, 4)}"}),
            )
        )

    # Dependencies only point backwards, making the generated graph acyclic.
    for index in range(2, len(items)):
        if rng.random() < graph_rate:
            dependency_index = rng.randrange(0, index)
            dependency = items[dependency_index].id
            items[index] = replace(items[index], dependencies=frozenset({dependency}))

    # Symmetric conflict edges are added after dependencies so direct
    # dependency/conflict contradictions can be skipped deterministically.
    conflicts: dict[str, set[str]] = {item.id: set(item.conflicts) for item in items}
    for left in range(1, len(items)):
        for right in range(left + 1, len(items)):
            if rng.random() >= conflict_rate:
                continue
            left_id, right_id = items[left].id, items[right].id
            if (
                right_id in items[left].dependencies
                or left_id in items[right].dependencies
            ):
                continue
            conflicts[left_id].add(right_id)
            conflicts[right_id].add(left_id)
    items = [replace(item, conflicts=frozenset(conflicts[item.id])) for item in items]

    problem = SelectionProblem(
        items=tuple(items),
        budget=budget,
        weights=weights or ObjectiveWeights(),
    )
    if problem.tokens_for(problem.mandatory_closure()) > budget:
        raise ValueError("budget cannot fit mandatory context")
    return SyntheticCase(
        case_id=f"synthetic-{seed}",
        problem=problem,
        critical_ids=frozenset(critical_ids),
    )
