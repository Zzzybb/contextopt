"""Core, model-agnostic data structures for context selection."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import Any


def _frozen_strings(value: Iterable[str] | None) -> frozenset[str]:
    return frozenset(str(item) for item in (value or ()))


def _unit_interval(name: str, value: float) -> None:
    if not isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and between 0 and 1, got {value!r}")


@dataclass(frozen=True, slots=True)
class ContextItem:
    """One candidate unit that may be packed into a model context window."""

    id: str
    tokens: int
    relevance: float
    importance: float = 0.5
    freshness: float = 1.0
    kind: str = "generic"
    source: str | None = None
    content: str = ""
    topics: frozenset[str] = field(default_factory=frozenset)
    dependencies: frozenset[str] = field(default_factory=frozenset)
    conflicts: frozenset[str] = field(default_factory=frozenset)
    duplicate_group: str | None = None
    mandatory: bool = False

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("context item id must not be empty")
        if self.tokens <= 0:
            raise ValueError(f"tokens must be positive for {self.id!r}")
        _unit_interval("relevance", self.relevance)
        _unit_interval("importance", self.importance)
        _unit_interval("freshness", self.freshness)
        if self.id in self.dependencies:
            raise ValueError(f"{self.id!r} cannot depend on itself")
        if self.id in self.conflicts:
            raise ValueError(f"{self.id!r} cannot conflict with itself")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ContextItem:
        return cls(
            id=str(data["id"]),
            tokens=int(data["tokens"]),
            relevance=float(data["relevance"]),
            importance=float(data.get("importance", 0.5)),
            freshness=float(data.get("freshness", 1.0)),
            kind=str(data.get("kind", "generic")),
            source=None if data.get("source") is None else str(data["source"]),
            content=str(data.get("content", "")),
            topics=_frozen_strings(data.get("topics")),
            dependencies=_frozen_strings(data.get("dependencies")),
            conflicts=_frozen_strings(data.get("conflicts")),
            duplicate_group=(
                None
                if data.get("duplicate_group") is None
                else str(data["duplicate_group"])
            ),
            mandatory=bool(data.get("mandatory", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tokens": self.tokens,
            "relevance": self.relevance,
            "importance": self.importance,
            "freshness": self.freshness,
            "kind": self.kind,
            "source": self.source,
            "content": self.content,
            "topics": sorted(self.topics),
            "dependencies": sorted(self.dependencies),
            "conflicts": sorted(self.conflicts),
            "duplicate_group": self.duplicate_group,
            "mandatory": self.mandatory,
        }


@dataclass(frozen=True, slots=True)
class ObjectiveWeights:
    """Weights for the transparent set-utility function used by v0.1."""

    relevance: float = 1.0
    importance: float = 0.8
    freshness: float = 0.2
    topic_coverage: float = 0.7
    duplicate_penalty: float = 0.6

    def __post_init__(self) -> None:
        for name in (
            "relevance",
            "importance",
            "freshness",
            "topic_coverage",
            "duplicate_penalty",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> ObjectiveWeights:
        return cls(**dict(data or {}))

    def to_dict(self) -> dict[str, float]:
        return {
            "relevance": self.relevance,
            "importance": self.importance,
            "freshness": self.freshness,
            "topic_coverage": self.topic_coverage,
            "duplicate_penalty": self.duplicate_penalty,
        }


@dataclass(slots=True)
class SelectionProblem:
    """A token-budgeted context packing problem with graph constraints."""

    items: tuple[ContextItem, ...]
    budget: int
    weights: ObjectiveWeights = field(default_factory=ObjectiveWeights)
    _by_id: dict[str, ContextItem] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.items = tuple(self.items)
        if self.budget <= 0:
            raise ValueError("budget must be positive")
        self._by_id = {item.id: item for item in self.items}
        if len(self._by_id) != len(self.items):
            raise ValueError("context item ids must be unique")
        known = set(self._by_id)
        for item in self.items:
            unknown_dependencies = item.dependencies - known
            unknown_conflicts = item.conflicts - known
            if unknown_dependencies:
                raise ValueError(
                    f"{item.id!r} has unknown dependencies: "
                    f"{sorted(unknown_dependencies)!r}"
                )
            if unknown_conflicts:
                raise ValueError(
                    f"{item.id!r} has unknown conflicts: {sorted(unknown_conflicts)!r}"
                )

    @property
    def by_id(self) -> Mapping[str, ContextItem]:
        return self._by_id

    @property
    def mandatory_ids(self) -> frozenset[str]:
        return frozenset(item.id for item in self.items if item.mandatory)

    def dependency_closure(self, item_ids: Iterable[str]) -> frozenset[str]:
        pending = list(item_ids)
        closure: set[str] = set()
        while pending:
            item_id = pending.pop()
            if item_id in closure:
                continue
            if item_id not in self._by_id:
                raise KeyError(f"unknown context item: {item_id}")
            closure.add(item_id)
            pending.extend(self._by_id[item_id].dependencies - closure)
        return frozenset(closure)

    def mandatory_closure(self) -> frozenset[str]:
        return self.dependency_closure(self.mandatory_ids)

    def tokens_for(self, item_ids: Iterable[str]) -> int:
        return sum(self._by_id[item_id].tokens for item_id in set(item_ids))

    def feasibility_issues(self, item_ids: Iterable[str]) -> tuple[str, ...]:
        selected = frozenset(item_ids)
        issues: list[str] = []
        unknown = selected - self._by_id.keys()
        if unknown:
            issues.append(f"unknown items: {sorted(unknown)!r}")
            return tuple(issues)
        used_tokens = self.tokens_for(selected)
        if used_tokens > self.budget:
            issues.append(f"token budget exceeded: {used_tokens} > {self.budget}")
        missing_mandatory = self.mandatory_ids - selected
        if missing_mandatory:
            issues.append(f"missing mandatory items: {sorted(missing_mandatory)!r}")
        seen_conflicts: set[tuple[str, str]] = set()
        for item_id in sorted(selected):
            item = self._by_id[item_id]
            missing = item.dependencies - selected
            if missing:
                issues.append(f"{item_id} missing dependencies: {sorted(missing)!r}")
            conflicts = item.conflicts & selected
            for other in sorted(conflicts):
                pair = (item_id, other) if item_id < other else (other, item_id)
                if pair not in seen_conflicts:
                    seen_conflicts.add(pair)
                    issues.append(f"conflict: {pair[0]} vs {pair[1]}")
        return tuple(issues)

    def is_feasible(self, item_ids: Iterable[str]) -> bool:
        return not self.feasibility_issues(item_ids)

    def standalone_utility(self, item: ContextItem) -> float:
        base = (
            self.weights.relevance * item.relevance
            + self.weights.importance * item.importance
            + self.weights.freshness * item.freshness
        )
        topic_signal = item.importance * item.freshness
        return base + self.weights.topic_coverage * len(item.topics) * topic_signal

    def score(self, item_ids: Iterable[str]) -> float:
        selected = [self._by_id[item_id] for item_id in set(item_ids)]
        base = sum(
            self.weights.relevance * item.relevance
            + self.weights.importance * item.importance
            + self.weights.freshness * item.freshness
            for item in selected
        )

        topic_best: dict[str, float] = {}
        duplicate_counts: dict[str, int] = {}
        for item in selected:
            signal = item.importance * item.freshness
            for topic in item.topics:
                topic_best[topic] = max(topic_best.get(topic, 0.0), signal)
            if item.duplicate_group:
                duplicate_counts[item.duplicate_group] = (
                    duplicate_counts.get(item.duplicate_group, 0) + 1
                )

        coverage = self.weights.topic_coverage * sum(topic_best.values())
        duplicate_extras = sum(max(0, count - 1) for count in duplicate_counts.values())
        penalty = self.weights.duplicate_penalty * duplicate_extras
        return base + coverage - penalty

    def incremental_bundle(
        self, item_id: str, selected_ids: Iterable[str]
    ) -> frozenset[str]:
        selected = frozenset(selected_ids)
        return self.dependency_closure((item_id,)) - selected

    def marginal_gain(
        self, selected_ids: Iterable[str], bundle_ids: Iterable[str]
    ) -> float:
        selected = frozenset(selected_ids)
        expanded = selected | frozenset(bundle_ids)
        return self.score(expanded) - self.score(selected)


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    item_id: str
    status: str
    reason: str
    marginal_gain: float | None = None
    added_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "status": self.status,
            "reason": self.reason,
            "marginal_gain": self.marginal_gain,
            "added_tokens": self.added_tokens,
        }


@dataclass(frozen=True, slots=True)
class ContextFrame:
    policy: str
    selected_ids: tuple[str, ...]
    used_tokens: int
    budget: int
    objective_score: float
    decisions: tuple[SelectionDecision, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def utilization(self) -> float:
        return self.used_tokens / self.budget

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "selected_ids": list(self.selected_ids),
            "used_tokens": self.used_tokens,
            "budget": self.budget,
            "utilization": self.utilization,
            "objective_score": self.objective_score,
            "decisions": [decision.to_dict() for decision in self.decisions],
            "metadata": dict(self.metadata),
        }
