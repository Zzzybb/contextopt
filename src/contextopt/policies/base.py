"""Shared policy machinery and receipt construction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from contextopt.models import (
    ContextFrame,
    SelectionDecision,
    SelectionProblem,
)


class UnsupportedProblemError(ValueError):
    """Raised when a baseline intentionally does not support graph constraints."""


class ContextPolicy(ABC):
    """Interface implemented by every interchangeable context selection policy."""

    name: str

    @abstractmethod
    def select(self, problem: SelectionProblem) -> ContextFrame:
        raise NotImplementedError

    def _seed_mandatory(
        self, problem: SelectionProblem
    ) -> tuple[set[str], list[str], dict[str, str]]:
        selected = set(problem.mandatory_closure())
        issues = problem.feasibility_issues(selected)
        if issues:
            raise ValueError("mandatory context is infeasible: " + "; ".join(issues))
        order = sorted(selected)
        reasons: dict[str, str] = {}
        for item_id in order:
            item = problem.by_id[item_id]
            reasons[item_id] = (
                "mandatory item" if item.mandatory else "dependency of a mandatory item"
            )
        return selected, order, reasons

    def _finalize(
        self,
        problem: SelectionProblem,
        selected_order: list[str],
        selected_reasons: Mapping[str, str],
        rejected_reasons: Mapping[str, str] | None = None,
        gains: Mapping[str, float] | None = None,
        added_tokens: Mapping[str, int] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ContextFrame:
        selected_ids = tuple(dict.fromkeys(selected_order))
        issues = problem.feasibility_issues(selected_ids)
        if issues:
            raise AssertionError(
                f"policy {self.name!r} emitted an infeasible frame: {'; '.join(issues)}"
            )

        rejected_reasons = rejected_reasons or {}
        gains = gains or {}
        added_tokens = added_tokens or {}
        selected_set = set(selected_ids)
        decisions: list[SelectionDecision] = []
        for item in problem.items:
            if item.id in selected_set:
                decisions.append(
                    SelectionDecision(
                        item_id=item.id,
                        status="selected",
                        reason=selected_reasons.get(item.id, "selected by policy"),
                        marginal_gain=gains.get(item.id),
                        added_tokens=added_tokens.get(item.id, item.tokens),
                    )
                )
            else:
                decisions.append(
                    SelectionDecision(
                        item_id=item.id,
                        status="rejected",
                        reason=rejected_reasons.get(item.id, "lower marginal value"),
                    )
                )

        return ContextFrame(
            policy=self.name,
            selected_ids=selected_ids,
            used_tokens=problem.tokens_for(selected_ids),
            budget=problem.budget,
            objective_score=problem.score(selected_ids),
            decisions=tuple(decisions),
            metadata=dict(metadata or {}),
        )


def proposed_issues(
    problem: SelectionProblem, selected: set[str], bundle: set[str] | frozenset[str]
) -> tuple[str, ...]:
    """Return deterministic reasons why adding a dependency-closed bundle is invalid."""

    return problem.feasibility_issues(selected | set(bundle))
