"""Dependency-aware marginal-gain packing for the transparent set objective."""

from __future__ import annotations

from contextopt.models import ContextFrame, SelectionProblem
from contextopt.policies.base import ContextPolicy, proposed_issues


class SubmodularGreedyPolicy(ContextPolicy):
    """Repeatedly add the feasible closure with best marginal gain per token.

    The topic-coverage term is monotone submodular. Duplicate penalties and hard graph
    constraints can make the complete objective non-monotone, so this implementation
    deliberately stops when no positive marginal bundle remains instead of claiming a
    universal approximation guarantee.
    """

    name = "submodular"

    def select(self, problem: SelectionProblem) -> ContextFrame:
        selected, order, reasons = self._seed_mandatory(problem)
        rejected: dict[str, str] = {}
        gains: dict[str, float] = {}
        added: dict[str, int] = {}
        iterations = 0

        while True:
            best: tuple[float, float, str, set[str], int] | None = None
            for item in problem.items:
                if item.id in selected:
                    continue
                bundle = set(problem.incremental_bundle(item.id, selected))
                issues = proposed_issues(problem, selected, bundle)
                if issues:
                    rejected[item.id] = issues[0]
                    continue
                bundle_tokens = problem.tokens_for(bundle)
                marginal = problem.marginal_gain(selected, bundle)
                density = marginal / bundle_tokens
                candidate = (density, marginal, item.id, bundle, bundle_tokens)
                if best is None:
                    best = candidate
                    continue
                if density > best[0] + 1e-12 or (
                    abs(density - best[0]) <= 1e-12
                    and (
                        marginal > best[1] + 1e-12
                        or (abs(marginal - best[1]) <= 1e-12 and item.id < best[2])
                    )
                ):
                    best = candidate

            if best is None or best[1] <= 1e-12:
                break

            _, marginal, item_id, bundle, bundle_tokens = best
            iterations += 1
            selected.update(bundle)
            for dependency_id in sorted(bundle - {item_id}):
                if dependency_id not in order:
                    order.append(dependency_id)
                reasons.setdefault(dependency_id, f"dependency of {item_id}")
            if item_id not in order:
                order.append(item_id)
            reasons[item_id] = "best feasible marginal objective gain per token"
            gains[item_id] = marginal
            added[item_id] = bundle_tokens
            rejected.pop(item_id, None)

        for item in problem.items:
            if item.id not in selected:
                rejected.setdefault(item.id, "non-positive marginal gain")

        return self._finalize(
            problem,
            order,
            reasons,
            rejected_reasons=rejected,
            gains=gains,
            added_tokens=added,
            metadata={"iterations": iterations},
        )
