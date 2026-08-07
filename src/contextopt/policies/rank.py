"""Simple ranking baselines used to measure the value of set-aware selection."""

from __future__ import annotations

from contextopt.models import ContextFrame, SelectionProblem
from contextopt.policies.base import ContextPolicy, proposed_issues


class RankedPolicy(ContextPolicy):
    """Pack candidates in a fixed order, adding dependency closures atomically."""

    mode: str

    def __init__(self, mode: str) -> None:
        if mode not in {"topk", "density"}:
            raise ValueError(f"unsupported ranking mode: {mode}")
        self.mode = mode
        self.name = mode

    def _rank_key(self, problem: SelectionProblem, item_id: str) -> tuple[float, str]:
        item = problem.by_id[item_id]
        utility = problem.standalone_utility(item)
        score = utility if self.mode == "topk" else utility / item.tokens
        return (-score, item.id)

    def select(self, problem: SelectionProblem) -> ContextFrame:
        selected, order, reasons = self._seed_mandatory(problem)
        rejected: dict[str, str] = {}
        gains: dict[str, float] = {}
        added: dict[str, int] = {}

        ranked_ids = sorted(
            (item.id for item in problem.items if item.id not in selected),
            key=lambda item_id: self._rank_key(problem, item_id),
        )
        for item_id in ranked_ids:
            if item_id in selected:
                continue
            bundle = set(problem.incremental_bundle(item_id, selected))
            issues = proposed_issues(problem, selected, bundle)
            if issues:
                rejected[item_id] = issues[0]
                continue

            marginal = problem.marginal_gain(selected, bundle)
            bundle_tokens = problem.tokens_for(bundle)
            selected.update(bundle)
            for dependency_id in sorted(bundle - {item_id}):
                if dependency_id not in order:
                    order.append(dependency_id)
                reasons.setdefault(dependency_id, f"dependency of {item_id}")
            if item_id not in order:
                order.append(item_id)
            reasons[item_id] = (
                "highest standalone utility"
                if self.mode == "topk"
                else "highest standalone utility per token"
            )
            gains[item_id] = marginal
            added[item_id] = bundle_tokens

        return self._finalize(
            problem,
            order,
            reasons,
            rejected_reasons=rejected,
            gains=gains,
            added_tokens=added,
            metadata={"ranking_mode": self.mode},
        )


class TopKPolicy(RankedPolicy):
    def __init__(self) -> None:
        super().__init__("topk")


class DensityPolicy(RankedPolicy):
    def __init__(self) -> None:
        super().__init__("density")
