"""An additive 0/1 knapsack baseline with exact dynamic programming."""

from __future__ import annotations

from contextopt.models import ContextFrame, SelectionProblem
from contextopt.policies.base import ContextPolicy, UnsupportedProblemError


class KnapsackPolicy(ContextPolicy):
    """Optimize independent item utility exactly under the token budget.

    This is intentionally an additive baseline. Graph dependencies and conflicts turn
    the problem into a different combinatorial program and are handled by the graph
    greedy and oracle policies instead.
    """

    name = "knapsack"

    def select(self, problem: SelectionProblem) -> ContextFrame:
        if any(item.dependencies or item.conflicts for item in problem.items):
            raise UnsupportedProblemError(
                "knapsack baseline supports independent items only"
            )

        selected, order, reasons = self._seed_mandatory(problem)
        used_by_mandatory = problem.tokens_for(selected)
        remaining_budget = problem.budget - used_by_mandatory
        candidates = [item for item in problem.items if item.id not in selected]

        # dp[cost] = (additive utility, selected ids)
        dp: list[tuple[float, tuple[str, ...]] | None] = [None] * (remaining_budget + 1)
        dp[0] = (0.0, ())
        for item in candidates:
            value = problem.standalone_utility(item)
            for cost in range(remaining_budget, item.tokens - 1, -1):
                previous = dp[cost - item.tokens]
                if previous is None:
                    continue
                proposal = (previous[0] + value, previous[1] + (item.id,))
                current = dp[cost]
                if (
                    current is None
                    or proposal[0] > current[0] + 1e-12
                    or (
                        current is not None
                        and abs(proposal[0] - current[0]) <= 1e-12
                        and proposal[1] < current[1]
                    )
                ):
                    dp[cost] = proposal

        best_cost = 0
        best_value = -1.0
        best_ids: tuple[str, ...] = ()
        for cost, state in enumerate(dp):
            if state is None:
                continue
            value, ids = state
            if value > best_value + 1e-12 or (
                abs(value - best_value) <= 1e-12
                and (cost < best_cost or (cost == best_cost and ids < best_ids))
            ):
                best_cost, best_value, best_ids = cost, value, ids

        order.extend(best_ids)
        for item_id in best_ids:
            reasons[item_id] = "selected by exact additive 0/1 knapsack DP"

        rejected = {
            item.id: "not present in the additive knapsack optimum"
            for item in candidates
            if item.id not in best_ids
        }
        return self._finalize(
            problem,
            order,
            reasons,
            rejected_reasons=rejected,
            metadata={
                "optimized_objective": "additive_standalone_utility",
                "dp_capacity": remaining_budget,
                "dp_best_value": best_value,
                "dp_selected_cost": best_cost,
            },
        )
