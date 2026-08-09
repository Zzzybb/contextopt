"""Exact exhaustive oracle for validating heuristic quality on small instances."""

from __future__ import annotations

from contextopt.models import ContextFrame, SelectionProblem
from contextopt.policies.base import ContextPolicy, UnsupportedProblemError


class ExactOraclePolicy(ContextPolicy):
    """Enumerate every subset and return the best feasible full-objective solution."""

    name = "oracle"

    def __init__(self, max_items: int = 20) -> None:
        self.max_items = max_items

    def select(self, problem: SelectionProblem) -> ContextFrame:
        item_count = len(problem.items)
        if item_count > self.max_items:
            raise UnsupportedProblemError(
                f"exact oracle is capped at {self.max_items} items, got {item_count}"
            )

        ids = tuple(item.id for item in problem.items)
        index = {item_id: position for position, item_id in enumerate(ids)}
        mandatory_mask = sum(1 << index[item_id] for item_id in problem.mandatory_ids)
        dependency_masks = [
            sum(1 << index[dependency] for dependency in item.dependencies)
            for item in problem.items
        ]
        conflict_masks = [
            sum(1 << index[conflict] for conflict in item.conflicts)
            for item in problem.items
        ]
        token_sums = [0] * (1 << item_count)

        best_mask = 0
        best_score = float("-inf")
        best_tokens = problem.budget + 1
        feasible_subsets = 0
        for mask in range(1 << item_count):
            if mask:
                low_bit = mask & -mask
                bit_index = low_bit.bit_length() - 1
                token_sums[mask] = (
                    token_sums[mask ^ low_bit] + problem.items[bit_index].tokens
                )
            if mask & mandatory_mask != mandatory_mask:
                continue
            used_tokens = token_sums[mask]
            if used_tokens > problem.budget:
                continue

            valid = True
            for item_index in range(item_count):
                bit = 1 << item_index
                if not mask & bit:
                    continue
                if dependency_masks[item_index] & mask != dependency_masks[item_index]:
                    valid = False
                    break
                if conflict_masks[item_index] & mask:
                    valid = False
                    break
            if not valid:
                continue

            feasible_subsets += 1
            selected_ids = tuple(ids[i] for i in range(item_count) if mask & (1 << i))
            score = problem.score(selected_ids)
            if score > best_score + 1e-12 or (
                abs(score - best_score) <= 1e-12
                and (
                    used_tokens < best_tokens
                    or (used_tokens == best_tokens and mask < best_mask)
                )
            ):
                best_mask, best_score, best_tokens = mask, score, used_tokens

        if best_score == float("-inf"):
            raise ValueError("selection problem has no feasible solution")

        selected_ids = tuple(ids[i] for i in range(item_count) if best_mask & (1 << i))
        reasons = {
            item_id: "member of the exact full-objective optimum"
            for item_id in selected_ids
        }
        rejected = {
            item_id: "not present in the exact full-objective optimum"
            for item_id in ids
            if item_id not in reasons
        }
        return self._finalize(
            problem,
            list(selected_ids),
            reasons,
            rejected_reasons=rejected,
            metadata={
                "search_space": 1 << item_count,
                "feasible_subsets": feasible_subsets,
                "max_items": self.max_items,
            },
        )
