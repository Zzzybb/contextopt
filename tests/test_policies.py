from __future__ import annotations

import unittest

from contextopt.models import ContextItem, ObjectiveWeights, SelectionProblem
from contextopt.policies import (
    DensityPolicy,
    ExactOraclePolicy,
    KnapsackPolicy,
    SubmodularGreedyPolicy,
    TopKPolicy,
    UnsupportedProblemError,
)


class PolicyTests(unittest.TestCase):
    def test_every_general_policy_respects_budget_and_mandatory_item(self) -> None:
        items = (
            ContextItem(id="goal", tokens=20, relevance=1.0, mandatory=True),
            ContextItem(id="a", tokens=70, relevance=0.9),
            ContextItem(id="b", tokens=60, relevance=0.8),
            ContextItem(id="c", tokens=50, relevance=0.7),
        )
        problem = SelectionProblem(items, budget=130)
        for policy in (
            TopKPolicy(),
            DensityPolicy(),
            KnapsackPolicy(),
            SubmodularGreedyPolicy(),
            ExactOraclePolicy(),
        ):
            with self.subTest(policy=policy.name):
                frame = policy.select(problem)
                self.assertLessEqual(frame.used_tokens, problem.budget)
                self.assertIn("goal", frame.selected_ids)
                self.assertTrue(problem.is_feasible(frame.selected_ids))
                self.assertEqual(len(frame.decisions), len(items))

    def test_knapsack_finds_additive_optimum(self) -> None:
        weights = ObjectiveWeights(
            relevance=1.0,
            importance=0.0,
            freshness=0.0,
            topic_coverage=0.0,
            duplicate_penalty=0.0,
        )
        items = (
            ContextItem(id="a", tokens=6, relevance=0.9),
            ContextItem(id="b", tokens=5, relevance=0.6),
            ContextItem(id="c", tokens=5, relevance=0.6),
        )
        frame = KnapsackPolicy().select(SelectionProblem(items, 10, weights))
        self.assertEqual(set(frame.selected_ids), {"b", "c"})

    def test_knapsack_refuses_graph_constraints(self) -> None:
        dependency = ContextItem(id="dep", tokens=10, relevance=0.4)
        item = ContextItem(
            id="item",
            tokens=10,
            relevance=0.8,
            dependencies=frozenset({"dep"}),
        )
        with self.assertRaises(UnsupportedProblemError):
            KnapsackPolicy().select(SelectionProblem((dependency, item), 30))

    def test_submodular_policy_adds_dependency_closure(self) -> None:
        dependency = ContextItem(
            id="interface", tokens=30, relevance=0.2, topics=frozenset({"api"})
        )
        implementation = ContextItem(
            id="implementation",
            tokens=40,
            relevance=1.0,
            importance=1.0,
            dependencies=frozenset({"interface"}),
            topics=frozenset({"service"}),
        )
        problem = SelectionProblem((dependency, implementation), 70)
        frame = SubmodularGreedyPolicy().select(problem)
        self.assertEqual(set(frame.selected_ids), {"interface", "implementation"})

    def test_oracle_obeys_conflicts(self) -> None:
        weights = ObjectiveWeights(topic_coverage=0.0, duplicate_penalty=0.0)
        first = ContextItem(
            id="first",
            tokens=40,
            relevance=1.0,
            importance=1.0,
            conflicts=frozenset({"second"}),
        )
        second = ContextItem(
            id="second",
            tokens=40,
            relevance=0.2,
            importance=0.1,
            conflicts=frozenset({"first"}),
        )
        frame = ExactOraclePolicy().select(
            SelectionProblem((first, second), 80, weights)
        )
        self.assertEqual(frame.selected_ids, ("first",))

    def test_oracle_rejects_large_search_space(self) -> None:
        items = tuple(
            ContextItem(id=f"item-{index}", tokens=10, relevance=0.5)
            for index in range(4)
        )
        with self.assertRaises(UnsupportedProblemError):
            ExactOraclePolicy(max_items=3).select(SelectionProblem(items, 40))


if __name__ == "__main__":
    unittest.main()
