from __future__ import annotations

import unittest

from contextopt.models import ContextItem, ObjectiveWeights, SelectionProblem


class ContextItemTests(unittest.TestCase):
    def test_rejects_invalid_token_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "tokens must be positive"):
            ContextItem(id="bad", tokens=0, relevance=0.5)

    def test_rejects_score_outside_unit_interval(self) -> None:
        with self.assertRaisesRegex(ValueError, "relevance"):
            ContextItem(id="bad", tokens=10, relevance=1.1)

    def test_round_trip_dict(self) -> None:
        original = ContextItem(
            id="service",
            tokens=120,
            relevance=0.8,
            importance=0.9,
            topics=frozenset({"auth", "service"}),
            dependencies=frozenset({"interface"}),
            source="src/service.py:10",
        )
        self.assertEqual(ContextItem.from_dict(original.to_dict()), original)


class SelectionProblemTests(unittest.TestCase):
    def setUp(self) -> None:
        self.goal = ContextItem(id="goal", tokens=40, relevance=1.0, mandatory=True)
        self.interface = ContextItem(
            id="interface",
            tokens=80,
            relevance=0.5,
            topics=frozenset({"api"}),
        )
        self.service = ContextItem(
            id="service",
            tokens=120,
            relevance=0.8,
            dependencies=frozenset({"interface"}),
            topics=frozenset({"auth"}),
        )

    def test_dependency_closure(self) -> None:
        problem = SelectionProblem((self.goal, self.interface, self.service), 300)
        self.assertEqual(
            problem.dependency_closure(("service",)),
            frozenset({"service", "interface"}),
        )

    def test_missing_dependency_is_infeasible(self) -> None:
        problem = SelectionProblem((self.goal, self.interface, self.service), 300)
        issues = problem.feasibility_issues(("goal", "service"))
        self.assertTrue(any("missing dependencies" in issue for issue in issues))

    def test_unknown_dependency_rejected_at_construction(self) -> None:
        broken = ContextItem(
            id="broken",
            tokens=10,
            relevance=0.5,
            dependencies=frozenset({"missing"}),
        )
        with self.assertRaisesRegex(ValueError, "unknown dependencies"):
            SelectionProblem((broken,), 100)

    def test_conflict_is_infeasible(self) -> None:
        old = ContextItem(
            id="old",
            tokens=50,
            relevance=0.7,
            conflicts=frozenset({"new"}),
        )
        new = ContextItem(id="new", tokens=50, relevance=0.8)
        problem = SelectionProblem((old, new), 100)
        self.assertFalse(problem.is_feasible(("old", "new")))

    def test_topic_coverage_rewards_diversity(self) -> None:
        weights = ObjectiveWeights(
            relevance=0.0,
            importance=0.0,
            freshness=0.0,
            topic_coverage=1.0,
            duplicate_penalty=0.0,
        )
        first = ContextItem(
            id="first",
            tokens=10,
            relevance=0.0,
            importance=1.0,
            topics=frozenset({"auth"}),
        )
        duplicate = ContextItem(
            id="duplicate",
            tokens=10,
            relevance=0.0,
            importance=1.0,
            topics=frozenset({"auth"}),
        )
        diverse = ContextItem(
            id="diverse",
            tokens=10,
            relevance=0.0,
            importance=1.0,
            topics=frozenset({"database"}),
        )
        problem = SelectionProblem((first, duplicate, diverse), 20, weights)
        self.assertGreater(
            problem.score(("first", "diverse")),
            problem.score(("first", "duplicate")),
        )


if __name__ == "__main__":
    unittest.main()
