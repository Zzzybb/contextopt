import unittest

from contextopt.search import CandidatePatch, TestResult, select_candidate_batch


def _candidate(candidate_id: str, parent_id: str = "root") -> CandidatePatch:
    return CandidatePatch(
        id=candidate_id,
        parent_id=parent_id,
        hypothesis=f"try {candidate_id}",
        files={"solver.py": f"# {candidate_id}\n"},
    )


class SchedulerTests(unittest.TestCase):
    def test_fixed_policy_is_sorted_and_deduplicates_workspace_states(self) -> None:
        first = _candidate("b")
        duplicate = CandidatePatch(
            id="a",
            parent_id="root",
            hypothesis="same workspace",
            files=first.files,
        )
        selected = select_candidate_batch(
            (first, duplicate, _candidate("c")),
            {},
            limit=3,
            policy="fixed",
        )
        self.assertEqual([candidate.id for candidate in selected], ["b", "c"])

    def test_adaptive_policy_promotes_children_of_higher_quality_parent(self) -> None:
        root_a = _candidate("root-a")
        root_b = _candidate("root-b")
        child_a = _candidate("child-a", "root-a")
        child_b = _candidate("child-b", "root-b")
        observations = {
            root_a.workspace_fingerprint: TestResult(
                suite="visible", failed_tests=("case",)
            ),
            root_b.workspace_fingerprint: TestResult(
                suite="visible", passed_tests=("case",)
            ),
        }
        selected = select_candidate_batch(
            (root_a, root_b, child_a, child_b),
            observations,
            limit=1,
            policy="adaptive",
        )
        self.assertEqual([candidate.id for candidate in selected], ["child-b"])

    def test_scheduler_rejects_invalid_policy_and_limit(self) -> None:
        with self.assertRaises(ValueError):
            select_candidate_batch((), {}, limit=0)
        with self.assertRaises(ValueError):
            select_candidate_batch((), {}, limit=1, policy="unknown")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
