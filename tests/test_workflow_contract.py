from __future__ import annotations

import unittest
from pathlib import Path

WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "workflows"
    / "real-agent-model-matrix.yml"
)


class RealProviderWorkflowContractTests(unittest.TestCase):
    def test_model_matrix_workflow_keeps_manual_matrix_and_compare_boundaries(
        self,
    ) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        required_fragments = (
            "workflow_dispatch:",
            "models:",
            "at least two entries",
            "duplicate model label",
            "repetitions must be an integer >= 3",
            "matrix: ${{ fromJSON(needs.prepare.outputs.matrix) }}",
            "CONTEXTOPT_API_KEY: ${{ secrets.CONTEXTOPT_API_KEY }}",
            "--record-transcript-dir",
            "agent-eval-compare",
            "if: always()",
            "if-no-files-found: warn",
            "concurrency:",
            "cancel-in-progress: false",
        )
        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, workflow)

        single_model_workflow = (WORKFLOW.parent / "real-agent-eval.yml").read_text(
            encoding="utf-8"
        )
        for fragment in ("if: always()", "if-no-files-found: warn"):
            with self.subTest(workflow="single-model", fragment=fragment):
                self.assertIn(fragment, single_model_workflow)
        self.assertIn("concurrency:", single_model_workflow)
        self.assertIn("cancel-in-progress: false", single_model_workflow)

    def test_model_matrix_workflow_preserves_per_model_and_final_artifacts(
        self,
    ) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        required_fragments = (
            "actions/upload-artifact@v4",
            "name: contextopt-model-${{ matrix.label }}",
            "actions/download-artifact@v4",
            "pattern: contextopt-model-*",
            "model-matrix.json",
            "model-matrix.md",
            "model-matrix.html",
            "contextopt-model-matrix-${{ github.run_id }}",
        )
        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, workflow)


if __name__ == "__main__":
    unittest.main()
