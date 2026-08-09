from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_model_matrix_demo import run


class ModelMatrixDemoTests(unittest.TestCase):
    def test_demo_builds_two_bundles_then_one_comparison_with_local_revision(
        self,
    ) -> None:
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch.dict(os.environ, {}, clear=False),
            patch(
                "scripts.run_model_matrix_demo.contextopt_main", return_value=0
            ) as contextopt_main,
        ):
            os.environ.pop("CONTEXTOPT_GIT_REVISION", None)
            self.assertEqual(run(Path(temp_dir)), 0)

            self.assertEqual(contextopt_main.call_count, 3)
            first_call = contextopt_main.call_args_list[0].args[0]
            second_call = contextopt_main.call_args_list[1].args[0]
            compare_call = contextopt_main.call_args_list[2].args[0]
            self.assertEqual(first_call[0], "agent-eval")
            self.assertEqual(second_call[0], "agent-eval")
            self.assertEqual(compare_call[0], "agent-eval-compare")
            self.assertIn("scripted-a", compare_call)
            self.assertIn("scripted-b", compare_call)
            self.assertEqual(
                os.environ["CONTEXTOPT_GIT_REVISION"],
                "provider-free-model-matrix-demo",
            )


if __name__ == "__main__":
    unittest.main()
