from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from contextopt.cli import main
from contextopt.evaluation import (
    AgentEvalConfig,
    AgentEvalModelMatrixReport,
    build_agent_eval_model_matrix,
    load_agent_eval_bundle,
    run_agent_evaluation,
)


def _write_bundle(
    root: Path,
    label: str,
    *,
    sandbox: str = "host",
) -> tuple[Path, Path]:
    report = run_agent_evaluation(
        AgentEvalConfig(
            fixtures=("two-sum",),
            strategies=("single_pass", "best_of_n"),
            repetitions=2,
        )
    )
    report_path = root / f"{label}.report.json"
    manifest_path = root / f"{label}.manifest.json"
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "1",
        "kind": "contextopt.agent-eval.manifest",
        "config": report.config.to_dict(),
        "provider": {
            "adapter": "scripted",
            "model": label,
            "planner_model": None,
            "solver_model": None,
            "reviewer_model": None,
            "base_url": "http://example.invalid/v1",
            "cancellation_url": None,
        },
        "runtime": {
            "temperature": 0.0,
            "timeout_seconds": 90.0,
            "max_retries": 2,
            "api_key_env": "CONTEXTOPT_API_KEY",
            "sandbox": sandbox,
            "container_image": "python:3.12-slim",
        },
        "transcript": {"mode": None, "directory": None, "layout": None},
        "repository_revision": "test-revision",
        "claim_boundary": "test manifest",
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report_path, manifest_path


class AgentEvalModelMatrixTests(unittest.TestCase):
    def test_matched_models_are_aggregated_and_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = [_write_bundle(root, "model-a"), _write_bundle(root, "model-b")]
            report = build_agent_eval_model_matrix(
                tuple(
                    load_agent_eval_bundle(label, report_path, manifest_path)
                    for label, (report_path, manifest_path) in zip(
                        ("model-a", "model-b"), paths, strict=True
                    )
                )
            )
            self.assertEqual(len(report.models), 2)
            self.assertEqual(report.generalization[0].model_count, 2)
            self.assertEqual(report.generalization[0].visible_positive_models, 2)
            self.assertEqual(
                AgentEvalModelMatrixReport.from_dict(report.to_dict()).to_dict(),
                report.to_dict(),
            )
            tampered = json.loads(json.dumps(report.to_dict()))
            tampered["protocol"]["fingerprint"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                AgentEvalModelMatrixReport.from_dict(tampered)

    def test_protocol_drift_is_rejected_before_pooling(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = _write_bundle(root, "model-a")
            second = _write_bundle(root, "model-b", sandbox="docker")
            with self.assertRaisesRegex(ValueError, "matched protocol fingerprint"):
                build_agent_eval_model_matrix(
                    (
                        load_agent_eval_bundle("model-a", *first),
                        load_agent_eval_bundle("model-b", *second),
                    )
                )

    def test_malformed_runtime_boundary_is_rejected_before_fingerprinting(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report_path, manifest_path = _write_bundle(root, "model-a")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            del manifest["runtime"]["timeout_seconds"]
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "runtime.timeout_seconds"):
                load_agent_eval_bundle("model-a", report_path, manifest_path)

    def test_cli_writes_machine_readable_and_dashboard_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = _write_bundle(root, "model-a")
            second = _write_bundle(root, "model-b")
            output = root / "matrix.json"
            markdown = root / "matrix.md"
            html = root / "matrix.html"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "agent-eval-compare",
                            "--bundle",
                            "model-a",
                            str(first[0]),
                            str(first[1]),
                            "--bundle",
                            "model-b",
                            str(second[0]),
                            str(second[1]),
                            "--output",
                            str(output),
                            "--markdown",
                            str(markdown),
                            "--html",
                            str(html),
                        ]
                    ),
                    0,
                )
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["models"]), 2)
            self.assertIn(
                "Cross-model consistency", markdown.read_text(encoding="utf-8")
            )
            self.assertIn("machine-readable analysis", html.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
