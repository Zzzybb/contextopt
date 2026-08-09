from __future__ import annotations

import io
import json
import os
import tempfile
import threading
import unittest
from collections.abc import Mapping
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import TracebackType
from typing import Any
from unittest.mock import patch

from contextopt.cli import main
from contextopt.evaluation import (
    AgentEvalCheckpoint,
    AgentEvalConfig,
    AgentEvalReport,
    build_agent_eval_comparisons,
    build_algorithm_fixtures,
    build_openai_model_factory,
    read_agent_evaluation_checkpoint,
    render_agent_evaluation_html,
    render_agent_evaluation_markdown,
    run_agent_evaluation,
    wilson_interval,
    write_agent_evaluation_checkpoint,
)


class _DynamicProviderServer:
    """Tiny local Chat Completions endpoint for the real-adapter harness path."""

    def __init__(self, fixture_files: Mapping[str, str]) -> None:
        self.fixture_files = dict(fixture_files)
        self.requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        state = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                with state._lock:
                    state.requests.append(
                        {
                            "path": self.path,
                            "authorization": self.headers.get("Authorization"),
                            "idempotency_key": self.headers.get("Idempotency-Key"),
                            "payload": payload,
                        }
                    )
                content = json.dumps(
                    {
                        "candidates": [
                            {
                                "id": "provider-smoke-good",
                                "parent_id": "root",
                                "hypothesis": (
                                    "apply the complete known-good fixture snapshot"
                                ),
                                "files": state.fixture_files,
                                "evidence": ["visible tests are the oracle"],
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
                response = json.dumps(
                    {
                        "id": "provider-smoke-response",
                        "choices": [
                            {
                                "message": {"role": "assistant", "content": content},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 12,
                            "completion_tokens": 8,
                        },
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

            def log_message(self, _format: str, *args: object) -> None:
                del args

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        port = int(self._server.server_address[1])
        self.base_url = f"http://127.0.0.1:{port}/v1"
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
        )

    def __enter__(self) -> _DynamicProviderServer:
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


class _OrchestratedProviderServer:
    """Local provider fixture that exercises all three orchestration roles."""

    def __init__(self, fixture_files: Mapping[str, str]) -> None:
        self.fixture_files = dict(fixture_files)
        self.requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        state = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                messages = payload.get("messages", [])
                system = (
                    str(messages[0].get("content", ""))
                    if messages and isinstance(messages[0], Mapping)
                    else ""
                )
                if "planning role" in system:
                    role = "planner"
                    content = json.dumps(
                        {
                            "goal": "repair the complete fixture snapshot",
                            "constraints": [
                                "preserve the public function signature",
                                "return a complete workspace snapshot",
                            ],
                            "hypotheses": [
                                "apply the standard algorithm and explicit edge cases"
                            ],
                            "test_focus": ["run every bundled visible test"],
                            "risks": ["do not weaken the test contract"],
                        },
                        ensure_ascii=False,
                    )
                elif "solver role" in system:
                    role = "solver"
                    content = json.dumps(
                        {
                            "candidates": [
                                {
                                    "id": "provider-smoke-good",
                                    "parent_id": "root",
                                    "hypothesis": (
                                        "apply the complete known-good fixture snapshot"
                                    ),
                                    "files": state.fixture_files,
                                    "evidence": ["visible tests are the oracle"],
                                }
                            ]
                        },
                        ensure_ascii=False,
                    )
                elif "reviewer role" in system:
                    role = "reviewer"
                    content = json.dumps(
                        {
                            "decision": "accept",
                            "candidate_id": "round-0-provider-smoke-good",
                            "confidence": 0.9,
                            "blocking_issues": [],
                            "required_checks": [],
                            "summary": "the visible oracle supports this decision",
                        },
                        ensure_ascii=False,
                    )
                else:
                    role = "unknown"
                    content = json.dumps(
                        {"error": "unexpected role system prompt"},
                        ensure_ascii=False,
                    )
                with state._lock:
                    state.requests.append(
                        {
                            "role": role,
                            "path": self.path,
                            "authorization": self.headers.get("Authorization"),
                            "idempotency_key": self.headers.get("Idempotency-Key"),
                            "payload": payload,
                        }
                    )
                response = json.dumps(
                    {
                        "id": f"provider-smoke-{role}-response",
                        "choices": [
                            {
                                "message": {"role": "assistant", "content": content},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 12,
                            "completion_tokens": 8,
                        },
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

            def log_message(self, _format: str, *args: object) -> None:
                del args

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        port = int(self._server.server_address[1])
        self.base_url = f"http://127.0.0.1:{port}/v1"
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
        )

    def __enter__(self) -> _OrchestratedProviderServer:
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


class AgentEvaluationFixtureTests(unittest.TestCase):
    def test_acm_and_math_fixtures_are_complete_snapshots(self) -> None:
        fixtures = build_algorithm_fixtures()

        self.assertEqual(
            [fixture.fixture_id for fixture in fixtures],
            ["two-sum", "extended-gcd", "merge-intervals", "modular-inverse"],
        )
        self.assertEqual(
            {fixture.category for fixture in fixtures}, {"acm-algorithm", "mathematics"}
        )
        for fixture in fixtures:
            self.assertEqual(set(fixture.root_files), set(fixture.bad_files))
            self.assertEqual(set(fixture.root_files), set(fixture.good_files))
            self.assertNotEqual(fixture.bad_files, fixture.good_files)
            self.assertIn("test_", " ".join(fixture.root_files))


class AgentEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = run_agent_evaluation(
            AgentEvalConfig(fixtures=("two-sum",), repetitions=1)
        )

    def test_strategy_matrix_has_visible_oracle_signal(self) -> None:
        report = self.report
        self.assertEqual(len(report.runs), 3)
        by_strategy = {summary.strategy: summary for summary in report.summaries}
        self.assertEqual(by_strategy["single_pass"].success_count, 0)
        self.assertEqual(by_strategy["best_of_n"].success_count, 1)
        self.assertEqual(by_strategy["orchestrated"].success_count, 1)
        self.assertEqual(by_strategy["best_of_n"].hidden_success_rate, 1.0)
        self.assertEqual(by_strategy["orchestrated"].hidden_success_rate, 1.0)
        self.assertEqual(by_strategy["single_pass"].hidden_run_count, 0)
        self.assertGreater(
            by_strategy["orchestrated"].mean_model_calls,
            by_strategy["best_of_n"].mean_model_calls,
        )
        self.assertTrue(all(run.duration_ms >= 0 for run in report.runs))
        self.assertTrue(
            all(summary.mean_duration_ms >= 0 for summary in report.summaries)
        )
        self.assertIn(
            "does not measure general model capability", report.claim_boundary
        )

    def test_statistical_summary_is_paired_and_bounded(self) -> None:
        comparisons = build_agent_eval_comparisons(self.report)
        by_strategy = {comparison.strategy: comparison for comparison in comparisons}
        self.assertEqual(by_strategy["best_of_n"].wins, 1)
        self.assertEqual(by_strategy["best_of_n"].losses, 0)
        self.assertEqual(by_strategy["best_of_n"].ties, 0)
        self.assertEqual(by_strategy["best_of_n"].visible_delta, 1.0)
        self.assertIsNone(by_strategy["best_of_n"].hidden_delta)
        self.assertEqual(by_strategy["best_of_n"].stddev_test_call_delta, 0.0)
        self.assertEqual(by_strategy["best_of_n"].stddev_token_delta, 0.0)
        self.assertTrue(
            all(comparison.stddev_duration_delta >= 0 for comparison in comparisons)
        )
        low, high = wilson_interval(1, 1)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)
        self.assertLess(low, high)
        markdown = render_agent_evaluation_markdown(self.report)
        self.assertIn("Visible 95% CI", markdown)
        self.assertIn("Paired comparisons", markdown)
        self.assertIn("Mean tokens Δ (stdev)", markdown)
        self.assertIn("Mean duration Δ ms (stdev)", markdown)

    def test_hidden_tests_can_be_disabled_without_leaking_the_grader(self) -> None:
        report = run_agent_evaluation(
            AgentEvalConfig(
                fixtures=("two-sum",),
                strategies=("best_of_n",),
                include_hidden_tests=False,
            )
        )
        self.assertEqual(report.summaries[0].hidden_run_count, 0)
        self.assertEqual(report.runs[0].hidden_test_calls, 0)
        self.assertNotIn("hidden_files", report.fixtures[0].to_dict())

    def test_docker_grader_identity_is_checkpointed_and_uses_container_python(
        self,
    ) -> None:
        config = AgentEvalConfig(
            fixtures=("two-sum",),
            strategies=("single_pass",),
            planner_model="planner-v1",
            solver_model="solver-v1",
            reviewer_model="reviewer-v1",
            sandbox="docker",
            container_image="python:3.12-slim@sha256:example",
        )
        self.assertEqual(AgentEvalConfig.from_dict(config.to_dict()), config)
        execution = build_algorithm_fixtures()[0].execution_config_for(
            sandbox=config.sandbox,
            container_image=config.container_image,
        )
        self.assertEqual(execution.command[0], "python")
        self.assertEqual(execution.sandbox, "docker")
        self.assertEqual(execution.container_image, config.container_image)

    def test_openai_factory_is_fresh_and_role_specific_without_calling_network(
        self,
    ) -> None:
        factory = build_openai_model_factory(
            base_url="https://example.invalid/v1",
            api_key="test-key",
            model="solver-model",
            planner_model="planner-model",
            reviewer_model="reviewer-model",
        )
        models = factory(build_algorithm_fixtures()[0], "orchestrated")
        self.assertEqual(
            [model.name for model in models],
            [
                "openai-compatible:planner-model",
                "openai-compatible:solver-model",
                "openai-compatible:reviewer-model",
            ],
        )
        self.assertIsNot(
            models[0], factory(build_algorithm_fixtures()[0], "orchestrated")[0]
        )

    def test_real_adapter_matrix_path_works_with_local_compatible_provider(
        self,
    ) -> None:
        fixtures = build_algorithm_fixtures()
        for fixture in fixtures:
            with (
                self.subTest(fixture=fixture.fixture_id),
                _DynamicProviderServer(fixture.good_files) as server,
            ):
                factory = build_openai_model_factory(
                    base_url=server.base_url,
                    api_key="provider-smoke-secret",
                    model="provider-smoke-model",
                )
                report = run_agent_evaluation(
                    AgentEvalConfig(
                        strategies=("single_pass",),
                        fixtures=(fixture.fixture_id,),
                        repetitions=1,
                        include_hidden_tests=True,
                        model_adapter="openai-compatible",
                    ),
                    model_factory=factory,
                )
                self.assertEqual(report.runs[0].status, "accepted")
                self.assertTrue(report.runs[0].success)
                self.assertTrue(report.runs[0].hidden_success)
                self.assertEqual(len(server.requests), 1)
                request = server.requests[0]
                self.assertEqual(request["path"], "/v1/chat/completions")
                self.assertEqual(
                    request["authorization"], "Bearer provider-smoke-secret"
                )
                self.assertTrue(request["idempotency_key"].startswith("contextopt-"))
                self.assertEqual(request["payload"]["model"], "provider-smoke-model")

    def test_real_adapter_orchestrated_matrix_path_works_with_local_provider(
        self,
    ) -> None:
        fixtures = build_algorithm_fixtures()
        expected_roles = ("planner", "solver", "reviewer")
        expected_models = (
            "provider-smoke-planner",
            "provider-smoke-solver",
            "provider-smoke-reviewer",
        )
        for fixture in fixtures:
            with (
                self.subTest(fixture=fixture.fixture_id),
                _OrchestratedProviderServer(fixture.good_files) as server,
            ):
                factory = build_openai_model_factory(
                    base_url=server.base_url,
                    api_key="provider-smoke-secret",
                    model="provider-smoke-solver",
                    planner_model=expected_models[0],
                    solver_model=expected_models[1],
                    reviewer_model=expected_models[2],
                )
                report = run_agent_evaluation(
                    AgentEvalConfig(
                        strategies=("orchestrated",),
                        fixtures=(fixture.fixture_id,),
                        repetitions=1,
                        include_hidden_tests=True,
                        model_adapter="openai-compatible",
                    ),
                    model_factory=factory,
                )
                self.assertEqual(report.runs[0].status, "accepted")
                self.assertTrue(report.runs[0].success)
                self.assertTrue(report.runs[0].hidden_success)
                self.assertEqual(
                    [request["role"] for request in server.requests],
                    list(expected_roles),
                )
                self.assertEqual(
                    [request["payload"]["model"] for request in server.requests],
                    list(expected_models),
                )
                self.assertTrue(
                    all(
                        request["path"] == "/v1/chat/completions"
                        for request in server.requests
                    )
                )
                self.assertTrue(
                    all(
                        request["authorization"] == "Bearer provider-smoke-secret"
                        for request in server.requests
                    )
                )
                keys = [request["idempotency_key"] for request in server.requests]
                self.assertEqual(len(keys), len(set(keys)))
                self.assertTrue(all(key.startswith("contextopt-") for key in keys))

    def test_provider_matrix_transcript_round_trip_is_cell_and_repetition_scoped(
        self,
    ) -> None:
        fixture = build_algorithm_fixtures()[0]
        config = AgentEvalConfig(
            strategies=("single_pass",),
            fixtures=(fixture.fixture_id,),
            repetitions=2,
            include_hidden_tests=True,
            model_adapter="openai-compatible",
        )
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            _DynamicProviderServer(fixture.good_files) as server,
        ):
            record_dir = Path(temp_dir) / "recorded"
            recording_factory = build_openai_model_factory(
                base_url=server.base_url,
                api_key="provider-smoke-secret",
                model="provider-smoke-model",
                record_transcript_dir=record_dir,
            )
            recorded = run_agent_evaluation(config, model_factory=recording_factory)
            self.assertTrue(all(run.success for run in recorded.runs))
            self.assertEqual(len(server.requests), 2)
            for repetition in range(2):
                cassette = (
                    record_dir
                    / fixture.fixture_id
                    / "single_pass"
                    / f"repetition-{repetition}"
                    / "solver.jsonl"
                )
                self.assertTrue(cassette.exists())
                self.assertEqual(
                    len(cassette.read_text(encoding="utf-8").splitlines()), 1
                )

            replay_factory = build_openai_model_factory(
                base_url=None,
                api_key=None,
                model=None,
                replay_transcript_dir=record_dir,
            )
            replayed = run_agent_evaluation(
                AgentEvalConfig(
                    strategies=("single_pass",),
                    fixtures=(fixture.fixture_id,),
                    repetitions=2,
                    include_hidden_tests=True,
                    model_adapter="replay",
                ),
                model_factory=replay_factory,
            )
            self.assertTrue(all(run.success for run in replayed.runs))
            self.assertIn(
                "replays a previously recorded provider transcript",
                replayed.claim_boundary,
            )
            recorded_payload = recorded.to_dict()
            replayed_payload = replayed.to_dict()
            for payload in (recorded_payload, replayed_payload):
                for run in payload["runs"]:
                    run.pop("duration_ms", None)
            self.assertEqual(replayed_payload["runs"], recorded_payload["runs"])
            self.assertEqual(len(server.requests), 2)

    def test_cli_agent_eval_can_record_and_replay_matrix_transcripts(self) -> None:
        fixture = build_algorithm_fixtures()[0]
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            _DynamicProviderServer(fixture.good_files) as server,
            patch.dict(
                os.environ, {"CONTEXTOPT_TEST_PROVIDER_KEY": "provider-smoke-secret"}
            ),
        ):
            root = Path(temp_dir)
            record_dir = root / "recorded"
            record_manifest = root / "record.manifest.json"
            replay_manifest = root / "replay.manifest.json"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "agent-eval",
                            "--strategies",
                            "single_pass",
                            "--fixtures",
                            fixture.fixture_id,
                            "--repetitions",
                            "2",
                            "--model",
                            "provider-smoke-model",
                            "--base-url",
                            server.base_url,
                            "--api-key-env",
                            "CONTEXTOPT_TEST_PROVIDER_KEY",
                            "--record-transcript-dir",
                            str(record_dir),
                            "--manifest",
                            str(record_manifest),
                        ]
                    ),
                    0,
                )
                self.assertEqual(
                    main(
                        [
                            "agent-eval",
                            "--strategies",
                            "single_pass",
                            "--fixtures",
                            fixture.fixture_id,
                            "--repetitions",
                            "2",
                            "--replay-transcript-dir",
                            str(record_dir),
                            "--manifest",
                            str(replay_manifest),
                        ]
                    ),
                    0,
                )
            self.assertEqual(len(server.requests), 2)
            self.assertEqual(
                json.loads(record_manifest.read_text(encoding="utf-8"))["transcript"][
                    "mode"
                ],
                "record",
            )
            replay_manifest_payload = json.loads(
                replay_manifest.read_text(encoding="utf-8")
            )
            self.assertEqual(replay_manifest_payload["provider"]["adapter"], "replay")
            self.assertEqual(replay_manifest_payload["transcript"]["mode"], "replay")
            self.assertIsNone(replay_manifest_payload["provider"]["model"])

    def test_report_roundtrip_and_tamper_detection(self) -> None:
        payload = self.report.to_dict()
        self.assertEqual(AgentEvalReport.from_dict(payload).to_dict(), payload)

        tampered = json.loads(json.dumps(payload))
        tampered["summaries"][1]["success_count"] = 0
        with self.assertRaisesRegex(ValueError, "success_rate|summary metrics"):
            AgentEvalReport.from_dict(tampered)

    def test_matrix_checkpoint_is_atomic_and_resume_reuses_completed_cells(
        self,
    ) -> None:
        config = AgentEvalConfig(
            fixtures=("two-sum",), strategies=("best_of_n",), repetitions=2
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "agent-eval.checkpoint.json"
            complete = run_agent_evaluation(config, checkpoint_path=checkpoint_path)
            checkpoint = read_agent_evaluation_checkpoint(checkpoint_path)
            self.assertEqual(len(checkpoint.runs), 2)
            self.assertEqual(checkpoint.config.to_dict(), config.to_dict())
            write_agent_evaluation_checkpoint(
                AgentEvalCheckpoint(
                    config=checkpoint.config,
                    fixture_ids=checkpoint.fixture_ids,
                    runs=checkpoint.runs[:1],
                ),
                checkpoint_path,
            )
            resumed = run_agent_evaluation(
                config, checkpoint_path=checkpoint_path, resume=True
            )
            complete_payload = complete.to_dict()
            resumed_payload = resumed.to_dict()
            for payload in (complete_payload, resumed_payload):
                for run in payload["runs"]:
                    run.pop("duration_ms", None)
                for summary in payload["summaries"]:
                    summary.pop("mean_duration_ms", None)
            self.assertEqual(resumed_payload, complete_payload)

    def test_configuration_rejects_duplicate_or_unknown_strategy(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            AgentEvalConfig(strategies=("single_pass", "single_pass"))
        with self.assertRaisesRegex(ValueError, "unknown"):
            AgentEvalConfig(strategies=("made_up",))  # type: ignore[arg-type]

    def test_html_dashboard_contains_summary_bars_and_durable_payload(self) -> None:
        html = render_agent_evaluation_html(self.report)
        self.assertIn("visible success", html)
        self.assertIn("hidden success", html)
        self.assertIn("<table>", html)
        self.assertIn("orchestrated", html)
        self.assertIn("durable evaluation JSON", html)
        self.assertIn("visible 95% CI", html)
        self.assertIn("Paired comparisons", html)

    def test_cli_writes_json_markdown_and_html_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "agent-eval.json"
            markdown = root / "agent-eval.md"
            html = root / "agent-eval.html"
            checkpoint = root / "agent-eval.checkpoint.json"
            manifest = root / "agent-eval.manifest.json"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "agent-eval",
                        "--fixtures",
                        "two-sum",
                        "--output",
                        str(output),
                        "--markdown",
                        str(markdown),
                        "--html",
                        str(html),
                        "--checkpoint",
                        str(checkpoint),
                        "--manifest",
                        str(manifest),
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertIn("orchestrated", stdout.getvalue())
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["config"]["fixtures"], ["two-sum"])
            self.assertEqual(len(report["runs"]), 3)
            rendered = markdown.read_text(encoding="utf-8")
            self.assertIn("Claim boundary", rendered)
            self.assertIn("best_of_n", rendered)
            self.assertIn("<table>", html.read_text(encoding="utf-8"))
            self.assertEqual(len(json.loads(checkpoint.read_text())["runs"]), 3)
            manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(manifest_payload["provider"]["adapter"], "scripted")
            self.assertIsNone(manifest_payload["provider"]["model"])
            self.assertEqual(
                manifest_payload["runtime"]["api_key_env"], "CONTEXTOPT_API_KEY"
            )
            self.assertNotIn('"api_key":', manifest.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
