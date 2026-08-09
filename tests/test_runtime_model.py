from __future__ import annotations

import asyncio
import io
import json
import threading
import unittest
import urllib.error
from collections import deque
from collections.abc import Mapping, Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Any
from unittest.mock import patch

from contextopt.runtime.errors import ModelError, RuntimeContractError
from contextopt.runtime.model import OpenAICompatibleModel, ScriptedModel
from contextopt.runtime.protocol import AgentMessage, ModelRequest, ToolDefinition


class _ServerState:
    def __init__(self, responses: Sequence[tuple[int, Mapping[str, Any]]]) -> None:
        self.responses = deque((status, dict(payload)) for status, payload in responses)
        self.requests: list[dict[str, Any]] = []
        self.lock = threading.Lock()


class _LocalModelServer:
    def __init__(self, responses: Sequence[tuple[int, Mapping[str, Any]]]) -> None:
        self.state = _ServerState(responses)
        state = self.state

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                content_length = int(self.headers.get("Content-Length", "0"))
                raw_request = self.rfile.read(content_length)
                request_payload = json.loads(raw_request.decode("utf-8"))
                with state.lock:
                    state.requests.append(
                        {
                            "path": self.path,
                            "authorization": self.headers.get("Authorization"),
                            "content_type": self.headers.get("Content-Type"),
                            "idempotency_key": self.headers.get("Idempotency-Key"),
                            "payload": request_payload,
                        }
                    )
                    if state.responses:
                        status, response_payload = state.responses.popleft()
                    else:
                        status = 500
                        response_payload = {
                            "error": {"message": "unexpected extra request"}
                        }

                encoded = json.dumps(response_payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, _format: str, *args: object) -> None:
                del args

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        port = int(self._server.server_address[1])
        self.base_url = f"http://127.0.0.1:{port}/v1"
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.01},
            name="contextopt-test-model-server",
            daemon=True,
        )

    @property
    def requests(self) -> tuple[dict[str, Any], ...]:
        with self.state.lock:
            return tuple(dict(request) for request in self.state.requests)

    def __enter__(self) -> _LocalModelServer:
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


class _BlockingResponse:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.closed = threading.Event()

    def read(self) -> bytes:
        self.started.set()
        if not self.closed.wait(timeout=5):
            raise TimeoutError("test response was not cancelled")
        raise OSError("test response closed")

    def close(self) -> None:
        self.closed.set()


class _StatusResponse:
    def __init__(self, status: int) -> None:
        self.status = status

    def getcode(self) -> int:
        return self.status

    def close(self) -> None:
        return None


class _CancelDuringBackoffEvent(threading.Event):
    def wait(self, timeout: float | None = None) -> bool:
        del timeout
        self.set()
        return True


def _model_request(*, turn: int = 1) -> ModelRequest:
    return ModelRequest(
        run_id="run-test",
        turn=turn,
        messages=(
            AgentMessage(role="system", content="You are a coding agent."),
            AgentMessage(role="user", content="Read src/app.py."),
        ),
        tools=(
            ToolDefinition(
                name="read_file",
                description="Read a UTF-8 file.",
                input_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            ),
        ),
        max_output_tokens=321,
    )


def _tool_call_response() -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-read",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path":"src/app.py"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 17,
            "completion_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 3},
            "completion_tokens_details": {"reasoning_tokens": 2},
        },
    }


class OpenAICompatibleModelTests(unittest.IsolatedAsyncioTestCase):
    def test_openai_adapter_rejects_non_finite_runtime_controls(self) -> None:
        base = {
            "base_url": "https://example.test/v1",
            "api_key": "key",
            "model": "unit-model",
        }
        for field, value, message in (
            ("timeout_seconds", float("nan"), "timeout_seconds"),
            ("timeout_seconds", float("inf"), "timeout_seconds"),
            ("temperature", float("nan"), "temperature"),
            ("temperature", -0.1, "temperature"),
        ):
            with (
                self.subTest(field=field, value=value),
                self.assertRaisesRegex(ValueError, message),
            ):
                OpenAICompatibleModel(**base, **{field: value})

        with self.assertRaisesRegex(ValueError, "max_retries"):
            OpenAICompatibleModel(**base, max_retries=True)

    async def test_parses_tool_call_and_sends_expected_request(self) -> None:
        secret = "unit-test-secret"
        with _LocalModelServer([(200, _tool_call_response())]) as server:
            model = OpenAICompatibleModel(
                base_url=server.base_url,
                api_key=secret,
                model="unit-model",
                timeout_seconds=5,
                max_retries=0,
            )
            response = await model.complete(_model_request())

        self.assertEqual(response.response_id, "chatcmpl-test")
        self.assertEqual(response.content, "")
        self.assertEqual(response.finish_reason, "tool_calls")
        self.assertEqual(len(response.tool_calls), 1)
        self.assertEqual(response.tool_calls[0].id, "call-read")
        self.assertEqual(response.tool_calls[0].name, "read_file")
        self.assertEqual(
            json.loads(response.tool_calls[0].arguments_json),
            {"path": "src/app.py"},
        )
        self.assertEqual(response.usage.input_tokens, 17)
        self.assertEqual(response.usage.output_tokens, 5)
        self.assertEqual(response.usage.cached_input_tokens, 3)
        self.assertEqual(response.usage.reasoning_tokens, 2)

        self.assertEqual(len(server.requests), 1)
        captured = server.requests[0]
        self.assertEqual(captured["path"], "/v1/chat/completions")
        self.assertEqual(captured["authorization"], f"Bearer {secret}")
        self.assertEqual(captured["content_type"], "application/json")
        expected_key = model.request_idempotency_key(_model_request())
        self.assertEqual(captured["idempotency_key"], expected_key)
        self.assertTrue(expected_key.startswith("contextopt-"))
        payload = captured["payload"]
        self.assertEqual(payload["model"], "unit-model")
        self.assertEqual(payload["max_tokens"], 321)
        self.assertEqual(payload["tool_choice"], "auto")
        self.assertEqual(payload["messages"][1]["content"], "Read src/app.py.")
        self.assertEqual(
            payload["tools"][0]["function"]["name"],
            "read_file",
        )

    async def test_retries_429_then_succeeds(self) -> None:
        responses = [
            (429, {"error": {"message": "rate limited"}}),
            (200, _tool_call_response()),
        ]
        with _LocalModelServer(responses) as server:
            model = OpenAICompatibleModel(
                base_url=server.base_url,
                api_key="retry-secret",
                model="unit-model",
                timeout_seconds=5,
                max_retries=2,
            )
            response = await model.complete(_model_request())

        self.assertEqual(response.response_id, "chatcmpl-test")
        self.assertEqual(len(server.requests), 2)
        self.assertEqual(
            server.requests[0]["idempotency_key"],
            server.requests[1]["idempotency_key"],
        )

    async def test_cancellation_interrupts_retry_backoff(self) -> None:
        model = OpenAICompatibleModel(
            base_url="https://example.test/v1",
            api_key="key",
            model="unit-model",
            max_retries=2,
        )
        cancellation = _CancelDuringBackoffEvent()
        error = urllib.error.HTTPError(
            "https://example.test/v1/chat/completions",
            429,
            "rate limited",
            hdrs=None,
            fp=io.BytesIO(b"rate limited"),
        )
        with (
            patch(
                "contextopt.runtime.model.urllib.request.urlopen",
                side_effect=error,
            ) as urlopen,
            self.assertRaises(ModelError) as raised,
        ):
            model._complete_sync(_model_request(), "request-key", cancellation)

        self.assertEqual(raised.exception.code, "cancelled")
        self.assertFalse(raised.exception.retryable)
        urlopen.assert_called_once()

    async def test_does_not_retry_401_or_expose_api_key(self) -> None:
        secret = "must-not-appear-in-errors"
        with _LocalModelServer(
            [(401, {"error": {"message": "invalid credentials"}})]
        ) as server:
            model = OpenAICompatibleModel(
                base_url=server.base_url,
                api_key=secret,
                model="unit-model",
                timeout_seconds=5,
                max_retries=3,
            )
            with self.assertRaises(ModelError) as raised:
                await model.complete(_model_request())

        self.assertNotIn(secret, json.dumps(model.configuration, sort_keys=True))
        self.assertEqual(len(model.configuration_fingerprint), 64)
        self.assertEqual(raised.exception.code, "http_401")
        self.assertFalse(raised.exception.retryable)
        self.assertNotIn(secret, str(raised.exception))
        self.assertEqual(
            await model.request_cancellation(_model_request()), "not_observed"
        )
        self.assertEqual(len(server.requests), 1)
        self.assertEqual(server.requests[0]["authorization"], f"Bearer {secret}")
        self.assertIsNotNone(server.requests[0]["idempotency_key"])

    async def test_idempotency_key_changes_when_logical_request_changes(self) -> None:
        with _LocalModelServer([(200, _tool_call_response())]) as server:
            model = OpenAICompatibleModel(
                base_url=server.base_url,
                api_key="key",
                model="unit-model",
                max_retries=0,
            )
            first = model.request_idempotency_key(_model_request(turn=1))
            second = model.request_idempotency_key(_model_request(turn=2))

        self.assertNotEqual(first, second)
        self.assertEqual(len(first), 75)
        self.assertEqual(len(second), 75)

    async def test_cancellation_interrupts_active_local_http_transport(self) -> None:
        response = _BlockingResponse()
        model = OpenAICompatibleModel(
            base_url="https://example.test/v1",
            api_key="key",
            model="unit-model",
        )
        request = _model_request()
        with patch(
            "contextopt.runtime.model.urllib.request.urlopen",
            return_value=response,
        ):
            task = asyncio.create_task(model.complete(request))
            self.assertTrue(await asyncio.to_thread(response.started.wait, 2))
            self.assertEqual(await model.request_cancellation(request), "acknowledged")
            with self.assertRaises(ModelError) as raised:
                await task
        self.assertEqual(raised.exception.code, "cancelled")

    async def test_cancellation_notifies_provider_specific_abort_endpoint(self) -> None:
        response = _BlockingResponse()
        model = OpenAICompatibleModel(
            base_url="https://example.test/v1",
            api_key="cancel-secret",
            model="unit-model",
            cancellation_url="https://example.test/v1/cancel",
        )
        request = _model_request()
        captured: list[urllib.request.Request] = []

        def urlopen(
            http_request: urllib.request.Request, *, timeout: float
        ) -> _BlockingResponse | _StatusResponse:
            del timeout
            if http_request.full_url.endswith("/cancel"):
                captured.append(http_request)
                return _StatusResponse(202)
            return response

        with patch(
            "contextopt.runtime.model.urllib.request.urlopen", side_effect=urlopen
        ):
            task = asyncio.create_task(model.complete(request))
            self.assertTrue(await asyncio.to_thread(response.started.wait, 2))
            self.assertEqual(await model.request_cancellation(request), "acknowledged")
            with self.assertRaises(ModelError) as raised:
                await task

        self.assertEqual(len(captured), 1)
        abort_request = captured[0]
        request_key = model.request_idempotency_key(request)
        self.assertEqual(
            abort_request.get_header("Authorization"), "Bearer cancel-secret"
        )
        self.assertEqual(abort_request.get_header("Idempotency-key"), request_key)
        self.assertEqual(
            json.loads(abort_request.data.decode("utf-8")),
            {"model": "unit-model", "request_idempotency_key": request_key},
        )
        self.assertEqual(
            model.configuration["cancellation_url"], "https://example.test/v1/cancel"
        )
        self.assertEqual(raised.exception.code, "cancelled")

    async def test_cancellation_endpoint_404_is_reported_as_unsupported(self) -> None:
        response = _BlockingResponse()
        model = OpenAICompatibleModel(
            base_url="https://example.test/v1",
            api_key="key",
            model="unit-model",
            cancellation_url="https://example.test/v1/cancel",
        )
        request = _model_request()
        error = urllib.error.HTTPError(
            "https://example.test/v1/cancel",
            404,
            "missing",
            hdrs=None,
            fp=io.BytesIO(b"missing"),
        )

        def urlopen(
            http_request: urllib.request.Request, *, timeout: float
        ) -> _BlockingResponse:
            del timeout
            if http_request.full_url.endswith("/cancel"):
                raise error
            return response

        with patch(
            "contextopt.runtime.model.urllib.request.urlopen", side_effect=urlopen
        ):
            task = asyncio.create_task(model.complete(request))
            self.assertTrue(await asyncio.to_thread(response.started.wait, 2))
            self.assertEqual(await model.request_cancellation(request), "unsupported")
            with self.assertRaises(ModelError):
                await task

    async def test_cancellation_reports_not_observed_without_active_request(
        self,
    ) -> None:
        model = OpenAICompatibleModel(
            base_url="https://example.test/v1",
            api_key="key",
            model="unit-model",
        )
        self.assertEqual(
            await model.request_cancellation(_model_request()), "not_observed"
        )

    async def test_transport_registry_survives_outer_task_cancellation(self) -> None:
        response = _BlockingResponse()
        model = OpenAICompatibleModel(
            base_url="https://example.test/v1",
            api_key="key",
            model="unit-model",
        )
        request = _model_request()
        with patch(
            "contextopt.runtime.model.urllib.request.urlopen",
            return_value=response,
        ):
            task = asyncio.create_task(model.complete(request))
            self.assertTrue(await asyncio.to_thread(response.started.wait, 2))
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(await model.request_cancellation(request), "acknowledged")
            self.assertTrue(await asyncio.to_thread(response.closed.wait, 2))


class ScriptedModelTests(unittest.IsolatedAsyncioTestCase):
    async def test_resume_cursor_skips_durable_responses(self) -> None:
        model = ScriptedModel(
            [
                {"response": {"content": "already durable"}},
                {
                    "expect": {"turn": 2},
                    "response": {"content": "continued after resume"},
                },
            ],
            name="scripted:resume-test",
        )
        fingerprint = model.configuration_fingerprint

        model.resume_from_turn(1)
        response = await model.complete(_model_request(turn=2))

        self.assertEqual(response.content, "continued after resume")
        self.assertEqual(model.configuration["adapter"], "scripted")
        self.assertEqual(model.configuration_fingerprint, fingerprint)

    async def test_resume_cursor_rejects_invalid_or_late_moves(self) -> None:
        model = ScriptedModel([{"response": {"content": "only"}}])
        with self.assertRaises(ValueError):
            model.resume_from_turn(2)

        await model.complete(_model_request())
        with self.assertRaises(ValueError):
            model.resume_from_turn(0)

    async def test_expectation_failure_raises_contract_error(self) -> None:
        model = ScriptedModel(
            [
                {
                    "expect": {"turn": 2},
                    "response": {"content": "should not be returned"},
                }
            ]
        )

        with self.assertRaisesRegex(
            RuntimeContractError,
            "script expected turn 2, got 1",
        ):
            await model.complete(_model_request(turn=1))


if __name__ == "__main__":
    unittest.main()
