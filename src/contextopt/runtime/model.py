"""Deterministic test model and an OpenAI-compatible HTTP adapter."""

from __future__ import annotations

import asyncio
import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from contextopt.runtime.errors import ModelError, RuntimeContractError
from contextopt.runtime.identity import model_request_idempotency_key, stable_hash
from contextopt.runtime.protocol import (
    AgentMessage,
    ModelRequest,
    ModelResponse,
    TokenUsage,
    ToolCall,
)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    return value


def _usage_from_mapping(value: Mapping[str, Any] | None) -> TokenUsage:
    data = value or {}
    prompt_details = data.get("prompt_tokens_details")
    completion_details = data.get("completion_tokens_details")
    cached = (
        int(prompt_details.get("cached_tokens", 0))
        if isinstance(prompt_details, dict)
        else int(data.get("cached_input_tokens", 0))
    )
    reasoning = (
        int(completion_details.get("reasoning_tokens", 0))
        if isinstance(completion_details, dict)
        else int(data.get("reasoning_tokens", 0))
    )
    return TokenUsage(
        input_tokens=int(data.get("prompt_tokens", data.get("input_tokens", 0))),
        output_tokens=int(data.get("completion_tokens", data.get("output_tokens", 0))),
        cached_input_tokens=cached,
        reasoning_tokens=reasoning,
    )


class ScriptedModel:
    """A conformance-test model whose steps can assert prior observations."""

    def __init__(
        self, steps: Sequence[Mapping[str, Any]], *, name: str = "scripted:v1"
    ):
        normalized: Any = json.loads(
            json.dumps(list(steps), ensure_ascii=False, sort_keys=True)
        )
        self._steps = tuple(
            dict(_mapping(step, f"steps[{index}]"))
            for index, step in enumerate(_sequence(normalized, "steps"))
        )
        self._position = 0
        self._name = name
        self._script_sha256 = stable_hash(self._steps)
        self.requests: list[ModelRequest] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def configuration(self) -> Mapping[str, Any]:
        """Non-secret identity persisted to reject mismatched resume attempts."""

        return {
            "adapter": "scripted",
            "name": self._name,
            "script_sha256": self._script_sha256,
        }

    @property
    def configuration_fingerprint(self) -> str:
        return stable_hash(self.configuration)

    def resume_from_turn(self, completed_turns: int) -> None:
        """Move the deterministic cursor to the next unconsumed response."""

        if completed_turns < 0 or completed_turns > len(self._steps):
            raise ValueError("completed_turns is outside the scripted response range")
        if self.requests:
            raise ValueError(
                "cannot move a scripted model after it has received requests"
            )
        self._position = completed_turns

    @classmethod
    def from_path(cls, path: str | Path) -> ScriptedModel:
        payload: Any = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            raw_steps = _sequence(payload.get("steps"), "steps")
            name = str(payload.get("name", "scripted:v1"))
        else:
            raw_steps = _sequence(payload, "script")
            name = "scripted:v1"
        steps = [
            _mapping(step, f"steps[{index}]") for index, step in enumerate(raw_steps)
        ]
        return cls(steps, name=name)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if self._position >= len(self._steps):
            raise ModelError(
                "scripted model exhausted before the run stopped",
                code="script_exhausted",
            )
        step = self._steps[self._position]
        self._position += 1
        self.requests.append(request)
        self._check_expectation(step.get("expect"), request)
        response = _mapping(step.get("response", {}), "script response")
        return self._parse_response(response, self._position)

    def _check_expectation(self, raw: Any, request: ModelRequest) -> None:
        if raw is None:
            return
        expect = _mapping(raw, "script expectation")
        if "turn" in expect and int(expect["turn"]) != request.turn:
            raise RuntimeContractError(
                f"script expected turn {expect['turn']}, got {request.turn}"
            )
        expected_tool = expect.get("last_tool")
        expected_call_id = expect.get("tool_call_id")
        contains = expect.get("observation_contains", [])
        needles: tuple[str, ...]
        if isinstance(contains, str):
            needles = (contains,)
        else:
            needles = tuple(
                str(item) for item in _sequence(contains, "observation_contains")
            )
        if expected_tool is None and expected_call_id is None and not needles:
            return
        if not request.messages or request.messages[-1].role != "tool":
            raise RuntimeContractError(
                "script expected the last message to be a tool observation"
            )
        observation = request.messages[-1]
        if expected_tool is not None and observation.tool_name != str(expected_tool):
            raise RuntimeContractError(
                f"script expected tool {expected_tool!r}, got {observation.tool_name!r}"
            )
        if expected_call_id is not None and observation.tool_call_id != str(
            expected_call_id
        ):
            raise RuntimeContractError(
                "scripted observation was associated with the wrong tool call"
            )
        missing = [needle for needle in needles if needle not in observation.content]
        if missing:
            raise RuntimeContractError(
                f"scripted observation is missing expected text: {missing!r}"
            )

    @staticmethod
    def _parse_response(response: Mapping[str, Any], position: int) -> ModelResponse:
        raw_calls = response.get("tool_calls", [])
        calls: list[ToolCall] = []
        for index, raw_call in enumerate(_sequence(raw_calls, "tool_calls")):
            call = _mapping(raw_call, f"tool_calls[{index}]")
            arguments = call.get("arguments", {})
            if "arguments_json" in call:
                arguments_json = str(call["arguments_json"])
            else:
                arguments_json = json.dumps(
                    arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
            calls.append(
                ToolCall(
                    id=str(call.get("id", f"script-{position}-{index}")),
                    name=str(call["name"]),
                    arguments_json=arguments_json,
                )
            )
        usage_value = response.get("usage")
        usage = _usage_from_mapping(
            _mapping(usage_value, "usage") if usage_value is not None else None
        )
        return ModelResponse(
            content=str(response.get("content", "")),
            tool_calls=tuple(calls),
            finish_reason=str(
                response.get("finish_reason", "tool_calls" if calls else "stop")
            ),
            usage=usage,
            response_id=(
                None
                if response.get("response_id") is None
                else str(response["response_id"])
            ),
        )


class OpenAICompatibleModel:
    """Minimal non-streaming Chat Completions adapter using the standard library."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 90.0,
        max_retries: int = 2,
        temperature: float | None = 0.0,
        idempotency_header: str | None = "Idempotency-Key",
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must use http or https")
        if not api_key:
            raise ValueError("api_key must not be empty")
        if not model:
            raise ValueError("model must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if idempotency_header is not None and not idempotency_header.strip():
            raise ValueError("idempotency_header must be non-empty when provided")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.temperature = temperature
        self.idempotency_header = idempotency_header
        self._cancellation_lock = threading.Lock()
        self._cancellation_events: dict[str, threading.Event] = {}
        self._active_responses: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return f"openai-compatible:{self.model}"

    @property
    def configuration(self) -> Mapping[str, Any]:
        """Return resume-relevant adapter settings without the API key."""

        return {
            "adapter": "openai-compatible-chat-completions",
            "base_url": self.base_url,
            "model": self.model,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "temperature": self.temperature,
            "idempotency_header": self.idempotency_header,
        }

    @property
    def configuration_fingerprint(self) -> str:
        return stable_hash(self.configuration)

    def resume_from_turn(self, completed_turns: int) -> None:
        """The HTTP adapter is stateless; validate only the restored counter."""

        if completed_turns < 0:
            raise ValueError("completed_turns must be non-negative")

    async def complete(self, request: ModelRequest) -> ModelResponse:
        request_key = self.request_idempotency_key(request)
        cancellation = threading.Event()
        with self._cancellation_lock:
            self._cancellation_events[request_key] = cancellation
        worker = asyncio.create_task(
            asyncio.to_thread(self._complete_sync, request, request_key, cancellation)
        )

        def cleanup(done: asyncio.Future[ModelResponse]) -> None:
            with self._cancellation_lock:
                if self._cancellation_events.get(request_key) is cancellation:
                    self._cancellation_events.pop(request_key, None)
                response = self._active_responses.pop(request_key, None)
            if response is not None:
                response.close()
            if not done.cancelled():
                # A caller may cancel the outer task while the shielded worker is
                # still running. Retrieve a late exception so it cannot become an
                # unhandled-task warning after the caller has moved on.
                done.exception()

        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            worker.add_done_callback(cleanup)
            raise
        except BaseException:
            cleanup(worker)
            raise
        else:
            cleanup(worker)

    async def request_cancellation(self, request: ModelRequest) -> str:
        """Interrupt the local HTTP transport for an active request when possible.

        Chat Completions has no standard remote-abort endpoint. The adapter can still
        close an active local ``urllib`` response, which unblocks the worker thread and
        prevents further retries. ``acknowledged`` therefore means local transport
        interruption, not proof that the provider stopped server-side generation.
        """

        request_key = self.request_idempotency_key(request)
        with self._cancellation_lock:
            cancellation = self._cancellation_events.get(request_key)
            response = self._active_responses.get(request_key)
        if cancellation is None:
            return "not_observed"
        cancellation.set()
        if response is not None:
            try:
                response.close()
            except OSError as exc:
                return f"failed:{type(exc).__name__}"
        return "acknowledged"

    def request_idempotency_key(self, request: ModelRequest) -> str:
        """Return the stable key used for provider retries and run recovery."""

        return model_request_idempotency_key(
            run_id=request.run_id,
            turn=request.turn,
            request_sha256=self._request_sha256(request),
        )

    def _complete_sync(
        self,
        request: ModelRequest,
        request_key: str,
        cancellation: threading.Event,
    ) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                self._message_payload(message) for message in request.messages
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": dict(tool.input_schema),
                    },
                }
                for tool in request.tools
            ],
            "tool_choice": "auto",
            "max_tokens": request.max_output_tokens,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "contextopt-runtime/0.2",
        }
        if self.idempotency_header is not None:
            headers[self.idempotency_header] = request_key
        endpoint = self.base_url + "/chat/completions"
        last_error: ModelError | None = None
        for attempt in range(self.max_retries + 1):
            if cancellation.is_set():
                raise ModelError(
                    "model request cancelled",
                    code="cancelled",
                    retryable=False,
                )
            http_request = urllib.request.Request(
                endpoint,
                data=encoded,
                headers=headers,
                method="POST",
            )
            try:
                response = urllib.request.urlopen(
                    http_request, timeout=self.timeout_seconds
                )
                with self._cancellation_lock:
                    self._active_responses[request_key] = response
                if cancellation.is_set():
                    response.close()
                    raise OSError("model request cancelled")
                try:
                    raw = response.read()
                finally:
                    with self._cancellation_lock:
                        if self._active_responses.get(request_key) is response:
                            self._active_responses.pop(request_key, None)
                    response.close()
                if cancellation.is_set():
                    raise OSError("model request cancelled")
                return self._parse_payload(raw)
            except urllib.error.HTTPError as exc:
                if cancellation.is_set():
                    raise ModelError(
                        "model request cancelled",
                        code="cancelled",
                        retryable=False,
                    ) from exc
                body = exc.read(2_000).decode("utf-8", errors="replace")
                retryable = exc.code in {408, 409, 429} or exc.code >= 500
                last_error = ModelError(
                    f"model endpoint returned HTTP {exc.code}: {body}",
                    code=f"http_{exc.code}",
                    retryable=retryable,
                )
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if cancellation.is_set():
                    raise ModelError(
                        "model request cancelled",
                        code="cancelled",
                        retryable=False,
                    ) from exc
                last_error = ModelError(
                    f"model endpoint request failed: {exc}",
                    code="network_error",
                    retryable=True,
                )
            if last_error is not None and (
                not last_error.retryable or attempt >= self.max_retries
            ):
                raise last_error
            time.sleep(min(0.25 * (2**attempt), 2.0))
        raise last_error or ModelError("model request failed", code="unknown")

    @staticmethod
    def _request_sha256(request: ModelRequest) -> str:
        return stable_hash(
            {
                "messages": [message.to_dict() for message in request.messages],
                "tools": [tool.to_dict() for tool in request.tools],
                "max_output_tokens": request.max_output_tokens,
            }
        )

    @staticmethod
    def _message_payload(message: AgentMessage) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.role == "assistant" and message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments_json,
                    },
                }
                for call in message.tool_calls
            ]
        if message.role == "tool":
            payload["tool_call_id"] = message.tool_call_id
            if message.tool_name:
                payload["name"] = message.tool_name
        return payload

    @staticmethod
    def _parse_payload(raw: bytes) -> ModelResponse:
        try:
            payload = _mapping(json.loads(raw), "model response")
            choices = _sequence(payload.get("choices"), "choices")
            choice = _mapping(choices[0], "choices[0]")
            message = _mapping(choice.get("message"), "message")
            raw_calls = message.get("tool_calls", [])
            calls: list[ToolCall] = []
            for index, raw_call in enumerate(_sequence(raw_calls, "tool_calls")):
                call = _mapping(raw_call, f"tool_calls[{index}]")
                function = _mapping(call.get("function"), "tool call function")
                calls.append(
                    ToolCall(
                        id=str(call["id"]),
                        name=str(function["name"]),
                        arguments_json=str(function.get("arguments", "{}")),
                    )
                )
            usage_value = payload.get("usage")
            usage = _usage_from_mapping(
                _mapping(usage_value, "usage") if usage_value is not None else None
            )
            content = message.get("content")
            return ModelResponse(
                content="" if content is None else str(content),
                tool_calls=tuple(calls),
                finish_reason=str(choice.get("finish_reason", "unknown")),
                usage=usage,
                response_id=(None if payload.get("id") is None else str(payload["id"])),
            )
        except (
            IndexError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            raise ModelError(
                f"invalid model response: {exc}", code="invalid_response"
            ) from exc
