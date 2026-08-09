"""Auditable provider request/response recording and deterministic replay.

The runtime event log proves what the agent did, but a provider response is still an
external boundary.  This module provides an opt-in JSONL cassette for that boundary:
successful model calls can be recorded without credentials, then replayed offline with
strict request-hash matching.  Replay is intentionally sequential and at-least-once;
it is a debugging and evaluation aid, not a claim that a remote provider was called
exactly once.
"""

from __future__ import annotations

import inspect
import json
import os
from collections.abc import Mapping
from pathlib import Path
from threading import RLock
from typing import Any

from contextopt.runtime.errors import ModelError
from contextopt.runtime.identity import stable_hash
from contextopt.runtime.protocol import ModelClient, ModelRequest, ModelResponse

MODEL_TRANSCRIPT_SCHEMA_VERSION = "1"


def model_request_fingerprint(request: ModelRequest) -> str:
    """Return the strict identity used to match a replayed model request."""

    return stable_hash(request.to_dict())


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _record_fingerprint(
    *,
    sequence: int,
    previous_sha256: str | None,
    request_sha256: str,
    response_sha256: str,
    model_name: str,
) -> str:
    return stable_hash(
        {
            "schema_version": MODEL_TRANSCRIPT_SCHEMA_VERSION,
            "sequence": sequence,
            "previous_sha256": previous_sha256,
            "request_sha256": request_sha256,
            "response_sha256": response_sha256,
            "model_name": model_name,
        }
    )


class ModelTranscriptRecord:
    """One validated provider request/response pair."""

    __slots__ = (
        "model_name",
        "previous_sha256",
        "record_sha256",
        "request",
        "request_sha256",
        "response",
        "response_sha256",
        "sequence",
    )

    def __init__(
        self,
        *,
        sequence: int,
        request: ModelRequest,
        request_sha256: str,
        response: ModelResponse,
        model_name: str,
        previous_sha256: str | None = None,
        response_sha256: str | None = None,
        record_sha256: str | None = None,
    ) -> None:
        self.sequence = _integer(sequence, "transcript sequence")
        self.request = request
        self.request_sha256 = _string(request_sha256, "transcript request_sha256")
        self.response = response
        self.model_name = _string(model_name, "transcript model_name")
        if self.sequence == 0 and previous_sha256 is not None:
            raise ValueError("first transcript record cannot have previous_sha256")
        if self.sequence > 0 and (
            not isinstance(previous_sha256, str) or not previous_sha256
        ):
            raise ValueError("non-first transcript record requires previous_sha256")
        self.previous_sha256 = previous_sha256
        expected = model_request_fingerprint(request)
        if self.request_sha256 != expected:
            raise ValueError("transcript request_sha256 does not match request")
        expected_response = stable_hash(response.to_dict())
        if response_sha256 is not None and response_sha256 != expected_response:
            raise ValueError("transcript response_sha256 does not match response")
        self.response_sha256 = expected_response
        expected_record = _record_fingerprint(
            sequence=self.sequence,
            previous_sha256=self.previous_sha256,
            request_sha256=self.request_sha256,
            response_sha256=self.response_sha256,
            model_name=self.model_name,
        )
        if record_sha256 is not None and record_sha256 != expected_record:
            raise ValueError("transcript record_sha256 does not match record")
        self.record_sha256 = expected_record

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MODEL_TRANSCRIPT_SCHEMA_VERSION,
            "sequence": self.sequence,
            "previous_sha256": self.previous_sha256,
            "record_sha256": self.record_sha256,
            "request_sha256": self.request_sha256,
            "request": self.request.to_dict(),
            "response_sha256": self.response_sha256,
            "response": self.response.to_dict(),
            "model_name": self.model_name,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelTranscriptRecord:
        value = _object(data, "transcript record")
        required = {
            "schema_version",
            "sequence",
            "previous_sha256",
            "record_sha256",
            "request_sha256",
            "request",
            "response_sha256",
            "response",
            "model_name",
        }
        if set(value) != required:
            raise ValueError("transcript record fields do not match the schema")
        schema_version = _string(value["schema_version"], "transcript schema_version")
        if schema_version != MODEL_TRANSCRIPT_SCHEMA_VERSION:
            raise ValueError(f"unsupported model transcript schema: {schema_version}")
        return cls(
            sequence=_integer(value["sequence"], "transcript sequence"),
            request=ModelRequest.from_dict(
                _object(value["request"], "transcript request")
            ),
            request_sha256=_string(
                value["request_sha256"], "transcript request_sha256"
            ),
            response=ModelResponse.from_dict(
                _object(value["response"], "transcript response")
            ),
            model_name=_string(value["model_name"], "transcript model_name"),
            previous_sha256=(
                None
                if value["previous_sha256"] is None
                else _string(value["previous_sha256"], "transcript previous_sha256")
            ),
            response_sha256=_string(
                value["response_sha256"], "transcript response_sha256"
            ),
            record_sha256=_string(value["record_sha256"], "transcript record_sha256"),
        )


class ModelTranscript:
    """A strict append-only JSONL cassette for model calls."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve(strict=False)
        self._lock = RLock()
        self._records = self._read()

    @property
    def records(self) -> tuple[ModelTranscriptRecord, ...]:
        with self._lock:
            return tuple(self._records)

    @property
    def fingerprint(self) -> str:
        with self._lock:
            return stable_hash([record.to_dict() for record in self._records])

    def append(
        self,
        request: ModelRequest,
        response: ModelResponse,
        *,
        model_name: str,
    ) -> ModelTranscriptRecord:
        with self._lock:
            record = ModelTranscriptRecord(
                sequence=len(self._records),
                request=request,
                request_sha256=model_request_fingerprint(request),
                response=response,
                model_name=model_name,
                previous_sha256=(
                    None if not self._records else self._records[-1].record_sha256
                ),
            )
            self.path.parent.mkdir(parents=True, exist_ok=True)
            encoded = (
                json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            )
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            self._records.append(record)
            return record

    def response_at(self, position: int, request: ModelRequest) -> ModelResponse:
        """Match and return one record at a caller-owned cursor position."""

        with self._lock:
            if position < 0 or position >= len(self._records):
                raise ModelError(
                    "model transcript is exhausted",
                    code="transcript_exhausted",
                    retryable=False,
                )
            record = self._records[position]
            actual = model_request_fingerprint(request)
            if record.request_sha256 != actual:
                raise ModelError(
                    "model transcript request mismatch at sequence "
                    f"{record.sequence}: expected {record.request_sha256}, "
                    f"got {actual}",
                    code="transcript_mismatch",
                    retryable=False,
                )
            return record.response

    def _read(self) -> list[ModelTranscriptRecord]:
        if not self.path.exists():
            return []
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ValueError(
                f"cannot read model transcript {self.path}: {exc}"
            ) from exc
        records: list[ModelTranscriptRecord] = []
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                raise ValueError(f"model transcript line {line_number} is blank")
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"model transcript line {line_number} is not valid JSON"
                ) from exc
            record = ModelTranscriptRecord.from_dict(
                _object(raw, f"model transcript line {line_number}")
            )
            if record.sequence != len(records):
                raise ValueError(
                    f"model transcript sequence is not contiguous at line {line_number}"
                )
            expected_previous = None if not records else records[-1].record_sha256
            if record.previous_sha256 != expected_previous:
                raise ValueError(
                    f"model transcript chain is broken at line {line_number}"
                )
            records.append(record)
        return records


class RecordingModel:
    """Wrap a model and durably record every successful response."""

    def __init__(self, model: ModelClient, path: str | Path) -> None:
        self.model = model
        self.transcript = ModelTranscript(path)

    @property
    def name(self) -> str:
        return f"recording:{self.model.name}"

    @property
    def configuration_fingerprint(self) -> str:
        return stable_hash(
            {
                "adapter": "recording-v1",
                "inner_model": self.model.name,
                "inner_fingerprint": self.model.configuration_fingerprint,
                "transcript": str(self.transcript.path),
            }
        )

    def resume_from_turn(self, completed_turns: int) -> None:
        self.model.resume_from_turn(completed_turns)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        response = await self.model.complete(request)
        try:
            self.transcript.append(request, response, model_name=self.model.name)
        except (OSError, ValueError) as exc:
            raise ModelError(
                f"provider response was not recorded: {exc}",
                code="transcript_write_failed",
                retryable=False,
            ) from exc
        return response

    async def request_cancellation(self, request: ModelRequest) -> str:
        method = getattr(self.model, "request_cancellation", None)
        if method is None:
            return "not_supported"
        result = method(request)
        if inspect.isawaitable(result):
            return str(await result)
        return str(result)


class ReplayModel:
    """Replay a recorded model trajectory with strict request matching."""

    def __init__(self, path: str | Path, *, name: str | None = None) -> None:
        self.transcript = ModelTranscript(path)
        self._position = 0
        self._name = name or f"replay:{self.transcript.fingerprint[:16]}"

    @property
    def name(self) -> str:
        return self._name

    @property
    def configuration_fingerprint(self) -> str:
        return stable_hash(
            {
                "adapter": "replay-v1",
                "name": self._name,
                "transcript": self.transcript.fingerprint,
            }
        )

    def resume_from_turn(self, completed_turns: int) -> None:
        if completed_turns < 0 or completed_turns > len(self.transcript.records):
            raise ValueError("completed_turns is outside the transcript range")
        self._position = completed_turns

    async def complete(self, request: ModelRequest) -> ModelResponse:
        response = self.transcript.response_at(self._position, request)
        self._position += 1
        return response

    async def request_cancellation(self, request: ModelRequest) -> str:
        del request
        return "not_observed"


__all__ = [
    "MODEL_TRANSCRIPT_SCHEMA_VERSION",
    "ModelTranscript",
    "ModelTranscriptRecord",
    "RecordingModel",
    "ReplayModel",
    "model_request_fingerprint",
]
