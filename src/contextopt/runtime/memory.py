"""Version-aware evidence derived deterministically from an agent transcript.

The ledger in this module is deliberately observational.  It records only file
versions and workspace generations witnessed through successful runtime tool
outcomes; it does not claim that the workspace has not changed out of band.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Literal, Protocol, cast

from contextopt.runtime.protocol import AgentMessage, ToolCall, ToolOutcome

MEMORY_SCHEMA_VERSION = "1"

EvidenceKind = Literal[
    "file_read",
    "file_write",
    "test_result",
    "tool_result",
    "tool_error",
]
EvidenceStatus = Literal["current", "stale"]

_EVIDENCE_KINDS = frozenset(
    {"file_read", "file_write", "test_result", "tool_result", "tool_error"}
)
_EVIDENCE_STATUSES = frozenset({"current", "stale"})
_SHA256_CHARACTERS = frozenset("0123456789abcdef")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256_CHARACTERS for character in value)
    )


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _optional_integer(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer or null")
    return value


def _string(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise ValueError(f"{label} must be {qualifier}")
    return value


def _optional_string(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _string(value, label)


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _number(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label} must be a number")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1")
    return number


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _valid_relative_path(value: Any) -> str | None:
    if not isinstance(value, str) or not value or "\\" in value:
        return None
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        return None
    return path.as_posix()


def _metadata_sha256(metadata: Mapping[str, Any], name: str) -> str | None:
    value = metadata.get(name)
    return value if _is_sha256(value) else None


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """One tool observation and its validity within the observed history."""

    evidence_id: str
    source_message_index: int | None
    call_id: str
    tool_name: str
    turn: int
    call_index: int
    kind: EvidenceKind
    status: EvidenceStatus
    workspace_generation: int
    path: str | None
    path_version: int | None
    content_sha256: str | None
    test_scope: str | None
    test_exit_code: int | None
    test_passed: bool | None
    stale_reason: str | None
    invalidated_at_generation: int | None
    outcome_sha256: str
    duplicate_group: str
    base_importance: float
    truncated: bool

    def __post_init__(self) -> None:
        if not self.evidence_id or not self.call_id or not self.tool_name:
            raise ValueError("evidence, call, and tool identifiers must be non-empty")
        if self.source_message_index is not None and self.source_message_index < 0:
            raise ValueError("source_message_index must be non-negative or null")
        if self.turn < 1 or self.call_index < 0 or self.workspace_generation < 0:
            raise ValueError("evidence position and generation are invalid")
        if self.kind not in _EVIDENCE_KINDS:
            raise ValueError(f"unsupported evidence kind: {self.kind!r}")
        if self.status not in _EVIDENCE_STATUSES:
            raise ValueError(f"unsupported evidence status: {self.status!r}")
        if not _is_sha256(self.outcome_sha256):
            raise ValueError("outcome_sha256 must be a lowercase SHA-256 digest")
        if not self.duplicate_group:
            raise ValueError("duplicate_group must be non-empty")
        if not 0.0 <= self.base_importance <= 1.0:
            raise ValueError("base_importance must be between 0 and 1")
        if self.status == "current":
            if (
                self.stale_reason is not None
                or self.invalidated_at_generation is not None
            ):
                raise ValueError("current evidence cannot have invalidation metadata")
        elif self.stale_reason is None or self.invalidated_at_generation is None:
            raise ValueError("stale evidence requires invalidation metadata")
        if (
            self.invalidated_at_generation is not None
            and self.invalidated_at_generation < self.workspace_generation
        ):
            raise ValueError("evidence cannot be invalidated before it was observed")

        file_kind = self.kind in {"file_read", "file_write"}
        if file_kind:
            if (
                self.path is None
                or self.path_version is None
                or self.path_version < 0
                or not _is_sha256(self.content_sha256)
            ):
                raise ValueError(
                    "file evidence requires path, version, and content SHA"
                )
            if any(
                value is not None
                for value in (self.test_scope, self.test_exit_code, self.test_passed)
            ):
                raise ValueError("file evidence cannot contain test metadata")
        elif self.kind == "test_result":
            if (
                self.test_scope is None
                or self.test_exit_code is None
                or self.test_passed is None
            ):
                raise ValueError("test evidence requires scope, exit code, and result")
            if self.test_passed != (self.test_exit_code == 0):
                raise ValueError("test_passed does not match test_exit_code")
            if any(
                value is not None
                for value in (self.path, self.path_version, self.content_sha256)
            ):
                raise ValueError("test evidence cannot contain file metadata")
        elif any(
            value is not None
            for value in (
                self.path,
                self.path_version,
                self.content_sha256,
                self.test_scope,
                self.test_exit_code,
                self.test_passed,
            )
        ):
            raise ValueError("generic tool evidence cannot contain file/test metadata")

    @property
    def current(self) -> bool:
        return self.status == "current"

    @property
    def freshness(self) -> float:
        return 1.0 if self.current else 0.0

    @property
    def importance(self) -> float:
        # Stale evidence remains auditable, but is strongly disfavoured for packing.
        return self.base_importance if self.current else self.base_importance * 0.1

    @property
    def topics(self) -> frozenset[str]:
        if self.path is not None:
            return frozenset(
                {f"path:{self.path}", f"file:{PurePosixPath(self.path).name}"}
            )
        if self.test_scope is not None:
            return frozenset({"tests", f"test:{self.test_scope}"})
        return frozenset({f"tool:{self.tool_name}"})

    def context_signals(self) -> dict[str, Any]:
        """Return JSON-friendly signals that map directly onto ``ContextItem``."""

        return {
            "kind": self.kind,
            "importance": self.importance,
            "freshness": self.freshness,
            "topics": sorted(self.topics),
            "duplicate_group": self.duplicate_group,
            "memory_status": self.status,
            "stale_reason": self.stale_reason,
            "workspace_generation": self.workspace_generation,
            "path_version": self.path_version,
            "content_sha256": self.content_sha256,
            "test_passed": self.test_passed,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvidenceRecord:
        required = {
            "evidence_id",
            "source_message_index",
            "call_id",
            "tool_name",
            "turn",
            "call_index",
            "kind",
            "status",
            "workspace_generation",
            "path",
            "path_version",
            "content_sha256",
            "test_scope",
            "test_exit_code",
            "test_passed",
            "stale_reason",
            "invalidated_at_generation",
            "outcome_sha256",
            "duplicate_group",
            "base_importance",
            "truncated",
        }
        if set(data) != required:
            raise ValueError("evidence record fields do not match the memory schema")
        source_index = _optional_integer(
            data["source_message_index"], "source_message_index"
        )
        path_version = _optional_integer(data["path_version"], "path_version")
        invalidated = _optional_integer(
            data["invalidated_at_generation"], "invalidated_at_generation"
        )
        test_exit_code = _optional_integer(data["test_exit_code"], "test_exit_code")
        raw_passed = data["test_passed"]
        test_passed = (
            None if raw_passed is None else _boolean(raw_passed, "test_passed")
        )
        kind = _string(data["kind"], "kind")
        status = _string(data["status"], "status")
        return cls(
            evidence_id=_string(data["evidence_id"], "evidence_id"),
            source_message_index=source_index,
            call_id=_string(data["call_id"], "call_id"),
            tool_name=_string(data["tool_name"], "tool_name"),
            turn=_integer(data["turn"], "turn", minimum=1),
            call_index=_integer(data["call_index"], "call_index"),
            kind=cast(EvidenceKind, kind),
            status=cast(EvidenceStatus, status),
            workspace_generation=_integer(
                data["workspace_generation"], "workspace_generation"
            ),
            path=_optional_string(data["path"], "path"),
            path_version=path_version,
            content_sha256=_optional_string(data["content_sha256"], "content_sha256"),
            test_scope=_optional_string(data["test_scope"], "test_scope"),
            test_exit_code=test_exit_code,
            test_passed=test_passed,
            stale_reason=_optional_string(data["stale_reason"], "stale_reason"),
            invalidated_at_generation=invalidated,
            outcome_sha256=_string(data["outcome_sha256"], "outcome_sha256"),
            duplicate_group=_string(data["duplicate_group"], "duplicate_group"),
            base_importance=_number(data["base_importance"], "base_importance"),
            truncated=_boolean(data["truncated"], "truncated"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "source_message_index": self.source_message_index,
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "turn": self.turn,
            "call_index": self.call_index,
            "kind": self.kind,
            "status": self.status,
            "workspace_generation": self.workspace_generation,
            "path": self.path,
            "path_version": self.path_version,
            "content_sha256": self.content_sha256,
            "test_scope": self.test_scope,
            "test_exit_code": self.test_exit_code,
            "test_passed": self.test_passed,
            "stale_reason": self.stale_reason,
            "invalidated_at_generation": self.invalidated_at_generation,
            "outcome_sha256": self.outcome_sha256,
            "duplicate_group": self.duplicate_group,
            "base_importance": self.base_importance,
            "truncated": self.truncated,
        }


class CompletedCallLike(Protocol):
    """Structural subset of ``CompletedToolCall`` needed by the ledger."""

    @property
    def call(self) -> ToolCall: ...

    @property
    def turn(self) -> int: ...

    @property
    def call_index(self) -> int: ...

    @property
    def outcome(self) -> ToolOutcome: ...


@dataclass(frozen=True, slots=True)
class _Observation:
    call: ToolCall
    turn: int
    call_index: int
    outcome: ToolOutcome
    source_message_index: int | None
    reused: bool = False

    @property
    def evidence_id(self) -> str:
        source = (
            f"message:{self.source_message_index}"
            if self.source_message_index is not None
            else f"call:{self.turn}:{self.call_index}"
        )
        return f"{source}:{self.call.id}"


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    """Immutable observed memory state rebuilt from durable tool outcomes."""

    workspace_generation: int
    path_versions: Mapping[str, int]
    records: tuple[EvidenceRecord, ...]
    schema_version: str = MEMORY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MEMORY_SCHEMA_VERSION:
            raise ValueError(f"unsupported memory schema: {self.schema_version!r}")
        if self.workspace_generation < 0:
            raise ValueError("workspace_generation must be non-negative")
        normalized_versions: dict[str, int] = {}
        for path, version in self.path_versions.items():
            normalized_path = _valid_relative_path(path)
            if normalized_path != path:
                raise ValueError(f"invalid observed path: {path!r}")
            normalized_versions[path] = _integer(version, f"version for {path!r}")
        object.__setattr__(
            self,
            "path_versions",
            MappingProxyType(dict(sorted(normalized_versions.items()))),
        )
        if len({record.evidence_id for record in self.records}) != len(self.records):
            raise ValueError("evidence ids must be unique")
        if tuple(sorted(self.records, key=_record_sort_key)) != self.records:
            raise ValueError("evidence records must be in deterministic order")
        for record in self.records:
            if record.workspace_generation > self.workspace_generation:
                raise ValueError("evidence generation is newer than its snapshot")
            if record.path is not None:
                current_version = self.path_versions.get(record.path)
                if current_version is None or record.path_version is None:
                    raise ValueError("file evidence path is absent from path_versions")
                if record.path_version > current_version:
                    raise ValueError("file evidence version is newer than its path")
                if record.current and record.path_version != current_version:
                    raise ValueError(
                        "current file evidence must match current path version"
                    )
            if (
                record.current
                and record.kind == "test_result"
                and record.workspace_generation != self.workspace_generation
            ):
                raise ValueError(
                    "current test evidence must match workspace generation"
                )

    @property
    def fingerprint(self) -> str:
        return _digest(self.to_dict())

    @property
    def current_records(self) -> tuple[EvidenceRecord, ...]:
        return tuple(record for record in self.records if record.current)

    @property
    def stale_records(self) -> tuple[EvidenceRecord, ...]:
        return tuple(record for record in self.records if not record.current)

    def records_for_call(self, call_id: str) -> tuple[EvidenceRecord, ...]:
        return tuple(record for record in self.records if record.call_id == call_id)

    def context_signals_for_message(self, index: int) -> dict[str, Any] | None:
        """Return signals for a tool-result message, or ``None`` for other messages."""

        matches = [
            record for record in self.records if record.source_message_index == index
        ]
        if not matches:
            return None
        if len(matches) != 1:
            raise ValueError("a transcript message mapped to multiple evidence records")
        return matches[0].context_signals()

    @classmethod
    def from_messages(cls, messages: Sequence[AgentMessage]) -> MemorySnapshot:
        """Replay canonical runtime messages without consulting the workspace.

        Incomplete assistant tool calls are ignored until their tool-result message
        appears.  Malformed or mismatched tool messages are rejected because silently
        accepting them would make resume-time memory non-deterministic.
        """

        pending: dict[str, tuple[ToolCall, int, int, bool]] = {}
        completed: dict[str, tuple[ToolCall, ToolOutcome]] = {}
        observations: list[_Observation] = []
        turn = 0
        for message_index, message in enumerate(messages):
            if message.role == "assistant":
                turn += 1
                for call_index, call in enumerate(message.tool_calls):
                    if call.id in pending:
                        raise ValueError(f"duplicate pending tool call id: {call.id!r}")
                    previous = completed.get(call.id)
                    if previous is not None and previous[0] != call:
                        raise ValueError(
                            f"tool call id {call.id!r} was reused with new arguments"
                        )
                    pending[call.id] = (
                        call,
                        turn,
                        call_index,
                        previous is not None,
                    )
                continue
            if message.role != "tool":
                continue
            try:
                raw_outcome = json.loads(message.content)
                outcome = ToolOutcome.from_dict(
                    _object(raw_outcome, f"tool message {message_index}")
                )
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"tool message {message_index} is not a canonical ToolOutcome"
                ) from exc
            if (
                outcome.call_id != message.tool_call_id
                or outcome.tool_name != message.tool_name
            ):
                raise ValueError(
                    f"tool message {message_index} metadata does not match its outcome"
                )
            position = pending.pop(outcome.call_id, None)
            if position is None:
                raise ValueError(
                    f"tool message {message_index} has no pending assistant call"
                )
            call, call_turn, call_index, reused = position
            if call.name != outcome.tool_name:
                raise ValueError(
                    f"tool message {message_index} tool name does not match its call"
                )
            previous = completed.get(call.id)
            if reused:
                if previous is None or previous != (call, outcome):
                    raise ValueError(
                        f"cached tool call {call.id!r} changed its durable outcome"
                    )
            else:
                completed[call.id] = (call, outcome)
            observations.append(
                _Observation(
                    call=call,
                    turn=call_turn,
                    call_index=call_index,
                    outcome=outcome,
                    source_message_index=message_index,
                    reused=reused,
                )
            )
        return cls._from_observations(observations)

    @classmethod
    def from_completed_calls(cls, calls: Iterable[CompletedCallLike]) -> MemorySnapshot:
        """Build from recovery records without message-level annotations."""

        ordered = sorted(
            calls,
            key=lambda completed: (
                completed.turn,
                completed.call_index,
                completed.call.id,
            ),
        )
        observations = [
            _Observation(
                call=completed.call,
                turn=completed.turn,
                call_index=completed.call_index,
                outcome=completed.outcome,
                source_message_index=None,
            )
            for completed in ordered
        ]
        return cls._from_observations(observations)

    @classmethod
    def _from_observations(cls, observations: Iterable[_Observation]) -> MemorySnapshot:
        records: list[EvidenceRecord] = []
        workspace_generation = 0
        path_versions: dict[str, int] = {}

        for observation in observations:
            if observation.reused:
                original = next(
                    (
                        record
                        for record in records
                        if record.call_id == observation.call.id
                    ),
                    None,
                )
                if original is None:
                    raise ValueError(
                        f"cached call {observation.call.id!r} has no original evidence"
                    )
                records.append(
                    replace(
                        original,
                        evidence_id=observation.evidence_id,
                        source_message_index=observation.source_message_index,
                        turn=observation.turn,
                        call_index=observation.call_index,
                    )
                )
                continue
            classification = _classify(observation, workspace_generation, path_versions)
            if classification.kind == "file_write":
                assert classification.path is not None
                workspace_generation += 1
                path_versions[classification.path] = (
                    path_versions.get(classification.path, 0) + 1
                )
                records = [
                    _invalidate_for_write(
                        record,
                        path=classification.path,
                        generation=workspace_generation,
                    )
                    for record in records
                ]
                classification = replace(
                    classification,
                    workspace_generation=workspace_generation,
                    path_version=path_versions[classification.path],
                )
            elif classification.kind == "file_read":
                assert classification.path is not None
                path_versions.setdefault(classification.path, 0)
                classification = replace(
                    classification,
                    path_version=path_versions[classification.path],
                )
            elif classification.kind == "test_result":
                records = [
                    _invalidate_for_test_rerun(record, classification)
                    for record in records
                ]
            records.append(classification)

        return cls(
            workspace_generation=workspace_generation,
            path_versions=path_versions,
            records=tuple(sorted(records, key=_record_sort_key)),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MemorySnapshot:
        required = {
            "schema_version",
            "workspace_generation",
            "path_versions",
            "records",
        }
        if set(data) != required:
            raise ValueError("memory snapshot fields do not match the schema")
        raw_versions = _object(data["path_versions"], "path_versions")
        versions = {
            _string(path, "path_versions key"): _integer(
                version, f"version for {path!r}"
            )
            for path, version in raw_versions.items()
        }
        raw_records = data["records"]
        if not isinstance(raw_records, list):
            raise ValueError("records must be an array")
        return cls(
            schema_version=_string(data["schema_version"], "schema_version"),
            workspace_generation=_integer(
                data["workspace_generation"], "workspace_generation"
            ),
            path_versions=versions,
            records=tuple(
                EvidenceRecord.from_dict(_object(record, f"records[{index}]"))
                for index, record in enumerate(raw_records)
            ),
        )

    @classmethod
    def from_json(cls, payload: str) -> MemorySnapshot:
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError("memory snapshot is not valid JSON") from exc
        return cls.from_dict(_object(value, "memory snapshot"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "workspace_generation": self.workspace_generation,
            "path_versions": dict(self.path_versions),
            "records": [record.to_dict() for record in self.records],
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict()).decode("utf-8")


def _record_sort_key(record: EvidenceRecord) -> tuple[int, int, int, str]:
    source_index = (
        record.source_message_index
        if record.source_message_index is not None
        else (1 << 62)
    )
    return source_index, record.turn, record.call_index, record.evidence_id


def _base_record(
    observation: _Observation,
    *,
    kind: EvidenceKind,
    workspace_generation: int,
    path: str | None = None,
    path_version: int | None = None,
    content_sha256: str | None = None,
    test_scope: str | None = None,
    test_exit_code: int | None = None,
    importance: float,
    duplicate_group: str,
) -> EvidenceRecord:
    outcome = observation.outcome
    return EvidenceRecord(
        evidence_id=observation.evidence_id,
        source_message_index=observation.source_message_index,
        call_id=observation.call.id,
        tool_name=observation.call.name,
        turn=observation.turn,
        call_index=observation.call_index,
        kind=kind,
        status="current",
        workspace_generation=workspace_generation,
        path=path,
        path_version=path_version,
        content_sha256=content_sha256,
        test_scope=test_scope,
        test_exit_code=test_exit_code,
        test_passed=(None if test_exit_code is None else test_exit_code == 0),
        stale_reason=None,
        invalidated_at_generation=None,
        outcome_sha256=_digest(outcome.to_dict()),
        duplicate_group=duplicate_group,
        base_importance=importance,
        truncated=outcome.truncated,
    )


def _classify(
    observation: _Observation,
    workspace_generation: int,
    path_versions: Mapping[str, int],
) -> EvidenceRecord:
    outcome = observation.outcome
    metadata = outcome.metadata
    outcome_sha256 = _digest(outcome.to_dict())

    if outcome.ok and observation.call.name == "read_file":
        path = _valid_relative_path(metadata.get("path"))
        content_sha256 = _metadata_sha256(metadata, "sha256")
        if path is not None and content_sha256 is not None:
            return _base_record(
                observation,
                kind="file_read",
                workspace_generation=workspace_generation,
                path=path,
                path_version=path_versions.get(path, 0),
                content_sha256=content_sha256,
                importance=0.78,
                duplicate_group=f"file-content:{path}:{content_sha256}",
            )

    if outcome.ok and observation.call.name in {"create_file", "replace_text"}:
        path = _valid_relative_path(metadata.get("path"))
        digest_name = (
            "sha256" if observation.call.name == "create_file" else "after_sha256"
        )
        content_sha256 = _metadata_sha256(metadata, digest_name)
        if path is not None and content_sha256 is not None:
            return _base_record(
                observation,
                kind="file_write",
                workspace_generation=workspace_generation,
                path=path,
                path_version=path_versions.get(path, 0),
                content_sha256=content_sha256,
                importance=0.92,
                duplicate_group=f"file-content:{path}:{content_sha256}",
            )

    if outcome.ok and observation.call.name == "run_tests":
        scope = metadata.get("scope")
        exit_code = metadata.get("exit_code")
        if (
            isinstance(scope, str)
            and scope
            and isinstance(exit_code, int)
            and not isinstance(exit_code, bool)
        ):
            return _base_record(
                observation,
                kind="test_result",
                workspace_generation=workspace_generation,
                test_scope=scope,
                test_exit_code=exit_code,
                importance=1.0 if exit_code != 0 else 0.9,
                duplicate_group=(
                    f"test:{scope}:generation:{workspace_generation}:"
                    f"result:{outcome_sha256}"
                ),
            )

    kind: EvidenceKind = "tool_error" if not outcome.ok else "tool_result"
    return _base_record(
        observation,
        kind=kind,
        workspace_generation=workspace_generation,
        importance=0.82 if not outcome.ok else 0.55,
        duplicate_group=f"tool:{observation.call.name}:{outcome_sha256}",
    )


def _invalidate(
    record: EvidenceRecord, *, reason: str, generation: int
) -> EvidenceRecord:
    if not record.current:
        return record
    return replace(
        record,
        status="stale",
        stale_reason=reason,
        invalidated_at_generation=generation,
    )


def _invalidate_for_write(
    record: EvidenceRecord, *, path: str, generation: int
) -> EvidenceRecord:
    if record.kind == "test_result":
        return _invalidate(
            record,
            reason="workspace_modified_after_test",
            generation=generation,
        )
    if record.kind == "tool_result" and record.tool_name in {
        "list_files",
        "search_text",
    }:
        return _invalidate(
            record,
            reason="workspace_modified_after_query",
            generation=generation,
        )
    if record.kind in {"file_read", "file_write"} and record.path == path:
        return _invalidate(
            record,
            reason="path_modified",
            generation=generation,
        )
    return record


def _invalidate_for_test_rerun(
    record: EvidenceRecord, newer: EvidenceRecord
) -> EvidenceRecord:
    if (
        record.current
        and record.kind == "test_result"
        and record.test_scope == newer.test_scope
        and record.workspace_generation == newer.workspace_generation
    ):
        return _invalidate(
            record,
            reason="test_rerun",
            generation=newer.workspace_generation,
        )
    return record
