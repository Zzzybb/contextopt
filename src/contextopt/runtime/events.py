"""Durable, append-only and corruption-evident events for auditing agent runs."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from threading import Lock
from typing import Any, BinaryIO

SCHEMA_VERSION = "2"
LEGACY_SCHEMA_VERSION = "1"
GENESIS_EVENT_SHA256 = "0" * 64
_TERMINAL_EVENT_TYPES = {
    "run.completed",
    "run.stopped",
    "run.failed",
    "run.cancelled",
}
_SHA256_HEX = frozenset("0123456789abcdef")
_ACTIVE_LEASES: set[str] = set()
_ACTIVE_LEASES_LOCK = Lock()
_WINDOWS_LOCK_OFFSET = (1 << 31) - 1


def _event_file_identity(path: Path) -> tuple[int, int, str]:
    stat = path.stat()
    fallback = os.path.normcase(str(path)) if stat.st_ino == 0 else ""
    return int(stat.st_dev), int(stat.st_ino), fallback


def _handle_identity(handle: BinaryIO, path: Path) -> tuple[int, int, str]:
    stat = os.fstat(handle.fileno())
    fallback = os.path.normcase(str(path)) if stat.st_ino == 0 else ""
    return int(stat.st_dev), int(stat.st_ino), fallback


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _event_sha256(value: Mapping[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("event_sha256", None)
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256_HEX for character in value)
    )


@dataclass(frozen=True, slots=True)
class RunEvent:
    run_id: str
    seq: int
    timestamp: str
    type: str
    data: Mapping[str, Any]
    prev_event_sha256: str = GENESIS_EVENT_SHA256
    event_sha256: str | None = None
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "seq": self.seq,
            "timestamp": self.timestamp,
            "type": self.type,
            "data": dict(self.data),
        }
        if self.schema_version == LEGACY_SCHEMA_VERSION:
            return value
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported event schema: {self.schema_version!r}")
        value["prev_event_sha256"] = self.prev_event_sha256
        value["event_sha256"] = self.event_sha256 or _event_sha256(value)
        return value


@dataclass(frozen=True, slots=True)
class EventScan:
    """Validated events plus the exact durable prefix discovered in a JSONL file."""

    events: tuple[dict[str, Any], ...]
    last_valid_byte_offset: int
    file_size: int
    has_truncated_tail: bool
    needs_trailing_newline: bool
    schema_version: str | None
    run_id: str | None
    last_event_sha256: str | None

    @property
    def truncated_bytes(self) -> int:
        return self.file_size - self.last_valid_byte_offset


class RunLeaseError(ValueError):
    """Raised when another instance or process already owns a run lease."""


def _acquire_os_lock(handle: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        # Windows permits locking beyond EOF. Keeping the advisory byte away from
        # JSONL writes also lets an empty new event file be locked without a sentinel.
        handle.seek(_WINDOWS_LOCK_OFFSET)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        return

    fcntl = import_module("fcntl")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _release_os_lock(handle: BinaryIO) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(_WINDOWS_LOCK_OFFSET)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return

    fcntl = import_module("fcntl")
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class RunLease:
    """A non-blocking cross-process lease on the underlying file identity."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve(strict=False)
        self._handle: BinaryIO | None = None
        self._registry_key: str | None = None

    @property
    def acquired(self) -> bool:
        return self._handle is not None

    def acquire(self) -> RunLease:
        if self._handle is not None:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle: BinaryIO | None = None
        key: str | None = None
        try:
            handle = self.path.open("a+b")
            key = repr(_handle_identity(handle, self.path))
            with _ACTIVE_LEASES_LOCK:
                if key in _ACTIVE_LEASES:
                    raise RunLeaseError(f"run lease is already held: {self.path}")
                _ACTIVE_LEASES.add(key)
            _acquire_os_lock(handle)
        except RunLeaseError:
            if handle is not None:
                handle.close()
            raise
        except OSError as exc:
            if handle is not None:
                handle.close()
            if key is not None:
                with _ACTIVE_LEASES_LOCK:
                    _ACTIVE_LEASES.discard(key)
            raise RunLeaseError(
                f"run lease is already held by another process: {self.path}"
            ) from exc
        assert key is not None
        self._handle = handle
        self._registry_key = key
        return self

    def release(self) -> None:
        handle = self._handle
        key = self._registry_key
        if handle is None:
            return
        try:
            _release_os_lock(handle)
        finally:
            handle.close()
            self._handle = None
            self._registry_key = None
            if key is not None:
                with _ACTIVE_LEASES_LOCK:
                    _ACTIVE_LEASES.discard(key)

    def __enter__(self) -> RunLease:
        return self.acquire()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()

    def __del__(self) -> None:
        with suppress(Exception):
            self.release()


def _validate_event(
    value: Any,
    *,
    line_number: int,
    expected_seq: int,
    schema_version: str | None,
    schema2_run_id: str | None,
    previous_sha256: str,
) -> tuple[dict[str, Any], str, str | None, str]:
    if not isinstance(value, dict):
        raise ValueError(f"event at line {line_number} is not an object")
    event = dict(value)
    version = event.get("schema_version")
    if version not in {LEGACY_SCHEMA_VERSION, SCHEMA_VERSION}:
        raise ValueError(f"unsupported event schema at line {line_number}: {version!r}")
    if schema_version is not None and version != schema_version:
        raise ValueError(f"mixed event schemas at line {line_number}")
    seq = event.get("seq")
    if not isinstance(seq, int) or isinstance(seq, bool) or seq != expected_seq:
        raise ValueError(f"non-contiguous event sequence at line {line_number}")

    if version == LEGACY_SCHEMA_VERSION:
        return event, version, schema2_run_id, previous_sha256

    run_id = event.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError(f"invalid schema 2 run_id at line {line_number}")
    if schema2_run_id is not None and run_id != schema2_run_id:
        raise ValueError(f"schema 2 run_id changed at line {line_number}")
    previous = event.get("prev_event_sha256")
    if not _valid_sha256(previous) or previous != previous_sha256:
        raise ValueError(f"broken event hash chain at line {line_number}")
    declared = event.get("event_sha256")
    if not _valid_sha256(declared):
        raise ValueError(f"invalid event_sha256 at line {line_number}")
    computed = _event_sha256(event)
    if declared != computed:
        raise ValueError(f"event hash mismatch at line {line_number}")
    return event, version, run_id, declared


def scan_events(path: str | Path) -> EventScan:
    """Scan and validate a log while retaining the last complete byte offset."""

    source = Path(path)
    if not source.exists():
        return EventScan((), 0, 0, False, False, None, None, None)
    raw = source.read_bytes()
    raw_lines = raw.splitlines(keepends=True)
    events: list[dict[str, Any]] = []
    last_valid_offset = 0
    current_offset = 0
    truncated_tail = False
    last_line_terminated = True
    schema_version: str | None = None
    schema2_run_id: str | None = None
    previous_sha256 = GENESIS_EVENT_SHA256

    for index, raw_line in enumerate(raw_lines):
        line_number = index + 1
        is_last = index == len(raw_lines) - 1
        current_offset += len(raw_line)
        terminated = raw_line.endswith((b"\n", b"\r"))
        try:
            decoded = raw_line.decode("utf-8")
            value = json.loads(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError):
            if is_last and not terminated:
                truncated_tail = True
                break
            raise ValueError(f"invalid event JSON at line {line_number}") from None
        event, version, run_id, event_sha256 = _validate_event(
            value,
            line_number=line_number,
            expected_seq=len(events),
            schema_version=schema_version,
            schema2_run_id=schema2_run_id,
            previous_sha256=previous_sha256,
        )
        schema_version = version
        schema2_run_id = run_id
        previous_sha256 = event_sha256
        events.append(event)
        last_valid_offset = current_offset
        last_line_terminated = terminated

    needs_newline = bool(events) and not truncated_tail and not last_line_terminated
    return EventScan(
        events=tuple(events),
        last_valid_byte_offset=last_valid_offset,
        file_size=len(raw),
        has_truncated_tail=truncated_tail,
        needs_trailing_newline=needs_newline,
        schema_version=schema_version,
        run_id=schema2_run_id,
        last_event_sha256=(
            previous_sha256 if schema_version == SCHEMA_VERSION and events else None
        ),
    )


def read_events(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Read validated events, tolerating but never modifying a truncated final line."""

    return scan_events(path).events


class EventLog:
    """Append schema 2 events while holding an exclusive run lease."""

    def __init__(
        self,
        path: str | Path,
        run_id: str,
        *,
        repair_truncated: bool = False,
    ) -> None:
        if not run_id:
            raise ValueError("run_id must not be empty")
        self.path = Path(path).resolve(strict=False)
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            try:
                with self.path.open("xb"):
                    pass
            except FileExistsError:
                pass
        self._file_identity = _event_file_identity(self.path)
        self._lock = Lock()
        self._closed = False
        self._lease = RunLease(self.path)
        self._lease.acquire()
        try:
            if _event_file_identity(self.path) != self._file_identity:
                raise ValueError("event log file changed while acquiring its lease")
            scanned = scan_events(self.path)
            if scanned.has_truncated_tail:
                if not repair_truncated:
                    raise ValueError(
                        "event log has a truncated final line; reopen with "
                        "repair_truncated=True to discard only that tail"
                    )
                self._truncate(scanned.last_valid_byte_offset)
            if scanned.events and scanned.schema_version == LEGACY_SCHEMA_VERSION:
                raise ValueError(
                    "schema 1 event logs are read-only; start a new schema 2 log"
                )
            if (
                scanned.events
                and scanned.events[-1].get("type") in _TERMINAL_EVENT_TYPES
            ):
                raise ValueError("terminal event logs are read-only")
            if scanned.run_id is not None and scanned.run_id != run_id:
                raise ValueError("event log belongs to another run")
            if scanned.needs_trailing_newline:
                self._append_newline()
            self._next_seq = len(scanned.events)
            self._previous_event_sha256 = (
                scanned.last_event_sha256 or GENESIS_EVENT_SHA256
            )
        except Exception:
            self._lease.release()
            self._closed = True
            raise

    @property
    def event_count(self) -> int:
        return self._next_seq

    @property
    def closed(self) -> bool:
        return self._closed

    def _truncate(self, offset: int) -> None:
        with self.path.open("r+b") as handle:
            handle.truncate(offset)
            handle.flush()
            os.fsync(handle.fileno())

    def _append_newline(self) -> None:
        with self.path.open("ab") as handle:
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())

    def append(self, event_type: str, data: Mapping[str, Any]) -> RunEvent:
        if not event_type:
            raise ValueError("event_type must not be empty")
        with self._lock:
            if self._closed:
                raise RuntimeError("event log is closed")
            if _event_file_identity(self.path) != self._file_identity:
                raise RuntimeError(
                    "event log file was replaced while the run was active"
                )
            timestamp = _utc_timestamp()
            unsigned: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "run_id": self.run_id,
                "seq": self._next_seq,
                "timestamp": timestamp,
                "type": event_type,
                "data": dict(data),
                "prev_event_sha256": self._previous_event_sha256,
            }
            digest = _event_sha256(unsigned)
            event = RunEvent(
                run_id=self.run_id,
                seq=self._next_seq,
                timestamp=timestamp,
                type=event_type,
                data=dict(data),
                prev_event_sha256=self._previous_event_sha256,
                event_sha256=digest,
            )
            encoded = _canonical_json(event.to_dict())
            with self.path.open("ab") as handle:
                handle.write(encoded + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._next_seq += 1
            self._previous_event_sha256 = digest
            if event_type in _TERMINAL_EVENT_TYPES:
                self._closed = True
                self._lease.release()
            return event

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._lease.release()

    def __enter__(self) -> EventLog:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()


def render_trace(events: tuple[dict[str, Any], ...]) -> str:
    """Render a compact, deterministic human view of a run trace."""

    lines: list[str] = []
    for event in events:
        event_type = str(event.get("type", "unknown"))
        seq = int(event.get("seq", -1))
        data = event.get("data")
        payload = data if isinstance(data, dict) else {}
        detail = ""
        if event_type == "model.responded":
            tool_calls = payload.get("tool_calls", [])
            detail = f" turn={payload.get('turn')} calls={len(tool_calls)}"
        elif event_type in {"tool.completed", "tool.failed", "tool.reused"}:
            detail = (
                f" tool={payload.get('tool_name')} call={payload.get('call_id')}"
                f" ok={payload.get('ok')}"
            )
            metadata = payload.get("metadata")
            if isinstance(metadata, dict) and "exit_code" in metadata:
                detail += f" exit={metadata['exit_code']}"
        elif event_type.startswith("run."):
            status = payload.get("status", "")
            reason = payload.get("reason", "")
            detail = f" status={status} reason={reason}"
        lines.append(f"{seq:04d} {event_type}{detail}".rstrip())
    return "\n".join(lines) + ("\n" if lines else "")
