"""Durable, append-only JSONL events for auditing agent runs."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

SCHEMA_VERSION = "1"


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class RunEvent:
    run_id: str
    seq: int
    timestamp: str
    type: str
    data: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "seq": self.seq,
            "timestamp": self.timestamp,
            "type": self.type,
            "data": dict(self.data),
        }


def read_events(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Read a JSONL trace, tolerating only a truncated final line."""

    source = Path(path)
    if not source.exists():
        return ()
    raw_lines = source.read_bytes().splitlines(keepends=True)
    events: list[dict[str, Any]] = []
    for index, raw_line in enumerate(raw_lines):
        is_last = index == len(raw_lines) - 1
        try:
            decoded = raw_line.decode("utf-8")
            value = json.loads(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError):
            if is_last and not raw_line.endswith((b"\n", b"\r")):
                break
            raise ValueError(f"invalid event JSON at line {index + 1}") from None
        if not isinstance(value, dict):
            raise ValueError(f"event at line {index + 1} is not an object")
        if value.get("seq") != len(events):
            raise ValueError(f"non-contiguous event sequence at line {index + 1}")
        events.append(value)
    return tuple(events)


class EventLog:
    """Append events with a per-process lock and a durability flush."""

    def __init__(self, path: str | Path, run_id: str) -> None:
        self.path = Path(path)
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existing = read_events(self.path)
        if existing and any(event.get("run_id") != run_id for event in existing):
            raise ValueError("event log belongs to another run")
        self._next_seq = len(existing)
        self._lock = Lock()

    @property
    def event_count(self) -> int:
        return self._next_seq

    def append(self, event_type: str, data: Mapping[str, Any]) -> RunEvent:
        with self._lock:
            event = RunEvent(
                run_id=self.run_id,
                seq=self._next_seq,
                timestamp=_utc_timestamp(),
                type=event_type,
                data=dict(data),
            )
            encoded = json.dumps(
                event.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(encoded + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._next_seq += 1
            return event


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
