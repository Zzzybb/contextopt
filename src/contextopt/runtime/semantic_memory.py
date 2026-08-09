"""Durable, deterministic semantic memory for long-running coding agents.

The observed-memory ledger in :mod:`contextopt.runtime.memory` is rebuilt from one
run's tool transcript and is invalidated when the workspace changes.  This module is
the deliberately separate cross-run layer: an operator or agent can save a compact
fact, decision, procedure, or failure, then retrieve it in a later run through an
explicit runtime tool.  Entries are append-only events, deduplicated by content, and
searched with a dependency-free lexical scorer so the result is reproducible without
an embedding service.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from threading import RLock
from typing import Any, Literal, cast

from contextopt.runtime.events import EventLog, read_events
from contextopt.runtime.identity import stable_hash

SEMANTIC_MEMORY_SCHEMA_VERSION = "1"
MemoryKind = Literal["fact", "decision", "procedure", "failure"]
MemoryStatus = Literal["active", "superseded", "invalidated"]

_KINDS = frozenset({"fact", "decision", "procedure", "failure"})
_STATUSES = frozenset({"active", "superseded", "invalidated"})
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u3400-\u4dbf\u4e00-\u9fff]")
_MAX_TEXT_BYTES = 16 * 1024
_MAX_SCOPE_CHARS = 128
_MAX_TAG_CHARS = 64
_MAX_TAGS = 24
_MAX_SOURCE_REFS = 32
_MAX_SOURCE_REF_CHARS = 256


def _string(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise ValueError(f"{label} must be {qualifier}")
    return value


def _non_negative_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be an integer >= 0")
    return value


def _unit_interval(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label} must be a number between 0 and 1")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be a number between 0 and 1")
    return result


def _key(value: Any, label: str, *, maximum: int) -> str:
    text = _string(value, label).strip()
    if len(text) > maximum:
        raise ValueError(f"{label} exceeds the {maximum}-character limit")
    if any(ord(character) < 32 for character in text):
        raise ValueError(f"{label} contains a control character")
    return text


def _string_tuple(
    value: Any,
    label: str,
    *,
    maximum_items: int,
    maximum_chars: int,
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be an array of strings")
    if len(value) > maximum_items:
        raise ValueError(f"{label} has too many items")
    normalized = tuple(
        sorted(
            {
                _key(item, f"{label}[{index}]", maximum=maximum_chars)
                for index, item in enumerate(value)
            }
        )
    )
    return normalized


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(piece.casefold() for piece in _TOKEN_RE.findall(text))


def _source_ref_key(value: str) -> str:
    """Normalize a workspace source reference for conservative exact matching."""

    normalized = value.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.casefold()


def _digest_identity(*, text: str, kind: str, scope: str, tags: Sequence[str]) -> str:
    return stable_hash(
        {
            "text": text,
            "kind": kind,
            "scope": scope,
            "tags": list(tags),
        }
    )


@dataclass(frozen=True, slots=True)
class SemanticMemoryEntry:
    """One cross-run memory item and its provenance."""

    memory_id: str
    text: str
    scope: str
    kind: MemoryKind
    tags: tuple[str, ...] = ()
    confidence: float = 0.7
    source_run_id: str | None = None
    source_refs: tuple[str, ...] = ()
    created_seq: int = 0
    updated_seq: int = 0
    status: MemoryStatus = "active"
    invalidation_reason: str | None = None
    supersedes: str | None = None
    schema_version: str = SEMANTIC_MEMORY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SEMANTIC_MEMORY_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported semantic memory schema: {self.schema_version}"
            )
        memory_id = _key(self.memory_id, "memory_id", maximum=96)
        text = _string(self.text, "text")
        if len(text.encode("utf-8")) > _MAX_TEXT_BYTES:
            raise ValueError(f"text exceeds the {_MAX_TEXT_BYTES}-byte limit")
        scope = _key(self.scope, "scope", maximum=_MAX_SCOPE_CHARS)
        if not isinstance(self.kind, str) or self.kind not in _KINDS:
            raise ValueError(f"unsupported memory kind: {self.kind!r}")
        tags = _string_tuple(
            self.tags,
            "tags",
            maximum_items=_MAX_TAGS,
            maximum_chars=_MAX_TAG_CHARS,
        )
        if not isinstance(self.status, str) or self.status not in _STATUSES:
            raise ValueError(f"unsupported memory status: {self.status!r}")
        confidence = _unit_interval(self.confidence, "confidence")
        created_seq = _non_negative_int(self.created_seq, "created_seq")
        updated_seq = _non_negative_int(self.updated_seq, "updated_seq")
        if updated_seq < created_seq:
            raise ValueError("updated_seq cannot be before created_seq")
        source_run_id = (
            None
            if self.source_run_id is None
            else _key(self.source_run_id, "source_run_id", maximum=128)
        )
        source_refs = _string_tuple(
            self.source_refs,
            "source_refs",
            maximum_items=_MAX_SOURCE_REFS,
            maximum_chars=256,
        )
        reason = (
            None
            if self.invalidation_reason is None
            else _key(self.invalidation_reason, "invalidation_reason", maximum=256)
        )
        supersedes = (
            None
            if self.supersedes is None
            else _key(self.supersedes, "supersedes", maximum=96)
        )
        if self.status == "invalidated" and reason is None:
            raise ValueError("invalidated memory requires invalidation_reason")
        if self.status == "active" and reason is not None:
            raise ValueError("active memory cannot have invalidation_reason")
        object.__setattr__(self, "memory_id", memory_id)
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "tags", tags)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "source_run_id", source_run_id)
        object.__setattr__(self, "source_refs", source_refs)
        object.__setattr__(self, "invalidation_reason", reason)
        object.__setattr__(self, "supersedes", supersedes)

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def identity(self) -> str:
        return _digest_identity(
            text=self.text,
            kind=self.kind,
            scope=self.scope,
            tags=self.tags,
        )

    @property
    def active(self) -> bool:
        return self.status == "active"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "memory_id": self.memory_id,
            "text": self.text,
            "scope": self.scope,
            "kind": self.kind,
            "tags": list(self.tags),
            "confidence": self.confidence,
            "source_run_id": self.source_run_id,
            "source_refs": list(self.source_refs),
            "created_seq": self.created_seq,
            "updated_seq": self.updated_seq,
            "status": self.status,
            "invalidation_reason": self.invalidation_reason,
            "supersedes": self.supersedes,
            "content_sha256": self.content_sha256,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SemanticMemoryEntry:
        required = {
            "schema_version",
            "memory_id",
            "text",
            "scope",
            "kind",
            "tags",
            "confidence",
            "source_run_id",
            "source_refs",
            "created_seq",
            "updated_seq",
            "status",
            "invalidation_reason",
            "supersedes",
            "content_sha256",
        }
        if set(data) != required:
            raise ValueError("semantic memory entry fields do not match the schema")
        raw_kind = _string(data["kind"], "kind")
        raw_status = _string(data["status"], "status")
        if raw_kind not in _KINDS:
            raise ValueError(f"unsupported memory kind: {raw_kind!r}")
        if raw_status not in _STATUSES:
            raise ValueError(f"unsupported memory status: {raw_status!r}")
        entry = cls(
            schema_version=_string(data["schema_version"], "schema_version"),
            memory_id=_string(data["memory_id"], "memory_id"),
            text=_string(data["text"], "text"),
            scope=_string(data["scope"], "scope"),
            kind=cast(MemoryKind, raw_kind),
            tags=_string_tuple(
                data["tags"],
                "tags",
                maximum_items=_MAX_TAGS,
                maximum_chars=_MAX_TAG_CHARS,
            ),
            confidence=_unit_interval(data["confidence"], "confidence"),
            source_run_id=(
                None
                if data["source_run_id"] is None
                else _string(data["source_run_id"], "source_run_id")
            ),
            source_refs=_string_tuple(
                data["source_refs"],
                "source_refs",
                maximum_items=_MAX_SOURCE_REFS,
                maximum_chars=256,
            ),
            created_seq=_non_negative_int(data["created_seq"], "created_seq"),
            updated_seq=_non_negative_int(data["updated_seq"], "updated_seq"),
            status=cast(MemoryStatus, raw_status),
            invalidation_reason=(
                None
                if data["invalidation_reason"] is None
                else _string(data["invalidation_reason"], "invalidation_reason")
            ),
            supersedes=(
                None
                if data["supersedes"] is None
                else _string(data["supersedes"], "supersedes")
            ),
        )
        if data["content_sha256"] != entry.content_sha256:
            raise ValueError("semantic memory content_sha256 does not match text")
        return entry


@dataclass(frozen=True, slots=True)
class SemanticMemoryMatch:
    """A retrieved memory with transparent lexical evidence."""

    entry: SemanticMemoryEntry
    score: float
    matched_terms: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.entry.memory_id,
            "text": self.entry.text,
            "scope": self.entry.scope,
            "kind": self.entry.kind,
            "tags": list(self.entry.tags),
            "confidence": self.entry.confidence,
            "source_run_id": self.entry.source_run_id,
            "source_refs": list(self.entry.source_refs),
            "content_sha256": self.entry.content_sha256,
            "score": self.score,
            "matched_terms": list(self.matched_terms),
        }


@dataclass(frozen=True, slots=True)
class MemoryWriteResult:
    entry: SemanticMemoryEntry
    created: bool
    revision: int


class SemanticMemoryStore:
    """An append-only cross-run memory log with a deterministic search view."""

    def __init__(
        self,
        path: str | Path,
        *,
        run_id: str | None = None,
        repair_truncated: bool = True,
    ) -> None:
        self.path = Path(path).resolve(strict=False)
        self._log = EventLog(
            self.path,
            run_id or "contextopt-semantic-memory-v1",
            repair_truncated=repair_truncated,
        )
        self._lock = RLock()
        self._entries: dict[str, SemanticMemoryEntry] = {}
        for event in read_events(self.path):
            self._apply_event(event)
        self._revision = self._log.event_count

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    @property
    def configuration_fingerprint(self) -> str:
        """Identity used for run resume, excluding mutable memory contents."""

        return stable_hash(
            {
                "schema_version": SEMANTIC_MEMORY_SCHEMA_VERSION,
                "path": str(self.path),
            }
        )

    @property
    def fingerprint(self) -> str:
        with self._lock:
            return stable_hash(
                {
                    "schema_version": SEMANTIC_MEMORY_SCHEMA_VERSION,
                    "revision": self._revision,
                    "entries": [
                        entry.to_dict()
                        for entry in sorted(
                            self._entries.values(), key=lambda item: item.memory_id
                        )
                    ],
                }
            )

    def active_entries(
        self, *, scope: str | None = None
    ) -> tuple[SemanticMemoryEntry, ...]:
        normalized_scope = (
            None if scope is None else _key(scope, "scope", maximum=_MAX_SCOPE_CHARS)
        )
        with self._lock:
            return tuple(
                sorted(
                    (
                        entry
                        for entry in self._entries.values()
                        if entry.active
                        and (
                            normalized_scope is None
                            or entry.scope in {"global", normalized_scope}
                        )
                    ),
                    key=lambda item: (item.scope, item.kind, item.memory_id),
                )
            )

    def get(self, memory_id: str) -> SemanticMemoryEntry | None:
        with self._lock:
            return self._entries.get(memory_id)

    def put(
        self,
        text: str,
        *,
        scope: str = "global",
        kind: MemoryKind = "fact",
        tags: Sequence[str] = (),
        confidence: float = 0.7,
        source_run_id: str | None = None,
        source_refs: Sequence[str] = (),
        supersedes: str | None = None,
    ) -> MemoryWriteResult:
        normalized_text = _string(text, "text").strip()
        normalized_scope = _key(scope, "scope", maximum=_MAX_SCOPE_CHARS)
        if not isinstance(kind, str) or kind not in _KINDS:
            raise ValueError(f"unsupported memory kind: {kind!r}")
        normalized_tags = _string_tuple(
            tags,
            "tags",
            maximum_items=_MAX_TAGS,
            maximum_chars=_MAX_TAG_CHARS,
        )
        identity = _digest_identity(
            text=normalized_text,
            kind=kind,
            scope=normalized_scope,
            tags=normalized_tags,
        )
        memory_id = f"mem-{identity[:32]}"
        with self._lock:
            existing = self._entries.get(memory_id)
            if existing is not None and existing.active:
                return MemoryWriteResult(existing, False, self._revision)
            if supersedes is not None and supersedes not in self._entries:
                raise ValueError(f"cannot supersede unknown memory: {supersedes}")
            sequence = self._revision + 1
            entry = SemanticMemoryEntry(
                memory_id=memory_id,
                text=normalized_text,
                scope=normalized_scope,
                kind=kind,
                tags=normalized_tags,
                confidence=confidence,
                source_run_id=source_run_id,
                source_refs=tuple(source_refs),
                created_seq=(
                    existing.created_seq if existing is not None else sequence
                ),
                updated_seq=sequence,
                supersedes=supersedes,
            )
            self._append(
                "memory.upserted",
                {"entry": entry.to_dict(), "supersedes": supersedes},
            )
            return MemoryWriteResult(entry, True, self._revision)

    def invalidate(self, memory_id: str, reason: str) -> SemanticMemoryEntry:
        normalized_reason = _key(reason, "reason", maximum=256)
        with self._lock:
            entry = self._entries.get(memory_id)
            if entry is None:
                raise ValueError(f"unknown memory: {memory_id}")
            if not entry.active:
                return entry
            self._append(
                "memory.invalidated",
                {"memory_id": memory_id, "reason": normalized_reason},
            )
            return self._entries[memory_id]

    def invalidate_source_refs(
        self, source_ref: str, reason: str
    ) -> tuple[SemanticMemoryEntry, ...]:
        """Invalidate active entries that cite one changed workspace source.

        Matching is deliberately exact after slash/case normalization.  A memory
        citing ``src/main.py`` is stale when that file changes, while a memory
        citing ``src/main.py.bak`` remains untouched.  The invalidations are
        ordinary append-only events, so a fresh process observes the same result.
        """

        normalized_source_ref = _key(
            source_ref, "source_ref", maximum=_MAX_SOURCE_REF_CHARS
        )
        normalized_reason = _key(reason, "reason", maximum=256)
        source_key = _source_ref_key(normalized_source_ref)
        with self._lock:
            matching = tuple(
                sorted(
                    (
                        entry
                        for entry in self._entries.values()
                        if entry.active
                        and any(
                            _source_ref_key(reference) == source_key
                            for reference in entry.source_refs
                        )
                    ),
                    key=lambda item: item.memory_id,
                )
            )
            for entry in matching:
                self._append(
                    "memory.invalidated",
                    {"memory_id": entry.memory_id, "reason": normalized_reason},
                )
            return tuple(self._entries[entry.memory_id] for entry in matching)

    def search(
        self,
        query: str,
        *,
        scope: str | None = None,
        tags: Sequence[str] = (),
        limit: int = 5,
    ) -> tuple[SemanticMemoryMatch, ...]:
        query_text = _string(query, "query").strip()
        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")
        query_terms = tuple(dict.fromkeys(_tokens(query_text)))
        if not query_terms:
            raise ValueError("query must contain searchable terms")
        wanted_tags = set(
            _string_tuple(
                tags,
                "tags",
                maximum_items=_MAX_TAGS,
                maximum_chars=_MAX_TAG_CHARS,
            )
        )
        query_phrase = " ".join(query_terms)
        matches: list[SemanticMemoryMatch] = []
        for entry in self.active_entries(scope=scope):
            if wanted_tags and not wanted_tags.intersection(entry.tags):
                continue
            entry_terms = _tokens(" ".join((entry.text, entry.scope, *entry.tags)))
            term_set = set(entry_terms)
            matched = tuple(sorted(set(query_terms).intersection(term_set)))
            if not matched:
                continue
            overlap = len(matched) / len(set(query_terms))
            phrase_bonus = 1.0 if query_phrase in " ".join(entry_terms) else 0.0
            score = round(
                0.68 * overlap + 0.17 * phrase_bonus + 0.15 * entry.confidence, 6
            )
            matches.append(SemanticMemoryMatch(entry, score, matched))
        matches.sort(
            key=lambda match: (
                -match.score,
                -match.entry.confidence,
                -match.entry.updated_seq,
                match.entry.memory_id,
            )
        )
        return tuple(matches[:limit])

    def _append(self, event_type: str, data: Mapping[str, Any]) -> None:
        event = self._log.append(event_type, data)
        self._apply_event(event.to_dict())
        self._revision = event.seq + 1

    def _apply_event(self, event: Mapping[str, Any]) -> None:
        event_type = event.get("type")
        data = event.get("data")
        if not isinstance(data, Mapping):
            raise ValueError("semantic memory event data must be an object")
        if event_type == "memory.upserted":
            raw_entry = data.get("entry")
            if not isinstance(raw_entry, Mapping):
                raise ValueError("memory.upserted entry must be an object")
            entry = SemanticMemoryEntry.from_dict(raw_entry)
            supersedes = data.get("supersedes")
            if supersedes is not None and supersedes != entry.supersedes:
                raise ValueError("memory supersedes metadata is inconsistent")
            if supersedes is not None:
                previous = self._entries.get(str(supersedes))
                if previous is None:
                    raise ValueError(
                        f"memory event supersedes unknown entry: {supersedes}"
                    )
                self._entries[str(supersedes)] = replace(
                    previous,
                    status="superseded",
                    updated_seq=entry.updated_seq,
                    invalidation_reason=None,
                )
            self._entries[entry.memory_id] = entry
            return
        if event_type == "memory.invalidated":
            memory_id = _key(data.get("memory_id"), "memory_id", maximum=96)
            reason = _key(data.get("reason"), "reason", maximum=256)
            current_entry = self._entries.get(memory_id)
            if current_entry is None:
                raise ValueError(f"memory event invalidates unknown entry: {memory_id}")
            self._entries[memory_id] = replace(
                current_entry,
                status="invalidated",
                updated_seq=_non_negative_int(event.get("seq"), "event seq") + 1,
                invalidation_reason=reason,
            )
            return
        raise ValueError(f"unknown semantic memory event type: {event_type!r}")

    def close(self) -> None:
        with self._lock:
            self._log.close()

    def __enter__(self) -> SemanticMemoryStore:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()


__all__ = [
    "SEMANTIC_MEMORY_SCHEMA_VERSION",
    "MemoryKind",
    "MemoryStatus",
    "MemoryWriteResult",
    "SemanticMemoryEntry",
    "SemanticMemoryMatch",
    "SemanticMemoryStore",
]
