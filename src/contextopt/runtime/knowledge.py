"""Deterministic workspace knowledge-base indexing.

The semantic-memory store is useful for durable experience, but a coding agent also
needs a searchable projection of the repository it is working on.  This module turns
bounded UTF-8 source files into provenance-carrying memory entries.  Indexing is
provider-free, append-only, and safe to repeat: unchanged chunks are reused while
removed or changed chunks are invalidated before the next context compilation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from contextopt.runtime.identity import stable_hash
from contextopt.runtime.semantic_memory import SemanticMemoryEntry, SemanticMemoryStore

KNOWLEDGE_INDEX_SCHEMA_VERSION = "1"
KNOWLEDGE_CHUNK_TAG = "knowledge-chunk-v1"
_DEFAULT_EXTENSIONS = (
    ".c",
    ".cc",
    ".cpp",
    ".css",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".java",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".py",
    ".rs",
    ".scss",
    ".sh",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
)
_DEFAULT_IGNORED_DIRS = (
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "venv",
)
_MAX_REPORT_SKIPS = 64
_MAX_CHUNK_BYTES = 16 * 1024


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _non_empty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _normalized_extensions(value: Sequence[str]) -> tuple[str, ...]:
    normalized: set[str] = set()
    for item in value:
        extension = _non_empty(item, "extension").casefold()
        if not extension.startswith("."):
            extension = "." + extension
        if "/" in extension or "\\" in extension:
            raise ValueError("extensions must be file suffixes")
        normalized.add(extension)
    if not normalized:
        raise ValueError("at least one extension is required")
    return tuple(sorted(normalized))


def _normalized_names(value: Sequence[str], label: str) -> tuple[str, ...]:
    names = tuple(sorted({_non_empty(item, label) for item in value}))
    if not names:
        raise ValueError(f"at least one {label} is required")
    if any("/" in item or "\\" in item for item in names):
        raise ValueError(f"{label} must contain directory names only")
    return names


@dataclass(frozen=True, slots=True)
class KnowledgeIndexConfig:
    """Bounded, deterministic indexing policy for one workspace."""

    workspace: str | Path
    scope: str = "project:workspace"
    chunk_lines: int = 80
    max_chunk_bytes: int = 8 * 1024
    max_file_bytes: int = 256 * 1024
    max_files: int = 500
    extensions: tuple[str, ...] = _DEFAULT_EXTENSIONS
    ignored_dirs: tuple[str, ...] = _DEFAULT_IGNORED_DIRS
    include_dotfiles: bool = False
    source_run_id: str = "knowledge-index-v1"

    def __post_init__(self) -> None:
        workspace = Path(self.workspace).resolve(strict=False)
        if not workspace.exists() or not workspace.is_dir():
            raise ValueError(f"workspace must be an existing directory: {workspace}")
        scope = _non_empty(self.scope, "scope")
        chunk_lines = _positive_int(self.chunk_lines, "chunk_lines")
        max_chunk_bytes = _positive_int(self.max_chunk_bytes, "max_chunk_bytes")
        if max_chunk_bytes > _MAX_CHUNK_BYTES:
            raise ValueError(f"max_chunk_bytes cannot exceed {_MAX_CHUNK_BYTES}")
        max_file_bytes = _positive_int(self.max_file_bytes, "max_file_bytes")
        max_files = _positive_int(self.max_files, "max_files")
        if not isinstance(self.include_dotfiles, bool):
            raise ValueError("include_dotfiles must be a boolean")
        object.__setattr__(self, "workspace", workspace)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "chunk_lines", chunk_lines)
        object.__setattr__(self, "max_chunk_bytes", max_chunk_bytes)
        object.__setattr__(self, "max_file_bytes", max_file_bytes)
        object.__setattr__(self, "max_files", max_files)
        object.__setattr__(self, "extensions", _normalized_extensions(self.extensions))
        object.__setattr__(
            self, "ignored_dirs", _normalized_names(self.ignored_dirs, "ignored_dirs")
        )
        object.__setattr__(
            self, "source_run_id", _non_empty(self.source_run_id, "source_run_id")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": KNOWLEDGE_INDEX_SCHEMA_VERSION,
            "workspace": str(self.workspace),
            "scope": self.scope,
            "chunk_lines": self.chunk_lines,
            "max_chunk_bytes": self.max_chunk_bytes,
            "max_file_bytes": self.max_file_bytes,
            "max_files": self.max_files,
            "extensions": list(self.extensions),
            "ignored_dirs": list(self.ignored_dirs),
            "include_dotfiles": self.include_dotfiles,
            "source_run_id": self.source_run_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeIndexConfig:
        required = {
            "schema_version",
            "workspace",
            "scope",
            "chunk_lines",
            "max_chunk_bytes",
            "max_file_bytes",
            "max_files",
            "extensions",
            "ignored_dirs",
            "include_dotfiles",
            "source_run_id",
        }
        if set(data) != required:
            raise ValueError("knowledge index config fields do not match the schema")
        if data["schema_version"] != KNOWLEDGE_INDEX_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported knowledge index schema: {data['schema_version']}"
            )
        extensions = data["extensions"]
        ignored_dirs = data["ignored_dirs"]
        if not isinstance(extensions, list) or not isinstance(ignored_dirs, list):
            raise ValueError(
                "knowledge index extensions and ignored_dirs must be arrays"
            )
        if not isinstance(data["include_dotfiles"], bool):
            raise ValueError("knowledge index include_dotfiles must be a boolean")
        return cls(
            workspace=_non_empty(data["workspace"], "workspace"),
            scope=_non_empty(data["scope"], "scope"),
            chunk_lines=_positive_int(data["chunk_lines"], "chunk_lines"),
            max_chunk_bytes=_positive_int(data["max_chunk_bytes"], "max_chunk_bytes"),
            max_file_bytes=_positive_int(data["max_file_bytes"], "max_file_bytes"),
            max_files=_positive_int(data["max_files"], "max_files"),
            extensions=tuple(_non_empty(item, "extension") for item in extensions),
            ignored_dirs=tuple(
                _non_empty(item, "ignored_dir") for item in ignored_dirs
            ),
            include_dotfiles=data["include_dotfiles"],
            source_run_id=_non_empty(data["source_run_id"], "source_run_id"),
        )


@dataclass(frozen=True, slots=True)
class KnowledgeIndexReport:
    """Auditable result of one index pass."""

    config: KnowledgeIndexConfig
    files_considered: int
    files_indexed: int
    chunks_created: int
    chunks_reused: int
    chunks_invalidated: int
    active_chunk_ids: tuple[str, ...]
    skipped_files: tuple[tuple[str, str], ...]
    snapshot_sha256: str
    schema_version: str = KNOWLEDGE_INDEX_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != KNOWLEDGE_INDEX_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported knowledge index schema: {self.schema_version}"
            )
        for name in (
            "files_considered",
            "files_indexed",
            "chunks_created",
            "chunks_reused",
            "chunks_invalidated",
        ):
            _positive_or_zero(getattr(self, name), name)
        if self.files_indexed > self.files_considered:
            raise ValueError("files_indexed cannot exceed files_considered")
        if not isinstance(self.snapshot_sha256, str) or len(self.snapshot_sha256) != 64:
            raise ValueError("snapshot_sha256 must be a SHA-256 hex digest")
        if any(
            not isinstance(path, str)
            or not path
            or not isinstance(reason, str)
            or not reason
            for path, reason in self.skipped_files
        ):
            raise ValueError("skipped_files must contain path/reason pairs")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "config": self.config.to_dict(),
            "files_considered": self.files_considered,
            "files_indexed": self.files_indexed,
            "chunks_created": self.chunks_created,
            "chunks_reused": self.chunks_reused,
            "chunks_invalidated": self.chunks_invalidated,
            "active_chunk_ids": list(self.active_chunk_ids),
            "skipped_files": [
                {"path": path, "reason": reason} for path, reason in self.skipped_files
            ],
            "snapshot_sha256": self.snapshot_sha256,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeIndexReport:
        required = {
            "schema_version",
            "config",
            "files_considered",
            "files_indexed",
            "chunks_created",
            "chunks_reused",
            "chunks_invalidated",
            "active_chunk_ids",
            "skipped_files",
            "snapshot_sha256",
        }
        if set(data) != required:
            raise ValueError("knowledge index report fields do not match the schema")
        if not isinstance(data["config"], dict):
            raise ValueError("knowledge index report config must be an object")
        active_ids = data["active_chunk_ids"]
        raw_skipped = data["skipped_files"]
        if not isinstance(active_ids, list) or not isinstance(raw_skipped, list):
            raise ValueError(
                "knowledge index active_chunk_ids and skipped_files must be arrays"
            )
        skipped: list[tuple[str, str]] = []
        for item in raw_skipped:
            if not isinstance(item, dict) or set(item) != {"path", "reason"}:
                raise ValueError("knowledge index skipped file has invalid fields")
            skipped.append(
                (
                    _non_empty(item["path"], "skipped path"),
                    _non_empty(item["reason"], "skip reason"),
                )
            )
        return cls(
            schema_version=_non_empty(data["schema_version"], "schema_version"),
            config=KnowledgeIndexConfig.from_dict(data["config"]),
            files_considered=_positive_or_zero(
                data["files_considered"], "files_considered"
            ),
            files_indexed=_positive_or_zero(data["files_indexed"], "files_indexed"),
            chunks_created=_positive_or_zero(data["chunks_created"], "chunks_created"),
            chunks_reused=_positive_or_zero(data["chunks_reused"], "chunks_reused"),
            chunks_invalidated=_positive_or_zero(
                data["chunks_invalidated"], "chunks_invalidated"
            ),
            active_chunk_ids=tuple(
                _non_empty(item, "active chunk id") for item in active_ids
            ),
            skipped_files=tuple(skipped),
            snapshot_sha256=_non_empty(data["snapshot_sha256"], "snapshot_sha256"),
        )


def _positive_or_zero(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _relative_path(root: Path, path: Path) -> str:
    relative = PurePosixPath(path.relative_to(root).as_posix())
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError(f"invalid workspace-relative path: {relative}")
    return relative.as_posix()


def _iter_candidate_files(
    config: KnowledgeIndexConfig,
) -> Iterable[tuple[Path, str]]:
    root = Path(config.workspace)
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix().casefold()):
        if not path.is_file() or path.is_symlink():
            continue
        relative = _relative_path(root, path)
        parts = PurePosixPath(relative).parts
        if any(part in config.ignored_dirs for part in parts[:-1]):
            continue
        if not config.include_dotfiles and any(part.startswith(".") for part in parts):
            continue
        if path.suffix.casefold() not in config.extensions:
            continue
        yield path, relative


def _split_long_line(line: str, maximum_bytes: int) -> list[str]:
    if len(line.encode("utf-8")) <= maximum_bytes:
        return [line]
    pieces: list[str] = []
    current: list[str] = []
    current_bytes = 0
    for character in line:
        character_bytes = len(character.encode("utf-8"))
        if current and current_bytes + character_bytes > maximum_bytes:
            pieces.append("".join(current))
            current = []
            current_bytes = 0
        current.append(character)
        current_bytes += character_bytes
    if current:
        pieces.append("".join(current))
    return pieces or [""]


def _chunk_text(
    relative_path: str,
    text: str,
    *,
    chunk_lines: int,
    max_chunk_bytes: int,
) -> tuple[str, ...]:
    raw_lines = text.splitlines()
    if not raw_lines:
        raw_lines = [""]
    lines: list[str] = []
    for line in raw_lines:
        lines.extend(_split_long_line(line, max_chunk_bytes))
    chunks: list[str] = []
    for start in range(0, len(lines), chunk_lines):
        selected = lines[start : start + chunk_lines]
        end = start + len(selected)
        header = f"source: {relative_path}\nlines: {start + 1}-{end}\n\n"
        body = "\n".join(selected)
        encoded = (header + body).encode("utf-8")
        if len(encoded) > max_chunk_bytes:
            # The line splitter guarantees individual lines fit.  A small header can
            # still consume the budget when the configured limit is tiny, so split
            # the body conservatively rather than writing an oversized memory entry.
            available = max(1, max_chunk_bytes - len(header.encode("utf-8")))
            body_pieces = _split_long_line(body, available)
            for _offset, piece in enumerate(body_pieces):
                piece_end = start + len(selected)
                chunks.append(
                    "source: "
                    f"{relative_path}\nlines: {start + 1}-{piece_end}\n\n{piece}"
                )
        else:
            chunks.append(header + body)
    return tuple(chunks)


def _is_knowledge_chunk(entry: SemanticMemoryEntry, scope: str) -> bool:
    return entry.scope == scope and KNOWLEDGE_CHUNK_TAG in entry.tags


def index_workspace(
    store: SemanticMemoryStore,
    config: KnowledgeIndexConfig,
) -> KnowledgeIndexReport:
    """Index eligible workspace files into ``store`` with source-aware invalidation."""

    candidates = tuple(_iter_candidate_files(config))
    files_considered = len(candidates)
    skipped: list[tuple[str, str]] = []
    file_digests: list[dict[str, Any]] = []
    chunks: list[tuple[str, str]] = []
    files_indexed = 0

    for index, (path, relative) in enumerate(candidates):
        if index >= config.max_files:
            skipped.append((relative, "max_files exceeded"))
            continue
        try:
            raw = path.read_bytes()
        except OSError as exc:
            skipped.append((relative, f"read failed: {type(exc).__name__}"))
            continue
        if len(raw) > config.max_file_bytes:
            skipped.append((relative, "max_file_bytes exceeded"))
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            skipped.append((relative, "not valid UTF-8"))
            continue
        digest = hashlib.sha256(raw).hexdigest()
        file_digests.append({"path": relative, "bytes": len(raw), "sha256": digest})
        file_chunks = _chunk_text(
            relative,
            text,
            chunk_lines=config.chunk_lines,
            max_chunk_bytes=config.max_chunk_bytes,
        )
        for chunk in file_chunks:
            chunks.append((relative, chunk))
        files_indexed += 1

    previous = tuple(
        entry
        for entry in store.active_entries(scope=config.scope)
        if _is_knowledge_chunk(entry, config.scope)
    )
    active_ids: list[str] = []
    created = 0
    reused = 0
    for relative, chunk in chunks:
        result = store.put(
            chunk,
            scope=config.scope,
            kind="fact",
            tags=(
                KNOWLEDGE_CHUNK_TAG,
                f"ext:{Path(relative).suffix.casefold()[1:] or 'none'}",
            ),
            confidence=1.0,
            source_run_id=config.source_run_id,
            source_refs=(relative,),
        )
        active_ids.append(result.entry.memory_id)
        if result.created:
            created += 1
        else:
            reused += 1

    active_id_set = set(active_ids)
    invalidated = 0
    for entry in previous:
        if entry.memory_id not in active_id_set:
            store.invalidate(entry.memory_id, "knowledge index refreshed")
            invalidated += 1

    snapshot_sha256 = stable_hash(
        {
            "schema_version": KNOWLEDGE_INDEX_SCHEMA_VERSION,
            "config": config.to_dict(),
            "files": file_digests,
        }
    )
    return KnowledgeIndexReport(
        config=config,
        files_considered=files_considered,
        files_indexed=files_indexed,
        chunks_created=created,
        chunks_reused=reused,
        chunks_invalidated=invalidated,
        active_chunk_ids=tuple(sorted(active_id_set)),
        skipped_files=tuple(skipped[:_MAX_REPORT_SKIPS]),
        snapshot_sha256=snapshot_sha256,
    )


def render_knowledge_index_console(report: KnowledgeIndexReport) -> str:
    lines = [
        "knowledge-index files | chunks created | reused | invalidated | snapshot",
        "--- | ---: | ---: | ---: | ---",
        f"{report.files_indexed}/{report.files_considered} | "
        f"{report.chunks_created} | {report.chunks_reused} | "
        f"{report.chunks_invalidated} | {report.snapshot_sha256[:16]}",
    ]
    if report.skipped_files:
        lines.append(
            f"skipped={len(report.skipped_files)} (showing at most {_MAX_REPORT_SKIPS})"
        )
    return "\n".join(lines) + "\n"


def render_knowledge_index_markdown(report: KnowledgeIndexReport) -> str:
    lines = [
        "# ContextOpt workspace knowledge index",
        "",
        "The index is a deterministic, provider-free projection of bounded UTF-8 "
        "workspace files.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Files indexed | {report.files_indexed}/{report.files_considered} |",
        f"| Chunks created | {report.chunks_created} |",
        f"| Chunks reused | {report.chunks_reused} |",
        f"| Chunks invalidated | {report.chunks_invalidated} |",
        f"| Active chunks | {len(report.active_chunk_ids)} |",
        f"| Snapshot | `{report.snapshot_sha256}` |",
        "",
        "## Claim boundary",
        "",
        "The index makes source snippets retrievable and invalidates stale chunks; it "
        "does not prove that a model used a retrieved chunk or that the code is "
        "correct.",
    ]
    if report.skipped_files:
        lines.extend(["", "## Skipped files", "", "| Path | Reason |", "|---|---|"])
        lines.extend(
            f"| `{path}` | {reason} |" for path, reason in report.skipped_files
        )
    return "\n".join(lines) + "\n"


__all__ = [
    "KNOWLEDGE_CHUNK_TAG",
    "KNOWLEDGE_INDEX_SCHEMA_VERSION",
    "KnowledgeIndexConfig",
    "KnowledgeIndexReport",
    "index_workspace",
    "render_knowledge_index_console",
    "render_knowledge_index_markdown",
]
