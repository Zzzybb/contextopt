"""Explicit, conflict-checked application and rollback of an accepted snapshot.

The search session deliberately evaluates immutable snapshots in disposable
directories.  This module is the separate operator boundary that can move one
accepted snapshot into a real workspace.  It never runs a command, never follows
symbolic links, and refuses to write when the files used as the session baseline have
changed.  The write set is applied with temporary files and an in-process rollback if
one of the writes fails; a durable receipt records the observed before/after state.

This is a safety boundary for a local developer workflow, not an OS sandbox or a
distributed transaction.  A process or machine failure during the small write window
still needs normal filesystem recovery and an operator decision.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from stat import S_IMODE
from typing import TYPE_CHECKING, Any, Literal

from contextopt.runtime.identity import stable_hash

if TYPE_CHECKING:
    from contextopt.search.session import SearchSessionReport


_MAX_FILE_BYTES = 2 * 1024 * 1024
_DENIED_TOP_LEVEL = frozenset({".git", ".contextopt"})
_MISSING = object()


class WorkspaceApplyError(RuntimeError):
    """Base error for a refused or failed workspace operation."""


class WorkspaceConflict(WorkspaceApplyError):
    """The target workspace no longer matches the expected baseline."""


def _non_empty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _normal_path(raw: Any, label: str) -> str:
    value = _non_empty(raw, label)
    if "\\" in value or value.startswith("/"):
        raise ValueError(f"{label} must be a relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} must not contain traversal or empty components")
    if path.parts[0].casefold() in _DENIED_TOP_LEVEL:
        raise ValueError(f"{label} may not target runtime or version-control internals")
    return path.as_posix()


def _canonical_text(content: str) -> str:
    """Compare and write text with platform-independent LF line endings."""

    return content.replace("\r\n", "\n").replace("\r", "\n")


def _normalize_files(value: Mapping[str, str], label: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object mapping paths to text")
    normalized: dict[str, str] = {}
    for raw_path, content in value.items():
        path = _normal_path(raw_path, f"{label} path")
        if not isinstance(content, str):
            raise ValueError(f"{label}[{path!r}] must be a string")
        content = _canonical_text(content)
        if len(content.encode("utf-8")) > _MAX_FILE_BYTES:
            raise ValueError(f"{label}[{path!r}] exceeds the 2 MiB file limit")
        if path in normalized:
            raise ValueError(f"{label} contains duplicate path {path!r}")
        normalized[path] = content
    return dict(sorted(normalized.items()))


def snapshot_fingerprint(files: Mapping[str, str]) -> str:
    """Return the same content identity used by candidate workspace snapshots."""

    normalized = _normalize_files(files, "snapshot")
    return stable_hash({"files": [[path, normalized[path]] for path in normalized]})


def _state_fingerprint(paths: Iterable[str], files: Mapping[str, str]) -> str:
    return stable_hash(
        {"paths": [[path, files.get(path, None)] for path in sorted(paths)]}
    )


def _workspace_root(workspace: str | Path) -> Path:
    root = Path(workspace).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("workspace must be a directory")
    return root


def _resolve_relative(root: Path, relative: str) -> Path:
    """Resolve a workspace-relative path while rejecting links and escapes."""

    path = _normal_path(relative, "workspace path")
    candidate = root.joinpath(*PurePosixPath(path).parts)
    current = root
    for part in PurePosixPath(path).parts:
        current = current / part
        if current.is_symlink():
            raise WorkspaceApplyError(f"symbolic links are denied: {path}")
    resolved = candidate.resolve(strict=False)
    try:
        common = os.path.commonpath(
            (os.path.normcase(str(root)), os.path.normcase(str(resolved)))
        )
    except ValueError as exc:
        raise WorkspaceApplyError(
            f"workspace path is outside the workspace: {path}"
        ) from exc
    if common != os.path.normcase(str(root)):
        raise WorkspaceApplyError(f"workspace path is outside the workspace: {path}")
    return candidate


def _read_text_paths(root: Path, paths: Iterable[str]) -> dict[str, str]:
    files: dict[str, str] = {}
    for relative in sorted(set(paths)):
        path = _resolve_relative(root, relative)
        if not path.exists():
            continue
        if not path.is_file():
            raise WorkspaceApplyError(
                f"workspace path is not a regular file: {relative}"
            )
        try:
            raw = path.read_bytes()
            text = _canonical_text(raw.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise WorkspaceApplyError(
                f"workspace file is not UTF-8 text: {relative}"
            ) from exc
        if len(raw) > _MAX_FILE_BYTES:
            raise WorkspaceApplyError(
                f"workspace file exceeds the 2 MiB limit: {relative}"
            )
        files[relative] = text
    return files


@dataclass(frozen=True, slots=True)
class _FileBackup:
    content: bytes
    mode: int


def _capture_backups(root: Path, paths: Iterable[str]) -> dict[str, _FileBackup | None]:
    backups: dict[str, _FileBackup | None] = {}
    for relative in sorted(set(paths)):
        path = _resolve_relative(root, relative)
        if not path.exists():
            backups[relative] = None
            continue
        if not path.is_file():
            raise WorkspaceApplyError(
                f"workspace path is not a regular file: {relative}"
            )
        raw = path.read_bytes()
        backups[relative] = _FileBackup(raw, S_IMODE(path.stat().st_mode))
    return backups


def _ensure_parent(root: Path, relative: str) -> Path:
    path = _resolve_relative(root, relative)
    parent = path.parent
    if not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise WorkspaceApplyError(f"file parent is not a regular directory: {relative}")
    # Check every component once more after mkdir, so a pre-existing link cannot be
    # silently followed by the atomic replace below.
    _resolve_relative(root, relative)
    return path


def _missing_parent_dirs(root: Path, relative: str) -> tuple[Path, ...]:
    """Return parent directories that an upcoming write will create."""

    _resolve_relative(root, relative)
    current = root
    missing: list[Path] = []
    for part in PurePosixPath(relative).parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise WorkspaceApplyError(f"symbolic links are denied: {relative}")
        if current.exists():
            if not current.is_dir():
                raise WorkspaceApplyError(
                    f"file parent is not a regular directory: {relative}"
                )
        else:
            missing.append(current)
    return tuple(missing)


def _atomic_write(path: Path, content: bytes, mode: int | None) -> None:
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        if mode is not None:
            os.chmod(temporary_name, mode)
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _restore(root: Path, backups: Mapping[str, _FileBackup | None]) -> None:
    for relative, backup in sorted(
        backups.items(), key=lambda item: item[0].count("/"), reverse=True
    ):
        path = _resolve_relative(root, relative)
        if backup is None:
            if path.exists() or path.is_symlink():
                path.unlink()
            continue
        _ensure_parent(root, relative)
        _atomic_write(path, backup.content, backup.mode)


def _categories(
    baseline: Mapping[str, str], target: Mapping[str, str]
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    created = tuple(sorted(set(target) - set(baseline)))
    deleted = tuple(sorted(set(baseline) - set(target)))
    modified = tuple(
        sorted(
            path
            for path in set(baseline) & set(target)
            if baseline[path] != target[path]
        )
    )
    changed = tuple(sorted((*created, *deleted, *modified)))
    return changed, created, modified, deleted


ApplyOperation = Literal["apply", "rollback"]


@dataclass(frozen=True, slots=True)
class ApplyReceipt:
    """Auditable result of one apply or rollback operation."""

    operation_id: str
    operation: ApplyOperation
    run_id: str
    candidate_id: str
    session_fingerprint: str
    workspace: str
    baseline_fingerprint: str
    target_fingerprint: str
    observed_before_fingerprint: str
    observed_after_fingerprint: str
    changed_paths: tuple[str, ...]
    created_paths: tuple[str, ...]
    modified_paths: tuple[str, ...]
    deleted_paths: tuple[str, ...]
    rollback_of: str | None = None
    schema_version: str = "1"

    def __post_init__(self) -> None:
        for name in (
            "operation_id",
            "run_id",
            "candidate_id",
            "session_fingerprint",
            "workspace",
            "baseline_fingerprint",
            "target_fingerprint",
            "observed_before_fingerprint",
            "observed_after_fingerprint",
        ):
            object.__setattr__(self, name, _non_empty(getattr(self, name), name))
        if self.schema_version != "1":
            raise ValueError(
                f"unsupported apply receipt schema: {self.schema_version!r}"
            )
        if self.operation not in {"apply", "rollback"}:
            raise ValueError(f"unsupported apply operation: {self.operation!r}")
        groups = (
            tuple(sorted(self.changed_paths)),
            tuple(sorted(self.created_paths)),
            tuple(sorted(self.modified_paths)),
            tuple(sorted(self.deleted_paths)),
        )
        for group in groups:
            for path in group:
                _normal_path(path, "receipt path")
            if len(set(group)) != len(group):
                raise ValueError("receipt paths must be unique")
        changed, created, modified, deleted = groups
        if set(changed) != set(created) | set(modified) | set(deleted):
            raise ValueError("receipt changed_paths do not match its path categories")
        if (
            set(created) & set(modified)
            or set(created) & set(deleted)
            or set(modified) & set(deleted)
        ):
            raise ValueError("receipt path categories must be disjoint")
        object.__setattr__(self, "changed_paths", changed)
        object.__setattr__(self, "created_paths", created)
        object.__setattr__(self, "modified_paths", modified)
        object.__setattr__(self, "deleted_paths", deleted)
        if self.rollback_of is not None:
            object.__setattr__(
                self, "rollback_of", _non_empty(self.rollback_of, "rollback_of")
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ApplyReceipt:
        if not isinstance(data, Mapping):
            raise ValueError("apply receipt must be an object")
        allowed = {
            "schema_version",
            "operation_id",
            "operation",
            "run_id",
            "candidate_id",
            "session_fingerprint",
            "workspace",
            "baseline_fingerprint",
            "target_fingerprint",
            "observed_before_fingerprint",
            "observed_after_fingerprint",
            "changed_paths",
            "created_paths",
            "modified_paths",
            "deleted_paths",
            "rollback_of",
        }
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"apply receipt has unknown fields: {sorted(unknown)!r}")
        paths = {
            key: data.get(key, [])
            for key in (
                "changed_paths",
                "created_paths",
                "modified_paths",
                "deleted_paths",
            )
        }
        if any(not isinstance(value, list) for value in paths.values()):
            raise ValueError("apply receipt path fields must be arrays")
        receipt = cls(
            schema_version=str(data.get("schema_version", "")),
            operation=data.get("operation"),  # type: ignore[arg-type]
            operation_id=_non_empty(data.get("operation_id"), "operation_id"),
            run_id=_non_empty(data.get("run_id"), "run_id"),
            candidate_id=_non_empty(data.get("candidate_id"), "candidate_id"),
            session_fingerprint=_non_empty(
                data.get("session_fingerprint"), "session_fingerprint"
            ),
            workspace=_non_empty(data.get("workspace"), "workspace"),
            baseline_fingerprint=_non_empty(
                data.get("baseline_fingerprint"), "baseline_fingerprint"
            ),
            target_fingerprint=_non_empty(
                data.get("target_fingerprint"), "target_fingerprint"
            ),
            observed_before_fingerprint=_non_empty(
                data.get("observed_before_fingerprint"), "observed_before_fingerprint"
            ),
            observed_after_fingerprint=_non_empty(
                data.get("observed_after_fingerprint"), "observed_after_fingerprint"
            ),
            changed_paths=tuple(paths["changed_paths"]),
            created_paths=tuple(paths["created_paths"]),
            modified_paths=tuple(paths["modified_paths"]),
            deleted_paths=tuple(paths["deleted_paths"]),
            rollback_of=(
                None if data.get("rollback_of") is None else str(data["rollback_of"])
            ),
        )
        expected_id = stable_hash(
            {
                "operation": receipt.operation,
                "run_id": receipt.run_id,
                "candidate_id": receipt.candidate_id,
                "session_fingerprint": receipt.session_fingerprint,
                "baseline_fingerprint": receipt.baseline_fingerprint,
                "target_fingerprint": receipt.target_fingerprint,
                "changed_paths": list(receipt.changed_paths),
                "rollback_of": receipt.rollback_of,
            }
        )
        if receipt.operation_id != expected_id:
            raise ValueError("apply receipt operation_id is inconsistent")
        return receipt

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "operation_id": self.operation_id,
            "operation": self.operation,
            "run_id": self.run_id,
            "candidate_id": self.candidate_id,
            "session_fingerprint": self.session_fingerprint,
            "workspace": self.workspace,
            "baseline_fingerprint": self.baseline_fingerprint,
            "target_fingerprint": self.target_fingerprint,
            "observed_before_fingerprint": self.observed_before_fingerprint,
            "observed_after_fingerprint": self.observed_after_fingerprint,
            "changed_paths": list(self.changed_paths),
            "created_paths": list(self.created_paths),
            "modified_paths": list(self.modified_paths),
            "deleted_paths": list(self.deleted_paths),
            "rollback_of": self.rollback_of,
        }


def _operation_id(
    *,
    operation: ApplyOperation,
    run_id: str,
    candidate_id: str,
    session_fingerprint: str,
    baseline_fingerprint: str,
    target_fingerprint: str,
    changed_paths: Iterable[str],
    rollback_of: str | None,
) -> str:
    return stable_hash(
        {
            "operation": operation,
            "run_id": run_id,
            "candidate_id": candidate_id,
            "session_fingerprint": session_fingerprint,
            "baseline_fingerprint": baseline_fingerprint,
            "target_fingerprint": target_fingerprint,
            "changed_paths": list(changed_paths),
            "rollback_of": rollback_of,
        }
    )


def apply_snapshot(
    workspace: str | Path,
    baseline_files: Mapping[str, str],
    target_files: Mapping[str, str],
    *,
    run_id: str,
    candidate_id: str,
    session_fingerprint: str = "standalone",
    allow_write: bool = False,
    allow_delete: bool = False,
    operation: ApplyOperation = "apply",
    rollback_of: str | None = None,
) -> ApplyReceipt:
    """Apply target files only when the relevant workspace matches the baseline."""

    if not allow_write:
        raise PermissionError("workspace apply requires explicit allow_write=True")
    baseline = _normalize_files(baseline_files, "baseline_files")
    target = _normalize_files(target_files, "target_files")
    if operation == "rollback":
        allow_delete = True
    changed, created, modified, deleted = _categories(baseline, target)
    if deleted and not allow_delete:
        raise ValueError(
            "target snapshot deletes files; pass allow_delete=True "
            "for explicit deletion"
        )
    root = _workspace_root(workspace)
    relevant = tuple(sorted(set(baseline) | set(target)))
    current = _read_text_paths(root, relevant)
    mismatches = [
        path
        for path in relevant
        if (current.get(path, _MISSING) != baseline.get(path, _MISSING))
    ]
    if mismatches:
        raise WorkspaceConflict(
            "workspace differs from the session baseline at: " + ", ".join(mismatches)
        )
    observed_before = _state_fingerprint(relevant, current)
    backups = _capture_backups(root, relevant)
    created_dirs: list[Path] = []
    try:
        for relative in (*modified, *created):
            created_dirs.extend(_missing_parent_dirs(root, relative))
            path = _ensure_parent(root, relative)
            backup = backups[relative]
            mode = None if backup is None else backup.mode
            _atomic_write(path, target[relative].encode("utf-8"), mode)
        for relative in deleted:
            path = _resolve_relative(root, relative)
            if path.exists() or path.is_symlink():
                path.unlink()
        after = _read_text_paths(root, relevant)
        if any(
            after.get(path, _MISSING) != target.get(path, _MISSING) for path in relevant
        ):
            raise WorkspaceApplyError(
                "workspace did not match target after atomic writes"
            )
    except Exception:
        try:
            _restore(root, backups)
        except Exception as restore_error:
            raise WorkspaceApplyError(
                f"workspace apply failed and rollback also failed: {restore_error}"
            ) from restore_error
        raise
    finally:
        for directory in sorted(
            created_dirs, key=lambda path: len(path.parts), reverse=True
        ):
            with suppress(OSError):
                directory.rmdir()
    observed_after = _state_fingerprint(relevant, after)
    baseline_fingerprint = snapshot_fingerprint(baseline)
    target_fingerprint = snapshot_fingerprint(target)
    operation_id = _operation_id(
        operation=operation,
        run_id=run_id,
        candidate_id=candidate_id,
        session_fingerprint=session_fingerprint,
        baseline_fingerprint=baseline_fingerprint,
        target_fingerprint=target_fingerprint,
        changed_paths=changed,
        rollback_of=rollback_of,
    )
    return ApplyReceipt(
        operation_id=operation_id,
        operation=operation,
        run_id=run_id,
        candidate_id=candidate_id,
        session_fingerprint=session_fingerprint,
        workspace=str(root),
        baseline_fingerprint=baseline_fingerprint,
        target_fingerprint=target_fingerprint,
        observed_before_fingerprint=observed_before,
        observed_after_fingerprint=observed_after,
        changed_paths=changed,
        created_paths=created,
        modified_paths=modified,
        deleted_paths=deleted,
        rollback_of=rollback_of,
    )


def apply_best_snapshot(
    report: SearchSessionReport,
    workspace: str | Path,
    *,
    allow_write: bool = False,
    allow_delete: bool = False,
) -> ApplyReceipt:
    """Apply the accepted candidate from a durable search-session report."""

    if (
        report.status != "accepted"
        or report.best_candidate_id is None
        or report.best_files is None
    ):
        raise ValueError("only an accepted session with a best snapshot can be applied")
    return apply_snapshot(
        workspace,
        report.root_files,
        report.best_files,
        run_id=report.run_id,
        candidate_id=report.best_candidate_id,
        session_fingerprint=stable_hash(report.to_dict()),
        allow_write=allow_write,
        allow_delete=allow_delete,
    )


def rollback_best_snapshot(
    report: SearchSessionReport,
    receipt: ApplyReceipt,
    workspace: str | Path,
    *,
    allow_write: bool = False,
) -> ApplyReceipt:
    """Restore the baseline after verifying the applied workspace is unchanged."""

    if receipt.operation != "apply":
        raise ValueError("rollback requires an apply receipt")
    if (
        report.status != "accepted"
        or report.best_candidate_id is None
        or report.best_files is None
    ):
        raise ValueError("only an accepted session can be rolled back")
    session_fingerprint = stable_hash(report.to_dict())
    if (
        receipt.run_id != report.run_id
        or receipt.candidate_id != report.best_candidate_id
        or receipt.session_fingerprint != session_fingerprint
    ):
        raise WorkspaceConflict(
            "apply receipt does not belong to this session checkpoint"
        )
    return apply_snapshot(
        workspace,
        report.best_files,
        report.root_files,
        run_id=report.run_id,
        candidate_id=report.best_candidate_id,
        session_fingerprint=session_fingerprint,
        allow_write=allow_write,
        allow_delete=True,
        operation="rollback",
        rollback_of=receipt.operation_id,
    )


def write_apply_receipt(receipt: ApplyReceipt, path: str | Path) -> None:
    """Atomically persist an apply/rollback receipt."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        receipt.to_dict(), ensure_ascii=False, sort_keys=True, indent=2
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload + "\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        if temporary_path is None:
            raise OSError("temporary receipt path was not created")
        os.replace(temporary_path, target)
    except OSError:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def read_apply_receipt(path: str | Path) -> ApplyReceipt:
    target = Path(path)
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read apply receipt {target}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ValueError("apply receipt must be a JSON object")
    return ApplyReceipt.from_dict(data)


__all__ = [
    "ApplyReceipt",
    "WorkspaceApplyError",
    "WorkspaceConflict",
    "apply_best_snapshot",
    "apply_snapshot",
    "read_apply_receipt",
    "rollback_best_snapshot",
    "snapshot_fingerprint",
    "write_apply_receipt",
]
