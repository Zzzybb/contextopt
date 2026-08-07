"""Workspace-bounded tools exposed to a coding agent."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unicodedata
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from contextopt.runtime.identity import stable_hash
from contextopt.runtime.protocol import (
    RunLimits,
    RunPermissions,
    ToolCall,
    ToolDefinition,
    ToolOutcome,
)
from contextopt.runtime.tool_state import (
    ReplayPolicy,
    ToolExecutionPlan,
    ToolReconciliation,
    tool_call_fingerprint,
    tool_replay_policy,
)

_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
_DENIED_TOP_LEVEL = {".git", ".contextopt"}
_SAFE_ENV_NAMES = {
    "HOME",
    "LANG",
    "LC_ALL",
    "PATH",
    "PYTHONIOENCODING",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "TMPDIR",
    "USERPROFILE",
    "WINDIR",
}


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _error(call: ToolCall, code: str, message: str) -> ToolOutcome:
    return ToolOutcome(
        call_id=call.id,
        tool_name=call.name,
        ok=False,
        content=message,
        error_code=code,
    )


def _require_string(arguments: Mapping[str, Any], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _optional_int(
    arguments: Mapping[str, Any], name: str, default: int, *, minimum: int = 1
) -> int:
    value = arguments.get(name, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _truncate_text(text: str, limit_bytes: int) -> tuple[str, dict[str, Any]]:
    encoded = text.encode("utf-8", errors="replace")
    digest = _sha256(encoded)
    if len(encoded) <= limit_bytes:
        return text, {
            "truncated": False,
            "total_bytes": len(encoded),
            "retained_bytes": len(encoded),
            "omitted_bytes": 0,
            "sha256": digest,
        }
    marker = b"\n... output truncated ...\n"
    retained_budget = max(0, limit_bytes - len(marker))
    head_size = retained_budget // 2
    tail_size = retained_budget - head_size
    retained = encoded[:head_size] + marker + encoded[-tail_size:]
    return retained.decode("utf-8", errors="replace"), {
        "truncated": True,
        "total_bytes": len(encoded),
        "retained_bytes": len(retained),
        "omitted_bytes": len(encoded) - head_size - tail_size,
        "sha256": digest,
    }


class _PathViolation(ValueError):
    pass


class _WorkspaceResolver:
    def __init__(self, root: Path) -> None:
        resolved = root.resolve(strict=True)
        if not resolved.is_dir():
            raise ValueError("workspace must be a directory")
        self.root = resolved

    @staticmethod
    def _validated_parts(raw: str) -> tuple[str, ...]:
        if not raw or raw != unicodedata.normalize("NFC", raw):
            raise _PathViolation("path must be non-empty normalized Unicode")
        if "\\" in raw:
            raise _PathViolation("paths must use forward slashes")
        if any(ord(character) < 32 for character in raw):
            raise _PathViolation("path contains control characters")
        posix = PurePosixPath(raw)
        windows = PureWindowsPath(raw)
        if posix.is_absolute() or windows.is_absolute() or windows.drive:
            raise _PathViolation("absolute and drive-relative paths are denied")
        if any(part in {"", ".."} for part in posix.parts):
            raise _PathViolation("parent traversal is denied")
        parts = tuple(part for part in posix.parts if part != ".")
        if parts and parts[0].casefold() in _DENIED_TOP_LEVEL:
            raise _PathViolation("runtime and version-control internals are denied")
        for part in parts:
            if part.endswith((" ", ".")) or ":" in part:
                raise _PathViolation("path contains a non-portable component")
            stem = part.split(".", 1)[0].upper()
            if stem in _WINDOWS_RESERVED:
                raise _PathViolation("path contains a reserved device name")
        return parts

    def lexical_relative(self, raw: str) -> str:
        """Validate a path lexically without consulting mutable filesystem state."""

        parts = self._validated_parts(raw)
        return PurePosixPath(*parts).as_posix()

    def resolve(
        self,
        raw: str,
        *,
        must_exist: bool,
        allow_directory: bool = False,
    ) -> tuple[Path, str]:
        parts = self._validated_parts(raw)

        candidate = self.root.joinpath(*parts)
        current = self.root
        for part in parts:
            current = current / part
            if current.exists() and current.is_symlink():
                raise _PathViolation("symbolic links are denied")
        resolved = candidate.resolve(strict=must_exist)
        try:
            common = os.path.commonpath(
                (os.path.normcase(str(self.root)), os.path.normcase(str(resolved)))
            )
        except ValueError as exc:
            raise _PathViolation("path is outside the workspace") from exc
        if common != os.path.normcase(str(self.root)):
            raise _PathViolation("path is outside the workspace")
        if must_exist and not allow_directory and not resolved.is_file():
            raise _PathViolation("path is not a regular file")
        if must_exist and allow_directory and not resolved.is_dir():
            raise _PathViolation("path is not a directory")
        relative = resolved.relative_to(self.root).as_posix()
        return resolved, relative or "."


class WorkspaceTools:
    """A narrow tool registry rooted at one trusted workspace."""

    def __init__(
        self,
        workspace: str | Path,
        *,
        permissions: RunPermissions | None = None,
        limits: RunLimits | None = None,
        test_commands: Mapping[str, tuple[str, ...]] | None = None,
    ) -> None:
        self._resolver = _WorkspaceResolver(Path(workspace))
        self.permissions = permissions or RunPermissions()
        self.limits = limits or RunLimits()
        self._test_commands = self._normalize_commands(test_commands or {})
        self._handlers: dict[
            str, Callable[[ToolCall, Mapping[str, Any]], ToolOutcome]
        ] = {
            "list_files": self._list_files,
            "search_text": self._search_text,
            "read_file": self._read_file,
            "create_file": self._create_file,
            "replace_text": self._replace_text,
        }
        if self._test_commands:
            self._handlers["run_tests"] = self._run_tests
        self._definitions = self._build_definitions()

    @property
    def workspace(self) -> Path:
        return self._resolver.root

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        return self._definitions

    @property
    def configuration_fingerprint(self) -> str:
        """Fingerprint the exact workspace/tool boundary used by a resumable run."""

        workspace_identity = stable_hash(
            {"resolved_path": os.path.normcase(str(self.workspace))}
        )
        return stable_hash(
            {
                "workspace_identity": workspace_identity,
                "permissions": self.permissions.to_dict(),
                "definitions": [item.to_dict() for item in self.definitions],
                "test_commands": {
                    scope: list(argv)
                    for scope, argv in sorted(self._test_commands.items())
                },
            }
        )

    async def execute(self, call: ToolCall) -> ToolOutcome:
        handler = self._handlers.get(call.name)
        if handler is None:
            return _error(call, "unknown_tool", f"unknown tool: {call.name}")
        try:
            decoded: Any = json.loads(call.arguments_json)
        except json.JSONDecodeError as exc:
            return _error(
                call, "invalid_arguments", f"arguments are not valid JSON: {exc.msg}"
            )
        if not isinstance(decoded, dict):
            return _error(
                call, "invalid_arguments", "tool arguments must be a JSON object"
            )
        try:
            return await asyncio.to_thread(handler, call, decoded)
        except _PathViolation as exc:
            return _error(call, "path_denied", str(exc))
        except FileNotFoundError:
            return _error(call, "not_found", "path does not exist")
        except UnicodeDecodeError:
            return _error(call, "decode_error", "file is not valid UTF-8 text")
        except ValueError as exc:
            return _error(call, "invalid_arguments", str(exc))
        except PermissionError:
            return _error(call, "permission_denied", "operating system denied access")
        except OSError as exc:
            return _error(
                call, "io_error", f"workspace operation failed: {exc.strerror or exc}"
            )

    async def prepare(
        self,
        call: ToolCall,
        operation_id: str,
        fingerprint: str,
    ) -> ToolExecutionPlan:
        """Seal the replay policy and expected state transition before execution.

        Calls that cannot currently succeed raise ``ValueError`` or
        ``PermissionError``; callers may pass those calls to :meth:`execute` to
        preserve its structured failure outcome.
        """

        return await asyncio.to_thread(
            self._prepare_sync, call, operation_id, fingerprint
        )

    async def execute_prepared(self, plan: ToolExecutionPlan) -> ToolOutcome:
        """Execute an intact plan once, preserving the existing tool API semantics."""

        plan.validate()
        self._validate_plan_semantics(plan)
        outcome = await self.execute(plan.call)
        if (
            plan.replay_policy == "reconcile"
            and outcome.ok
            and outcome.to_dict() != plan.planned_outcome.to_dict()
        ):
            raise RuntimeError(
                "tool result did not match its planned successful outcome"
            )
        return outcome

    async def reconcile(self, plan: ToolExecutionPlan) -> ToolReconciliation:
        """Inspect an interrupted plan without executing its side effect."""

        return await asyncio.to_thread(self._reconcile_sync, plan)

    @staticmethod
    def _decode_arguments(call: ToolCall) -> dict[str, Any]:
        try:
            decoded: Any = json.loads(call.arguments_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"arguments are not valid JSON: {exc.msg}") from None
        if not isinstance(decoded, dict):
            raise ValueError("tool arguments must be a JSON object")
        return decoded

    @staticmethod
    def _generic_planned_outcome(
        call: ToolCall, replay_policy: ReplayPolicy
    ) -> ToolOutcome:
        return ToolOutcome(
            call_id=call.id,
            tool_name=call.name,
            ok=True,
            content=f"planned successful {call.name} execution",
            metadata={"replay_policy": replay_policy},
        )

    @staticmethod
    def _create_success_outcome(
        call: ToolCall, relative: str, encoded: bytes, digest: str
    ) -> ToolOutcome:
        return ToolOutcome(
            call_id=call.id,
            tool_name=call.name,
            ok=True,
            content=f"created {relative} ({len(encoded)} bytes, sha256={digest})",
            metadata={
                "path": relative,
                "bytes_written": len(encoded),
                "sha256": digest,
            },
        )

    @staticmethod
    def _replace_success_outcome(
        call: ToolCall,
        relative: str,
        before_sha256: str,
        after_sha256: str,
        replacements: int,
        bytes_written: int,
    ) -> ToolOutcome:
        return ToolOutcome(
            call_id=call.id,
            tool_name=call.name,
            ok=True,
            content=(
                f"replaced {replacements} occurrence(s) in {relative}; "
                f"sha256={after_sha256}"
            ),
            metadata={
                "path": relative,
                "before_sha256": before_sha256,
                "after_sha256": after_sha256,
                "replacements": replacements,
                "bytes_written": bytes_written,
            },
        )

    def _prepare_sync(
        self,
        call: ToolCall,
        operation_id: str,
        fingerprint: str,
    ) -> ToolExecutionPlan:
        expected_fingerprint = tool_call_fingerprint(call)
        if fingerprint != expected_fingerprint:
            raise ValueError("tool call fingerprint does not match the call")
        if call.name not in self._handlers:
            raise ValueError(f"unknown tool: {call.name}")
        arguments = self._decode_arguments(call)
        replay_policy = tool_replay_policy(call.name)
        if call.name == "create_file":
            return self._prepare_create(call, arguments, operation_id, fingerprint)
        if call.name == "replace_text":
            return self._prepare_replace(call, arguments, operation_id, fingerprint)
        if call.name == "run_tests":
            if not self.permissions.allow_command:
                raise PermissionError("command permission is disabled")
            scope = _require_string(arguments, "scope")
            if scope not in self._test_commands:
                raise ValueError(f"unknown test scope: {scope}")
            return ToolExecutionPlan(
                operation_id=operation_id,
                call=call,
                fingerprint=fingerprint,
                replay_policy="never",
                preconditions={"scope": scope},
                postconditions={},
                planned_outcome=self._generic_planned_outcome(call, "never"),
            )
        return ToolExecutionPlan(
            operation_id=operation_id,
            call=call,
            fingerprint=fingerprint,
            replay_policy=replay_policy,
            preconditions={},
            postconditions={},
            planned_outcome=self._generic_planned_outcome(call, replay_policy),
        )

    def _prepare_create(
        self,
        call: ToolCall,
        arguments: Mapping[str, Any],
        operation_id: str,
        fingerprint: str,
    ) -> ToolExecutionPlan:
        if not self.permissions.allow_write:
            raise PermissionError("write permission is disabled")
        raw_path = _require_string(arguments, "path")
        content = _require_string(arguments, "content")
        path, relative = self._resolver.resolve(raw_path, must_exist=False)
        if path.exists():
            raise ValueError("file already exists")
        if not path.parent.exists() or not path.parent.is_dir():
            raise ValueError("parent directory does not exist")
        encoded = content.encode("utf-8")
        if len(encoded) > 2 * 1024 * 1024:
            raise ValueError("new file exceeds the 2 MiB limit")
        digest = _sha256(encoded)
        return ToolExecutionPlan(
            operation_id=operation_id,
            call=call,
            fingerprint=fingerprint,
            replay_policy="reconcile",
            preconditions={"path": relative, "exists": False},
            postconditions={
                "path": relative,
                "exists": True,
                "sha256": digest,
                "bytes": len(encoded),
            },
            planned_outcome=self._create_success_outcome(
                call, relative, encoded, digest
            ),
        )

    def _prepare_replace(
        self,
        call: ToolCall,
        arguments: Mapping[str, Any],
        operation_id: str,
        fingerprint: str,
    ) -> ToolExecutionPlan:
        if not self.permissions.allow_write:
            raise PermissionError("write permission is disabled")
        raw_path = _require_string(arguments, "path")
        old_text = _require_string(arguments, "old_text")
        new_text = _require_string(arguments, "new_text")
        expected_sha256 = _require_string(arguments, "expected_sha256")
        expected_occurrences = _optional_int(arguments, "expected_occurrences", 1)
        if not old_text:
            raise ValueError("old_text must not be empty")
        path, relative = self._resolver.resolve(raw_path, must_exist=True)
        raw = self._read_bytes(path)
        digest = _sha256(raw)
        if digest != expected_sha256:
            raise ValueError(f"file changed since it was read; current sha256={digest}")
        text = raw.decode("utf-8-sig")
        occurrences = text.count(old_text)
        if occurrences != expected_occurrences:
            raise ValueError(
                f"expected {expected_occurrences} exact occurrence(s), "
                f"found {occurrences}"
            )
        encoded = text.replace(old_text, new_text, expected_occurrences).encode("utf-8")
        if len(encoded) > 2 * 1024 * 1024:
            raise ValueError("updated file exceeds the 2 MiB limit")
        new_digest = _sha256(encoded)
        return ToolExecutionPlan(
            operation_id=operation_id,
            call=call,
            fingerprint=fingerprint,
            replay_policy="reconcile",
            preconditions={
                "path": relative,
                "exists": True,
                "sha256": digest,
            },
            postconditions={
                "path": relative,
                "exists": True,
                "sha256": new_digest,
                "bytes": len(encoded),
            },
            planned_outcome=self._replace_success_outcome(
                call,
                relative,
                digest,
                new_digest,
                expected_occurrences,
                len(encoded),
            ),
        )

    @staticmethod
    def _condition_matches(
        current: Mapping[str, Any], expected: Mapping[str, Any]
    ) -> bool:
        return all(current.get(key) == value for key, value in expected.items())

    def _observe_file(self, raw_path: str) -> tuple[dict[str, Any] | None, str | None]:
        try:
            path, relative = self._resolver.resolve(raw_path, must_exist=False)
        except (OSError, _PathViolation) as exc:
            return None, str(exc)
        if not path.exists():
            return {"path": relative, "exists": False}, None
        if path.is_symlink() or not path.is_file():
            return None, "path is no longer a regular file"
        try:
            raw = self._read_bytes_for_reconciliation(path)
        except (OSError, ValueError) as exc:
            return None, str(exc)
        return {
            "path": relative,
            "exists": True,
            "sha256": _sha256(raw),
            "bytes": len(raw),
        }, None

    def _validate_plan_semantics(self, plan: ToolExecutionPlan) -> None:
        expected_policy = tool_replay_policy(plan.call.name)
        if plan.replay_policy != expected_policy:
            raise ValueError("tool execution plan has an invalid replay policy")
        arguments = self._decode_arguments(plan.call)
        if expected_policy == "safe":
            expected_outcome = self._generic_planned_outcome(plan.call, "safe")
            if (
                dict(plan.preconditions)
                or dict(plan.postconditions)
                or plan.planned_outcome.to_dict() != expected_outcome.to_dict()
            ):
                raise ValueError("read-only tool execution plan is inconsistent")
            return
        if expected_policy == "never":
            scope = _require_string(arguments, "scope")
            expected_outcome = self._generic_planned_outcome(plan.call, "never")
            if (
                dict(plan.preconditions) != {"scope": scope}
                or dict(plan.postconditions)
                or plan.planned_outcome.to_dict() != expected_outcome.to_dict()
            ):
                raise ValueError("non-replayable tool execution plan is inconsistent")
            return

        raw_path = _require_string(arguments, "path")
        relative = self._resolver.lexical_relative(raw_path)
        if plan.call.name == "create_file":
            content = _require_string(arguments, "content")
            encoded = content.encode("utf-8")
            digest = _sha256(encoded)
            expected_pre = {"path": relative, "exists": False}
            expected_post = {
                "path": relative,
                "exists": True,
                "sha256": digest,
                "bytes": len(encoded),
            }
            expected_outcome = self._create_success_outcome(
                plan.call, relative, encoded, digest
            )
        else:
            expected_sha256 = _require_string(arguments, "expected_sha256")
            expected_occurrences = _optional_int(arguments, "expected_occurrences", 1)
            expected_pre = {
                "path": relative,
                "exists": True,
                "sha256": expected_sha256,
            }
            expected_post = dict(plan.postconditions)
            after_sha256 = expected_post.get("sha256")
            bytes_written = expected_post.get("bytes")
            if not isinstance(after_sha256, str) or not isinstance(bytes_written, int):
                raise ValueError("replace plan postcondition is incomplete")
            expected_post = {
                "path": relative,
                "exists": True,
                "sha256": after_sha256,
                "bytes": bytes_written,
            }
            expected_outcome = self._replace_success_outcome(
                plan.call,
                relative,
                expected_sha256,
                after_sha256,
                expected_occurrences,
                bytes_written,
            )
        if (
            dict(plan.preconditions) != expected_pre
            or dict(plan.postconditions) != expected_post
            or plan.planned_outcome.to_dict() != expected_outcome.to_dict()
        ):
            raise ValueError("write tool execution plan is inconsistent")

    def _reconcile_sync(self, plan: ToolExecutionPlan) -> ToolReconciliation:
        plan.validate()
        self._validate_plan_semantics(plan)
        if plan.replay_policy == "safe":
            return ToolReconciliation(
                action="retry",
                reason="read-only tool is safe to execute again",
            )
        if plan.replay_policy == "never":
            return ToolReconciliation(
                action="paused",
                reason=(
                    "tool may have produced an external effect; automatic replay is "
                    "disabled"
                ),
            )

        arguments = self._decode_arguments(plan.call)
        raw_path = _require_string(arguments, "path")
        current, observation_error = self._observe_file(raw_path)
        if current is None:
            return ToolReconciliation(
                action="divergence",
                reason=f"cannot verify workspace state: {observation_error}",
            )
        if self._condition_matches(current, plan.postconditions):
            return ToolReconciliation(
                action="completed",
                reason="workspace already matches the planned postcondition",
                outcome=plan.planned_outcome,
            )
        if self._condition_matches(current, plan.preconditions):
            return ToolReconciliation(
                action="retry",
                reason="workspace still matches the planned precondition",
            )
        return ToolReconciliation(
            action="divergence",
            reason=(
                "workspace matches neither the planned precondition nor "
                f"postcondition: current={json.dumps(current, sort_keys=True)}"
            ),
        )

    def _build_definitions(self) -> tuple[ToolDefinition, ...]:
        definitions = [
            ToolDefinition(
                name="list_files",
                description="List regular files under a workspace-relative directory.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "default": "."},
                        "max_entries": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 1000,
                        },
                    },
                    "additionalProperties": False,
                },
            ),
            ToolDefinition(
                name="search_text",
                description="Find a literal string in UTF-8 workspace files.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "path": {"type": "string", "default": "."},
                        "case_sensitive": {"type": "boolean", "default": True},
                        "max_matches": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 500,
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            ),
            ToolDefinition(
                name="read_file",
                description="Read numbered lines and a SHA-256 from a UTF-8 file.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start_line": {"type": "integer", "minimum": 1},
                        "max_lines": {"type": "integer", "minimum": 1, "maximum": 2000},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
            ),
            ToolDefinition(
                name="create_file",
                description=(
                    "Create one new UTF-8 file. Existing files are never overwritten."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
            ),
            ToolDefinition(
                name="replace_text",
                description=(
                    "Atomically replace exact text in a UTF-8 file using the SHA-256 "
                    "returned by read_file as a compare-and-swap precondition."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "old_text": {"type": "string"},
                        "new_text": {"type": "string"},
                        "expected_sha256": {"type": "string"},
                        "expected_occurrences": {"type": "integer", "minimum": 1},
                    },
                    "required": ["path", "old_text", "new_text", "expected_sha256"],
                    "additionalProperties": False,
                },
            ),
        ]
        if self._test_commands:
            definitions.append(
                ToolDefinition(
                    name="run_tests",
                    description="Run one trusted, pre-registered test command.",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "scope": {
                                "type": "string",
                                "enum": sorted(self._test_commands),
                            }
                        },
                        "required": ["scope"],
                        "additionalProperties": False,
                    },
                )
            )
        return tuple(definitions)

    @staticmethod
    def _normalize_commands(
        commands: Mapping[str, tuple[str, ...]],
    ) -> dict[str, tuple[str, ...]]:
        normalized: dict[str, tuple[str, ...]] = {}
        for scope, argv in commands.items():
            if (
                not scope
                or not argv
                or any(not isinstance(part, str) or not part for part in argv)
            ):
                raise ValueError("test commands require a scope and non-empty argv")
            executable = argv[0]
            resolved = (
                str(Path(executable).resolve(strict=True))
                if Path(executable).is_absolute()
                else shutil.which(executable)
            )
            if not resolved:
                raise ValueError(f"test executable was not found: {executable}")
            normalized[scope] = (resolved, *argv[1:])
        return normalized

    def _read_bytes(self, path: Path) -> bytes:
        size = path.stat().st_size
        hard_limit = max(self.limits.max_tool_output_bytes * 8, 2 * 1024 * 1024)
        if size > hard_limit:
            raise ValueError(f"file exceeds the {hard_limit}-byte safety limit")
        content = path.read_bytes()
        if b"\x00" in content:
            raise ValueError("binary files are not supported")
        return content

    def _read_bytes_for_reconciliation(self, path: Path) -> bytes:
        hard_limit = max(self.limits.max_tool_output_bytes * 8, 2 * 1024 * 1024)
        with path.open("rb") as handle:
            content = handle.read(hard_limit + 1)
        if len(content) > hard_limit:
            raise ValueError(f"file exceeds the {hard_limit}-byte safety limit")
        return content

    def _list_files(self, call: ToolCall, arguments: Mapping[str, Any]) -> ToolOutcome:
        raw_path = str(arguments.get("path", "."))
        max_entries = _optional_int(arguments, "max_entries", 200)
        directory, relative_root = self._resolver.resolve(
            raw_path, must_exist=True, allow_directory=True
        )
        files: list[str] = []
        for candidate in sorted(directory.rglob("*"), key=lambda item: item.as_posix()):
            try:
                relative = candidate.relative_to(self.workspace)
            except ValueError:
                continue
            if not relative.parts or relative.parts[0].casefold() in _DENIED_TOP_LEVEL:
                continue
            if candidate.is_symlink() or not candidate.is_file():
                continue
            files.append(relative.as_posix())
            if len(files) >= max_entries:
                break
        content = "\n".join(files) if files else "No files found."
        return ToolOutcome(
            call_id=call.id,
            tool_name=call.name,
            ok=True,
            content=content,
            metadata={
                "path": relative_root,
                "entries": len(files),
                "truncated": len(files) >= max_entries,
            },
            truncated=len(files) >= max_entries,
        )

    def _search_text(self, call: ToolCall, arguments: Mapping[str, Any]) -> ToolOutcome:
        query = _require_string(arguments, "query")
        if not query:
            raise ValueError("query must not be empty")
        raw_path = str(arguments.get("path", "."))
        case_sensitive = arguments.get("case_sensitive", True)
        if not isinstance(case_sensitive, bool):
            raise ValueError("case_sensitive must be a boolean")
        max_matches = _optional_int(arguments, "max_matches", 100)
        directory, relative_root = self._resolver.resolve(
            raw_path, must_exist=True, allow_directory=True
        )
        needle = query if case_sensitive else query.casefold()
        matches: list[str] = []
        files_scanned = 0
        for candidate in sorted(directory.rglob("*"), key=lambda item: item.as_posix()):
            try:
                relative = candidate.relative_to(self.workspace)
            except ValueError:
                continue
            if not relative.parts or relative.parts[0].casefold() in _DENIED_TOP_LEVEL:
                continue
            if candidate.is_symlink() or not candidate.is_file():
                continue
            try:
                raw = self._read_bytes(candidate)
                text = raw.decode("utf-8-sig")
            except (UnicodeDecodeError, ValueError, OSError):
                continue
            files_scanned += 1
            for line_number, line in enumerate(text.splitlines(), start=1):
                haystack = line if case_sensitive else line.casefold()
                column = haystack.find(needle)
                if column < 0:
                    continue
                matches.append(
                    f"{relative.as_posix()}:{line_number}:{column + 1}:{line[:500]}"
                )
                if len(matches) >= max_matches:
                    break
            if len(matches) >= max_matches:
                break
        content = "\n".join(matches) if matches else "No matches found."
        content, capture = _truncate_text(content, self.limits.max_tool_output_bytes)
        truncated = len(matches) >= max_matches or bool(capture["truncated"])
        return ToolOutcome(
            call_id=call.id,
            tool_name=call.name,
            ok=True,
            content=content,
            metadata={
                "path": relative_root,
                "matches": len(matches),
                "files_scanned": files_scanned,
                "capture": capture,
            },
            truncated=truncated,
        )

    def _read_file(self, call: ToolCall, arguments: Mapping[str, Any]) -> ToolOutcome:
        raw_path = _require_string(arguments, "path")
        start_line = _optional_int(arguments, "start_line", 1)
        max_lines = _optional_int(arguments, "max_lines", 400)
        path, relative = self._resolver.resolve(raw_path, must_exist=True)
        raw = self._read_bytes(path)
        text = raw.decode("utf-8-sig")
        lines = text.splitlines()
        selected = lines[start_line - 1 : start_line - 1 + max_lines]
        numbered = "\n".join(
            f"{line_number}: {line}"
            for line_number, line in enumerate(selected, start=start_line)
        )
        content, capture = _truncate_text(numbered, self.limits.max_tool_output_bytes)
        end_line = start_line + len(selected) - 1 if selected else start_line - 1
        digest = _sha256(raw)
        header = (
            f"path={relative}\nsha256={digest}\nlines={start_line}-{end_line}"
            f"/{len(lines)}\n"
        )
        return ToolOutcome(
            call_id=call.id,
            tool_name=call.name,
            ok=True,
            content=header + content,
            metadata={
                "path": relative,
                "sha256": digest,
                "file_bytes": len(raw),
                "start_line": start_line,
                "end_line": end_line,
                "total_lines": len(lines),
                "capture": capture,
            },
            truncated=(end_line < len(lines) or bool(capture["truncated"])),
        )

    def _create_file(self, call: ToolCall, arguments: Mapping[str, Any]) -> ToolOutcome:
        if not self.permissions.allow_write:
            return _error(call, "write_denied", "write permission is disabled")
        raw_path = _require_string(arguments, "path")
        content = _require_string(arguments, "content")
        path, relative = self._resolver.resolve(raw_path, must_exist=False)
        if path.exists():
            return _error(call, "already_exists", "file already exists")
        if not path.parent.exists() or not path.parent.is_dir():
            return _error(call, "parent_not_found", "parent directory does not exist")
        encoded = content.encode("utf-8")
        if len(encoded) > 2 * 1024 * 1024:
            return _error(call, "too_large", "new file exceeds the 2 MiB limit")
        with path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        digest = _sha256(encoded)
        return self._create_success_outcome(call, relative, encoded, digest)

    def _replace_text(
        self, call: ToolCall, arguments: Mapping[str, Any]
    ) -> ToolOutcome:
        if not self.permissions.allow_write:
            return _error(call, "write_denied", "write permission is disabled")
        raw_path = _require_string(arguments, "path")
        old_text = _require_string(arguments, "old_text")
        new_text = _require_string(arguments, "new_text")
        expected_sha256 = _require_string(arguments, "expected_sha256")
        expected_occurrences = _optional_int(arguments, "expected_occurrences", 1)
        if not old_text:
            raise ValueError("old_text must not be empty")
        path, relative = self._resolver.resolve(raw_path, must_exist=True)
        raw = self._read_bytes(path)
        digest = _sha256(raw)
        if digest != expected_sha256:
            return _error(
                call,
                "content_conflict",
                f"file changed since it was read; current sha256={digest}",
            )
        text = raw.decode("utf-8-sig")
        occurrences = text.count(old_text)
        if occurrences != expected_occurrences:
            message = (
                f"expected {expected_occurrences} exact occurrence(s), "
                f"found {occurrences}"
            )
            return _error(
                call,
                "content_conflict",
                message,
            )
        updated = text.replace(old_text, new_text, expected_occurrences)
        encoded = updated.encode("utf-8")
        if len(encoded) > 2 * 1024 * 1024:
            return _error(call, "too_large", "updated file exceeds the 2 MiB limit")
        mode = path.stat().st_mode
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
            ) as handle:
                temporary_name = handle.name
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary_name, mode)
            os.replace(temporary_name, path)
            temporary_name = None
        finally:
            if temporary_name is not None:
                Path(temporary_name).unlink(missing_ok=True)
        new_digest = _sha256(encoded)
        return self._replace_success_outcome(
            call,
            relative,
            digest,
            new_digest,
            expected_occurrences,
            len(encoded),
        )

    def _run_tests(self, call: ToolCall, arguments: Mapping[str, Any]) -> ToolOutcome:
        if not self.permissions.allow_command:
            return _error(call, "command_denied", "command permission is disabled")
        scope = _require_string(arguments, "scope")
        argv = self._test_commands.get(scope)
        if argv is None:
            return _error(call, "unknown_test_scope", f"unknown test scope: {scope}")
        display_argv = [Path(argv[0]).name, *argv[1:]]
        environment = {
            key: value
            for key, value in os.environ.items()
            if key.upper() in _SAFE_ENV_NAMES
        }
        environment.setdefault("PYTHONIOENCODING", "utf-8")
        try:
            completed = subprocess.run(
                argv,
                cwd=self.workspace,
                env=environment,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.limits.command_timeout_seconds,
                shell=False,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")
            output, capture = _truncate_text(
                f"stdout:\n{stdout}\nstderr:\n{stderr}",
                self.limits.max_tool_output_bytes,
            )
            return ToolOutcome(
                call_id=call.id,
                tool_name=call.name,
                ok=False,
                content=f"test command timed out\n{output}",
                error_code="timeout",
                metadata={"scope": scope, "argv": display_argv, "capture": capture},
                truncated=bool(capture["truncated"]),
            )
        output, capture = _truncate_text(
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
            self.limits.max_tool_output_bytes,
        )
        content = f"scope={scope}\nexit_code={completed.returncode}\n{output}"
        return ToolOutcome(
            call_id=call.id,
            tool_name=call.name,
            ok=True,
            content=content,
            metadata={
                "scope": scope,
                "argv": display_argv,
                "exit_code": completed.returncode,
                "capture": capture,
            },
            truncated=bool(capture["truncated"]),
        )
