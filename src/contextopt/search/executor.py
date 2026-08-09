"""A deliberately explicit adapter for running branch candidates in temp workspaces.

``BranchSearch`` itself is pure: it consumes candidate snapshots and test observations.
This module is the narrow side-effect boundary for a local experiment. It materializes
one complete snapshot per unique workspace fingerprint, runs a caller-supplied argv
without a shell, starts it in a killable process group, removes common credential
environment variables, bounds the report excerpt, and converts the exit status into
the same ``TestResult`` consumed by the search core.

The command is still a trusted host process, not a sandbox. Callers must opt into it at
the CLI with ``--allow-command`` and should use a disposable workspace or CI runner.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import signal
import subprocess
import tempfile
import uuid
from collections.abc import Iterable, Mapping
from contextlib import suppress
from ctypes import wintypes
from dataclasses import dataclass
from math import isfinite
from pathlib import Path, PurePosixPath
from time import perf_counter
from typing import Any, Literal, cast

from contextopt.search.branching import (
    BranchCase,
    BranchSearch,
    BranchSearchConfig,
    BranchSearchReport,
    CandidatePatch,
    TestResult,
)

ExecutionSandbox = Literal["host", "docker"]


def _non_empty(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _bounded_output(stdout: bytes, stderr: bytes, limit: int) -> tuple[str, str]:
    combined = stdout + b"\n--- stderr ---\n" + stderr
    digest = hashlib.sha256(combined).hexdigest()
    if len(combined) > limit:
        marker = b"\n... output excerpt truncated ...\n"
        if limit <= len(marker):
            combined = marker[:limit]
        else:
            retained = limit - len(marker)
            head = retained // 2
            tail = retained - head
            tail_bytes = combined[-tail:] if tail else b""
            combined = combined[:head] + marker + tail_bytes
    excerpt = combined.decode("utf-8", errors="replace")
    if len(excerpt) > 4096:
        excerpt = excerpt[:2048] + "\n... excerpt shortened ...\n" + excerpt[-2048:]
    return digest, excerpt


_SENSITIVE_ENV_NAMES = frozenset(
    {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AZURE_CLIENT_SECRET",
        "CI_JOB_TOKEN",
        "CONTEXTOPT_API_KEY",
        "GITHUB_TOKEN",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "OPENAI_API_KEY",
    }
)
_SENSITIVE_ENV_SUFFIXES = (
    "_ACCESS_TOKEN",
    "_API_KEY",
    "_PASSWORD",
    "_PRIVATE_KEY",
    "_SECRET",
    "_TOKEN",
)


def _child_environment() -> dict[str, str]:
    """Return a host-compatible environment without common credential variables."""

    environment = os.environ.copy()
    for name in tuple(environment):
        upper = name.upper()
        if upper in _SENSITIVE_ENV_NAMES or upper.endswith(_SENSITIVE_ENV_SUFFIXES):
            environment.pop(name, None)
    return environment


def _new_process_group_kwargs() -> dict[str, Any]:
    """Start a candidate command in a group that can be terminated on timeout."""

    if os.name == "nt":
        # The constant is only exported by the Windows stdlib implementation;
        # keep the fallback explicit so the same module type-checks on POSIX.
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200)}
    return {"start_new_session": True}


def _attach_windows_job(process: subprocess.Popen[bytes]) -> tuple[Any, Any] | None:
    """Attach a Windows process to a kill-on-close Job Object when available."""

    if os.name != "nt":
        return None

    win_dll = getattr(ctypes, "WinDLL", None)
    if win_dll is None:
        return None
    kernel32: Any = win_dll("kernel32", use_last_error=True)

    class BasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class ExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BasicLimitInformation),
            ("IoInfo", IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        wintypes.INT,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel32.AssignProcessToJobObject.argtypes = [
        wintypes.HANDLE,
        wintypes.HANDLE,
    ]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return None
    information = ExtendedLimitInformation()
    information.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
    configured = kernel32.SetInformationJobObject(
        job,
        9,  # JobObjectExtendedLimitInformation
        ctypes.byref(information),
        ctypes.sizeof(information),
    )
    assigned = configured and kernel32.AssignProcessToJobObject(
        job,
        process._handle,  # type: ignore[attr-defined]
    )
    if not assigned:
        kernel32.CloseHandle(job)
        return None
    return kernel32, job


def _close_windows_job(handle: tuple[Any, Any] | None) -> None:
    if handle is not None:
        handle[0].CloseHandle(handle[1])


def _terminate_process_tree(process: subprocess.Popen[bytes]) -> None:
    """Best-effort terminate the process group without weakening the timeout result."""

    if process.poll() is not None:
        return
    if os.name == "nt":
        taskkill = shutil.which("taskkill")
        if taskkill is not None:
            with suppress(OSError, subprocess.TimeoutExpired):
                subprocess.run(
                    [taskkill, "/PID", str(process.pid), "/T", "/F"],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                    check=False,
                )
        if process.poll() is None:
            process.kill()
    else:
        killpg = getattr(os, "killpg", None)
        getpgid = getattr(os, "getpgid", None)
        if killpg is None or getpgid is None:
            process.terminate()
        else:
            sigterm = signal.SIGTERM
            sigkill = getattr(signal, "SIGKILL", sigterm)
            try:
                killpg(getpgid(process.pid), sigterm)
            except (OSError, ProcessLookupError):
                process.terminate()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                try:
                    killpg(getpgid(process.pid), sigkill)
                except (OSError, ProcessLookupError):
                    process.kill()

    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


@dataclass(frozen=True, slots=True)
class ExecutableSearchConfig:
    """Trusted test command, bounded observations, and optional Docker isolation."""

    command: tuple[str, ...]
    suite: str = "visible-tests"
    test_name: str = "all-visible-tests"
    timeout_seconds: float = 120.0
    max_report_bytes: int = 64 * 1024
    sandbox: ExecutionSandbox = "host"
    container_image: str = "python:3.12-slim"

    def __post_init__(self) -> None:
        if isinstance(self.command, str):
            raise ValueError(
                "command must be a sequence of argv strings, not a shell string"
            )
        command = tuple(_non_empty(value, "command argument") for value in self.command)
        if not command:
            raise ValueError("command must not be empty")
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "suite", _non_empty(self.suite, "suite"))
        object.__setattr__(self, "test_name", _non_empty(self.test_name, "test_name"))
        if not isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        if (
            not isinstance(self.max_report_bytes, int)
            or isinstance(self.max_report_bytes, bool)
            or self.max_report_bytes <= 0
        ):
            raise ValueError("max_report_bytes must be a positive integer")
        if not isinstance(self.sandbox, str) or self.sandbox not in {
            "host",
            "docker",
        }:
            raise ValueError("sandbox must be 'host' or 'docker'")
        object.__setattr__(
            self,
            "container_image",
            _non_empty(self.container_image, "container_image"),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExecutableSearchConfig:
        if not isinstance(data, Mapping):
            raise ValueError("execution config must be an object")
        allowed = {
            "command",
            "suite",
            "test_name",
            "timeout_seconds",
            "max_report_bytes",
            "sandbox",
            "container_image",
        }
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(
                f"execution config has unknown fields: {sorted(unknown)!r}"
            )
        command = data.get("command")
        if not isinstance(command, list):
            raise ValueError("execution config command must be an array")
        return cls(
            command=tuple(command),
            suite=str(data.get("suite", "visible-tests")),
            test_name=str(data.get("test_name", "all-visible-tests")),
            timeout_seconds=float(data.get("timeout_seconds", 120.0)),
            max_report_bytes=int(data.get("max_report_bytes", 64 * 1024)),
            sandbox=cast(ExecutionSandbox, str(data.get("sandbox", "host"))),
            container_image=str(data.get("container_image", "python:3.12-slim")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": list(self.command),
            "suite": self.suite,
            "test_name": self.test_name,
            "timeout_seconds": self.timeout_seconds,
            "max_report_bytes": self.max_report_bytes,
            "sandbox": self.sandbox,
            "container_image": self.container_image,
        }


def _materialize(files: Iterable[tuple[str, str]], root: Path) -> None:
    for relative, content in files:
        target = root.joinpath(*PurePosixPath(relative).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def _docker_command(
    config: ExecutableSearchConfig, workspace: Path
) -> tuple[list[str], str, str]:
    docker = shutil.which("docker")
    if docker is None:
        raise OSError("docker executable was not found on PATH")
    container_name = f"contextopt-{uuid.uuid4().hex[:20]}"
    mount = f"type=bind,source={workspace.resolve()},destination=/workspace"
    command = [
        docker,
        "run",
        "--rm",
        "--init",
        "--name",
        container_name,
        "--network=none",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--read-only",
        "--pids-limit",
        "256",
        "--memory",
        "1g",
        "--cpus",
        "2",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=64m",
        "--mount",
        mount,
        "--workdir",
        "/workspace",
        config.container_image,
        *config.command,
    ]
    return command, docker, container_name


def _remove_docker_container(docker: str, container_name: str) -> None:
    with suppress(OSError, subprocess.TimeoutExpired):
        subprocess.run(
            [docker, "rm", "--force", container_name],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )


def evaluate_candidate(
    candidate: CandidatePatch, config: ExecutableSearchConfig
) -> TestResult:
    """Materialize and test one candidate, returning a bounded auditable observation."""

    started = perf_counter()
    with tempfile.TemporaryDirectory(prefix="contextopt-branch-") as temporary:
        workspace = Path(temporary)
        _materialize(candidate.files.items(), workspace)
        windows_job: tuple[Any, Any] | None = None
        sandbox_cleanup: tuple[str, str] | None = None
        try:
            if config.sandbox == "docker":
                command, docker, container_name = _docker_command(config, workspace)
                sandbox_cleanup = (docker, container_name)
                environment = _child_environment()
            else:
                command = list(config.command)
                environment = _child_environment()
            process = subprocess.Popen(
                command,
                cwd=workspace,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                **_new_process_group_kwargs(),
            )
            windows_job = _attach_windows_job(process)
            try:
                stdout, stderr = process.communicate(timeout=config.timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                _close_windows_job(windows_job)
                windows_job = None
                _terminate_process_tree(process)
                if sandbox_cleanup is not None:
                    _remove_docker_container(*sandbox_cleanup)
                    sandbox_cleanup = None
                tail_stdout, tail_stderr = process.communicate()
                stdout = tail_stdout or (
                    exc.output if isinstance(exc.output, bytes) else b""
                )
                stderr = tail_stderr or (
                    exc.stderr if isinstance(exc.stderr, bytes) else b""
                )
                digest, excerpt = _bounded_output(
                    stdout, stderr, config.max_report_bytes
                )
                return TestResult(
                    suite=config.suite,
                    error=(
                        "test command timed out after "
                        f"{config.timeout_seconds:g}s; process group terminated"
                    ),
                    duration_ms=(perf_counter() - started) * 1_000,
                    output_sha256=digest,
                    output_excerpt=excerpt,
                )
            returncode = process.returncode
            digest, excerpt = _bounded_output(stdout, stderr, config.max_report_bytes)
            if returncode == 0:
                return TestResult(
                    suite=config.suite,
                    passed_tests=(config.test_name,),
                    duration_ms=(perf_counter() - started) * 1_000,
                    output_sha256=digest,
                    output_excerpt=excerpt,
                )
            return TestResult(
                suite=config.suite,
                failed_tests=(config.test_name,),
                duration_ms=(perf_counter() - started) * 1_000,
                output_sha256=digest,
                output_excerpt=excerpt,
            )
        except OSError as exc:
            return TestResult(
                suite=config.suite,
                error=f"test command could not start: {type(exc).__name__}: {exc}",
                duration_ms=(perf_counter() - started) * 1_000,
            )
        finally:
            _close_windows_job(windows_job)
            if sandbox_cleanup is not None:
                _remove_docker_container(*sandbox_cleanup)


def evaluate_case(case: BranchCase, config: ExecutableSearchConfig) -> BranchCase:
    """Run each unique candidate snapshot and return a case with fresh observations."""

    results: dict[str, TestResult] = {}
    cached: dict[str, TestResult] = {}
    for candidate in case.candidates:
        fingerprint = candidate.workspace_fingerprint
        if fingerprint in cached:
            results[candidate.id] = cached[fingerprint]
            continue
        result = evaluate_candidate(candidate, config)
        cached[fingerprint] = result
        results[candidate.id] = result
    return BranchCase(
        task=case.task,
        root_files=case.root_files,
        candidates=case.candidates,
        tests=results,
    )


def run_executable_search(
    case: BranchCase,
    search_config: BranchSearchConfig | None = None,
    execution_config: ExecutableSearchConfig | None = None,
) -> BranchSearchReport:
    """Evaluate candidates in disposable workspaces, then run the pure search core."""

    if execution_config is None:
        raise ValueError("execution_config is required for executable branch search")
    observed = evaluate_case(case, execution_config)
    return BranchSearch(search_config).run(observed)
