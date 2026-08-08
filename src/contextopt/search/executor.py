"""A deliberately explicit adapter for running branch candidates in temp workspaces.

``BranchSearch`` itself is pure: it consumes candidate snapshots and test observations.
This module is the narrow side-effect boundary for a local experiment. It materializes
one complete snapshot per unique workspace fingerprint, runs a caller-supplied argv
without a shell, bounds the report excerpt, and converts the exit status into the same
``TestResult`` consumed by the search core.

The command is a trusted host process, not a sandbox. Callers must opt into it at the
CLI with ``--allow-command`` and should use a disposable workspace or CI runner.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite
from pathlib import Path, PurePosixPath
from time import perf_counter

from contextopt.search.branching import (
    BranchCase,
    BranchSearch,
    BranchSearchConfig,
    BranchSearchReport,
    CandidatePatch,
    TestResult,
)


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


@dataclass(frozen=True, slots=True)
class ExecutableSearchConfig:
    """Trusted local test command and bounded observation settings."""

    command: tuple[str, ...]
    suite: str = "visible-tests"
    test_name: str = "all-visible-tests"
    timeout_seconds: float = 120.0
    max_report_bytes: int = 64 * 1024

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


def _materialize(files: Iterable[tuple[str, str]], root: Path) -> None:
    for relative, content in files:
        target = root.joinpath(*PurePosixPath(relative).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def evaluate_candidate(
    candidate: CandidatePatch, config: ExecutableSearchConfig
) -> TestResult:
    """Materialize and test one candidate, returning a bounded auditable observation."""

    started = perf_counter()
    with tempfile.TemporaryDirectory(prefix="contextopt-branch-") as temporary:
        workspace = Path(temporary)
        _materialize(candidate.files.items(), workspace)
        try:
            completed = subprocess.run(
                list(config.command),
                cwd=workspace,
                env=os.environ.copy(),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=config.timeout_seconds,
                check=False,
            )
            stdout = completed.stdout
            stderr = completed.stderr
            digest, excerpt = _bounded_output(stdout, stderr, config.max_report_bytes)
            duration_ms = (perf_counter() - started) * 1_000
            if completed.returncode == 0:
                return TestResult(
                    suite=config.suite,
                    passed_tests=(config.test_name,),
                    duration_ms=duration_ms,
                    output_sha256=digest,
                    output_excerpt=excerpt,
                )
            return TestResult(
                suite=config.suite,
                failed_tests=(config.test_name,),
                duration_ms=duration_ms,
                output_sha256=digest,
                output_excerpt=excerpt,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout if isinstance(exc.stdout, bytes) else b""
            stderr = exc.stderr if isinstance(exc.stderr, bytes) else b""
            digest, excerpt = _bounded_output(stdout, stderr, config.max_report_bytes)
            return TestResult(
                suite=config.suite,
                error=f"test command timed out after {config.timeout_seconds:g}s",
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
