"""Run the CI quality gates with actionable GitHub annotations on failure."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ruff check", ("ruff", "check", "src", "tests")),
    ("ruff format", ("ruff", "format", "--check", "src", "tests")),
    ("mypy", ("mypy", "src/contextopt")),
)


def _workflow_escape(value: str) -> str:
    return (
        value.replace("%", "%25")
        .replace("\r", "%0D")
        .replace("\n", "%0A")
        .replace(":", "%3A")
        .replace(",", "%2C")
    )


def _emit_annotations(label: str, output: str) -> None:
    if os.environ.get("GITHUB_ACTIONS", "").lower() != "true":
        return
    lines = output.splitlines() or ["(command produced no output)"]
    for line in lines[:50]:
        print(f"::error title={_workflow_escape(label)}::{_workflow_escape(line)}")
    if len(lines) > 50:
        print(
            f"::error title={_workflow_escape(label)}::"
            f"{_workflow_escape(f'... {len(lines) - 50} more lines omitted')}",
        )


def _run(label: str, command: Sequence[str]) -> int:
    argv = (sys.executable, "-m", *command)
    print(f"$ {' '.join(argv)}")
    completed = subprocess.run(
        argv,
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    output = completed.stdout + completed.stderr
    print(output, end="" if output.endswith("\n") else "\n")
    if completed.returncode:
        _emit_annotations(label, output)
    return completed.returncode


def main() -> int:
    for label, command in CHECKS:
        status = _run(label, command)
        if status:
            return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
