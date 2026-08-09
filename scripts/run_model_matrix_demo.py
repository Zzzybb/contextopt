"""Generate a provider-free two-bundle model-matrix demonstration."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from contextopt.cli import main as contextopt_main


LABELS = ("scripted-a", "scripted-b")


def _bundle_args(root: Path, label: str) -> tuple[str, ...]:
    directory = root / label
    return (
        "agent-eval",
        "--fixtures",
        "all",
        "--strategies",
        "single_pass,best_of_n,orchestrated",
        "--repetitions",
        "3",
        "--output",
        str(directory / "report.json"),
        "--markdown",
        str(directory / "report.md"),
        "--html",
        str(directory / "report.html"),
        "--checkpoint",
        str(directory / "checkpoint.json"),
        "--manifest",
        str(directory / "manifest.json"),
    )


def run(output_dir: Path) -> int:
    # A local shell has no GITHUB_SHA.  Keep the demo's revision explicit and deterministic
    # instead of weakening the analyzer's requirement for a provenance anchor.
    os.environ.setdefault(
        "CONTEXTOPT_GIT_REVISION", "provider-free-model-matrix-demo"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for label in LABELS:
        status = contextopt_main(list(_bundle_args(output_dir, label)))
        if status:
            return status

    compare_args: list[str] = [
        "agent-eval-compare",
        "--output",
        str(output_dir / "model-matrix.json"),
        "--markdown",
        str(output_dir / "model-matrix.md"),
        "--html",
        str(output_dir / "model-matrix.html"),
    ]
    for label in LABELS:
        directory = output_dir / label
        compare_args.extend(
            (
                "--bundle",
                label,
                str(directory / "report.json"),
                str(directory / "manifest.json"),
            )
        )
    return contextopt_main(compare_args)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/model-matrix-demo"),
        help="directory for the generated bundles and comparison artifacts",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    return run(args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
