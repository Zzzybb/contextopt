"""Command-line interface for reproducible context packing experiments."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from contextopt import __version__
from contextopt.benchmark import (
    BenchmarkConfig,
    render_console,
    render_markdown,
    run_benchmark,
)
from contextopt.models import ContextItem, ObjectiveWeights, SelectionProblem
from contextopt.policies import POLICIES, create_policy


def _write(path: str | None, content: str) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def _benchmark(args: argparse.Namespace) -> int:
    policy_names = tuple(
        name.strip() for name in args.policies.split(",") if name.strip()
    )
    config = BenchmarkConfig(
        instances=args.instances,
        item_count=args.items,
        critical_count=args.critical,
        budget=args.budget,
        seed=args.seed,
        graph_rate=args.graph_rate,
        conflict_rate=args.conflict_rate,
    )
    report = run_benchmark(config, policy_names)
    print(render_console(report))
    _write(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write(args.markdown, render_markdown(report))
    return 0


def _load_problem(path: str, budget_override: int | None) -> SelectionProblem:
    data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    budget = budget_override if budget_override is not None else int(data["budget"])
    items = tuple(ContextItem.from_dict(item) for item in data["items"])
    return SelectionProblem(
        items=items,
        budget=budget,
        weights=ObjectiveWeights.from_dict(data.get("weights")),
    )


def _pack(args: argparse.Namespace) -> int:
    problem = _load_problem(args.input, args.budget)
    frame = create_policy(args.policy).select(problem)
    payload = json.dumps(frame.to_dict(), indent=2, sort_keys=True) + "\n"
    print(payload, end="")
    _write(args.output, payload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="contextopt",
        description="Algorithmic context optimization for long-horizon agents.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark = subparsers.add_parser(
        "benchmark", help="run a deterministic paired synthetic benchmark"
    )
    benchmark.add_argument("--instances", type=int, default=50)
    benchmark.add_argument("--items", type=int, default=14)
    benchmark.add_argument("--critical", type=int, default=4)
    benchmark.add_argument("--budget", type=int, default=1_200)
    benchmark.add_argument("--seed", type=int, default=42)
    benchmark.add_argument("--graph-rate", type=float, default=0.0)
    benchmark.add_argument("--conflict-rate", type=float, default=0.0)
    benchmark.add_argument(
        "--policies", default=",".join(POLICIES), help="comma-separated policy names"
    )
    benchmark.add_argument("--output", help="write the full JSON report")
    benchmark.add_argument("--markdown", help="write a compact Markdown report")
    benchmark.set_defaults(handler=_benchmark)

    pack = subparsers.add_parser("pack", help="select context from a JSON problem")
    pack.add_argument("input")
    pack.add_argument("--policy", choices=sorted(POLICIES), default="submodular")
    pack.add_argument("--budget", type=int, help="override the JSON token budget")
    pack.add_argument("--output", help="write the selection receipt as JSON")
    pack.set_defaults(handler=_pack)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
