"""Command-line interface for the auditable agent runtime and experiments."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import uuid
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
from contextopt.runtime import (
    AgentRunner,
    EventLog,
    ModelClient,
    OpenAICompatibleModel,
    RunLimits,
    RunPermissions,
    ScriptedModel,
    WorkspaceTools,
    read_events,
    render_trace,
)


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


def _split_command(command: str) -> tuple[str, ...]:
    parts = shlex.split(command, posix=os.name != "nt")
    if not parts:
        raise ValueError("test command must not be empty")
    return tuple(parts)


def _run_agent(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve(strict=True)
    if not workspace.is_dir():
        raise ValueError("workspace must be a directory")
    run_id = args.run_id or uuid.uuid4().hex[:12]
    limits = RunLimits(
        max_turns=args.max_turns,
        max_tool_calls=args.max_tool_calls,
        max_total_tokens=args.max_total_tokens,
        max_output_tokens_per_call=args.max_output_tokens,
        wall_timeout_seconds=args.wall_timeout,
        command_timeout_seconds=args.command_timeout,
        max_tool_output_bytes=args.max_tool_output_bytes,
    )
    permissions = RunPermissions(
        allow_write=args.allow_write,
        allow_command=args.allow_command,
    )
    commands = (
        {"visible": _split_command(args.test_command)} if args.test_command else {}
    )
    tools = WorkspaceTools(
        workspace,
        permissions=permissions,
        limits=limits,
        test_commands=commands,
    )
    model: ModelClient
    if args.script:
        model = ScriptedModel.from_path(args.script)
    else:
        if not args.model or not args.base_url:
            raise ValueError("--model and --base-url are required without --script")
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise ValueError(
                f"model API key is missing from environment variable {args.api_key_env}"
            )
        model = OpenAICompatibleModel(
            base_url=args.base_url,
            api_key=api_key,
            model=args.model,
            timeout_seconds=args.model_timeout,
            max_retries=args.model_retries,
            temperature=args.temperature,
        )
    event_path = (
        Path(args.event_log)
        if args.event_log
        else workspace / ".contextopt" / "runs" / run_id / "events.jsonl"
    )
    event_log = EventLog(event_path, run_id)
    runner = AgentRunner(model=model, tools=tools, event_log=event_log, limits=limits)
    result = asyncio.run(runner.run(args.task))
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.status == "completed" else 2


def _trace(args: argparse.Namespace) -> int:
    events = read_events(args.event_log)
    print(render_trace(events), end="")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="contextopt",
        description="Auditable runtime and context lab for long-horizon coding agents.",
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

    run = subparsers.add_parser(
        "run", help="run one auditable coding-agent loop in a workspace"
    )
    run.add_argument("task", help="coding task or issue text")
    run.add_argument("--workspace", default=".")
    run.add_argument(
        "--script", help="deterministic scripted-model JSON for offline runs"
    )
    run.add_argument("--model", help="OpenAI-compatible model name")
    run.add_argument("--base-url", help="OpenAI-compatible API base URL")
    run.add_argument(
        "--api-key-env",
        default="CONTEXTOPT_API_KEY",
        help="environment variable containing the API key",
    )
    run.add_argument("--model-timeout", type=float, default=90.0)
    run.add_argument("--model-retries", type=int, default=2)
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--allow-write", action="store_true")
    run.add_argument("--allow-command", action="store_true")
    run.add_argument(
        "--test-command",
        help="trusted visible-test command registered as the run_tests tool",
    )
    run.add_argument("--run-id")
    run.add_argument("--event-log")
    run.add_argument("--max-turns", type=int, default=20)
    run.add_argument("--max-tool-calls", type=int, default=50)
    run.add_argument("--max-total-tokens", type=int, default=100_000)
    run.add_argument("--max-output-tokens", type=int, default=2_048)
    run.add_argument("--wall-timeout", type=float, default=900.0)
    run.add_argument("--command-timeout", type=float, default=120.0)
    run.add_argument("--max-tool-output-bytes", type=int, default=256 * 1024)
    run.set_defaults(handler=_run_agent)

    trace = subparsers.add_parser("trace", help="render a compact JSONL run trace")
    trace.add_argument("event_log")
    trace.set_defaults(handler=_trace)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        return 130
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
