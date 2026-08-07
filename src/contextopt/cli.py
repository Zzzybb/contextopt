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
from contextopt.runtime.recovery import replay_events_with_checkpoint


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


def _build_model(args: argparse.Namespace) -> ModelClient:
    if args.script:
        return ScriptedModel.from_path(args.script)
    if not args.model or not args.base_url:
        raise ValueError("--model and --base-url are required without --script")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(
            f"model API key is missing from environment variable {args.api_key_env}"
        )
    return OpenAICompatibleModel(
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
        timeout_seconds=args.model_timeout,
        max_retries=args.model_retries,
        temperature=args.temperature,
    )


def _result_exit_code(status: str) -> int:
    if status == "completed":
        return 0
    if status == "paused":
        return 4
    return 2


def _default_checkpoint_path(event_path: Path) -> Path:
    return event_path.with_name(event_path.name + ".checkpoint.json")


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
    model = _build_model(args)
    event_path = (
        Path(args.event_log)
        if args.event_log
        else workspace / ".contextopt" / "runs" / run_id / "events.jsonl"
    )
    event_log = EventLog(event_path, run_id)
    try:
        runner = AgentRunner(
            model=model,
            tools=tools,
            event_log=event_log,
            limits=limits,
        )
        result = asyncio.run(runner.run(args.task))
    finally:
        event_log.close()
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return _result_exit_code(result.status)


def _resume_agent(args: argparse.Namespace) -> int:
    event_path = Path(args.event_log)
    events = read_events(event_path)
    if not events:
        raise ValueError("cannot resume an empty or missing event log")
    checkpoint_path = _default_checkpoint_path(event_path)
    state = replay_events_with_checkpoint(events, checkpoint_path)
    if state.terminal is not None:
        terminal_result = {
            "run_id": state.run_id,
            "status": state.terminal.status,
            "reason": state.terminal.reason,
            "final_text": state.terminal.final_text,
            "turns": state.turn,
            "tool_calls": state.tool_calls,
            "usage": state.usage.to_dict(),
            "event_log": event_path.as_posix(),
        }
        print(json.dumps(terminal_result, indent=2, sort_keys=True))
        return _result_exit_code(state.terminal.status)

    workspace = Path(args.workspace).resolve(strict=True)
    if not workspace.is_dir():
        raise ValueError("workspace must be a directory")
    commands = (
        {"visible": _split_command(args.test_command)} if args.test_command else {}
    )
    tools = WorkspaceTools(
        workspace,
        permissions=state.config.permissions,
        limits=state.config.limits,
        test_commands=commands,
    )
    model = _build_model(args)
    event_log = EventLog(event_path, state.run_id, repair_truncated=True)
    try:
        runner = AgentRunner(
            model=model,
            tools=tools,
            event_log=event_log,
            limits=state.config.limits,
            checkpoint_path=checkpoint_path,
        )
        run_result = asyncio.run(
            runner.resume(pending_tool_resolution=args.pending_tool_resolution)
        )
    finally:
        event_log.close()
    print(json.dumps(run_result.to_dict(), indent=2, sort_keys=True))
    return _result_exit_code(run_result.status)


def _status(args: argparse.Namespace) -> int:
    event_path = Path(args.event_log)
    events = read_events(event_path)
    if not events:
        raise ValueError("event log is empty or missing")
    schema = events[0].get("schema_version")
    if schema != "2":
        payload = {
            "schema_version": schema,
            "resumable": False,
            "reason": "schema 1 traces are audit-only",
            "events": len(events),
            "last_event_type": events[-1].get("type"),
        }
    else:
        state = replay_events_with_checkpoint(
            events, _default_checkpoint_path(event_path)
        )
        payload = {
            "schema_version": schema,
            "run_id": state.run_id,
            "phase": state.phase,
            "resumable": state.phase != "terminal",
            "turns": state.turn,
            "tool_calls": state.tool_calls,
            "usage": state.usage.to_dict(),
            "pending_model": (
                None if state.pending_model is None else state.pending_model.to_dict()
            ),
            "pending_tools": [
                {
                    "call_id": item.call.id,
                    "tool_name": item.call.name,
                    "turn": item.turn,
                    "call_index": item.call_index,
                    "started": item.started,
                    "replay_policy": (
                        None if item.plan is None else item.plan.replay_policy
                    ),
                }
                for item in state.pending_tools
            ],
            "terminal": (None if state.terminal is None else state.terminal.to_dict()),
            "through_seq": state.through_seq,
            "through_event_sha256": state.through_event_sha256,
        }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _trace(args: argparse.Namespace) -> int:
    events = read_events(args.event_log)
    print(render_trace(events), end="")
    return 0


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--script", help="deterministic scripted-model JSON for offline runs"
    )
    parser.add_argument("--model", help="OpenAI-compatible model name")
    parser.add_argument("--base-url", help="OpenAI-compatible API base URL")
    parser.add_argument(
        "--api-key-env",
        default="CONTEXTOPT_API_KEY",
        help="environment variable containing the API key",
    )
    parser.add_argument("--model-timeout", type=float, default=90.0)
    parser.add_argument("--model-retries", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)


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
    _add_model_arguments(run)
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

    resume = subparsers.add_parser(
        "resume", help="resume a non-terminal schema-v2 coding-agent run"
    )
    resume.add_argument("event_log")
    resume.add_argument("--workspace", default=".")
    _add_model_arguments(resume)
    resume.add_argument(
        "--test-command",
        help="the same trusted visible-test command used by the original run",
    )
    resume.add_argument(
        "--pending-tool-resolution",
        choices=("retry", "mark_failed"),
        help="explicitly resolve an indeterminate non-replayable tool",
    )
    resume.set_defaults(handler=_resume_agent)

    status = subparsers.add_parser(
        "status", help="inspect the recoverable projection of an event log"
    )
    status.add_argument("event_log")
    status.set_defaults(handler=_status)

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
