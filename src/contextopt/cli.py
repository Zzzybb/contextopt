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
from urllib.parse import urlsplit, urlunsplit

from contextopt import __version__
from contextopt.benchmark import (
    BenchmarkConfig,
    render_console,
    render_markdown,
    run_benchmark,
)
from contextopt.evaluation import (
    AgentEvalConfig,
    ContextRoutingEvalConfig,
    RecoveryEvalConfig,
    SemanticMemoryEvalConfig,
    build_openai_model_factory,
    render_agent_evaluation_console,
    render_agent_evaluation_html,
    render_agent_evaluation_markdown,
    render_context_routing_console,
    render_context_routing_markdown,
    render_recovery_console,
    render_recovery_html,
    render_recovery_markdown,
    render_semantic_memory_console,
    render_semantic_memory_html,
    render_semantic_memory_markdown,
    run_agent_evaluation,
    run_context_routing_evaluation,
    run_recovery_evaluation,
    run_semantic_memory_evaluation,
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
    render_trace_html,
)
from contextopt.runtime.context import ContextCompiler, ContextCompilerConfig
from contextopt.runtime.recovery import replay_events_with_checkpoint
from contextopt.runtime.semantic_memory import SemanticMemoryStore
from contextopt.search import (
    BranchCase,
    BranchSearch,
    BranchSearchConfig,
    ExecutableSearchConfig,
    OrchestrationConfig,
    PlannerConfig,
    ProposalConfig,
    ReviewerConfig,
    SearchSessionConfig,
    WorkspaceApplyError,
    apply_best_snapshot,
    demo_case,
    evaluate_case,
    propose_case,
    read_apply_receipt,
    read_session_checkpoint,
    render_branch_console,
    render_branch_html,
    render_branch_markdown,
    render_orchestration_console,
    render_orchestration_html,
    render_orchestration_markdown,
    render_session_console,
    render_session_markdown,
    rollback_best_snapshot,
    run_orchestration,
    run_search_session,
    write_apply_receipt,
)


def _write(path: str | None, content: str) -> None:
    if not path:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def _safe_endpoint(value: str | None) -> str | None:
    """Keep provider origin/path while dropping query and fragment secrets."""

    if not value:
        return None
    parts = urlsplit(value)
    if not parts.scheme or not parts.netloc:
        return value.split("?", 1)[0].split("#", 1)[0]
    safe_netloc = parts.netloc.rsplit("@", 1)[-1]
    return urlunsplit((parts.scheme, safe_netloc, parts.path, "", ""))


def _agent_eval_manifest(
    args: argparse.Namespace, config: AgentEvalConfig, model_adapter: str
) -> dict[str, Any]:
    return {
        "schema_version": "1",
        "kind": "contextopt.agent-eval.manifest",
        "config": config.to_dict(),
        "provider": {
            "adapter": model_adapter,
            "model": args.model,
            "planner_model": args.planner_model,
            "solver_model": args.solver_model,
            "reviewer_model": args.reviewer_model,
            "base_url": _safe_endpoint(args.base_url),
        },
        "runtime": {
            "temperature": args.temperature,
            "timeout_seconds": args.model_timeout,
            "max_retries": args.model_retries,
            "api_key_env": args.api_key_env,
        },
        "repository_revision": os.environ.get("CONTEXTOPT_GIT_REVISION")
        or os.environ.get("GITHUB_SHA"),
        "claim_boundary": (
            "The manifest records experiment identity and configuration; it does not "
            "prove model quality or provider reproducibility."
        ),
    }


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


def _context_eval(args: argparse.Namespace) -> int:
    policies = tuple(name.strip() for name in args.policies.split(",") if name.strip())
    try:
        budgets = tuple(
            int(value.strip()) for value in args.budgets.split(",") if value.strip()
        )
    except ValueError as exc:
        raise ValueError("--budgets must be comma-separated integers") from exc
    config = ContextRoutingEvalConfig(
        policies=policies,
        budgets=budgets,
        repetitions=args.repetitions,
        recent_blocks=args.recent_blocks,
        max_tool_output_tokens=args.max_tool_output_tokens,
    )
    report = run_context_routing_evaluation(config)
    print(render_context_routing_console(report))
    _write(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write(args.markdown, render_context_routing_markdown(report))
    return 0


def _agent_eval(args: argparse.Namespace) -> int:
    strategies = tuple(
        name.strip() for name in args.strategies.split(",") if name.strip()
    )
    raw_fixtures = tuple(
        name.strip() for name in args.fixtures.split(",") if name.strip()
    )
    fixtures = ("two-sum", "extended-gcd") if "all" in raw_fixtures else raw_fixtures
    model_factory = None
    model_adapter = "scripted"
    if args.model:
        if not args.base_url:
            raise ValueError("--base-url is required when --model is supplied")
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise ValueError(
                f"model API key is missing from environment variable {args.api_key_env}"
            )
        model_factory = build_openai_model_factory(
            base_url=args.base_url,
            api_key=api_key,
            model=args.model,
            planner_model=args.planner_model,
            solver_model=args.solver_model,
            reviewer_model=args.reviewer_model,
            timeout_seconds=args.model_timeout,
            max_retries=args.model_retries,
            temperature=args.temperature,
        )
        model_adapter = "openai-compatible"
    config = AgentEvalConfig(
        strategies=strategies,
        fixtures=fixtures,
        repetitions=args.repetitions,
        max_rounds=args.max_rounds,
        max_model_calls=args.max_model_calls,
        max_candidates=args.max_candidates,
        max_test_calls=args.max_test_calls,
        search_policy=args.search_policy,
        exploration_constant=args.exploration_constant,
        include_hidden_tests=not args.no_hidden_tests,
        model_adapter=model_adapter,
    )
    _write(
        args.manifest,
        json.dumps(
            _agent_eval_manifest(args, config, model_adapter),
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    report = run_agent_evaluation(
        config,
        model_factory=model_factory,
        checkpoint_path=args.checkpoint,
        resume=args.resume,
    )
    print(render_agent_evaluation_console(report), end="")
    _write(args.output, json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    _write(args.markdown, render_agent_evaluation_markdown(report))
    _write(args.html, render_agent_evaluation_html(report))
    # A completed evaluation is useful even when an intentionally weak baseline
    # fails; callers should inspect the success-rate columns.
    return 0


def _recovery_eval(args: argparse.Namespace) -> int:
    raw = tuple(item.strip() for item in args.scenarios.split(",") if item.strip())
    scenarios = RecoveryEvalConfig().scenarios if "all" in raw else raw
    report = run_recovery_evaluation(RecoveryEvalConfig(scenarios=scenarios))
    print(render_recovery_console(report), end="")
    _write(args.output, json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    _write(args.markdown, render_recovery_markdown(report))
    _write(args.html, render_recovery_html(report))
    _write(
        args.manifest,
        json.dumps(
            {
                "schema_version": "1",
                "kind": "contextopt.recovery-eval.manifest",
                "config": report.config.to_dict(),
                "fault_injection": {
                    "adapter": "durable-event-hook-v1",
                    "fresh_runner_on_resume": True,
                    "model_adapter": "scripted",
                },
                "claim_boundary": report.claim_boundary,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return 0 if report.failed_count == 0 else 1


def _semantic_memory_eval(args: argparse.Namespace) -> int:
    report = run_semantic_memory_evaluation(
        SemanticMemoryEvalConfig(
            repetitions=args.repetitions,
            limit=args.limit,
        )
    )
    print(render_semantic_memory_console(report), end="\n")
    _write(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    _write(args.markdown, render_semantic_memory_markdown(report))
    _write(args.html, render_semantic_memory_html(report))
    return 0


def _load_branch_case(path: str) -> BranchCase:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("branch search input must be a JSON object")
    return BranchCase.from_dict(data)


def _load_root_files(path: str) -> dict[str, str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("root files input must be a JSON object")
    files: dict[str, str] = {}
    for file_path, content in data.items():
        if not isinstance(file_path, str) or not isinstance(content, str):
            raise ValueError("root files must map strings to strings")
        files[file_path] = content
    return files


def _branch_search(args: argparse.Namespace) -> int:
    case = demo_case() if args.input is None else _load_branch_case(args.input)
    if args.test_command:
        if not args.allow_command:
            raise ValueError("--allow-command is required when executing branch tests")
        case = evaluate_case(
            case,
            ExecutableSearchConfig(
                command=_split_command(args.test_command),
                suite=args.test_suite,
                test_name=args.test_name,
                timeout_seconds=args.test_timeout,
                max_report_bytes=args.max_report_bytes,
            ),
        )
    report = BranchSearch(
        BranchSearchConfig(
            beam_width=args.beam_width,
            max_depth=args.max_depth,
            max_candidates=args.max_candidates,
            test_environment_fingerprint=args.test_environment,
            stop_on_pass=not args.no_stop_on_pass,
            search_policy=args.search_policy,
            exploration_constant=args.exploration_constant,
        )
    ).run(case)
    print(render_branch_console(report))
    payload = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    _write(args.output, payload)
    _write(args.markdown, render_branch_markdown(report))
    _write(args.html, render_branch_html(report))
    return 0


def _propose_case(args: argparse.Namespace) -> int:
    root_files = _load_root_files(args.root_files)
    model = _build_model(args)
    config = ProposalConfig(
        max_candidates=args.max_candidates,
        max_files_per_candidate=args.max_files_per_candidate,
        max_file_chars=args.max_file_chars,
        max_total_prompt_chars=args.max_total_prompt_chars,
        max_output_tokens=args.max_output_tokens,
    )
    case, _ = asyncio.run(
        propose_case(
            model,
            args.task,
            root_files,
            config,
            run_id=args.run_id,
        )
    )
    payload = json.dumps(case.to_dict(), indent=2, sort_keys=True) + "\n"
    print(payload, end="")
    _write(args.output, payload)
    return 0


def _search_session(args: argparse.Namespace) -> int:
    if args.resume:
        root_files = None
    else:
        if not args.root_files:
            raise ValueError("--root-files is required for a new search session")
        root_files = _load_root_files(args.root_files)
    execution_config = None
    if args.test_command:
        if not args.allow_command:
            raise ValueError("--allow-command is required when executing session tests")
        execution_config = ExecutableSearchConfig(
            command=_split_command(args.test_command),
            suite=args.test_suite,
            test_name=args.test_name,
            timeout_seconds=args.test_timeout,
            max_report_bytes=args.max_report_bytes,
        )
    elif not args.resume:
        raise ValueError("--test-command is required for a new search session")
    model = _build_model(args)
    session_config = None
    proposal_config = None
    search_config = None
    if not args.resume:
        session_config = SearchSessionConfig(
            max_rounds=args.max_rounds,
            max_model_calls=args.max_model_calls,
            max_candidates=args.session_max_candidates,
            max_test_calls=args.session_max_test_calls,
            max_parallel_tests=args.max_parallel_tests,
            scheduler_policy=args.scheduler_policy,
        )
        proposal_config = ProposalConfig(
            max_candidates=args.proposal_max_candidates,
            max_files_per_candidate=args.max_files_per_candidate,
            max_file_chars=args.max_file_chars,
            max_total_prompt_chars=args.max_total_prompt_chars,
            max_output_tokens=args.proposal_output_tokens,
        )
        search_config = BranchSearchConfig(
            beam_width=args.beam_width,
            max_depth=args.max_depth,
            max_candidates=args.branch_max_candidates,
            test_environment_fingerprint=args.test_environment,
            stop_on_pass=not args.no_stop_on_pass,
            search_policy=args.search_policy,
            exploration_constant=args.exploration_constant,
        )
    report = asyncio.run(
        run_search_session(
            model,
            task=args.task,
            root_files=root_files,
            execution_config=execution_config,
            config=session_config,
            proposal_config=proposal_config,
            search_config=search_config,
            run_id=args.run_id or uuid.uuid4().hex,
            checkpoint_path=args.checkpoint,
            resume=args.resume,
            retry_pending=args.retry_pending,
        )
    )
    print(render_session_console(report), end="")
    _write(args.output, json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    _write(args.markdown, render_session_markdown(report))
    if report.status == "accepted":
        return 0
    if report.status == "paused":
        return 4
    return 3 if report.status == "failed" else 2


def _apply_best(args: argparse.Namespace) -> int:
    if not args.allow_write:
        raise ValueError("--allow-write is required when applying a snapshot")
    checkpoint = Path(args.checkpoint)
    report = read_session_checkpoint(checkpoint)
    receipt = apply_best_snapshot(
        report,
        args.workspace,
        allow_write=True,
        allow_delete=args.allow_delete,
    )
    receipt_path = (
        Path(args.receipt)
        if args.receipt
        else checkpoint.with_name(checkpoint.name + ".apply.json")
    )
    write_apply_receipt(receipt, receipt_path)
    print(json.dumps(receipt.to_dict(), indent=2, sort_keys=True))
    print(f"receipt={receipt_path}")
    return 0


def _rollback_best(args: argparse.Namespace) -> int:
    if not args.allow_write:
        raise ValueError("--allow-write is required when rolling back a snapshot")
    checkpoint = Path(args.checkpoint)
    report = read_session_checkpoint(checkpoint)
    receipt_path = (
        Path(args.receipt)
        if args.receipt
        else checkpoint.with_name(checkpoint.name + ".apply.json")
    )
    receipt = read_apply_receipt(receipt_path)
    rollback_receipt = rollback_best_snapshot(
        report,
        receipt,
        args.workspace,
        allow_write=True,
    )
    rollback_path = (
        Path(args.rollback_receipt)
        if args.rollback_receipt
        else receipt_path.with_name(receipt_path.name + ".rollback.json")
    )
    write_apply_receipt(rollback_receipt, rollback_path)
    print(json.dumps(rollback_receipt.to_dict(), indent=2, sort_keys=True))
    print(f"receipt={rollback_path}")
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


def _build_role_model(args: argparse.Namespace, role: str) -> ModelClient:
    script = getattr(args, f"{role}_script")
    if script:
        return ScriptedModel.from_path(script)
    model_name = getattr(args, f"{role}_model")
    if not model_name or not args.base_url:
        raise ValueError(
            f"--{role}-model and --base-url are required without --{role}-script"
        )
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise ValueError(
            f"model API key is missing from environment variable {args.api_key_env}"
        )
    return OpenAICompatibleModel(
        base_url=args.base_url,
        api_key=api_key,
        model=model_name,
        timeout_seconds=args.model_timeout,
        max_retries=args.model_retries,
        temperature=args.temperature,
    )


def _orchestrate(args: argparse.Namespace) -> int:
    if args.resume:
        root_files = None
        execution_config = None
    else:
        if not args.root_files or not args.task:
            raise ValueError(
                "--task and --root-files are required for a new orchestration"
            )
        root_files = _load_root_files(args.root_files)
        if not args.test_command:
            raise ValueError("--test-command is required for a new orchestration")
        if not args.allow_command:
            raise ValueError(
                "--allow-command is required when executing orchestration tests"
            )
        execution_config = ExecutableSearchConfig(
            command=_split_command(args.test_command),
            suite=args.test_suite,
            test_name=args.test_name,
            timeout_seconds=args.test_timeout,
            max_report_bytes=args.max_report_bytes,
        )
    planner = _build_role_model(args, "planner")
    solver = _build_role_model(args, "solver")
    reviewer = _build_role_model(args, "reviewer")
    config = planner_config = solver_config = reviewer_config = search_config = None
    if not args.resume:
        config = OrchestrationConfig(
            max_rounds=args.max_rounds,
            max_model_calls=args.max_model_calls,
            max_planner_calls=args.max_planner_calls,
            max_solver_calls=args.max_solver_calls,
            max_reviewer_calls=args.max_reviewer_calls,
            max_candidates=args.max_candidates,
            max_test_calls=args.max_test_calls,
            max_parallel_tests=args.max_parallel_tests,
            speculative_solver_width=args.speculative_solver_width,
            speculative_solver_stop_on_valid=args.speculative_solver_stop_on_valid,
            scheduler_policy=args.scheduler_policy,
            merge_policy=args.merge_policy,
            max_total_tokens=args.max_total_tokens,
            context_config=ContextCompilerConfig(
                policy=args.context_policy,
                budget_tokens=args.context_budget,
                recent_blocks=args.context_recent_blocks,
                max_tool_output_tokens=args.context_max_tool_output_tokens,
                memory_policy=args.context_memory,
            ),
        )
        planner_config = PlannerConfig(
            max_items=args.planner_max_items,
            max_item_chars=args.planner_max_item_chars,
            max_prompt_chars=args.planner_max_prompt_chars,
            max_output_tokens=args.planner_output_tokens,
        )
        solver_config = ProposalConfig(
            max_candidates=args.solver_max_candidates,
            max_files_per_candidate=args.max_files_per_candidate,
            max_file_chars=args.max_file_chars,
            max_total_prompt_chars=args.max_total_prompt_chars,
            max_output_tokens=args.solver_output_tokens,
        )
        reviewer_config = ReviewerConfig(
            max_prompt_chars=args.reviewer_max_prompt_chars,
            max_output_tokens=args.reviewer_output_tokens,
            max_issue_items=args.reviewer_max_issue_items,
            max_item_chars=args.reviewer_max_item_chars,
        )
        search_config = BranchSearchConfig(
            beam_width=args.beam_width,
            max_depth=args.max_depth,
            max_candidates=args.branch_max_candidates,
            test_environment_fingerprint=args.test_environment,
            stop_on_pass=not args.no_stop_on_pass,
            search_policy=args.search_policy,
            exploration_constant=args.exploration_constant,
        )
    report = asyncio.run(
        run_orchestration(
            planner,
            solver,
            reviewer,
            task=args.task,
            root_files=root_files,
            execution_config=execution_config,
            config=config,
            planner_config=planner_config,
            solver_config=solver_config,
            reviewer_config=reviewer_config,
            search_config=search_config,
            run_id=args.run_id or uuid.uuid4().hex,
            checkpoint_path=args.checkpoint,
            resume=args.resume,
            retry_pending=args.retry_pending,
        )
    )
    print(render_orchestration_console(report), end="")
    _write(args.output, json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    _write(args.markdown, render_orchestration_markdown(report))
    _write(args.html, render_orchestration_html(report))
    if report.status == "accepted":
        return 0
    if report.status == "paused":
        return 4
    return 3 if report.status == "failed" else 2


def _result_exit_code(status: str) -> int:
    if status == "completed":
        return 0
    if status == "paused":
        return 4
    return 2


def _default_checkpoint_path(event_path: Path) -> Path:
    return event_path.with_name(event_path.name + ".checkpoint.json")


def _context_compiler_from_args(args: argparse.Namespace) -> ContextCompiler:
    return ContextCompiler(
        ContextCompilerConfig(
            policy=args.context_policy,
            budget_tokens=args.context_budget,
            recent_blocks=args.context_recent_blocks,
            max_tool_output_tokens=args.context_max_tool_output_tokens,
            memory_policy=args.context_memory,
        )
    )


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
    event_path = (
        Path(args.event_log)
        if args.event_log
        else workspace / ".contextopt" / "runs" / run_id / "events.jsonl"
    )
    memory_store: SemanticMemoryStore | None = None
    tools: WorkspaceTools | None = None
    event_log: EventLog | None = None
    try:
        memory_store = (
            None
            if args.memory_store is None
            else SemanticMemoryStore(args.memory_store)
        )
        tools = WorkspaceTools(
            workspace,
            permissions=permissions,
            limits=limits,
            test_commands=commands,
            memory_store=memory_store,
            memory_scope=args.memory_scope,
        )
        model = _build_model(args)
        event_log = EventLog(event_path, run_id)
        runner = AgentRunner(
            model=model,
            tools=tools,
            event_log=event_log,
            limits=limits,
            context_compiler=_context_compiler_from_args(args),
        )
        result = asyncio.run(runner.run(args.task))
    finally:
        if event_log is not None:
            event_log.close()
        if tools is not None:
            tools.close()
        elif memory_store is not None:
            memory_store.close()
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
    context_compiler = (
        None
        if state.config.context_config is None
        else ContextCompiler(
            ContextCompilerConfig.from_dict(state.config.context_config)
        )
    )
    memory_store: SemanticMemoryStore | None = None
    tools: WorkspaceTools | None = None
    event_log: EventLog | None = None
    try:
        memory_store = (
            None
            if args.memory_store is None
            else SemanticMemoryStore(args.memory_store)
        )
        tools = WorkspaceTools(
            workspace,
            permissions=state.config.permissions,
            limits=state.config.limits,
            test_commands=commands,
            memory_store=memory_store,
            memory_scope=args.memory_scope,
        )
        model = _build_model(args)
        event_log = EventLog(event_path, state.run_id, repair_truncated=True)
        runner = AgentRunner(
            model=model,
            tools=tools,
            event_log=event_log,
            limits=state.config.limits,
            checkpoint_path=checkpoint_path,
            context_compiler=context_compiler,
        )
        run_result = asyncio.run(
            runner.resume(pending_tool_resolution=args.pending_tool_resolution)
        )
    finally:
        if event_log is not None:
            event_log.close()
        if tools is not None:
            tools.close()
        elif memory_store is not None:
            memory_store.close()
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
            "context": {
                "config": state.config.context_config,
                "fingerprint": state.config.context_fingerprint,
                "last_receipt": next(
                    (
                        event["data"].get("context")
                        for event in reversed(events)
                        if event.get("type") == "model.requested"
                        and isinstance(event.get("data"), dict)
                        and event["data"].get("context") is not None
                    ),
                    None,
                ),
            },
        }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _trace(args: argparse.Namespace) -> int:
    events = read_events(args.event_log)
    _write(args.html, render_trace_html(events))
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

    context_eval = subparsers.add_parser(
        "context-eval",
        help="compare live context policies on fixed, model-free coding traces",
    )
    context_eval.add_argument("--policies", default="recent,topk,density,submodular")
    context_eval.add_argument("--budgets", default="512,1024")
    context_eval.add_argument("--repetitions", type=int, default=3)
    context_eval.add_argument("--recent-blocks", type=int, default=2)
    context_eval.add_argument("--max-tool-output-tokens", type=int, default=96)
    context_eval.add_argument("--output", help="write the complete JSON report")
    context_eval.add_argument("--markdown", help="write the summary as Markdown")
    context_eval.set_defaults(handler=_context_eval)

    agent_eval = subparsers.add_parser(
        "agent-eval",
        help="compare coding-agent search strategies on fixed ACM/math fixtures",
    )
    agent_eval.add_argument(
        "--strategies",
        default="single_pass,best_of_n,orchestrated",
        help="comma-separated strategies: single_pass,best_of_n,orchestrated",
    )
    agent_eval.add_argument(
        "--fixtures",
        default="all",
        help="all or comma-separated fixture ids (two-sum,extended-gcd)",
    )
    agent_eval.add_argument("--repetitions", type=int, default=1)
    agent_eval.add_argument("--max-rounds", type=int, default=2)
    agent_eval.add_argument("--max-model-calls", type=int, default=6)
    agent_eval.add_argument("--max-candidates", type=int, default=2)
    agent_eval.add_argument("--max-test-calls", type=int, default=2)
    agent_eval.add_argument(
        "--search-policy",
        choices=("beam", "mcts"),
        default="beam",
        help="branch selection policy used inside each strategy",
    )
    agent_eval.add_argument("--exploration-constant", type=float, default=1.0)
    agent_eval.add_argument(
        "--no-hidden-tests",
        action="store_true",
        help=(
            "skip the independent fixture grader (visible tests remain "
            "authoritative for search)"
        ),
    )
    agent_eval.add_argument(
        "--model",
        help=(
            "optional OpenAI-compatible model; omit for deterministic "
            "ScriptedModel mode"
        ),
    )
    agent_eval.add_argument("--planner-model")
    agent_eval.add_argument("--solver-model")
    agent_eval.add_argument("--reviewer-model")
    agent_eval.add_argument("--base-url")
    agent_eval.add_argument("--api-key-env", default="CONTEXTOPT_API_KEY")
    agent_eval.add_argument("--model-timeout", type=float, default=90.0)
    agent_eval.add_argument("--model-retries", type=int, default=2)
    agent_eval.add_argument("--temperature", type=float, default=0.0)
    agent_eval.add_argument("--output", help="write the complete JSON report")
    agent_eval.add_argument(
        "--manifest", help="write reproducibility metadata without API credentials"
    )
    agent_eval.add_argument("--markdown", help="write the summary as Markdown")
    agent_eval.add_argument(
        "--html", help="write a self-contained evaluation dashboard"
    )
    agent_eval.add_argument(
        "--checkpoint", help="atomically persist completed evaluation cells"
    )
    agent_eval.add_argument(
        "--resume", action="store_true", help="resume completed cells from --checkpoint"
    )
    agent_eval.set_defaults(handler=_agent_eval)

    recovery_eval = subparsers.add_parser(
        "recovery-eval",
        help="inject durable-boundary stops and evaluate fresh-run recovery",
    )
    recovery_eval.add_argument(
        "--scenarios",
        default="all",
        help="all or comma-separated deterministic recovery scenario ids",
    )
    recovery_eval.add_argument("--output", help="write the complete JSON report")
    recovery_eval.add_argument(
        "--markdown", help="write the recovery report as Markdown"
    )
    recovery_eval.add_argument(
        "--html", help="write a self-contained recovery dashboard"
    )
    recovery_eval.add_argument(
        "--manifest", help="write recovery protocol metadata without secrets"
    )
    recovery_eval.set_defaults(handler=_recovery_eval)

    memory_eval = subparsers.add_parser(
        "memory-eval",
        help="measure deterministic cross-run semantic-memory retrieval conformance",
    )
    memory_eval.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="repeated searches per fixed query (minimum 2)",
    )
    memory_eval.add_argument(
        "--limit",
        type=int,
        default=3,
        help="maximum results returned for each query",
    )
    memory_eval.add_argument("--output", help="write the complete JSON report")
    memory_eval.add_argument("--markdown", help="write the summary as Markdown")
    memory_eval.add_argument(
        "--html", help="write a self-contained retrieval dashboard"
    )
    memory_eval.set_defaults(handler=_semantic_memory_eval)

    propose = subparsers.add_parser(
        "propose-case",
        help="ask a model for bounded coding candidates without executing them",
    )
    propose.add_argument("task", help="coding task or issue text")
    propose.add_argument(
        "--root-files",
        required=True,
        help="JSON object mapping relative paths to complete source text",
    )
    _add_model_arguments(propose)
    propose.add_argument("--run-id", default="branch-proposal")
    propose.add_argument("--max-candidates", type=int, default=4)
    propose.add_argument("--max-files-per-candidate", type=int, default=32)
    propose.add_argument("--max-file-chars", type=int, default=200_000)
    propose.add_argument("--max-total-prompt-chars", type=int, default=400_000)
    propose.add_argument("--max-output-tokens", type=int, default=8_192)
    propose.add_argument("--output", help="write the proposed BranchCase as JSON")
    propose.set_defaults(handler=_propose_case)

    session = subparsers.add_parser(
        "search-session",
        help="run or resume iterative model proposal and test-guided search",
    )
    session.add_argument(
        "--task",
        help="coding task or issue text; omitted when resuming from a checkpoint",
    )
    session.add_argument(
        "--root-files",
        help="JSON object mapping relative paths to complete source text",
    )
    session.add_argument("--checkpoint", required=True)
    session.add_argument("--resume", action="store_true")
    session.add_argument(
        "--retry-pending",
        action="store_true",
        help="retry a model request that was pending when the process stopped",
    )
    _add_model_arguments(session)
    session.add_argument("--run-id")
    session.add_argument("--test-command")
    session.add_argument(
        "--allow-command",
        action="store_true",
        help="explicitly allow the trusted host test command",
    )
    session.add_argument("--test-suite", default="visible-tests")
    session.add_argument("--test-name", default="all-visible-tests")
    session.add_argument("--test-timeout", type=float, default=120.0)
    session.add_argument("--max-report-bytes", type=int, default=64 * 1024)
    session.add_argument("--max-rounds", type=int, default=3)
    session.add_argument("--max-model-calls", type=int, default=4)
    session.add_argument("--session-max-candidates", type=int, default=16)
    session.add_argument("--session-max-test-calls", type=int, default=16)
    session.add_argument(
        "--max-parallel-tests",
        type=int,
        default=1,
        help="maximum number of isolated candidate test workspaces in flight",
    )
    session.add_argument(
        "--scheduler-policy",
        choices=("fixed", "adaptive"),
        default="fixed",
        help="fixed candidate order or adaptive parent-quality scheduling",
    )
    session.add_argument("--proposal-max-candidates", type=int, default=4)
    session.add_argument("--max-files-per-candidate", type=int, default=32)
    session.add_argument("--max-file-chars", type=int, default=200_000)
    session.add_argument("--max-total-prompt-chars", type=int, default=400_000)
    session.add_argument("--proposal-output-tokens", type=int, default=8_192)
    session.add_argument("--beam-width", type=int, default=2)
    session.add_argument(
        "--search-policy",
        choices=("beam", "mcts"),
        default="beam",
        help="beam baseline or observed-quality MCTS traversal",
    )
    session.add_argument("--exploration-constant", type=float, default=1.0)
    session.add_argument("--max-depth", type=int, default=4)
    session.add_argument("--branch-max-candidates", type=int, default=32)
    session.add_argument("--test-environment", default="visible-tests-v1")
    session.add_argument("--no-stop-on-pass", action="store_true")
    session.add_argument("--output", help="write the complete session report")
    session.add_argument("--markdown", help="write a Markdown session report")
    session.set_defaults(handler=_search_session)

    orchestrate = subparsers.add_parser(
        "orchestrate",
        help="run a bounded planner/solver/reviewer coding search",
    )
    orchestrate.add_argument(
        "--task",
        help="coding task or issue text; omitted when resuming from a checkpoint",
    )
    orchestrate.add_argument("--root-files")
    orchestrate.add_argument("--checkpoint", required=True)
    orchestrate.add_argument("--resume", action="store_true")
    orchestrate.add_argument(
        "--retry-pending",
        action="store_true",
        help="retry a planner request that was pending when the process stopped",
    )
    orchestrate.add_argument("--planner-script")
    orchestrate.add_argument("--solver-script")
    orchestrate.add_argument("--reviewer-script")
    orchestrate.add_argument("--planner-model")
    orchestrate.add_argument("--solver-model")
    orchestrate.add_argument("--reviewer-model")
    orchestrate.add_argument("--base-url")
    orchestrate.add_argument("--api-key-env", default="CONTEXTOPT_API_KEY")
    orchestrate.add_argument("--model-timeout", type=float, default=90.0)
    orchestrate.add_argument("--model-retries", type=int, default=2)
    orchestrate.add_argument("--temperature", type=float, default=0.0)
    orchestrate.add_argument("--run-id")
    orchestrate.add_argument("--test-command")
    orchestrate.add_argument("--allow-command", action="store_true")
    orchestrate.add_argument("--test-suite", default="visible-tests")
    orchestrate.add_argument("--test-name", default="all-visible-tests")
    orchestrate.add_argument("--test-timeout", type=float, default=120.0)
    orchestrate.add_argument("--max-report-bytes", type=int, default=64 * 1024)
    orchestrate.add_argument("--max-rounds", type=int, default=3)
    orchestrate.add_argument("--max-model-calls", type=int, default=12)
    orchestrate.add_argument("--max-planner-calls", type=int, default=3)
    orchestrate.add_argument("--max-solver-calls", type=int, default=3)
    orchestrate.add_argument("--max-reviewer-calls", type=int, default=3)
    orchestrate.add_argument("--max-candidates", type=int, default=16)
    orchestrate.add_argument("--max-test-calls", type=int, default=16)
    orchestrate.add_argument(
        "--max-parallel-tests",
        type=int,
        default=1,
        help="maximum number of isolated candidate test workspaces in flight",
    )
    orchestrate.add_argument(
        "--speculative-solver-width",
        type=int,
        default=1,
        help="number of independent solver provider calls to run concurrently",
    )
    orchestrate.add_argument(
        "--speculative-solver-stop-on-valid",
        action="store_true",
        help=(
            "cancel unfinished solver lanes after the first parseable candidate "
            "(best-effort provider cancellation)"
        ),
    )
    orchestrate.add_argument(
        "--scheduler-policy",
        choices=("fixed", "adaptive"),
        default="fixed",
        help="fixed candidate order or adaptive parent-quality scheduling",
    )
    orchestrate.add_argument(
        "--merge-policy",
        choices=("disabled", "disjoint"),
        default="disabled",
        help="disabled or bounded three-way merge of independent solver branches",
    )
    orchestrate.add_argument("--max-total-tokens", type=int, default=100_000)
    orchestrate.add_argument(
        "--context-policy",
        choices=("full", "recent", "topk", "density", "submodular"),
        default="submodular",
        help="context-routing policy applied independently to each role history",
    )
    orchestrate.add_argument(
        "--context-budget",
        type=int,
        default=16_000,
        help="estimated input-token budget for each compiled role request",
    )
    orchestrate.add_argument(
        "--context-recent-blocks",
        type=int,
        default=2,
        help="newest protocol blocks retained as mandatory context",
    )
    orchestrate.add_argument(
        "--context-max-tool-output-tokens",
        type=int,
        default=2_048,
        help="estimated-token cap per tool observation before compaction",
    )
    orchestrate.add_argument(
        "--context-memory",
        choices=("none", "versioned-v1"),
        default="versioned-v1",
        help="observed-memory validity signals supplied to role context routing",
    )
    orchestrate.add_argument("--planner-max-items", type=int, default=6)
    orchestrate.add_argument("--planner-max-item-chars", type=int, default=600)
    orchestrate.add_argument("--planner-max-prompt-chars", type=int, default=400_000)
    orchestrate.add_argument("--planner-output-tokens", type=int, default=4_096)
    orchestrate.add_argument("--solver-max-candidates", type=int, default=4)
    orchestrate.add_argument("--max-files-per-candidate", type=int, default=32)
    orchestrate.add_argument("--max-file-chars", type=int, default=200_000)
    orchestrate.add_argument("--max-total-prompt-chars", type=int, default=400_000)
    orchestrate.add_argument("--solver-output-tokens", type=int, default=8_192)
    orchestrate.add_argument("--reviewer-max-prompt-chars", type=int, default=400_000)
    orchestrate.add_argument("--reviewer-output-tokens", type=int, default=2_048)
    orchestrate.add_argument("--reviewer-max-issue-items", type=int, default=8)
    orchestrate.add_argument("--reviewer-max-item-chars", type=int, default=600)
    orchestrate.add_argument("--beam-width", type=int, default=2)
    orchestrate.add_argument(
        "--search-policy",
        choices=("beam", "mcts"),
        default="beam",
        help="beam baseline or observed-quality MCTS traversal",
    )
    orchestrate.add_argument("--exploration-constant", type=float, default=1.0)
    orchestrate.add_argument("--max-depth", type=int, default=4)
    orchestrate.add_argument("--branch-max-candidates", type=int, default=32)
    orchestrate.add_argument("--test-environment", default="visible-tests-v1")
    orchestrate.add_argument("--no-stop-on-pass", action="store_true")
    orchestrate.add_argument("--output")
    orchestrate.add_argument("--markdown")
    orchestrate.add_argument("--html")
    orchestrate.set_defaults(handler=_orchestrate)

    apply_best = subparsers.add_parser(
        "apply-best",
        help="explicitly apply an accepted search-session snapshot to a workspace",
    )
    apply_best.add_argument("--checkpoint", required=True)
    apply_best.add_argument("--workspace", default=".")
    apply_best.add_argument(
        "--allow-write",
        action="store_true",
        help="explicitly authorize writes to the target workspace",
    )
    apply_best.add_argument(
        "--allow-delete",
        action="store_true",
        help="authorize deletion when the accepted snapshot removes a file",
    )
    apply_best.add_argument("--receipt", help="write the apply receipt to this path")
    apply_best.set_defaults(handler=_apply_best)

    rollback_best = subparsers.add_parser(
        "rollback-best",
        help="restore the baseline after an explicit apply-best operation",
    )
    rollback_best.add_argument("--checkpoint", required=True)
    rollback_best.add_argument("--workspace", default=".")
    rollback_best.add_argument(
        "--allow-write",
        action="store_true",
        help="explicitly authorize writes to the target workspace",
    )
    rollback_best.add_argument(
        "--receipt",
        help="apply receipt; defaults to <checkpoint>.apply.json",
    )
    rollback_best.add_argument(
        "--rollback-receipt",
        help="write the rollback receipt to this path",
    )
    rollback_best.set_defaults(handler=_rollback_best)

    branch_search = subparsers.add_parser(
        "branch-search",
        help="run deterministic test-guided coding-candidate branch search",
    )
    branch_search.add_argument(
        "input",
        nargs="?",
        help="JSON branch case; omit it to run the built-in repair demo",
    )
    branch_search.add_argument("--beam-width", type=int, default=2)
    branch_search.add_argument(
        "--search-policy",
        choices=("beam", "mcts"),
        default="beam",
        help="beam baseline or observed-quality MCTS traversal",
    )
    branch_search.add_argument("--exploration-constant", type=float, default=1.0)
    branch_search.add_argument("--max-depth", type=int, default=4)
    branch_search.add_argument("--max-candidates", type=int, default=32)
    branch_search.add_argument(
        "--test-environment",
        default="visible-tests-v1",
        help="fingerprint of the deterministic test environment used for deduplication",
    )
    branch_search.add_argument(
        "--no-stop-on-pass",
        action="store_true",
        help="continue exploring the current frontier after a passing candidate",
    )
    branch_search.add_argument(
        "--test-command",
        help=(
            "trusted argv command to execute in one disposable workspace per "
            "unique candidate"
        ),
    )
    branch_search.add_argument(
        "--allow-command",
        action="store_true",
        help="explicitly allow the trusted host test command to execute",
    )
    branch_search.add_argument("--test-suite", default="visible-tests")
    branch_search.add_argument("--test-name", default="all-visible-tests")
    branch_search.add_argument("--test-timeout", type=float, default=120.0)
    branch_search.add_argument("--max-report-bytes", type=int, default=64 * 1024)
    branch_search.add_argument("--output", help="write the complete JSON report")
    branch_search.add_argument("--markdown", help="write a Markdown search report")
    branch_search.add_argument("--html", help="write a self-contained SVG HTML report")
    branch_search.set_defaults(handler=_branch_search)

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
    run.add_argument(
        "--memory-store",
        help=(
            "append-only JSONL store exposed through memory_search; use "
            "--context-memory versioned-v1+semantic for bounded automatic candidates; "
            "memory_save and memory_feedback also require --allow-write"
        ),
    )
    run.add_argument(
        "--memory-scope",
        help=(
            "optional semantic-memory scope for automatic context candidates; "
            "global entries remain visible"
        ),
    )
    run.add_argument("--max-turns", type=int, default=20)
    run.add_argument("--max-tool-calls", type=int, default=50)
    run.add_argument("--max-total-tokens", type=int, default=100_000)
    run.add_argument("--max-output-tokens", type=int, default=2_048)
    run.add_argument("--wall-timeout", type=float, default=900.0)
    run.add_argument("--command-timeout", type=float, default=120.0)
    run.add_argument("--max-tool-output-bytes", type=int, default=256 * 1024)
    run.add_argument(
        "--context-policy",
        choices=("full", "recent", "topk", "density", "submodular"),
        default="submodular",
        help="live context-routing policy used for every model request",
    )
    run.add_argument(
        "--context-budget",
        type=int,
        default=16_000,
        help="estimated input-token budget for compiled message history",
    )
    run.add_argument(
        "--context-recent-blocks",
        type=int,
        default=2,
        help="newest protocol blocks retained as mandatory context",
    )
    run.add_argument(
        "--context-max-tool-output-tokens",
        type=int,
        default=2_048,
        help="estimated-token cap per tool observation before head/tail compaction",
    )
    run.add_argument(
        "--context-memory",
        choices=("none", "versioned-v1", "versioned-v1+semantic"),
        default="versioned-v1",
        help=(
            "context memory policy: observed evidence only, or opt-in bounded "
            "semantic candidates from --memory-store"
        ),
    )
    run.set_defaults(handler=_run_agent)

    resume = subparsers.add_parser(
        "resume", help="resume a non-terminal schema-v2 coding-agent run"
    )
    resume.add_argument("event_log")
    resume.add_argument("--workspace", default=".")
    _add_model_arguments(resume)
    resume.add_argument(
        "--memory-store",
        help="the same semantic-memory JSONL store configured for the original run",
    )
    resume.add_argument(
        "--memory-scope",
        help=(
            "the same optional semantic-memory scope used by the original run; "
            "required for an equivalent tool configuration when it was set"
        ),
    )
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
    trace.add_argument("--html", help="write a self-contained trace timeline")
    trace.set_defaults(handler=_trace)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        return 130
    except (PermissionError, ValueError, WorkspaceApplyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
