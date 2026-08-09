"""Deterministic long-horizon recovery fault-injection evaluation.

The runtime already has a recovery contract and unit-level crash tests.  This module
turns the most important durable boundaries into a small executable matrix that can
be shown, regenerated, and inspected without a model API key.  The injected failure
is a process-like stop after a durable event; the second half always uses a fresh
runner, model adapter, and tool registry.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Literal

from contextopt.runtime.events import EventLog, read_events
from contextopt.runtime.model import ScriptedModel
from contextopt.runtime.protocol import RunLimits, RunPermissions
from contextopt.runtime.recovery import RunProjection, replay_events_with_checkpoint
from contextopt.runtime.runner import AgentRunner
from contextopt.runtime.tools import WorkspaceTools

FaultEvent = Literal[
    "model.requested",
    "model.responded",
    "tool.started",
    "tool.completed",
]

_CLAIM_BOUNDARY = (
    "This is a deterministic recovery-contract evaluation with injected process-like "
    "stops and ScriptedModel responses. It measures durable event boundaries, pending "
    "work reconstruction, conservative tool replay, pause/resolution semantics, and "
    "duplicate-effect accounting on one local fixture; it does not prove arbitrary "
    "machine-loss recovery, exactly-once provider effects, sandbox security, or model "
    "coding quality."
)


@dataclass(frozen=True, slots=True)
class RecoveryScenario:
    """One durable boundary and the expected recovery policy."""

    scenario_id: str
    title: str
    fault_event: FaultEvent
    tool_name: str | None = None
    expected_first_resume: Literal["completed", "paused"] = "completed"
    resolution: Literal["none", "mark_failed"] = "none"

    def __post_init__(self) -> None:
        if not self.scenario_id.strip():
            raise ValueError("scenario_id must not be empty")
        if not self.title.strip():
            raise ValueError("title must not be empty")
        if self.tool_name is None and self.fault_event.startswith("tool."):
            raise ValueError("tool fault scenarios require a tool name")
        if self.expected_first_resume == "paused" and self.resolution == "none":
            raise ValueError("paused scenarios require an explicit resolution")

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "title": self.title,
            "fault_event": self.fault_event,
            "tool_name": self.tool_name,
            "expected_first_resume": self.expected_first_resume,
            "resolution": self.resolution,
        }


DEFAULT_SCENARIOS: tuple[RecoveryScenario, ...] = (
    RecoveryScenario(
        scenario_id="pending-model-request",
        title="stop after model.requested and reuse the pending request",
        fault_event="model.requested",
    ),
    RecoveryScenario(
        scenario_id="durable-final-response",
        title="stop after model.responded without making a duplicate call",
        fault_event="model.responded",
    ),
    RecoveryScenario(
        scenario_id="reconcile-write-before-effect",
        title="stop before replace_text and reconcile the precondition",
        fault_event="tool.started",
        tool_name="replace_text",
    ),
    RecoveryScenario(
        scenario_id="durable-write-result",
        title="stop after replace_text result and avoid a duplicate write",
        fault_event="tool.completed",
        tool_name="replace_text",
    ),
    RecoveryScenario(
        scenario_id="nonreplayable-test-pause",
        title="pause an interrupted run_tests call, then mark it failed",
        fault_event="tool.started",
        tool_name="run_tests",
        expected_first_resume="paused",
        resolution="mark_failed",
    ),
)

_SCENARIO_BY_ID = {item.scenario_id: item for item in DEFAULT_SCENARIOS}


@dataclass(frozen=True, slots=True)
class RecoveryEvalConfig:
    """Configuration for a deterministic recovery matrix."""

    scenarios: tuple[str, ...] = tuple(item.scenario_id for item in DEFAULT_SCENARIOS)

    def __post_init__(self) -> None:
        if not self.scenarios:
            raise ValueError("scenarios must not be empty")
        if len(set(self.scenarios)) != len(self.scenarios):
            raise ValueError("scenarios must be unique")
        unknown = sorted(set(self.scenarios) - set(_SCENARIO_BY_ID))
        if unknown:
            raise ValueError(f"unknown recovery scenarios: {unknown!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"scenarios": list(self.scenarios)}


@dataclass(frozen=True, slots=True)
class RecoveryCaseResult:
    """Audit result for one injected-stop scenario."""

    scenario_id: str
    fault_event: str
    tool_name: str | None
    first_resume_status: str
    final_status: str
    expected_first_resume: str
    expected_final_status: str
    pending_before_resume: Mapping[str, Any]
    event_types_before_resume: tuple[str, ...]
    event_types_after_resume: tuple[str, ...]
    model_calls_before_crash: int
    model_calls_after_resume: int
    durable_model_requests: int
    tool_event_counts: Mapping[str, int]
    workspace_digest: str
    expected_workspace_digest: str
    passed: bool
    failure: str | None = None

    def __post_init__(self) -> None:
        if (
            self.final_status == "completed"
            and self.expected_final_status != "completed"
        ):
            raise ValueError("completed result has an invalid expected status")
        object.__setattr__(
            self, "pending_before_resume", dict(self.pending_before_resume)
        )
        object.__setattr__(self, "tool_event_counts", dict(self.tool_event_counts))

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "fault_event": self.fault_event,
            "tool_name": self.tool_name,
            "first_resume_status": self.first_resume_status,
            "final_status": self.final_status,
            "expected_first_resume": self.expected_first_resume,
            "expected_final_status": self.expected_final_status,
            "pending_before_resume": dict(self.pending_before_resume),
            "event_types_before_resume": list(self.event_types_before_resume),
            "event_types_after_resume": list(self.event_types_after_resume),
            "model_calls_before_crash": self.model_calls_before_crash,
            "model_calls_after_resume": self.model_calls_after_resume,
            "durable_model_requests": self.durable_model_requests,
            "tool_event_counts": dict(self.tool_event_counts),
            "workspace_digest": self.workspace_digest,
            "expected_workspace_digest": self.expected_workspace_digest,
            "passed": self.passed,
            "failure": self.failure,
        }


@dataclass(frozen=True, slots=True)
class RecoveryMatrixReport:
    """Serializable recovery matrix with an explicit claim boundary."""

    config: RecoveryEvalConfig
    results: tuple[RecoveryCaseResult, ...]
    claim_boundary: str = _CLAIM_BOUNDARY
    schema_version: str = "1"

    @property
    def passed_count(self) -> int:
        return sum(1 for result in self.results if result.passed)

    @property
    def failed_count(self) -> int:
        return len(self.results) - self.passed_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": "contextopt.recovery-eval.report",
            "config": self.config.to_dict(),
            "summary": {
                "scenario_count": len(self.results),
                "passed_count": self.passed_count,
                "failed_count": self.failed_count,
            },
            "results": [result.to_dict() for result in self.results],
            "claim_boundary": self.claim_boundary,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RecoveryMatrixReport:
        required = {
            "schema_version",
            "kind",
            "config",
            "summary",
            "results",
            "claim_boundary",
        }
        if set(data) != required:
            raise ValueError("recovery report has an invalid schema")
        if (
            data["schema_version"] != "1"
            or data["kind"] != "contextopt.recovery-eval.report"
        ):
            raise ValueError("unsupported recovery report schema")
        raw_config = data["config"]
        raw_results = data["results"]
        if not isinstance(raw_config, Mapping) or not isinstance(raw_results, list):
            raise ValueError("recovery report config/results have invalid types")
        config_values = raw_config.get("scenarios")
        if not isinstance(config_values, list) or not all(
            isinstance(value, str) for value in config_values
        ):
            raise ValueError("recovery report scenarios must be strings")
        config = RecoveryEvalConfig(tuple(config_values))
        results = tuple(_result_from_dict(item) for item in raw_results)
        if tuple(result.scenario_id for result in results) != config.scenarios:
            raise ValueError("recovery report result order does not match config")
        summary = data["summary"]
        if not isinstance(summary, Mapping):
            raise ValueError("recovery report summary must be an object")
        if summary.get("scenario_count") != len(results):
            raise ValueError("recovery report scenario count is inconsistent")
        if summary.get("passed_count") != sum(result.passed for result in results):
            raise ValueError("recovery report passed count is inconsistent")
        if summary.get("failed_count") != len(results) - sum(
            result.passed for result in results
        ):
            raise ValueError("recovery report failed count is inconsistent")
        boundary = data["claim_boundary"]
        if not isinstance(boundary, str) or not boundary:
            raise ValueError("recovery report claim boundary must be non-empty")
        return cls(config=config, results=results, claim_boundary=boundary)


def _result_from_dict(data: Any) -> RecoveryCaseResult:
    if not isinstance(data, Mapping):
        raise ValueError("recovery result must be an object")
    required = {
        "scenario_id",
        "fault_event",
        "tool_name",
        "first_resume_status",
        "final_status",
        "expected_first_resume",
        "expected_final_status",
        "pending_before_resume",
        "event_types_before_resume",
        "event_types_after_resume",
        "model_calls_before_crash",
        "model_calls_after_resume",
        "durable_model_requests",
        "tool_event_counts",
        "workspace_digest",
        "expected_workspace_digest",
        "passed",
        "failure",
    }
    if set(data) != required:
        raise ValueError("recovery result has an invalid schema")
    event_before = data["event_types_before_resume"]
    event_after = data["event_types_after_resume"]
    if not isinstance(event_before, list) or not all(
        isinstance(value, str) for value in event_before
    ):
        raise ValueError("recovery result event_types_before_resume is invalid")
    if not isinstance(event_after, list) or not all(
        isinstance(value, str) for value in event_after
    ):
        raise ValueError("recovery result event_types_after_resume is invalid")
    pending = data["pending_before_resume"]
    counts = data["tool_event_counts"]
    if not isinstance(pending, Mapping) or not isinstance(counts, Mapping):
        raise ValueError("recovery result pending/count fields are invalid")
    return RecoveryCaseResult(
        scenario_id=str(data["scenario_id"]),
        fault_event=str(data["fault_event"]),
        tool_name=None if data["tool_name"] is None else str(data["tool_name"]),
        first_resume_status=str(data["first_resume_status"]),
        final_status=str(data["final_status"]),
        expected_first_resume=str(data["expected_first_resume"]),
        expected_final_status=str(data["expected_final_status"]),
        pending_before_resume=dict(pending),
        event_types_before_resume=tuple(event_before),
        event_types_after_resume=tuple(event_after),
        model_calls_before_crash=int(data["model_calls_before_crash"]),
        model_calls_after_resume=int(data["model_calls_after_resume"]),
        durable_model_requests=int(data["durable_model_requests"]),
        tool_event_counts={str(key): int(value) for key, value in counts.items()},
        workspace_digest=str(data["workspace_digest"]),
        expected_workspace_digest=str(data["expected_workspace_digest"]),
        passed=bool(data["passed"]),
        failure=None if data["failure"] is None else str(data["failure"]),
    )


class _InjectedProcessStop(BaseException):
    """A deliberate stop after a durable event, bypassing normal error handling."""


class _CrashAfterEventRunner(AgentRunner):
    def __init__(self, *args: Any, fault_event: FaultEvent, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._fault_event = fault_event
        self._injected = False

    def _append(
        self,
        state: RunProjection | None,
        event_type: str,
        data: dict[str, object],
    ) -> RunProjection:
        updated = super()._append(state, event_type, data)
        if event_type == self._fault_event and not self._injected:
            self._injected = True
            raise _InjectedProcessStop(event_type)
        return updated


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _response(
    *,
    content: str = "recovered",
    tool_calls: Sequence[Mapping[str, Any]] = (),
    expect: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "response": {
            "content": content,
            "tool_calls": list(tool_calls),
            "usage": {"prompt_tokens": 7, "completion_tokens": 11},
        }
    }
    if expect is not None:
        value["expect"] = dict(expect)
    return value


def _scenario_steps(
    scenario: RecoveryScenario, *, source_text: str
) -> tuple[dict[str, Any], ...]:
    if scenario.tool_name is None:
        return (_response(content="durable final response"),)

    if scenario.tool_name == "replace_text":
        call = {
            "id": "replace-recovery",
            "name": "replace_text",
            "arguments": {
                "path": "module.py",
                "old_text": 'VALUE = "bad"\n',
                "new_text": 'VALUE = "good"\n',
                "expected_sha256": _sha256_text(source_text),
                "expected_occurrences": 1,
            },
        }
        return (
            _response(content="apply the bounded replacement", tool_calls=(call,)),
            _response(
                content="write recovered",
                expect={
                    "last_tool": "replace_text",
                    "tool_call_id": "replace-recovery",
                },
            ),
        )

    if scenario.tool_name == "run_tests":
        call = {
            "id": "tests-recovery",
            "name": "run_tests",
            "arguments": {"scope": "visible"},
        }
        return (
            _response(content="run the visible oracle", tool_calls=(call,)),
            _response(
                content="tests were resolved conservatively",
                expect={
                    "last_tool": "run_tests",
                    "tool_call_id": "tests-recovery",
                    "observation_contains": "indeterminate_previous_execution",
                },
            ),
        )
    raise ValueError(f"unsupported recovery scenario tool: {scenario.tool_name}")


def _limits() -> RunLimits:
    return RunLimits(
        max_turns=5,
        max_tool_calls=5,
        max_total_tokens=1_000,
        max_output_tokens_per_call=128,
        wall_timeout_seconds=10.0,
        command_timeout_seconds=5.0,
        max_tool_output_bytes=16_384,
    )


def _make_tools(root: Path, scenario: RecoveryScenario) -> WorkspaceTools:
    limits = _limits()
    permissions = RunPermissions(
        allow_write=scenario.tool_name == "replace_text",
        allow_command=scenario.tool_name == "run_tests",
    )
    test_commands = (
        {"visible": (sys.executable, "-c", "raise SystemExit(0)")}
        if scenario.tool_name == "run_tests"
        else {}
    )
    return WorkspaceTools(
        root,
        permissions=permissions,
        limits=limits,
        test_commands=test_commands,
    )


def _make_runner(
    *,
    root: Path,
    event_path: Path,
    checkpoint_path: Path,
    run_id: str,
    model: ScriptedModel,
    scenario: RecoveryScenario,
    injected: bool,
) -> tuple[AgentRunner, EventLog]:
    log = EventLog(event_path, run_id)
    common: dict[str, Any] = {
        "model": model,
        "tools": _make_tools(root, scenario),
        "event_log": log,
        "limits": _limits(),
        "checkpoint_path": checkpoint_path,
    }
    runner: AgentRunner
    if injected:
        runner = _CrashAfterEventRunner(
            **common,
            fault_event=scenario.fault_event,
        )
    else:
        runner = AgentRunner(**common)
    return runner, log


def _pending_payload(state: RunProjection) -> dict[str, Any]:
    return {
        "phase": state.phase,
        "model": None
        if state.pending_model is None
        else {"turn": state.pending_model.turn},
        "tools": [
            {
                "name": pending.call.name,
                "started": pending.started,
                "replay_policy": (
                    None if pending.plan is None else pending.plan.replay_policy
                ),
            }
            for pending in state.pending_tools
        ],
    }


def _run_case(scenario: RecoveryScenario) -> RecoveryCaseResult:
    source_text = 'VALUE = "bad"\n'
    task = "Recover the fixture after an injected process stop."
    run_id = f"recovery-matrix:{scenario.scenario_id}:v1"
    with tempfile.TemporaryDirectory(prefix="contextopt-recovery-") as temp_dir:
        root = Path(temp_dir)
        # Preserve LF bytes so the scripted precondition is the exact digest that
        # WorkspaceTools observes on every supported platform.
        (root / "module.py").write_bytes(source_text.encode("utf-8"))
        event_path = root / "events.jsonl"
        checkpoint_path = root / "events.checkpoint.json"
        steps = _scenario_steps(scenario, source_text=source_text)
        initial_model = ScriptedModel(steps, name=f"recovery:{scenario.scenario_id}:v1")
        crash_runner, crash_log = _make_runner(
            root=root,
            event_path=event_path,
            checkpoint_path=checkpoint_path,
            run_id=run_id,
            model=initial_model,
            scenario=scenario,
            injected=True,
        )
        try:
            with suppress(_InjectedProcessStop):
                asyncio.run(crash_runner.run(task))
        finally:
            crash_log.close()

        before_events = read_events(event_path)
        before_state = replay_events_with_checkpoint(before_events, checkpoint_path)
        first_model = ScriptedModel(steps, name=f"recovery:{scenario.scenario_id}:v1")
        first_runner, first_log = _make_runner(
            root=root,
            event_path=event_path,
            checkpoint_path=checkpoint_path,
            run_id=run_id,
            model=first_model,
            scenario=scenario,
            injected=False,
        )
        try:
            first_result = asyncio.run(first_runner.resume())
        finally:
            first_log.close()

        first_resume_status = str(first_result.status)
        final_result = first_result
        model_calls_after_resume = len(first_model.requests)
        if first_result.status == "paused":
            if scenario.resolution == "none":
                raise AssertionError("a paused recovery case has no resolution")
            second_model = ScriptedModel(
                steps, name=f"recovery:{scenario.scenario_id}:v1"
            )
            second_runner, second_log = _make_runner(
                root=root,
                event_path=event_path,
                checkpoint_path=checkpoint_path,
                run_id=run_id,
                model=second_model,
                scenario=scenario,
                injected=False,
            )
            try:
                final_result = asyncio.run(
                    second_runner.resume(pending_tool_resolution=scenario.resolution)
                )
            finally:
                second_log.close()
            model_calls_after_resume += len(second_model.requests)

        after_events = read_events(event_path)
        after_state = replay_events_with_checkpoint(after_events, checkpoint_path)
        event_counts = Counter(str(event.get("type")) for event in after_events)
        tool_event_counts = {
            name: count
            for name, count in sorted(event_counts.items())
            if name.startswith("tool.")
        }
        expected_text = (
            'VALUE = "good"\n' if scenario.tool_name == "replace_text" else source_text
        )
        # Compare only the fixture file; event/checkpoint and temporary metadata are
        # intentionally excluded from the claimable workspace digest.
        actual_file_digest = hashlib.sha256(
            (root / "module.py").read_bytes()
        ).hexdigest()
        expected_file_digest = _sha256_text(expected_text)
        pending = _pending_payload(before_state)
        final_status = str(final_result.status)
        durable_model_requests = event_counts.get("model.requested", 0)
        passed = (
            first_resume_status == scenario.expected_first_resume
            and final_status == "completed"
            and actual_file_digest == expected_file_digest
            and after_state.phase == "terminal"
            and (
                scenario.tool_name != "replace_text"
                or event_counts.get("tool.completed", 0) == 1
            )
            and (
                scenario.tool_name != "run_tests"
                or event_counts.get("tool.failed", 0) == 1
            )
        )
        failure: str | None = None
        if not passed:
            failure = (
                f"first_resume={first_resume_status!r}, final={final_status!r}, "
                f"workspace={actual_file_digest[:12]} "
                f"expected={expected_file_digest[:12]}, "
                f"phase={after_state.phase!r}"
            )
        return RecoveryCaseResult(
            scenario_id=scenario.scenario_id,
            fault_event=scenario.fault_event,
            tool_name=scenario.tool_name,
            first_resume_status=first_resume_status,
            final_status=final_status,
            expected_first_resume=scenario.expected_first_resume,
            expected_final_status="completed",
            pending_before_resume=pending,
            event_types_before_resume=tuple(
                str(event.get("type")) for event in before_events
            ),
            event_types_after_resume=tuple(
                str(event.get("type")) for event in after_events[len(before_events) :]
            ),
            model_calls_before_crash=len(initial_model.requests),
            model_calls_after_resume=model_calls_after_resume,
            durable_model_requests=durable_model_requests,
            tool_event_counts=tool_event_counts,
            workspace_digest=actual_file_digest,
            expected_workspace_digest=expected_file_digest,
            passed=passed,
            failure=failure,
        )


def run_recovery_evaluation(
    config: RecoveryEvalConfig | None = None,
) -> RecoveryMatrixReport:
    """Run the selected deterministic crash/recovery scenarios."""

    selected = config or RecoveryEvalConfig()
    results = tuple(_run_case(_SCENARIO_BY_ID[item]) for item in selected.scenarios)
    return RecoveryMatrixReport(config=selected, results=results)


def render_recovery_console(report: RecoveryMatrixReport) -> str:
    lines = [
        "scenario | crash after | first resume | final | "
        "model calls before/after | pass",
        "--- | --- | --- | --- | ---: | ---",
    ]
    for result in report.results:
        lines.append(
            f"{result.scenario_id} | {result.fault_event} | "
            f"{result.first_resume_status} | {result.final_status} | "
            f"{result.model_calls_before_crash}/{result.model_calls_after_resume} | "
            f"{'yes' if result.passed else 'no'}"
        )
    lines.append(
        f"passed={report.passed_count}/{len(report.results)} "
        f"failed={report.failed_count}"
    )
    return "\n".join(lines) + "\n"


def render_recovery_markdown(report: RecoveryMatrixReport) -> str:
    lines = [
        "# Long-horizon recovery fault-injection evaluation",
        "",
        "This report stops a fresh run after a durable event, opens the same log "
        "with a "
        "new runner, and records the recovery contract that follows.",
        "",
        "| Scenario | Fault boundary | Pending before resume | First resume | Final | "
        "Model calls before/after | Tool events | Result |",
        "|---|---|---|---|---|---:|---|---|",
    ]
    for result in report.results:
        pending = (
            ", ".join(
                [
                    "model" if result.pending_before_resume.get("model") else "",
                    *(
                        str(item.get("name"))
                        for item in result.pending_before_resume.get("tools", [])
                        if isinstance(item, Mapping)
                    ),
                ]
            ).strip(", ")
            or "none"
        )
        tools = (
            ", ".join(
                f"{name}={count}" for name, count in result.tool_event_counts.items()
            )
            or "none"
        )
        lines.append(
            f"| `{result.scenario_id}` | `{result.fault_event}` | {pending} | "
            f"`{result.first_resume_status}` | `{result.final_status}` | "
            f"{result.model_calls_before_crash}/{result.model_calls_after_resume} | "
            f"{tools} | {'PASS' if result.passed else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Durable event evidence",
            "",
        ]
    )
    for result in report.results:
        lines.append(f"### `{result.scenario_id}`")
        lines.append("")
        lines.append(
            "Before resume: "
            + " → ".join(f"`{item}`" for item in result.event_types_before_resume)
        )
        lines.append("")
        lines.append(
            "After resume: "
            + " → ".join(f"`{item}`" for item in result.event_types_after_resume)
        )
        lines.append("")
        if result.failure:
            lines.append(f"Failure: `{result.failure}`")
            lines.append("")
    lines.extend(
        [
            "## Claim boundary",
            "",
            f"> {report.claim_boundary}",
            "",
            f"Summary: **{report.passed_count}/{len(report.results)} scenarios "
            "passed**.",
            "",
        ]
    )
    return "\n".join(lines)


def render_recovery_html(report: RecoveryMatrixReport) -> str:
    payload = escape(json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True))
    rows: list[str] = []
    for result in report.results:
        status_class = "pass" if result.passed else "fail"
        tool_summary = (
            ", ".join(
                f"{key}={value}" for key, value in result.tool_event_counts.items()
            )
            or "none"
        )
        rows.append(
            "<tr>"
            f"<td><code>{escape(result.scenario_id)}</code></td>"
            f"<td><code>{escape(result.fault_event)}</code></td>"
            f"<td>{escape(result.first_resume_status)} → "
            f"{escape(result.final_status)}</td>"
            f"<td>{result.model_calls_before_crash}/"
            f"{result.model_calls_after_resume}</td>"
            f"<td>{escape(tool_summary)}</td>"
            f'<td class="{status_class}">{"PASS" if result.passed else "FAIL"}</td>'
            "</tr>"
        )
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en"><head><meta charset="utf-8">',
            "<title>ContextOpt recovery evaluation</title>",
            "<style>",
            "body { background:#0b1020; color:#e6edf7; "
            "font:15px/1.5 system-ui,sans-serif; margin:0; }",
            "main { max-width:1180px; margin:0 auto; padding:32px; }",
            ".cards { display:flex; gap:12px; flex-wrap:wrap; margin:18px 0; }",
            ".card { background:#151d32; border:1px solid #2b3857; "
            "border-radius:10px; padding:14px 18px; min-width:150px; }",
            ".label { color:#9badc9; font-size:12px; text-transform:uppercase; "
            "letter-spacing:.08em; }",
            ".value { font-size:26px; font-weight:700; margin-top:4px; }",
            "table { border-collapse:collapse; width:100%; background:#11182a; }",
            "th,td { border-bottom:1px solid #2b3857; padding:10px 12px; "
            "text-align:left; }",
            "th { color:#9badc9; font-size:12px; text-transform:uppercase; }",
            ".pass { color:#6ee7a0; font-weight:700; } "
            ".fail { color:#ff8d8d; font-weight:700; }",
            ".boundary { border-left:3px solid #6ea8fe; background:#151d32; "
            "padding:12px 16px; margin-top:22px; }",
            "code { color:#c5d8ff; }",
            "</style></head><body><main>",
            "<h1>Long-horizon recovery evaluation</h1>",
            "<p>Injected process-like stops are followed by a fresh runner and a "
            "durable-log resume.</p>",
            '<section class="cards">',
            f'<div class="card"><div class="label">scenarios</div><div '
            f'class="value">{len(report.results)}</div></div>',
            f'<div class="card"><div class="label">passed</div><div '
            f'class="value pass">{report.passed_count}</div></div>',
            f'<div class="card"><div class="label">failed</div><div '
            f'class="value fail">{report.failed_count}</div></div>',
            "</section>",
            "<table><thead><tr><th>Scenario</th><th>Fault boundary</th>"
            "<th>Recovery</th><th>Model calls</th><th>Tool events</th>"
            "<th>Result</th></tr></thead>",
            f"<tbody>{''.join(rows)}</tbody></table>",
            f'<div class="boundary"><strong>Claim boundary:</strong> '
            f"{escape(report.claim_boundary)}</div>",
            f'<script type="application/json" id="recovery-report">{payload}</script>',
            "</main></body></html>",
        ]
    )


__all__ = [
    "DEFAULT_SCENARIOS",
    "RecoveryCaseResult",
    "RecoveryEvalConfig",
    "RecoveryMatrixReport",
    "RecoveryScenario",
    "render_recovery_console",
    "render_recovery_html",
    "render_recovery_markdown",
    "run_recovery_evaluation",
]
