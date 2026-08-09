"""Deterministic Level 4 robustness fault-injection evaluation.

The runtime has focused unit tests for context compaction, stale evidence, protocol
validation, and compare-and-swap writes.  This module turns those boundaries into one
small executable ledger so a portfolio or PR can show the failure policy and the
observed evidence together.  It never calls a model provider.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Literal, cast

from contextopt.runtime.context import (
    ContextCompiler,
    ContextCompilerConfig,
    estimate_text_tokens,
)
from contextopt.runtime.protocol import (
    AgentMessage,
    RunLimits,
    RunPermissions,
    ToolCall,
)
from contextopt.runtime.semantic_memory import SemanticMemoryStore
from contextopt.runtime.tools import WorkspaceTools

RobustnessScenarioId = Literal[
    "tool-output-compaction",
    "stale-memory-invalidation",
    "duplicate-tool-result",
    "cas-write-conflict",
]

ROBUSTNESS_EVAL_SCHEMA_VERSION = "1"
DEFAULT_SCENARIOS: tuple[RobustnessScenarioId, ...] = (
    "tool-output-compaction",
    "stale-memory-invalidation",
    "duplicate-tool-result",
    "cas-write-conflict",
)
_SCENARIO_TITLES: Mapping[str, str] = {
    "tool-output-compaction": (
        "Compact oversized tool output while preserving sentinels"
    ),
    "stale-memory-invalidation": (
        "Invalidate source-aware memory after a workspace change"
    ),
    "duplicate-tool-result": "Reject a repeated tool result inside one exchange",
    "cas-write-conflict": "Refuse a stale compare-and-swap workspace write",
}
_SCENARIO_SET = frozenset(DEFAULT_SCENARIOS)
_CLAIM_BOUNDARY = (
    "This is a deterministic Level 4 runtime robustness evaluation. It measures local "
    "fault-injection outcomes for context compaction, source-aware memory "
    "invalidation, "
    "duplicate tool-result rejection, and compare-and-swap write conflicts without a "
    "model provider; it does not prove machine-loss recovery, sandbox security, "
    "provider "
    "reliability, exactly-once external effects, or coding quality."
)


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


@dataclass(frozen=True, slots=True)
class RobustnessEvalConfig:
    """Selected deterministic fault-injection scenarios."""

    scenarios: tuple[RobustnessScenarioId, ...] = DEFAULT_SCENARIOS

    def __post_init__(self) -> None:
        if not self.scenarios:
            raise ValueError("robustness scenarios must not be empty")
        if len(self.scenarios) != len(set(self.scenarios)):
            raise ValueError("robustness scenarios must be unique")
        unknown = sorted(set(self.scenarios) - _SCENARIO_SET)
        if unknown:
            raise ValueError(f"unknown robustness scenarios: {unknown!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"scenarios": list(self.scenarios)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RobustnessEvalConfig:
        value = _mapping(data, "robustness config")
        if set(value) != {"scenarios"}:
            raise ValueError("robustness config has an invalid schema")
        raw = value["scenarios"]
        if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
            raise ValueError("robustness scenarios must be an array of strings")
        return cls(tuple(cast(RobustnessScenarioId, item) for item in raw))


@dataclass(frozen=True, slots=True)
class RobustnessCaseResult:
    """One scenario result with JSON-safe observed evidence."""

    scenario_id: RobustnessScenarioId
    title: str
    observations: Mapping[str, Any]
    passed: bool
    error: str | None = None

    def __post_init__(self) -> None:
        if self.scenario_id not in _SCENARIO_SET:
            raise ValueError(f"unknown robustness scenario: {self.scenario_id!r}")
        if self.title != _SCENARIO_TITLES[self.scenario_id]:
            raise ValueError("robustness result title is not canonical")
        if not isinstance(self.observations, Mapping):
            raise ValueError("robustness observations must be an object")
        if not isinstance(self.passed, bool):
            raise ValueError("robustness passed must be a boolean")
        if self.error is not None:
            _text(self.error, "robustness error")
        # Fail early if a future scenario accidentally adds a non-serializable value.
        try:
            json.dumps(dict(self.observations), ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "robustness observations must be JSON serializable"
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "title": self.title,
            "observations": dict(self.observations),
            "passed": self.passed,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RobustnessCaseResult:
        value = _mapping(data, "robustness result")
        required = {"scenario_id", "title", "observations", "passed", "error"}
        if set(value) != required:
            raise ValueError("robustness result has an invalid schema")
        observations = _mapping(value["observations"], "robustness observations")
        return cls(
            scenario_id=cast(
                RobustnessScenarioId, _text(value["scenario_id"], "scenario_id")
            ),
            title=_text(value["title"], "title"),
            observations=dict(observations),
            passed=bool(value["passed"]),
            error=None if value["error"] is None else _text(value["error"], "error"),
        )


@dataclass(frozen=True, slots=True)
class RobustnessMatrixReport:
    """Serializable fault-injection report with an explicit claim boundary."""

    config: RobustnessEvalConfig
    results: tuple[RobustnessCaseResult, ...]
    claim_boundary: str = _CLAIM_BOUNDARY
    schema_version: str = ROBUSTNESS_EVAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ROBUSTNESS_EVAL_SCHEMA_VERSION:
            raise ValueError(f"unsupported robustness schema: {self.schema_version!r}")
        if self.claim_boundary != _CLAIM_BOUNDARY:
            raise ValueError("robustness claim boundary is not canonical")
        if (
            tuple(result.scenario_id for result in self.results)
            != self.config.scenarios
        ):
            raise ValueError("robustness result order does not match config")

    @property
    def passed_count(self) -> int:
        return sum(result.passed for result in self.results)

    @property
    def failed_count(self) -> int:
        return len(self.results) - self.passed_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": "contextopt.robustness-eval.report",
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
    def from_dict(cls, data: Mapping[str, Any]) -> RobustnessMatrixReport:
        value = _mapping(data, "robustness report")
        required = {
            "schema_version",
            "kind",
            "config",
            "summary",
            "results",
            "claim_boundary",
        }
        if set(value) != required:
            raise ValueError("robustness report has an invalid schema")
        if (
            value["schema_version"] != ROBUSTNESS_EVAL_SCHEMA_VERSION
            or value["kind"] != "contextopt.robustness-eval.report"
        ):
            raise ValueError("unsupported robustness report schema")
        raw_results = value["results"]
        if not isinstance(raw_results, list):
            raise ValueError("robustness results must be an array")
        results = tuple(
            RobustnessCaseResult.from_dict(_mapping(item, "robustness result"))
            for item in raw_results
        )
        summary = _mapping(value["summary"], "robustness summary")
        if summary.get("scenario_count") != len(results):
            raise ValueError("robustness scenario count is inconsistent")
        if summary.get("passed_count") != sum(result.passed for result in results):
            raise ValueError("robustness passed count is inconsistent")
        if summary.get("failed_count") != len(results) - sum(
            result.passed for result in results
        ):
            raise ValueError("robustness failed count is inconsistent")
        return cls(
            schema_version=str(value["schema_version"]),
            config=RobustnessEvalConfig.from_dict(
                _mapping(value["config"], "robustness config")
            ),
            results=results,
            claim_boundary=_text(value["claim_boundary"], "claim_boundary"),
        )


def _run_compaction() -> Mapping[str, Any]:
    content = "HEAD-SENTINEL\n" + ("middle payload " * 800) + "\nTAIL-SENTINEL"
    call = ToolCall("large-output", "read_file", '{"path":"large.txt"}')
    messages = (
        AgentMessage(role="user", content="inspect the large tool result"),
        AgentMessage(role="assistant", content="", tool_calls=(call,)),
        AgentMessage(
            role="tool",
            content=content,
            tool_call_id=call.id,
            tool_name=call.name,
        ),
    )
    compiler = ContextCompiler(
        ContextCompilerConfig(
            policy="full", budget_tokens=1_000, max_tool_output_tokens=80
        )
    )
    first = compiler.compile(messages)
    second = compiler.compile(messages)
    compacted = first.messages[-1].content
    return {
        "compacted_block_ids": list(first.receipt.compacted_block_ids),
        "deterministic": first == second,
        "has_head_sentinel": "HEAD-SENTINEL" in compacted,
        "has_tail_sentinel": "TAIL-SENTINEL" in compacted,
        "selected_tokens": first.receipt.estimated_selected_tokens,
        "compacted_tokens": estimate_text_tokens(compacted),
        "max_tool_output_tokens": 80,
    }


def _run_stale_memory() -> Mapping[str, Any]:
    with tempfile.TemporaryDirectory(prefix="contextopt-robust-memory-") as directory:
        path = Path(directory) / "memory.jsonl"
        store = SemanticMemoryStore(path)
        entry = store.put(
            "Two-sum hash-map invariant handles duplicate values.",
            scope="project:acm",
            kind="procedure",
            tags=("two-sum", "hash-map"),
            source_refs=("two_sum.py",),
        ).entry
        before = tuple(
            item.entry.memory_id for item in store.search("two sum hash map")
        )
        invalidated = store.invalidate_source_refs(
            "two_sum.py", "workspace file changed during the run"
        )
        after = tuple(item.entry.memory_id for item in store.search("two sum hash map"))
        revision = store.revision
        store.close()
        reopened = SemanticMemoryStore(path)
        persisted_status = reopened.get(entry.memory_id)
        persisted = None if persisted_status is None else persisted_status.status
        reopened.close()
    return {
        "memory_id": entry.memory_id,
        "retrieved_before": list(before),
        "invalidated_ids": [item.memory_id for item in invalidated],
        "retrieved_after": list(after),
        "persisted_status": persisted,
        "revision": revision,
    }


def _run_duplicate_tool_result() -> Mapping[str, Any]:
    call = ToolCall("same-call", "read_file", '{"path":"module.py"}')
    messages = (
        AgentMessage(role="assistant", content="", tool_calls=(call,)),
        AgentMessage(
            role="tool",
            content="first result",
            tool_call_id=call.id,
            tool_name=call.name,
        ),
        AgentMessage(
            role="tool",
            content="duplicate result",
            tool_call_id=call.id,
            tool_name=call.name,
        ),
    )
    try:
        ContextCompiler(ContextCompilerConfig(policy="full")).compile(messages)
    except ValueError as exc:
        return {"rejected": True, "error": str(exc)}
    return {"rejected": False, "error": None}


def _run_cas_conflict() -> Mapping[str, Any]:
    with tempfile.TemporaryDirectory(prefix="contextopt-robust-cas-") as directory:
        root = Path(directory)
        path = root / "module.py"
        path.write_text("VALUE = 'old'\n", encoding="utf-8")
        tools = WorkspaceTools(
            root,
            permissions=RunPermissions(allow_write=True),
            limits=RunLimits(max_tool_output_bytes=8_192),
        )
        read_call = ToolCall("read-before-write", "read_file", '{"path":"module.py"}')
        read = asyncio.run(tools.execute(read_call))
        stale_sha = read.metadata.get("sha256")
        path.write_text("VALUE = 'external'\n", encoding="utf-8")
        replace_call = ToolCall(
            "stale-write",
            "replace_text",
            json.dumps(
                {
                    "path": "module.py",
                    "old_text": "VALUE = 'old'\n",
                    "new_text": "VALUE = 'new'\n",
                    "expected_sha256": stale_sha,
                    "expected_occurrences": 1,
                }
            ),
        )
        outcome = asyncio.run(tools.execute(replace_call))
        unchanged = path.read_text(encoding="utf-8") == "VALUE = 'external'\n"
        tools.close()
    return {
        "read_ok": read.ok,
        "stale_sha_present": isinstance(stale_sha, str) and bool(stale_sha),
        "write_ok": outcome.ok,
        "error_code": outcome.error_code,
        "workspace_unchanged": unchanged,
    }


def _scenario_observation(scenario_id: RobustnessScenarioId) -> Mapping[str, Any]:
    if scenario_id == "tool-output-compaction":
        return _run_compaction()
    if scenario_id == "stale-memory-invalidation":
        return _run_stale_memory()
    if scenario_id == "duplicate-tool-result":
        return _run_duplicate_tool_result()
    return _run_cas_conflict()


def _scenario_passes(
    scenario_id: RobustnessScenarioId, observations: Mapping[str, Any]
) -> bool:
    if scenario_id == "tool-output-compaction":
        return all(
            (
                observations.get("compacted_block_ids") == ["block-000001"],
                observations.get("deterministic") is True,
                observations.get("has_head_sentinel") is True,
                observations.get("has_tail_sentinel") is True,
                isinstance(observations.get("compacted_tokens"), int)
                and observations["compacted_tokens"] <= 80,
            )
        )
    if scenario_id == "stale-memory-invalidation":
        invalidated = observations.get("invalidated_ids", [])
        return (
            isinstance(invalidated, list)
            and len(invalidated) == 1
            and observations.get("retrieved_before") == invalidated
            and observations.get("retrieved_after") == []
            and observations.get("persisted_status") == "invalidated"
        )
    if scenario_id == "duplicate-tool-result":
        return observations.get("rejected") is True and "duplicate tool result" in str(
            observations.get("error", "")
        )
    return (
        observations.get("read_ok") is True
        and observations.get("stale_sha_present") is True
        and observations.get("write_ok") is False
        and observations.get("error_code") == "content_conflict"
        and observations.get("workspace_unchanged") is True
    )


def run_robustness_evaluation(
    config: RobustnessEvalConfig | None = None,
) -> RobustnessMatrixReport:
    """Run the selected local fault-injection scenarios."""

    selected = config or RobustnessEvalConfig()
    results: list[RobustnessCaseResult] = []
    for scenario_id in selected.scenarios:
        try:
            observations = _scenario_observation(scenario_id)
            passed = _scenario_passes(scenario_id, observations)
            error = None if passed else "observed evidence did not satisfy the contract"
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            observations = {}
            passed = False
            error = f"{type(exc).__name__}: {str(exc)[:800]}"
        results.append(
            RobustnessCaseResult(
                scenario_id=scenario_id,
                title=_SCENARIO_TITLES[scenario_id],
                observations=observations,
                passed=passed,
                error=error,
            )
        )
    return RobustnessMatrixReport(config=selected, results=tuple(results))


def render_robustness_console(report: RobustnessMatrixReport) -> str:
    lines = [
        "scenario | result | key evidence",
        "--- | --- | ---",
    ]
    for result in report.results:
        evidence = json.dumps(
            dict(result.observations), ensure_ascii=False, sort_keys=True
        )
        lines.append(
            f"{result.scenario_id} | {'PASS' if result.passed else 'FAIL'} | {evidence}"
        )
    lines.append(
        f"passed={report.passed_count}/{len(report.results)} "
        f"failed={report.failed_count}"
    )
    return "\n".join(lines) + "\n"


def render_robustness_markdown(report: RobustnessMatrixReport) -> str:
    lines = [
        "# ContextOpt Level 4 robustness evaluation",
        "",
        report.claim_boundary,
        "",
        "| Scenario | Fault contract | Result | Observed evidence | Error |",
        "|---|---|---|---|---|",
    ]
    for result in report.results:
        evidence = json.dumps(
            dict(result.observations), ensure_ascii=False, sort_keys=True
        )
        lines.append(
            f"| `{result.scenario_id}` | {result.title} | "
            f"{'PASS' if result.passed else 'FAIL'} | `{evidence}` | "
            f"{result.error or ''} |"
        )
    lines.extend(
        (
            "",
            "## Summary",
            "",
            f"- Scenarios: `{len(report.results)}`; passed: `{report.passed_count}`; "
            f"failed: `{report.failed_count}`.",
            "- The matrix is provider-free and intentionally does not claim "
            "machine-loss "
            "recovery, exactly-once effects, security isolation, or coding quality.",
            "",
        )
    )
    return "\n".join(lines)


def render_robustness_html(report: RobustnessMatrixReport) -> str:
    payload = json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True)
    rows_list: list[str] = []
    for result in report.results:
        evidence = escape(
            json.dumps(dict(result.observations), ensure_ascii=False, sort_keys=True)
        )
        rows_list.append(
            "<tr>"
            f"<td><code>{escape(result.scenario_id)}</code></td>"
            f"<td>{escape(result.title)}</td>"
            f"<td class={'pass' if result.passed else 'fail'}>"
            f"{'PASS' if result.passed else 'FAIL'}</td>"
            f"<td><code>{evidence}</code></td>"
            f"<td>{escape(result.error or '')}</td>"
            "</tr>"
        )
    rows = "".join(rows_list)
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>ContextOpt robustness evaluation</title>"
        "<style>body{font:14px system-ui,sans-serif;margin:2rem;color:#172033;"
        "max-width:1400px}"
        "table{border-collapse:collapse;width:100%}th,td{border:1px solid #d7dce8;"
        "padding:.55rem;text-align:left;vertical-align:top}th{background:#f5f7fb}.pass{color:#087f5b;"
        "font-weight:700}.fail{color:#b42318;font-weight:700}code{white-space:pre-wrap}"
        ".boundary{background:#fff8e1;border-left:4px solid #d99a00;"
        "padding:1rem;margin:1rem 0}"
        "</style></head><body><h1>ContextOpt Level 4 robustness evaluation</h1>"
        f"<p class='boundary'>{escape(report.claim_boundary)}</p>"
        f"<p>Passed <strong>{report.passed_count}/{len(report.results)}</strong>; "
        f"failed <strong>{report.failed_count}</strong>.</p>"
        f"<table><thead><tr><th>Scenario</th><th>Contract</th><th>Result</th>"
        f"<th>Observed evidence</th><th>Error</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
        f"<script type='application/json' id='robustness-report'>"
        f"{escape(payload)}</script>"
        "</body></html>\n"
    )


__all__ = [
    "DEFAULT_SCENARIOS",
    "ROBUSTNESS_EVAL_SCHEMA_VERSION",
    "RobustnessCaseResult",
    "RobustnessEvalConfig",
    "RobustnessMatrixReport",
    "render_robustness_console",
    "render_robustness_html",
    "render_robustness_markdown",
    "run_robustness_evaluation",
]
