"""Deterministic conformance evaluation for context routing and compilation.

The evaluator never calls a model.  It compares compiler policies on paired,
fixed traces and reports only observable context-retention and protocol metrics.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Any, cast

from contextopt.runtime.context import (
    CONTEXT_COMPILER_VERSION,
    MIN_TOOL_OUTPUT_TOKENS,
    ContextBudgetError,
    ContextCompiler,
    ContextCompilerConfig,
    ContextPolicyName,
    estimate_messages_tokens,
)
from contextopt.runtime.protocol import AgentMessage, ToolCall

_BOUNDED_POLICIES = frozenset({"recent", "topk", "density", "submodular"})
_KNOWN_POLICIES = _BOUNDED_POLICIES | {"full"}
_DEFAULT_POLICIES: tuple[ContextPolicyName, ...] = (
    "recent",
    "topk",
    "density",
    "submodular",
)


@dataclass(frozen=True, slots=True)
class EvidenceProbe:
    """One exact fact whose survival can be checked without a model judge."""

    probe_id: str
    exact_text: str

    def __post_init__(self) -> None:
        if not self.probe_id.strip():
            raise ValueError("evidence probe_id must not be empty")
        if not self.exact_text.strip():
            raise ValueError("evidence exact_text must not be empty")


@dataclass(frozen=True, slots=True)
class RoutingTraceCase:
    """A fixed coding trace and evaluation-only evidence labels."""

    case_id: str
    query: str
    messages: tuple[AgentMessage, ...]
    evidence: tuple[EvidenceProbe, ...]

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("routing case_id must not be empty")
        if not self.query.strip():
            raise ValueError("routing query must not be empty")
        if not self.messages:
            raise ValueError("routing case messages must not be empty")
        if not self.evidence:
            raise ValueError("routing case evidence must not be empty")
        probe_ids = [probe.probe_id for probe in self.evidence]
        if len(probe_ids) != len(set(probe_ids)):
            raise ValueError("routing case evidence probe ids must be unique")
        probe_texts = [probe.exact_text for probe in self.evidence]
        if len(probe_texts) != len(set(probe_texts)):
            raise ValueError("routing case evidence probe texts must be unique")
        missing = [
            probe.probe_id
            for probe in self.evidence
            if not any(probe.exact_text in message.content for message in self.messages)
        ]
        if missing:
            raise ValueError(f"evidence probes are absent from source: {missing!r}")
        issues = tool_protocol_issues(self.messages)
        if issues:
            raise ValueError(
                "routing case must be protocol-valid: " + "; ".join(issues)
            )


@dataclass(frozen=True, slots=True)
class ContextRoutingEvalConfig:
    """Paired compiler settings used by every trace and policy."""

    policies: tuple[ContextPolicyName, ...] = _DEFAULT_POLICIES
    budgets: tuple[int, ...] = (512, 1_024)
    repetitions: int = 3
    recent_blocks: int = 2
    max_tool_output_tokens: int = 96

    def __post_init__(self) -> None:
        if not self.policies:
            raise ValueError("at least one context policy is required")
        if any(not isinstance(policy, str) for policy in self.policies):
            raise ValueError("context policies must be strings")
        if len(self.policies) != len(set(self.policies)):
            raise ValueError("context policies must be unique")
        unknown = set(self.policies) - _KNOWN_POLICIES
        if unknown:
            raise ValueError(f"unknown context policies: {sorted(unknown)!r}")
        if not self.budgets or any(
            not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0
            for budget in self.budgets
        ):
            raise ValueError("context budgets must be positive")
        if len(self.budgets) != len(set(self.budgets)):
            raise ValueError("context budgets must be unique")
        if (
            not isinstance(self.repetitions, int)
            or isinstance(self.repetitions, bool)
            or self.repetitions < 2
        ):
            raise ValueError("determinism requires at least two repetitions")
        if (
            not isinstance(self.recent_blocks, int)
            or isinstance(self.recent_blocks, bool)
            or self.recent_blocks < 0
        ):
            raise ValueError("recent_blocks must be non-negative")
        if (
            not isinstance(self.max_tool_output_tokens, int)
            or isinstance(self.max_tool_output_tokens, bool)
            or self.max_tool_output_tokens < MIN_TOOL_OUTPUT_TOKENS
        ):
            raise ValueError(
                f"max_tool_output_tokens must be at least {MIN_TOOL_OUTPUT_TOKENS}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policies": list(self.policies),
            "budgets": list(self.budgets),
            "repetitions": self.repetitions,
            "recent_blocks": self.recent_blocks,
            "max_tool_output_tokens": self.max_tool_output_tokens,
        }


@dataclass(frozen=True, slots=True)
class RoutingRunMetrics:
    """Metrics for one case/policy/budget cell in the paired evaluation."""

    case_id: str
    policy: str
    budget_tokens: int
    supported: bool
    original_estimated_tokens: int
    compiled_estimated_tokens: int | None
    evidence_total: int
    evidence_retained: int | None
    evidence_recall: float | None
    protocol_valid: bool
    protocol_issues: tuple[str, ...]
    budget_compliant: bool
    budget_overage_tokens: int | None
    compression_ratio: float | None
    deterministic: bool
    output_sha256: str | None
    receipt: Mapping[str, Any] | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "policy": self.policy,
            "budget_tokens": self.budget_tokens,
            "supported": self.supported,
            "original_estimated_tokens": self.original_estimated_tokens,
            "compiled_estimated_tokens": self.compiled_estimated_tokens,
            "evidence_total": self.evidence_total,
            "evidence_retained": self.evidence_retained,
            "evidence_recall": self.evidence_recall,
            "protocol_valid": self.protocol_valid,
            "protocol_issues": list(self.protocol_issues),
            "budget_compliant": self.budget_compliant,
            "budget_overage_tokens": self.budget_overage_tokens,
            "compression_ratio": self.compression_ratio,
            "deterministic": self.deterministic,
            "output_sha256": self.output_sha256,
            "receipt": None if self.receipt is None else dict(self.receipt),
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class _EvidenceStep:
    position: int
    probe_id: str
    exact_text: str
    tool_name: str
    target: str


def tool_protocol_issues(messages: Sequence[AgentMessage]) -> tuple[str, ...]:
    """Return deterministic provider-protocol issues for assistant/tool exchanges."""

    if not messages:
        return ("compiled message sequence is empty",)

    issues: list[str] = []
    pending: list[tuple[str, str]] = []
    pending_start = -1
    for index, message in enumerate(messages):
        if pending:
            if message.role != "tool":
                unresolved = [call_id for call_id, _ in pending]
                issues.append(
                    f"assistant tool calls at index {pending_start} are incomplete: "
                    f"{unresolved!r}"
                )
                pending.clear()
            else:
                expected_id, expected_name = pending.pop(0)
                if message.tool_call_id != expected_id:
                    issues.append(
                        f"tool result at index {index} has id "
                        f"{message.tool_call_id!r}; expected {expected_id!r}"
                    )
                if message.tool_name != expected_name:
                    issues.append(
                        f"tool result at index {index} has name "
                        f"{message.tool_name!r}; expected {expected_name!r}"
                    )
                continue

        if message.role == "tool":
            issues.append(f"orphan tool result at index {index}")
            continue
        if message.tool_calls and message.role != "assistant":
            issues.append(f"non-assistant message at index {index} has tool calls")
            continue
        if message.role == "assistant" and message.tool_calls:
            call_ids = [call.id for call in message.tool_calls]
            if len(call_ids) != len(set(call_ids)):
                issues.append(f"assistant message at index {index} repeats a tool id")
            pending = [(call.id, call.name) for call in message.tool_calls]
            pending_start = index

    if pending:
        unresolved = [call_id for call_id, _ in pending]
        issues.append(
            f"assistant tool calls at index {pending_start} are incomplete: "
            f"{unresolved!r}"
        )
    return tuple(issues)


def _long_tool_output(prefix: str, step: int) -> str:
    filler = " ".join(
        f"archive_{step:02d}_{offset:02d}=stable_unrelated_telemetry"
        for offset in range(36)
    )
    return f"{prefix}\n{filler}\nend_of_observation={step:02d}"


def _tool_arguments(tool_name: str, target: str, step: int) -> dict[str, object]:
    if tool_name == "search_text":
        return {"path": target, "query": f"trace_probe_{step:02d}"}
    if tool_name == "run_tests":
        return {"scope": target}
    return {"path": target}


def _build_case(
    case_id: str,
    query: str,
    evidence_steps: Sequence[_EvidenceStep],
) -> RoutingTraceCase:
    evidence_by_position = {step.position: step for step in evidence_steps}
    if len(evidence_by_position) != len(evidence_steps):
        raise ValueError("evidence positions must be unique")

    messages: list[AgentMessage] = [
        AgentMessage(
            role="system",
            content=(
                "You are a coding agent. Preserve tool-call protocol, use exact "
                "workspace evidence, and make the smallest verified change."
            ),
        ),
        AgentMessage(role="user", content=query),
    ]
    generic_targets = (
        "docs/overview.md",
        "src/telemetry.py",
        "tests/test_formatting.py",
        "pyproject.toml",
        "src/logging_helpers.py",
        "tests/test_docs.py",
    )
    generic_tools = ("read_file", "search_text", "run_tests", "list_files")
    for position in range(18):
        evidence = evidence_by_position.get(position)
        primary_name = (
            evidence.tool_name
            if evidence is not None
            else generic_tools[position % len(generic_tools)]
        )
        primary_target = (
            evidence.target
            if evidence is not None
            else generic_targets[position % len(generic_targets)]
        )
        primary_id = f"{case_id}-call-{position:02d}-a"
        calls = [
            ToolCall(
                id=primary_id,
                name=primary_name,
                arguments_json=json.dumps(
                    _tool_arguments(primary_name, primary_target, position),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        ]
        if position in {5, 11}:
            calls.append(
                ToolCall(
                    id=f"{case_id}-call-{position:02d}-b",
                    name="list_files",
                    arguments_json='{"path":"examples"}',
                )
            )
        messages.append(
            AgentMessage(
                role="assistant",
                content=(
                    f"Inspecting deterministic trace step {position:02d}; "
                    "gathering bounded workspace observations."
                ),
                tool_calls=tuple(calls),
            )
        )
        prefix = (
            evidence.exact_text
            if evidence is not None
            else (
                f"Routine observation {position:02d}: no relevant defect signal "
                "was found in this unrelated artifact."
            )
        )
        messages.append(
            AgentMessage(
                role="tool",
                content=_long_tool_output(prefix, position),
                tool_call_id=primary_id,
                tool_name=primary_name,
            )
        )
        if len(calls) == 2:
            secondary = calls[1]
            messages.append(
                AgentMessage(
                    role="tool",
                    content=_long_tool_output(
                        "Secondary directory listing contained only sample files.",
                        position,
                    ),
                    tool_call_id=secondary.id,
                    tool_name=secondary.name,
                )
            )

    probes = tuple(
        EvidenceProbe(step.probe_id, step.exact_text) for step in evidence_steps
    )
    return RoutingTraceCase(case_id, query, tuple(messages), probes)


def build_long_coding_traces() -> tuple[RoutingTraceCase, ...]:
    """Return fixed, protocol-valid traces with evaluation-only evidence probes."""

    return (
        _build_case(
            "retry-timeout",
            (
                "Fix the retry budget in src/retry.py, preserve TimeoutError "
                "semantics, and add the failing regression test."
            ),
            (
                _EvidenceStep(
                    2,
                    "retry-limit",
                    "FACT retry MAX_ATTEMPTS is 3 at src/retry.py:18.",
                    "read_file",
                    "src/retry.py",
                ),
                _EvidenceStep(
                    7,
                    "timeout-contract",
                    "FACT exhausted retries must re-raise TimeoutError unchanged.",
                    "search_text",
                    "src/retry.py",
                ),
                _EvidenceStep(
                    12,
                    "retry-regression",
                    (
                        "FACT failing test is "
                        "tests/test_retry.py::test_timeout_after_three_attempts."
                    ),
                    "run_tests",
                    "tests/test_retry.py",
                ),
            ),
        ),
        _build_case(
            "tenant-cache",
            (
                "Repair tenant cache invalidation in src/cache.py without crossing "
                "tenant boundaries, then cover the stale version-key regression."
            ),
            (
                _EvidenceStep(
                    1,
                    "cache-key",
                    "FACT cache keys are tuples of tenant_id, object_id, and version.",
                    "read_file",
                    "src/cache.py",
                ),
                _EvidenceStep(
                    8,
                    "invalidation-order",
                    "FACT invalidation increments version before deleting the old key.",
                    "search_text",
                    "src/cache.py",
                ),
                _EvidenceStep(
                    13,
                    "tenant-regression",
                    (
                        "FACT failing test is "
                        "tests/test_cache.py::test_invalidation_is_tenant_scoped."
                    ),
                    "run_tests",
                    "tests/test_cache.py",
                ),
            ),
        ),
        _build_case(
            "lease-alias",
            (
                "Fix EventLog lease alias locking for hard links while preserving "
                "resume token and tool-call budgets."
            ),
            (
                _EvidenceStep(
                    3,
                    "lease-identity",
                    "FACT the lease must bind the physical event file identity.",
                    "read_file",
                    "src/contextopt/runtime/events.py",
                ),
                _EvidenceStep(
                    9,
                    "alias-reproduction",
                    "FACT two hard-link paths currently acquire independent leases.",
                    "run_tests",
                    "tests/test_runtime_events.py",
                ),
                _EvidenceStep(
                    14,
                    "budget-contract",
                    "FACT resumed usage and tool_calls are cumulative, never reset.",
                    "search_text",
                    "src/contextopt/runtime/recovery.py",
                ),
            ),
        ),
    )


def _compiled_payload(
    messages: Sequence[AgentMessage], receipt: Mapping[str, Any]
) -> str:
    return json.dumps(
        {
            "messages": [message.to_dict() for message in messages],
            "receipt": dict(receipt),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _evaluate_cell(
    case: RoutingTraceCase,
    policy: ContextPolicyName,
    budget: int,
    config: ContextRoutingEvalConfig,
) -> RoutingRunMetrics:
    original_tokens = estimate_messages_tokens(case.messages)
    compiler_config = ContextCompilerConfig(
        policy=policy,
        budget_tokens=budget,
        recent_blocks=config.recent_blocks,
        max_tool_output_tokens=config.max_tool_output_tokens,
    )

    payloads: list[str] = []
    first_messages: tuple[AgentMessage, ...] | None = None
    first_receipt: dict[str, Any] | None = None
    try:
        for _ in range(config.repetitions):
            compiled = ContextCompiler(compiler_config).compile(
                case.messages,
                query=case.query,
            )
            receipt = compiled.receipt.to_dict()
            payloads.append(_compiled_payload(compiled.messages, receipt))
            if first_messages is None:
                first_messages = compiled.messages
                first_receipt = receipt
    except ContextBudgetError as exc:
        return RoutingRunMetrics(
            case_id=case.case_id,
            policy=policy,
            budget_tokens=budget,
            supported=False,
            original_estimated_tokens=original_tokens,
            compiled_estimated_tokens=None,
            evidence_total=len(case.evidence),
            evidence_retained=None,
            evidence_recall=None,
            protocol_valid=False,
            protocol_issues=(),
            budget_compliant=False,
            budget_overage_tokens=None,
            compression_ratio=None,
            deterministic=False,
            output_sha256=None,
            receipt=None,
            error=f"{type(exc).__name__}: {exc}",
        )

    assert first_messages is not None
    assert first_receipt is not None
    compiled_tokens = estimate_messages_tokens(first_messages)
    retained = sum(
        any(probe.exact_text in message.content for message in first_messages)
        for probe in case.evidence
    )
    issues = tool_protocol_issues(first_messages)
    canonical = payloads[0]
    return RoutingRunMetrics(
        case_id=case.case_id,
        policy=policy,
        budget_tokens=budget,
        supported=True,
        original_estimated_tokens=original_tokens,
        compiled_estimated_tokens=compiled_tokens,
        evidence_total=len(case.evidence),
        evidence_retained=retained,
        evidence_recall=retained / len(case.evidence),
        protocol_valid=not issues,
        protocol_issues=issues,
        budget_compliant=compiled_tokens <= budget,
        budget_overage_tokens=max(0, compiled_tokens - budget),
        compression_ratio=1.0 - compiled_tokens / original_tokens,
        deterministic=all(payload == canonical for payload in payloads[1:]),
        output_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        receipt=first_receipt,
    )


def _summary(
    runs: Sequence[RoutingRunMetrics],
    policies: Sequence[ContextPolicyName],
    budgets: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for budget in budgets:
        for policy in policies:
            cells = [
                run
                for run in runs
                if run.policy == policy and run.budget_tokens == budget
            ]
            supported = [run for run in cells if run.supported]
            recalls = [
                run.evidence_recall
                for run in supported
                if run.evidence_recall is not None
            ]
            compression = [
                run.compression_ratio
                for run in supported
                if run.compression_ratio is not None
            ]
            rows.append(
                {
                    "policy": policy,
                    "budget_tokens": budget,
                    "cases": len(cells),
                    "supported_cases": len(supported),
                    "error_cases": len(cells) - len(supported),
                    "mean_evidence_recall": fmean(recalls) if recalls else None,
                    "protocol_valid_rate": (
                        sum(run.protocol_valid for run in cells) / len(cells)
                    ),
                    "budget_compliance_rate": (
                        sum(run.budget_compliant for run in cells) / len(cells)
                    ),
                    "mean_compression_ratio": (
                        fmean(compression) if compression else None
                    ),
                    "determinism_rate": (
                        sum(run.deterministic for run in cells) / len(cells)
                    ),
                }
            )
    return rows


def run_context_routing_evaluation(
    config: ContextRoutingEvalConfig | None = None,
    cases: Iterable[RoutingTraceCase] | None = None,
) -> dict[str, Any]:
    """Run the paired, model-free context compiler conformance evaluation."""

    selected_config = config or ContextRoutingEvalConfig()
    selected_cases = tuple(build_long_coding_traces() if cases is None else cases)
    if not selected_cases:
        raise ValueError("at least one routing trace case is required")

    runs = tuple(
        _evaluate_cell(case, policy, budget, selected_config)
        for budget in selected_config.budgets
        for case in selected_cases
        for policy in selected_config.policies
    )
    return {
        "schema_version": "1",
        "evaluation": "context-routing-compiler-conformance",
        "context_compiler_version": CONTEXT_COMPILER_VERSION,
        "model_calls": 0,
        "token_measure": "contextopt deterministic estimate",
        "config": selected_config.to_dict(),
        "cases": [
            {
                "case_id": case.case_id,
                "message_count": len(case.messages),
                "original_estimated_tokens": estimate_messages_tokens(case.messages),
                "evidence_probe_ids": [probe.probe_id for probe in case.evidence],
            }
            for case in selected_cases
        ],
        "summary": _summary(runs, selected_config.policies, selected_config.budgets),
        "runs": [run.to_dict() for run in runs],
        "interpretation": (
            "Deterministic compiler conformance only; this report does not measure "
            "model quality, coding correctness, or task success."
        ),
    }


def _metric(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def render_context_routing_console(report: Mapping[str, Any]) -> str:
    """Render summary rows without hiding unsupported policy/budget cells."""

    headers = ("budget", "policy", "recall", "protocol", "compliance", "compress")
    rows = [
        (
            str(row["budget_tokens"]),
            str(row["policy"]),
            _metric(row["mean_evidence_recall"]),
            _metric(row["protocol_valid_rate"]),
            _metric(row["budget_compliance_rate"]),
            _metric(row["mean_compression_ratio"]),
        )
        for row in cast(Sequence[Mapping[str, Any]], report["summary"])
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def line(values: Sequence[str]) -> str:
        return "  ".join(
            value.ljust(widths[index]) for index, value in enumerate(values)
        ).rstrip()

    separator = line(tuple("-" * width for width in widths))
    return "\n".join((line(headers), separator, *(line(row) for row in rows)))


def render_context_routing_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact report whose heading states the model-free boundary."""

    rows = cast(Sequence[Mapping[str, Any]], report["summary"])
    output = [
        "# Context routing/compiler conformance",
        "",
        "> Model calls: 0. Evidence retention is not coding success or model quality.",
        "",
        (
            "| Budget | Policy | Evidence recall | Protocol valid | "
            "Budget compliant | Compression |"
        ),
        "|---:|---|---:|---:|---:|---:|",
    ]
    output.extend(
        (
            "| {budget} | {policy} | {recall} | {protocol} | "
            "{compliance} | {compression} |"
        ).format(
            budget=row["budget_tokens"],
            policy=row["policy"],
            recall=_metric(row["mean_evidence_recall"]),
            protocol=_metric(row["protocol_valid_rate"]),
            compliance=_metric(row["budget_compliance_rate"]),
            compression=_metric(row["mean_compression_ratio"]),
        )
        for row in rows
    )
    output.extend(
        (
            "",
            (
                "Token counts use ContextOpt's deterministic estimator, "
                "not a provider tokenizer."
            ),
            "",
        )
    )
    return "\n".join(output)


__all__ = [
    "ContextRoutingEvalConfig",
    "EvidenceProbe",
    "RoutingRunMetrics",
    "RoutingTraceCase",
    "build_long_coding_traces",
    "render_context_routing_console",
    "render_context_routing_markdown",
    "run_context_routing_evaluation",
    "tool_protocol_issues",
]
