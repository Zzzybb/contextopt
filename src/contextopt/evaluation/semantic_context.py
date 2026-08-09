"""Model-free conformance for automatic durable-memory context candidates.

This evaluator exercises the opt-in ``versioned-v1+semantic`` boundary without a
provider.  It measures which durable candidates were retrieved, which survived the
normal context budget, and whether the receipt can reproduce the exact request after
the live store is removed.  It deliberately does not claim that a model used the
candidate or that the candidate improved coding quality.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from html import escape
from statistics import fmean
from tempfile import TemporaryDirectory
from typing import Any, Literal

from contextopt.runtime.context import (
    ContextCompiler,
    ContextCompilerConfig,
    compile_runtime_context,
    durable_memory_matches_from_receipt,
)
from contextopt.runtime.identity import stable_hash
from contextopt.runtime.protocol import AgentMessage
from contextopt.runtime.semantic_memory import SemanticMemoryEntry, SemanticMemoryStore

SEMANTIC_CONTEXT_EVAL_SCHEMA_VERSION = "1"
SemanticContextPolicy = Literal["recent", "topk", "density", "submodular"]
_POLICIES = frozenset({"recent", "topk", "density", "submodular"})
_DEFAULT_POLICIES: tuple[SemanticContextPolicy, ...] = (
    "recent",
    "topk",
    "density",
    "submodular",
)
_CLAIM_BOUNDARY = (
    "This is a deterministic context-projection conformance evaluation. It measures "
    "lexical candidate retrieval, budgeted selection, receipt determinism, and replay "
    "without a provider; it does not measure model reasoning, memory usefulness, "
    "semantic understanding, or coding success."
)


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _recall(found: tuple[str, ...], expected: tuple[str, ...]) -> float:
    if not expected:
        return 1.0
    return round(len(set(found).intersection(expected)) / len(expected), 6)


@dataclass(frozen=True, slots=True)
class SemanticContextEvalConfig:
    """Fixed policy, budget, and repetition settings for the conformance run."""

    policies: tuple[SemanticContextPolicy, ...] = _DEFAULT_POLICIES
    budgets: tuple[int, ...] = (128, 256, 512)
    repetitions: int = 3
    recent_blocks: int = 0
    max_tool_output_tokens: int = 96

    def __post_init__(self) -> None:
        if not self.policies:
            raise ValueError("at least one semantic context policy is required")
        if any(policy not in _POLICIES for policy in self.policies):
            raise ValueError(
                "unknown semantic context policies: "
                f"{sorted(set(self.policies) - _POLICIES)!r}"
            )
        if len(self.policies) != len(set(self.policies)):
            raise ValueError("semantic context policies must be unique")
        if not self.budgets or any(
            not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0
            for budget in self.budgets
        ):
            raise ValueError("semantic context budgets must be positive")
        if len(self.budgets) != len(set(self.budgets)):
            raise ValueError("semantic context budgets must be unique")
        if (
            not isinstance(self.repetitions, int)
            or isinstance(self.repetitions, bool)
            or self.repetitions < 2
        ):
            raise ValueError("repetitions must be an integer >= 2")
        if (
            not isinstance(self.recent_blocks, int)
            or isinstance(self.recent_blocks, bool)
            or self.recent_blocks < 0
        ):
            raise ValueError("recent_blocks must be a non-negative integer")
        if self.max_tool_output_tokens < 48:
            raise ValueError("max_tool_output_tokens must be at least 48")

    def to_dict(self) -> dict[str, Any]:
        return {
            "policies": list(self.policies),
            "budgets": list(self.budgets),
            "repetitions": self.repetitions,
            "recent_blocks": self.recent_blocks,
            "max_tool_output_tokens": self.max_tool_output_tokens,
        }


@dataclass(frozen=True, slots=True)
class SemanticContextEvalCase:
    """One task and its evaluation-only expected durable-memory ids."""

    case_id: str
    task: str
    scope: str
    messages: tuple[AgentMessage, ...]
    expected_memory_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _text(self.case_id, "case_id"))
        object.__setattr__(self, "task", _text(self.task, "task"))
        object.__setattr__(self, "scope", _text(self.scope, "scope"))
        if not self.messages:
            raise ValueError("semantic context case messages must not be empty")
        if any(not isinstance(message, AgentMessage) for message in self.messages):
            raise ValueError(
                "semantic context case messages must be AgentMessage values"
            )
        if not self.expected_memory_ids:
            raise ValueError("semantic context case needs an expected memory id")
        if len(self.expected_memory_ids) != len(set(self.expected_memory_ids)):
            raise ValueError("expected memory ids must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "task": self.task,
            "scope": self.scope,
            "messages": [message.to_dict() for message in self.messages],
            "expected_memory_ids": list(self.expected_memory_ids),
        }


@dataclass(frozen=True, slots=True)
class SemanticContextEvalFixture:
    """Fixed memory entries and task cases used by the evaluator."""

    entries: tuple[SemanticMemoryEntry, ...]
    cases: tuple[SemanticContextEvalCase, ...]
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "entries": [entry.to_dict() for entry in self.entries],
            "cases": [case.to_dict() for case in self.cases],
        }


@dataclass(frozen=True, slots=True)
class SemanticContextEvalRun:
    """One policy/budget/repetition observation."""

    case_id: str
    policy: str
    budget_tokens: int
    repetition: int
    expected_memory_ids: tuple[str, ...]
    retrieved_memory_ids: tuple[str, ...]
    selected_memory_ids: tuple[str, ...]
    retrieved_recall: float | None
    selected_recall: float | None
    candidate_tokens: int | None
    selected_candidate_tokens: int | None
    selected_context_tokens: int | None
    evicted_candidates: int | None
    budget_compliant: bool
    deterministic: bool
    replayable: bool
    messages_sha256: str | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "policy": self.policy,
            "budget_tokens": self.budget_tokens,
            "repetition": self.repetition,
            "expected_memory_ids": list(self.expected_memory_ids),
            "retrieved_memory_ids": list(self.retrieved_memory_ids),
            "selected_memory_ids": list(self.selected_memory_ids),
            "retrieved_recall": self.retrieved_recall,
            "selected_recall": self.selected_recall,
            "candidate_tokens": self.candidate_tokens,
            "selected_candidate_tokens": self.selected_candidate_tokens,
            "selected_context_tokens": self.selected_context_tokens,
            "evicted_candidates": self.evicted_candidates,
            "budget_compliant": self.budget_compliant,
            "deterministic": self.deterministic,
            "replayable": self.replayable,
            "messages_sha256": self.messages_sha256,
            "error": self.error,
        }


def build_semantic_context_fixture(
    store: SemanticMemoryStore,
) -> SemanticContextEvalFixture:
    """Populate a tiny ACM/math memory notebook and return fixed cases."""

    entries = (
        store.put(
            "Two-sum complement invariant: save each prior value in a hash map "
            "before scanning the next value; duplicate values must remain valid.",
            scope="project:acm",
            kind="procedure",
            tags=("two-sum", "hash-map"),
            confidence=0.95,
            source_refs=("two_sum.py",),
            source_run_id="semantic-context-fixture",
        ).entry,
        store.put(
            "Extended Euclid invariant: a*x + b*y equals the current gcd, and "
            "the b == 0 base case returns Bezout coefficients.",
            scope="project:math",
            kind="fact",
            tags=("extended-gcd", "bezout"),
            confidence=0.95,
            source_refs=("extended_gcd.py",),
            source_run_id="semantic-context-fixture",
        ).entry,
        store.put(
            "Binary search keeps a half-open interval and recomputes the midpoint "
            "without changing the public API.",
            scope="global",
            kind="procedure",
            tags=("binary-search",),
            confidence=0.6,
            source_run_id="semantic-context-fixture",
        ).entry,
        store.put(
            "Visible tests are the authoritative gate before accepting a complete "
            "coding-agent snapshot.",
            scope="global",
            kind="decision",
            tags=("testing",),
            confidence=0.7,
            source_run_id="semantic-context-fixture",
        ).entry,
    )
    cases = (
        SemanticContextEvalCase(
            case_id="two-sum",
            task=(
                "Implement two-sum with the hash-map complement invariant and "
                "duplicate values."
            ),
            scope="project:acm",
            messages=(
                AgentMessage(role="system", content="You are a coding agent."),
                AgentMessage(
                    role="user",
                    content=(
                        "Implement two-sum with the hash-map complement invariant "
                        "and duplicate values."
                    ),
                ),
            ),
            expected_memory_ids=(entries[0].memory_id,),
        ),
        SemanticContextEvalCase(
            case_id="extended-gcd",
            task=(
                "Implement extended gcd and preserve the Bezout invariant at the "
                "b == 0 base case."
            ),
            scope="project:math",
            messages=(
                AgentMessage(role="system", content="You are a coding agent."),
                AgentMessage(
                    role="user",
                    content=(
                        "Implement extended gcd and preserve the Bezout invariant "
                        "at the b == 0 base case."
                    ),
                ),
            ),
            expected_memory_ids=(entries[1].memory_id,),
        ),
    )
    fingerprint = stable_hash(
        {
            "entries": [entry.to_dict() for entry in entries],
            "cases": [case.to_dict() for case in cases],
        }
    )
    return SemanticContextEvalFixture(entries, cases, fingerprint)


def _candidate_tokens(
    compiled: Any, source_count: int, *, selected_only: bool = False
) -> int:
    return sum(
        block.estimated_tokens
        for block in compiled.receipt.blocks
        if block.start_index >= source_count and (block.selected or not selected_only)
    )


def _run_cell(
    store: SemanticMemoryStore,
    case: SemanticContextEvalCase,
    policy: SemanticContextPolicy,
    budget: int,
    repetition: int,
    config: SemanticContextEvalConfig,
) -> SemanticContextEvalRun:
    try:
        compiler = ContextCompiler(
            ContextCompilerConfig(
                policy=policy,
                budget_tokens=budget,
                recent_blocks=config.recent_blocks,
                max_tool_output_tokens=config.max_tool_output_tokens,
                memory_policy="versioned-v1+semantic",
            )
        )
        live = compile_runtime_context(
            compiler,
            case.messages,
            task=case.task,
            memory_store=store,
            memory_scope=case.scope,
        )
        metadata = live.receipt.frame.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ValueError("semantic context receipt metadata is missing")
        retrieved = tuple(str(item) for item in metadata["durable_memory_ids"])
        selected = tuple(str(item) for item in metadata["durable_memory_selected_ids"])
        second = compile_runtime_context(
            compiler,
            case.messages,
            task=case.task,
            memory_store=store,
            memory_scope=case.scope,
        )
        replay = compile_runtime_context(
            compiler,
            case.messages,
            task=case.task,
            durable_memory_matches=durable_memory_matches_from_receipt(
                metadata["durable_memory_matches"]
            ),
            durable_memory_store_fingerprint=metadata[
                "durable_memory_store_fingerprint"
            ],
        )
        candidate_count = len(retrieved)
        return SemanticContextEvalRun(
            case_id=case.case_id,
            policy=policy,
            budget_tokens=budget,
            repetition=repetition,
            expected_memory_ids=case.expected_memory_ids,
            retrieved_memory_ids=retrieved,
            selected_memory_ids=selected,
            retrieved_recall=_recall(retrieved, case.expected_memory_ids),
            selected_recall=_recall(selected, case.expected_memory_ids),
            candidate_tokens=_candidate_tokens(live, len(case.messages)),
            selected_candidate_tokens=_candidate_tokens(
                live, len(case.messages), selected_only=True
            ),
            selected_context_tokens=live.receipt.estimated_selected_tokens,
            evicted_candidates=candidate_count - len(selected),
            budget_compliant=(
                live.receipt.estimated_selected_tokens <= live.receipt.budget_tokens
            ),
            deterministic=live == second,
            replayable=live == replay,
            messages_sha256=live.receipt.messages_sha256,
        )
    except (TypeError, ValueError, OSError) as exc:
        return SemanticContextEvalRun(
            case_id=case.case_id,
            policy=policy,
            budget_tokens=budget,
            repetition=repetition,
            expected_memory_ids=case.expected_memory_ids,
            retrieved_memory_ids=(),
            selected_memory_ids=(),
            retrieved_recall=None,
            selected_recall=None,
            candidate_tokens=None,
            selected_candidate_tokens=None,
            selected_context_tokens=None,
            evicted_candidates=None,
            budget_compliant=False,
            deterministic=False,
            replayable=False,
            messages_sha256=None,
            error=f"{type(exc).__name__}: {str(exc)[:800]}",
        )


def run_semantic_context_evaluation(
    config: SemanticContextEvalConfig | None = None,
) -> dict[str, Any]:
    """Run the deterministic automatic-memory context matrix."""

    selected = config or SemanticContextEvalConfig()
    with TemporaryDirectory(prefix="contextopt-semantic-context-") as directory:
        store = SemanticMemoryStore(f"{directory}/memory.jsonl")
        fixture = build_semantic_context_fixture(store)
        runs = tuple(
            _run_cell(store, case, policy, budget, repetition, selected)
            for case in fixture.cases
            for policy in selected.policies
            for budget in selected.budgets
            for repetition in range(selected.repetitions)
        )
        store_revision = store.revision
        store.close()

    successful = [run for run in runs if run.error is None]
    summary = {
        "model_calls": 0,
        "cell_count": len(runs),
        "failed_count": len(runs) - len(successful),
        "retrieved_recall_rate": round(
            fmean(run.retrieved_recall or 0.0 for run in successful), 6
        )
        if successful
        else 0.0,
        "selected_recall_rate": round(
            fmean(run.selected_recall or 0.0 for run in successful), 6
        )
        if successful
        else 0.0,
        "budget_compliant_rate": round(
            sum(run.budget_compliant for run in successful) / len(successful), 6
        )
        if successful
        else 0.0,
        "deterministic_rate": round(
            sum(run.deterministic for run in successful) / len(successful), 6
        )
        if successful
        else 0.0,
        "replayable_rate": round(
            sum(run.replayable for run in successful) / len(successful), 6
        )
        if successful
        else 0.0,
        "mean_retrieved_candidates": round(
            fmean(len(run.retrieved_memory_ids) for run in successful), 6
        )
        if successful
        else 0.0,
        "mean_selected_candidates": round(
            fmean(len(run.selected_memory_ids) for run in successful), 6
        )
        if successful
        else 0.0,
        "mean_evicted_candidates": round(
            fmean(run.evicted_candidates or 0 for run in successful), 6
        )
        if successful
        else 0.0,
    }
    return {
        "schema_version": SEMANTIC_CONTEXT_EVAL_SCHEMA_VERSION,
        "kind": "contextopt.semantic-context-eval.report",
        "config": selected.to_dict(),
        "fixture": fixture.to_dict(),
        "store_revision": store_revision,
        "model_calls": 0,
        "runs": [run.to_dict() for run in runs],
        "summary": summary,
        "claim_boundary": _CLAIM_BOUNDARY,
    }


def render_semantic_context_console(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "policy budget | retrieved | selected | selected candidate tokens | "
        "budget | deterministic | replay",
        "--- | ---: | ---: | ---: | ---: | ---: | ---:",
    ]
    for run in report["runs"]:
        lines.append(
            f"{run['policy']} {run['budget_tokens']} | "
            f"{len(run['retrieved_memory_ids'])} | {len(run['selected_memory_ids'])} | "
            f"{run['selected_candidate_tokens']!s} | "
            f"{run['budget_compliant']} | {run['deterministic']} | {run['replayable']}"
        )
    lines.append(
        f"summary: cells={summary['cell_count']} failed={summary['failed_count']} "
        f"selected_recall={summary['selected_recall_rate']:.3f} "
        f"replayable={summary['replayable_rate']:.3f}"
    )
    return "\n".join(lines) + "\n"


def render_semantic_context_markdown(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Semantic-context candidate evaluation",
        "",
        report["claim_boundary"],
        "",
        "| Policy | Budget | Retrieved | Selected | Candidate tokens | "
        "Selected candidate tokens | Context tokens | Evicted | "
        "Deterministic | Replayable |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in report["runs"]:
        lines.append(
            f"| `{run['policy']}` | {run['budget_tokens']} | "
            f"{len(run['retrieved_memory_ids'])} | {len(run['selected_memory_ids'])} | "
            f"{run['candidate_tokens']} | {run['selected_candidate_tokens']} | "
            f"{run['selected_context_tokens']} | "
            f"{run['evicted_candidates']} | {run['deterministic']} | "
            f"{run['replayable']} |"
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Cells: `{summary['cell_count']}`; failures: "
            f"`{summary['failed_count']}`.",
            f"- Retrieved recall: `{summary['retrieved_recall_rate']:.3f}`; "
            f"selected recall: `{summary['selected_recall_rate']:.3f}`.",
            f"- Budget compliant: `{summary['budget_compliant_rate']:.3f}`; "
            f"deterministic: `{summary['deterministic_rate']:.3f}`; "
            f"replayable: `{summary['replayable_rate']:.3f}`.",
            f"- Mean retrieved/selected/evicted candidates: "
            f"`{summary['mean_retrieved_candidates']:.3f}` / "
            f"`{summary['mean_selected_candidates']:.3f}` / "
            f"`{summary['mean_evicted_candidates']:.3f}`.",
            "",
            "The recall fields are fixture-label retention metrics, not semantic "
            "understanding or model-use metrics.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_semantic_context_html(report: Mapping[str, Any]) -> str:
    payload = json.dumps(report, ensure_ascii=False, sort_keys=True).replace(
        "</", "<\\/"
    )
    rows = "".join(
        "<tr>"
        f"<td>{escape(str(run['policy']))}</td>"
        f"<td>{run['budget_tokens']}</td>"
        f"<td>{len(run['retrieved_memory_ids'])}</td>"
        f"<td>{len(run['selected_memory_ids'])}</td>"
        f"<td>{run['candidate_tokens']}</td>"
        f"<td>{run['selected_candidate_tokens']}</td>"
        f"<td>{run['selected_context_tokens']}</td>"
        f"<td>{run['retrieved_recall']}</td>"
        f"<td>{run['selected_recall']}</td>"
        f"<td>{run['evicted_candidates']}</td>"
        f"<td>{run['deterministic']}</td>"
        f"<td>{run['replayable']}</td>"
        "</tr>"
        for run in report["runs"]
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Semantic-context evaluation</title>"
        "<style>body{font:14px system-ui;margin:2rem}table{border-collapse:collapse}"
        "th,td{border:1px solid #ddd;padding:.35rem .5rem}"
        ".boundary{background:#fff8e1;padding:1rem;border-left:4px solid #d99a00}"
        "</style></head><body>"
        "<h1>Semantic-context candidate evaluation</h1>"
        f"<div class='boundary'>{escape(str(report['claim_boundary']))}</div>"
        "<table><thead><tr><th>Policy</th><th>Budget</th><th>Retrieved</th>"
        "<th>Selected</th><th>Candidate tokens</th>"
        "<th>Selected candidate tokens</th><th>Context tokens</th>"
        "<th>Retrieved recall</th><th>Selected recall</th>"
        "<th>Evicted</th><th>Deterministic</th><th>Replayable</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
        f"<details><summary>Full JSON ledger</summary><pre>"
        f"{escape(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))}"
        "</pre></details>"
        "<script type='application/json' id='semantic-context-report'>"
        f"{payload}</script>"
        "</body></html>"
    )


__all__ = [
    "SEMANTIC_CONTEXT_EVAL_SCHEMA_VERSION",
    "SemanticContextEvalCase",
    "SemanticContextEvalConfig",
    "SemanticContextEvalFixture",
    "SemanticContextEvalRun",
    "build_semantic_context_fixture",
    "render_semantic_context_console",
    "render_semantic_context_html",
    "render_semantic_context_markdown",
    "run_semantic_context_evaluation",
]
