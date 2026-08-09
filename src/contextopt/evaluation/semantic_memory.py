"""Deterministic retrieval conformance for the durable semantic-memory store.

This evaluator deliberately measures the memory boundary, not model intelligence.  It
uses fixed facts, project scopes, tags, and an explicitly invalidated entry to check
retrieval evidence, scope isolation, negative queries, invalidation exclusion, and
byte-level determinism.  The lexical scorer is a transparent baseline; the report does
not claim semantic understanding or coding-task improvement.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory
from typing import Any, cast

from contextopt.runtime.identity import stable_hash
from contextopt.runtime.semantic_memory import (
    SemanticMemoryEntry,
    SemanticMemoryMatch,
    SemanticMemoryStore,
)

SEMANTIC_MEMORY_EVAL_SCHEMA_VERSION = "1"


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _strings(value: Sequence[str], label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be an array of strings")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{label} must contain non-empty strings")
    normalized = tuple(sorted(set(item.strip() for item in value)))
    if len(normalized) != len(value):
        raise ValueError(f"{label} must not contain duplicates")
    return normalized


@dataclass(frozen=True, slots=True)
class SemanticMemoryEvalConfig:
    """Fixed repetition and retrieval settings for the memory conformance run."""

    repetitions: int = 3
    limit: int = 3

    def __post_init__(self) -> None:
        if (
            not isinstance(self.repetitions, int)
            or isinstance(self.repetitions, bool)
            or self.repetitions < 2
        ):
            raise ValueError("repetitions must be an integer >= 2")
        if (
            not isinstance(self.limit, int)
            or isinstance(self.limit, bool)
            or not 1 <= self.limit <= 100
        ):
            raise ValueError("limit must be an integer between 1 and 100")

    def to_dict(self) -> dict[str, int]:
        return {"repetitions": self.repetitions, "limit": self.limit}


@dataclass(frozen=True, slots=True)
class SemanticMemoryEvalCase:
    """One query with positive or negative retrieval expectations."""

    case_id: str
    query: str
    scope: str
    expected_memory_ids: tuple[str, ...] = ()
    excluded_memory_ids: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    limit: int = 3

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _text(self.case_id, "case_id"))
        object.__setattr__(self, "query", _text(self.query, "query"))
        object.__setattr__(self, "scope", _text(self.scope, "scope"))
        object.__setattr__(
            self,
            "expected_memory_ids",
            _strings(self.expected_memory_ids, "expected_memory_ids"),
        )
        object.__setattr__(
            self,
            "excluded_memory_ids",
            _strings(self.excluded_memory_ids, "excluded_memory_ids"),
        )
        object.__setattr__(self, "tags", _strings(self.tags, "tags"))
        if set(self.expected_memory_ids) & set(self.excluded_memory_ids):
            raise ValueError("expected and excluded memory ids must be disjoint")
        if not isinstance(self.limit, int) or isinstance(self.limit, bool):
            raise ValueError("case limit must be an integer")
        if not 1 <= self.limit <= 100:
            raise ValueError("case limit must be between 1 and 100")

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "query": self.query,
            "scope": self.scope,
            "expected_memory_ids": list(self.expected_memory_ids),
            "excluded_memory_ids": list(self.excluded_memory_ids),
            "tags": list(self.tags),
            "limit": self.limit,
        }


@dataclass(frozen=True, slots=True)
class SemanticMemoryEvalFixture:
    """Memory entries and query cases used by the deterministic evaluator."""

    entries: tuple[SemanticMemoryEntry, ...]
    cases: tuple[SemanticMemoryEvalCase, ...]
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "entries": [entry.to_dict() for entry in self.entries],
            "cases": [case.to_dict() for case in self.cases],
        }


@dataclass(frozen=True, slots=True)
class SemanticMemoryEvalRun:
    """Aggregated repeated observations for one query case."""

    case_id: str
    query: str
    scope: str
    limit: int
    expected_memory_ids: tuple[str, ...]
    excluded_memory_ids: tuple[str, ...]
    result_memory_ids: tuple[str, ...]
    matched_terms: tuple[tuple[str, ...], ...]
    scores: tuple[float, ...]
    rank: int | None
    hit_at_1: bool | None
    hit_at_k: bool | None
    negative_pass: bool | None
    scope_leakage_count: int
    excluded_returned_count: int
    deterministic: bool
    repetition_digests: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "query": self.query,
            "scope": self.scope,
            "limit": self.limit,
            "expected_memory_ids": list(self.expected_memory_ids),
            "excluded_memory_ids": list(self.excluded_memory_ids),
            "result_memory_ids": list(self.result_memory_ids),
            "matched_terms": [list(terms) for terms in self.matched_terms],
            "scores": list(self.scores),
            "rank": self.rank,
            "hit_at_1": self.hit_at_1,
            "hit_at_k": self.hit_at_k,
            "negative_pass": self.negative_pass,
            "scope_leakage_count": self.scope_leakage_count,
            "excluded_returned_count": self.excluded_returned_count,
            "deterministic": self.deterministic,
            "repetition_digests": list(self.repetition_digests),
        }


def _fixture(store: SemanticMemoryStore, limit: int) -> SemanticMemoryEvalFixture:
    global_edit = store.put(
        "Before replace_text, read the file and use its latest sha256.",
        scope="global",
        kind="procedure",
        tags=("cas", "editing"),
        confidence=0.95,
        source_refs=("memory-eval",),
    ).entry
    parser_rule = store.put(
        "Parser rejects duplicate tool call ids before execution.",
        scope="project:parser",
        kind="decision",
        tags=("parser", "protocol"),
        confidence=0.92,
        source_refs=("memory-eval",),
    ).entry
    math_rule = store.put(
        "Extended gcd must satisfy the Bezout identity for negative inputs.",
        scope="project:math",
        kind="fact",
        tags=("number-theory", "acm"),
        confidence=0.9,
        source_refs=("memory-eval",),
    ).entry
    test_procedure = store.put(
        "Run visible tests with a bounded timeout before the final response.",
        scope="project:parser",
        kind="procedure",
        tags=("tests", "runtime"),
        confidence=0.8,
        source_refs=("memory-eval",),
    ).entry
    stale_rule = store.put(
        "Old parser behavior allowed duplicate tool call ids.",
        scope="project:parser",
        kind="failure",
        tags=("parser", "stale"),
        confidence=0.5,
        source_refs=("memory-eval",),
    ).entry
    store.invalidate(stale_rule.memory_id, "replaced by the stricter parser rule")
    entries = tuple(
        entry
        for entry in (
            store.get(global_edit.memory_id),
            store.get(parser_rule.memory_id),
            store.get(math_rule.memory_id),
            store.get(test_procedure.memory_id),
            store.get(stale_rule.memory_id),
        )
        if entry is not None
    )
    cases = (
        SemanticMemoryEvalCase(
            case_id="global-cas-procedure",
            query="latest sha256 replace_text",
            scope="project:parser",
            expected_memory_ids=(global_edit.memory_id,),
            tags=("cas",),
            limit=limit,
        ),
        SemanticMemoryEvalCase(
            case_id="parser-protocol-rule",
            query="duplicate tool call ids",
            scope="project:parser",
            expected_memory_ids=(parser_rule.memory_id,),
            tags=("protocol",),
            limit=limit,
        ),
        SemanticMemoryEvalCase(
            case_id="math-bezout-rule",
            query="negative Bezout",
            scope="project:math",
            expected_memory_ids=(math_rule.memory_id,),
            tags=("number-theory",),
            limit=limit,
        ),
        SemanticMemoryEvalCase(
            case_id="scope-isolation",
            query="visible tests bounded timeout",
            scope="project:math",
            limit=limit,
        ),
        SemanticMemoryEvalCase(
            case_id="invalidated-rule-excluded",
            query="old allowed",
            scope="project:parser",
            excluded_memory_ids=(stale_rule.memory_id,),
            limit=limit,
        ),
    )
    fingerprint = stable_hash(
        {
            "entries": [entry.to_dict() for entry in entries],
            "cases": [case.to_dict() for case in cases],
        }
    )
    return SemanticMemoryEvalFixture(entries, cases, fingerprint)


def build_semantic_memory_fixture(
    store: SemanticMemoryStore, *, limit: int = 3
) -> SemanticMemoryEvalFixture:
    """Populate *store* with fixed facts and return the query fixture."""

    if not isinstance(store, SemanticMemoryStore):
        raise ValueError("store must be a SemanticMemoryStore")
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
        raise ValueError("limit must be an integer between 1 and 100")
    return _fixture(store, limit)


def _search_digest(matches: Sequence[SemanticMemoryMatch]) -> str:
    return stable_hash([match.to_dict() for match in matches])


def _run_case(
    store: SemanticMemoryStore,
    case: SemanticMemoryEvalCase,
    repetitions: int,
) -> SemanticMemoryEvalRun:
    observations: list[tuple[SemanticMemoryMatch, ...]] = []
    digests: list[str] = []
    for _ in range(repetitions):
        matches = store.search(
            case.query,
            scope=case.scope,
            tags=case.tags,
            limit=case.limit,
        )
        observations.append(matches)
        digests.append(_search_digest(matches))
    first = observations[0]
    result_ids = tuple(match.entry.memory_id for match in first)
    expected = set(case.expected_memory_ids)
    rank = next(
        (
            index
            for index, memory_id in enumerate(result_ids, start=1)
            if memory_id in expected
        ),
        None,
    )
    allowed_scopes = {"global", case.scope}
    scope_leakage_count = sum(
        match.entry.scope not in allowed_scopes for match in first
    )
    excluded = set(case.excluded_memory_ids)
    excluded_returned_count = len(excluded.intersection(result_ids))
    return SemanticMemoryEvalRun(
        case_id=case.case_id,
        query=case.query,
        scope=case.scope,
        limit=case.limit,
        expected_memory_ids=case.expected_memory_ids,
        excluded_memory_ids=case.excluded_memory_ids,
        result_memory_ids=result_ids,
        matched_terms=tuple(match.matched_terms for match in first),
        scores=tuple(match.score for match in first),
        rank=rank,
        hit_at_1=(bool(result_ids) and result_ids[0] in expected) if expected else None,
        hit_at_k=(rank is not None) if expected else None,
        negative_pass=(not result_ids) if not expected else None,
        scope_leakage_count=scope_leakage_count,
        excluded_returned_count=excluded_returned_count,
        deterministic=len(set(digests)) == 1
        and all(
            tuple(match.entry.memory_id for match in matches) == result_ids
            for matches in observations
        ),
        repetition_digests=tuple(digests),
    )


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else round(fmean(values), 6)


def _summary(runs: Sequence[SemanticMemoryEvalRun]) -> dict[str, Any]:
    positive = [run for run in runs if run.expected_memory_ids]
    negative = [run for run in runs if not run.expected_memory_ids]
    excluded = [run for run in runs if run.excluded_memory_ids]
    return {
        "case_count": len(runs),
        "positive_case_count": len(positive),
        "negative_case_count": len(negative),
        "hit_at_1_rate": _mean(
            [float(run.hit_at_1) for run in positive if run.hit_at_1 is not None]
        ),
        "hit_at_k_rate": _mean(
            [float(run.hit_at_k) for run in positive if run.hit_at_k is not None]
        ),
        "mean_reciprocal_rank": _mean(
            [0.0 if run.rank is None else 1.0 / run.rank for run in positive]
        ),
        "negative_pass_rate": _mean(
            [
                float(run.negative_pass)
                for run in negative
                if run.negative_pass is not None
            ]
        ),
        "scope_isolation_rate": _mean(
            [float(run.scope_leakage_count == 0) for run in runs]
        ),
        "invalidated_exclusion_rate": _mean(
            [float(run.excluded_returned_count == 0) for run in excluded]
        ),
        "deterministic_rate": _mean([float(run.deterministic) for run in runs]),
        "total_scope_leakage": sum(run.scope_leakage_count for run in runs),
        "total_excluded_returned": sum(run.excluded_returned_count for run in runs),
    }


def run_semantic_memory_evaluation(
    config: SemanticMemoryEvalConfig | None = None,
) -> dict[str, Any]:
    """Run the fixed retrieval fixture in a temporary durable store."""

    selected = config or SemanticMemoryEvalConfig()
    with (
        TemporaryDirectory(prefix="contextopt-memory-eval-") as directory,
        SemanticMemoryStore(Path(directory) / "memory.jsonl") as store,
    ):
        fixture = build_semantic_memory_fixture(store, limit=selected.limit)
        runs = tuple(
            _run_case(store, case, selected.repetitions) for case in fixture.cases
        )
        report = {
            "schema_version": SEMANTIC_MEMORY_EVAL_SCHEMA_VERSION,
            "evaluation": "semantic-memory-retrieval-conformance",
            "model_calls": 0,
            "retrieval": "deterministic-lexical-v1",
            "embedding_used": False,
            "config": selected.to_dict(),
            "store_revision": store.revision,
            "store_fingerprint": store.fingerprint,
            "fixture": fixture.to_dict(),
            "summary": _summary(runs),
            "runs": [run.to_dict() for run in runs],
            "claim_boundary": (
                "This is a provider-free memory retrieval and persistence "
                "conformance fixture. It measures hit@k, reciprocal rank, "
                "scope isolation, invalidation exclusion, and deterministic "
                "replay; it does not measure embedding quality, model use of "
                "memory, or coding success."
            ),
        }
    return report


def _metric(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _run_mrr(run: Mapping[str, Any]) -> str:
    return _metric(None if run["rank"] is None else 1.0 / int(run["rank"]))


def render_semantic_memory_console(report: Mapping[str, Any]) -> str:
    """Render per-case retrieval metrics for a terminal."""

    runs = cast(Sequence[Mapping[str, Any]], report["runs"])
    headers = (
        "case",
        "hit@1",
        "hit@k",
        "mrr",
        "negative",
        "deterministic",
        "leak",
        "excluded",
    )
    rows = [
        (
            str(run["case_id"]),
            "n/a" if run["hit_at_1"] is None else str(run["hit_at_1"]),
            "n/a" if run["hit_at_k"] is None else str(run["hit_at_k"]),
            _run_mrr(run),
            "n/a" if run["negative_pass"] is None else str(run["negative_pass"]),
            str(run["deterministic"]),
            str(run["scope_leakage_count"]),
            str(run["excluded_returned_count"]),
        )
        for run in runs
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def line(values: Sequence[str]) -> str:
        return "  ".join(
            value.ljust(widths[index]) for index, value in enumerate(values)
        ).rstrip()

    summary = cast(Mapping[str, Any], report["summary"])
    return "\n".join(
        (
            line(headers),
            line(tuple("-" * width for width in widths)),
            *(line(row) for row in rows),
            "",
            "summary: "
            f"hit@1={_metric(summary['hit_at_1_rate'])} "
            f"hit@k={_metric(summary['hit_at_k_rate'])} "
            f"mrr={_metric(summary['mean_reciprocal_rank'])} "
            f"negative={_metric(summary['negative_pass_rate'])} "
            f"scope={_metric(summary['scope_isolation_rate'])} "
            f"invalidated={_metric(summary['invalidated_exclusion_rate'])} "
            f"deterministic={_metric(summary['deterministic_rate'])}",
        )
    )


def _md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_semantic_memory_markdown(report: Mapping[str, Any]) -> str:
    """Render a reviewable Markdown report with the claim boundary up front."""

    runs = cast(Sequence[Mapping[str, Any]], report["runs"])
    summary = cast(Mapping[str, Any], report["summary"])
    output = [
        "# Semantic-memory retrieval conformance",
        "",
        "> Model calls: 0. Lexical retrieval evidence is not model quality "
        "or coding success.",
        "",
        "| Case | Scope | Hit@1 | Hit@k | MRR | Negative pass | Deterministic | "
        "Scope leakage | Invalidated returned |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    output.extend(
        (
            "| {case} | {scope} | {hit1} | {hitk} | {mrr} | {negative} | "
            "{deterministic} | {leak} | {excluded} |"
        ).format(
            case=_md(run["case_id"]),
            scope=_md(run["scope"]),
            hit1="n/a" if run["hit_at_1"] is None else run["hit_at_1"],
            hitk="n/a" if run["hit_at_k"] is None else run["hit_at_k"],
            mrr=_run_mrr(run),
            negative=("n/a" if run["negative_pass"] is None else run["negative_pass"]),
            deterministic=run["deterministic"],
            leak=run["scope_leakage_count"],
            excluded=run["excluded_returned_count"],
        )
        for run in runs
    )
    output.extend(
        (
            "",
            "## Summary",
            "",
            f"- Positive hit@1 rate: `{_metric(summary['hit_at_1_rate'])}`",
            f"- Positive hit@k rate: `{_metric(summary['hit_at_k_rate'])}`",
            f"- Mean reciprocal rank: `{_metric(summary['mean_reciprocal_rank'])}`",
            f"- Negative-query pass rate: `{_metric(summary['negative_pass_rate'])}`",
            f"- Scope-isolation rate: `{_metric(summary['scope_isolation_rate'])}`",
            "- Invalidated-exclusion rate: "
            f"`{_metric(summary['invalidated_exclusion_rate'])}`",
            f"- Deterministic replay rate: `{_metric(summary['deterministic_rate'])}`",
            "",
            str(report["claim_boundary"]),
        )
    )
    return "\n".join(output) + "\n"


def render_semantic_memory_html(report: Mapping[str, Any]) -> str:
    """Render a dependency-free HTML table with the full JSON report embedded."""

    runs = cast(Sequence[Mapping[str, Any]], report["runs"])
    summary = cast(Mapping[str, Any], report["summary"])
    rows = "".join(
        "<tr>"
        f"<td>{escape(str(run['case_id']))}</td>"
        f"<td>{escape(str(run['scope']))}</td>"
        f"<td>{escape(str(run['hit_at_1']))}</td>"
        f"<td>{escape(str(run['hit_at_k']))}</td>"
        f"<td>{escape(_run_mrr(run))}"
        "</td>"
        f"<td>{escape(str(run['negative_pass']))}</td>"
        f"<td>{escape(str(run['deterministic']))}</td>"
        f"<td>{run['scope_leakage_count']}</td>"
        f"<td>{run['excluded_returned_count']}</td>"
        "</tr>"
        for run in runs
    )
    report_json = escape(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    )
    hit_at_1 = escape(_metric(summary["hit_at_1_rate"]))
    hit_at_k = escape(_metric(summary["hit_at_k_rate"]))
    mean_mrr = escape(_metric(summary["mean_reciprocal_rank"]))
    negative_pass = escape(_metric(summary["negative_pass_rate"]))
    scope_isolation = escape(_metric(summary["scope_isolation_rate"]))
    invalidated_excluded = escape(_metric(summary["invalidated_exclusion_rate"]))
    deterministic = escape(_metric(summary["deterministic_rate"]))
    html_lines = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        "<title>Semantic-memory retrieval conformance</title>",
        "<style>",
        "body{font:15px system-ui,sans-serif;max-width:1180px;",
        "margin:2rem auto;padding:0 1rem;color:#172033}",
        "table{border-collapse:collapse;width:100%;margin:1rem 0}",
        "th,td{border:1px solid #ccd3df;padding:.45rem;text-align:left}",
        "th{background:#e8eef7}",
        "code{background:#f1f4f8;padding:.12rem .3rem;border-radius:.2rem}",
        ".metrics{display:grid;grid-template-columns:repeat(auto-fit,",
        "minmax(160px,1fr));gap:.7rem}",
        ".metric{background:#f5f7fa;border:1px solid #d9e0ea;",
        "border-radius:.35rem;padding:.7rem}",
        ".metric b{display:block;font-size:1.3rem}",
        "pre{white-space:pre-wrap;background:#101827;color:#e6edf7;",
        "padding:1rem;overflow:auto}",
        "</style></head><body>",
        "<h1>Semantic-memory retrieval conformance</h1>",
        "<p>Model calls: <code>0</code>. Lexical retrieval evidence is not "
        "model quality or coding success.</p>",
        '<section class="metrics">',
        f'<div class="metric"><b>{hit_at_1}</b>hit@1</div>',
        f'<div class="metric"><b>{hit_at_k}</b>hit@k</div>',
        f'<div class="metric"><b>{mean_mrr}</b>mean MRR</div>',
        f'<div class="metric"><b>{negative_pass}</b>negative pass</div>',
        f'<div class="metric"><b>{scope_isolation}</b>scope isolation</div>',
        f'<div class="metric"><b>{invalidated_excluded}</b>invalidated excluded</div>',
        f'<div class="metric"><b>{deterministic}</b>deterministic</div>',
        "</section>",
        "<table><thead><tr><th>Case</th><th>Scope</th><th>Hit@1</th>",
        "<th>Hit@k</th><th>MRR</th><th>Negative pass</th>",
        "<th>Deterministic</th><th>Scope leakage</th>",
        "<th>Invalidated returned</th></tr></thead>",
        f"<tbody>{rows}</tbody></table>",
        f"<p>{escape(str(report['claim_boundary']))}</p>",
        f"<details><summary>Full JSON ledger</summary><pre>{report_json}</pre>",
        "</details>",
        "</body></html>",
    ]
    return "\n".join(html_lines) + "\n"


__all__ = [
    "SEMANTIC_MEMORY_EVAL_SCHEMA_VERSION",
    "SemanticMemoryEvalCase",
    "SemanticMemoryEvalConfig",
    "SemanticMemoryEvalFixture",
    "SemanticMemoryEvalRun",
    "build_semantic_memory_fixture",
    "render_semantic_memory_console",
    "render_semantic_memory_html",
    "render_semantic_memory_markdown",
    "run_semantic_memory_evaluation",
]
