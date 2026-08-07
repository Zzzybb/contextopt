"""Reproducible, paired evaluation of interchangeable context policies."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from statistics import fmean, median
from time import perf_counter
from typing import Any

from contextopt.models import ContextFrame, ObjectiveWeights, SelectionProblem
from contextopt.policies import UnsupportedProblemError, create_policy
from contextopt.synthetic import SyntheticCase, generate_case


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    instances: int = 50
    item_count: int = 14
    critical_count: int = 4
    budget: int = 1_200
    seed: int = 42
    graph_rate: float = 0.0
    conflict_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.instances <= 0:
            raise ValueError("instances must be positive")


@dataclass(frozen=True, slots=True)
class RunMetrics:
    case_id: str
    policy: str
    supported: bool
    objective_score: float | None
    approximation_ratio: float | None
    critical_recall: float | None
    redundancy_ratio: float | None
    stale_item_ratio: float | None
    budget_utilization: float | None
    latency_ms: float
    selected_count: int | None
    error: str | None = None


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * percentile)))
    return ordered[index]


def _critical_recall(frame: ContextFrame, case: SyntheticCase) -> float:
    if not case.critical_ids:
        return 1.0
    return len(set(frame.selected_ids) & case.critical_ids) / len(case.critical_ids)


def _redundancy_ratio(frame: ContextFrame, problem: SelectionProblem) -> float:
    selected = [problem.by_id[item_id] for item_id in frame.selected_ids]
    groups: dict[str, list[Any]] = {}
    for item in selected:
        if item.duplicate_group:
            groups.setdefault(item.duplicate_group, []).append(item)
    redundant_tokens = 0
    for members in groups.values():
        if len(members) <= 1:
            continue
        keep = max(members, key=problem.standalone_utility)
        redundant_tokens += sum(item.tokens for item in members if item.id != keep.id)
    return redundant_tokens / frame.used_tokens if frame.used_tokens else 0.0


def _stale_ratio(frame: ContextFrame, problem: SelectionProblem) -> float:
    if not frame.selected_ids:
        return 0.0
    stale = sum(
        problem.by_id[item_id].freshness < 0.25 for item_id in frame.selected_ids
    )
    return stale / len(frame.selected_ids)


def _run_policy(
    policy_name: str,
    case: SyntheticCase,
    oracle_frame: ContextFrame,
    cached_oracle_latency_ms: float,
) -> RunMetrics:
    if policy_name == "oracle":
        frame = oracle_frame
        latency_ms = cached_oracle_latency_ms
    else:
        policy = create_policy(policy_name)
        started = perf_counter()
        try:
            frame = policy.select(case.problem)
        except UnsupportedProblemError as exc:
            latency_ms = (perf_counter() - started) * 1_000
            return RunMetrics(
                case_id=case.case_id,
                policy=policy_name,
                supported=False,
                objective_score=None,
                approximation_ratio=None,
                critical_recall=None,
                redundancy_ratio=None,
                stale_item_ratio=None,
                budget_utilization=None,
                latency_ms=latency_ms,
                selected_count=None,
                error=str(exc),
            )
        latency_ms = (perf_counter() - started) * 1_000

    issues = case.problem.feasibility_issues(frame.selected_ids)
    if issues:
        raise AssertionError(f"{policy_name} returned infeasible context: {issues}")
    ratio = (
        frame.objective_score / oracle_frame.objective_score
        if oracle_frame.objective_score > 0
        else 1.0
    )
    return RunMetrics(
        case_id=case.case_id,
        policy=policy_name,
        supported=True,
        objective_score=frame.objective_score,
        approximation_ratio=ratio,
        critical_recall=_critical_recall(frame, case),
        redundancy_ratio=_redundancy_ratio(frame, case.problem),
        stale_item_ratio=_stale_ratio(frame, case.problem),
        budget_utilization=frame.utilization,
        latency_ms=latency_ms,
        selected_count=len(frame.selected_ids),
    )


def run_benchmark(
    config: BenchmarkConfig,
    policy_names: Iterable[str] = (
        "topk",
        "density",
        "knapsack",
        "submodular",
        "oracle",
    ),
) -> dict[str, Any]:
    names = tuple(dict.fromkeys(policy_names))
    unknown = set(names) - {"topk", "density", "knapsack", "submodular", "oracle"}
    if unknown:
        raise ValueError(f"unknown policies: {sorted(unknown)!r}")

    runs: list[RunMetrics] = []
    for offset in range(config.instances):
        case = generate_case(
            seed=config.seed + offset,
            item_count=config.item_count,
            critical_count=config.critical_count,
            budget=config.budget,
            graph_rate=config.graph_rate,
            conflict_rate=config.conflict_rate,
            weights=ObjectiveWeights(),
        )
        oracle = create_policy("oracle")
        oracle_started = perf_counter()
        oracle_frame = oracle.select(case.problem)
        oracle_latency_ms = (perf_counter() - oracle_started) * 1_000
        for name in names:
            runs.append(_run_policy(name, case, oracle_frame, oracle_latency_ms))

    summaries: list[dict[str, Any]] = []
    for name in names:
        policy_runs = [run for run in runs if run.policy == name]
        successful = [run for run in policy_runs if run.supported]
        unsupported = len(policy_runs) - len(successful)
        latencies = [run.latency_ms for run in policy_runs]
        summary: dict[str, Any] = {
            "policy": name,
            "runs": len(policy_runs),
            "supported_runs": len(successful),
            "unsupported_runs": unsupported,
            "mean_objective": None,
            "mean_approximation_ratio": None,
            "mean_critical_recall": None,
            "mean_redundancy_ratio": None,
            "mean_stale_item_ratio": None,
            "mean_budget_utilization": None,
            "median_latency_ms": median(latencies),
            "p95_latency_ms": _percentile(latencies, 0.95),
        }
        if successful:
            summary.update(
                {
                    "mean_objective": fmean(
                        run.objective_score
                        for run in successful
                        if run.objective_score is not None
                    ),
                    "mean_approximation_ratio": fmean(
                        run.approximation_ratio
                        for run in successful
                        if run.approximation_ratio is not None
                    ),
                    "mean_critical_recall": fmean(
                        run.critical_recall
                        for run in successful
                        if run.critical_recall is not None
                    ),
                    "mean_redundancy_ratio": fmean(
                        run.redundancy_ratio
                        for run in successful
                        if run.redundancy_ratio is not None
                    ),
                    "mean_stale_item_ratio": fmean(
                        run.stale_item_ratio
                        for run in successful
                        if run.stale_item_ratio is not None
                    ),
                    "mean_budget_utilization": fmean(
                        run.budget_utilization
                        for run in successful
                        if run.budget_utilization is not None
                    ),
                }
            )
        summaries.append(summary)

    return {
        "schema_version": "1",
        "config": asdict(config),
        "policies": list(names),
        "summary": summaries,
        "runs": [asdict(run) for run in runs],
    }


def render_markdown(report: dict[str, Any]) -> str:
    config = report["config"]
    lines = [
        "# ContextOpt synthetic benchmark",
        "",
        (
            f"Paired evaluation over **{config['instances']}** instances with "
            f"**{config['item_count']}** candidates and a "
            f"**{config['budget']} token** budget."
        ),
        "",
        (
            "| Policy | Approx. ratio | Critical recall | Redundancy | "
            "Budget used | Median ms | p95 ms |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["summary"]:
        if row["mean_approximation_ratio"] is None:
            lines.append(
                f"| {row['policy']} | unsupported | — | — | — | "
                f"{row['median_latency_ms']:.3f} | {row['p95_latency_ms']:.3f} |"
            )
            continue
        lines.append(
            f"| {row['policy']} | {row['mean_approximation_ratio']:.3f} | "
            f"{row['mean_critical_recall']:.3f} | "
            f"{row['mean_redundancy_ratio']:.3f} | "
            f"{row['mean_budget_utilization']:.3f} | "
            f"{row['median_latency_ms']:.3f} | {row['p95_latency_ms']:.3f} |"
        )
    lines.extend(
        [
            "",
            (
                "The exact oracle is used only on small instances to measure "
                "heuristic quality."
            ),
            (
                "Critical-fact recall is an evaluation label and is never "
                "exposed to a policy."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def render_console(report: dict[str, Any]) -> str:
    headers = (
        "policy",
        "approx",
        "critical",
        "redundancy",
        "budget",
        "median_ms",
    )
    rows = [headers]
    for row in report["summary"]:
        if row["mean_approximation_ratio"] is None:
            rows.append(
                (
                    row["policy"],
                    "n/a",
                    "n/a",
                    "n/a",
                    "n/a",
                    f"{row['median_latency_ms']:.3f}",
                )
            )
        else:
            rows.append(
                (
                    row["policy"],
                    f"{row['mean_approximation_ratio']:.3f}",
                    f"{row['mean_critical_recall']:.3f}",
                    f"{row['mean_redundancy_ratio']:.3f}",
                    f"{row['mean_budget_utilization']:.3f}",
                    f"{row['median_latency_ms']:.3f}",
                )
            )
    widths = [
        max(len(str(row[index])) for row in rows) for index in range(len(headers))
    ]
    rendered = []
    for row_index, row in enumerate(rows):
        rendered.append(
            "  ".join(
                str(value).ljust(widths[index]) for index, value in enumerate(row)
            )
        )
        if row_index == 0:
            rendered.append("  ".join("-" * width for width in widths))
    return "\n".join(rendered)
