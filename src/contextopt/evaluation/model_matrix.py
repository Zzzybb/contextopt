"""Matched multi-model analysis for completed agent-eval artifacts.

The real-provider workflow intentionally keeps data collection separate from analysis.
This
module consumes report/manifest pairs and refuses to pool artifacts whose task protocol,
source revision, budget, or execution policy differs. It makes a future multi-model
comparison auditable without pretending that a small fixed matrix is a general
capability benchmark.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, cast

from contextopt.evaluation.agent_search import (
    AgentEvalComparison,
    AgentEvalReport,
    AgentEvalSummary,
    AgentStrategy,
    build_agent_eval_comparisons,
)

MODEL_MATRIX_SCHEMA_VERSION = "1"
_MODEL_IDENTITY_FIELDS = (
    "planner_model",
    "solver_model",
    "reviewer_model",
)
_CLAIM_BOUNDARY = (
    "This is a matched artifact analysis over fixed agent-eval ledgers. It measures "
    "whether observed strategy deltas are directionally consistent across supplied "
    "model runs; it "
    "does not establish general model capability, a causal strategy effect, a "
    "random-effects estimate, a multiple-comparison correction, or production safety."
)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _finite(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if result != result or result in {float("inf"), float("-inf")}:
        raise ValueError(f"{label} must be finite")
    return result


def _integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    return value


def _non_negative_integer(value: Any, label: str) -> int:
    result = _integer(value, label)
    if result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _read_json(path: str | Path, label: str) -> Mapping[str, Any]:
    target = Path(path)
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}: {target}") from exc
    return _mapping(data, label)


def _protocol_payload(
    report: AgentEvalReport, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    config = dict(report.config.to_dict())
    # Model identity is the independent variable.  All other config fields are fixed
    # protocol controls and must match before deltas are pooled.
    for field in _MODEL_IDENTITY_FIELDS:
        config[field] = None
    provider = _mapping(manifest.get("provider"), "agent-eval manifest provider")
    runtime = _mapping(manifest.get("runtime"), "agent-eval manifest runtime")
    temperature = _finite(runtime.get("temperature"), "runtime.temperature")
    if temperature < 0:
        raise ValueError("runtime.temperature must be non-negative")
    timeout_seconds = _finite(runtime.get("timeout_seconds"), "runtime.timeout_seconds")
    if timeout_seconds <= 0:
        raise ValueError("runtime.timeout_seconds must be positive")
    sandbox = _text(runtime.get("sandbox"), "runtime.sandbox")
    if sandbox not in {"host", "docker"}:
        raise ValueError("runtime.sandbox must be 'host' or 'docker'")
    return {
        "adapter": _text(provider.get("adapter"), "provider.adapter"),
        "config": config,
        "repository_revision": _text(
            manifest.get("repository_revision"), "repository_revision"
        ),
        "runtime": {
            "temperature": temperature,
            "timeout_seconds": timeout_seconds,
            "max_retries": _non_negative_integer(
                runtime.get("max_retries"), "runtime.max_retries"
            ),
            "sandbox": sandbox,
            "container_image": _text(
                runtime.get("container_image"), "runtime.container_image"
            ),
        },
    }


def _fingerprint(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class AgentEvalBundle:
    """One report and its matching manifest, with provider identity kept separate."""

    label: str
    report: AgentEvalReport
    manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        _text(self.label, "bundle label")
        if _text(self.manifest.get("schema_version"), "manifest.schema_version") != "1":
            raise ValueError("unsupported agent-eval manifest schema")
        if (
            _text(self.manifest.get("kind"), "manifest.kind")
            != "contextopt.agent-eval.manifest"
        ):
            raise ValueError("manifest.kind is not an agent-eval manifest")
        manifest_config = _mapping(self.manifest.get("config"), "manifest.config")
        if dict(manifest_config) != self.report.config.to_dict():
            raise ValueError("manifest config does not match the agent-eval report")
        provider = _mapping(self.manifest.get("provider"), "manifest.provider")
        adapter = _text(provider.get("adapter"), "provider.adapter")
        if adapter != self.report.config.model_adapter:
            raise ValueError("manifest provider adapter does not match report config")
        _protocol_payload(self.report, self.manifest)

    @property
    def provider(self) -> Mapping[str, Any]:
        return _mapping(self.manifest.get("provider"), "manifest.provider")

    @property
    def protocol(self) -> dict[str, Any]:
        return _protocol_payload(self.report, self.manifest)

    @property
    def protocol_fingerprint(self) -> str:
        return _fingerprint(self.protocol)

    def to_dict(self) -> dict[str, Any]:
        provider = self.provider
        return {
            "label": self.label,
            "adapter": provider.get("adapter"),
            "model": provider.get("model"),
            "planner_model": provider.get("planner_model"),
            "solver_model": provider.get("solver_model"),
            "reviewer_model": provider.get("reviewer_model"),
            "base_url": provider.get("base_url"),
            "repository_revision": self.protocol["repository_revision"],
            "protocol_fingerprint": self.protocol_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class AgentEvalGeneralizationSummary:
    """Direction-consistency summary for one strategy across model artifacts."""

    strategy: AgentStrategy
    model_count: int
    visible_positive_models: int
    visible_negative_models: int
    visible_tie_models: int
    mean_visible_delta: float
    stddev_visible_delta: float
    hidden_model_count: int
    hidden_positive_models: int
    hidden_negative_models: int
    hidden_tie_models: int
    mean_hidden_delta: float | None
    stddev_hidden_delta: float | None

    def __post_init__(self) -> None:
        if self.strategy not in ("single_pass", "best_of_n", "orchestrated"):
            raise ValueError(f"unknown model-matrix strategy: {self.strategy!r}")
        if self.model_count <= 0:
            raise ValueError("model_count must be positive")
        if (
            self.visible_positive_models
            + self.visible_negative_models
            + self.visible_tie_models
            != self.model_count
        ):
            raise ValueError("visible model direction counts are inconsistent")
        if not 0 <= self.hidden_model_count <= self.model_count:
            raise ValueError("hidden_model_count must be within model_count")
        if (
            self.hidden_positive_models
            + self.hidden_negative_models
            + self.hidden_tie_models
            != self.hidden_model_count
        ):
            raise ValueError("hidden model direction counts are inconsistent")
        _finite(self.mean_visible_delta, "mean_visible_delta")
        _finite(self.stddev_visible_delta, "stddev_visible_delta")
        if self.mean_hidden_delta is None:
            if self.stddev_hidden_delta is not None:
                raise ValueError("hidden standard deviation requires a hidden mean")
        else:
            _finite(self.mean_hidden_delta, "mean_hidden_delta")
            if self.stddev_hidden_delta is None:
                raise ValueError("hidden mean requires a hidden standard deviation")
            _finite(self.stddev_hidden_delta, "stddev_hidden_delta")

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "model_count": self.model_count,
            "visible_positive_models": self.visible_positive_models,
            "visible_negative_models": self.visible_negative_models,
            "visible_tie_models": self.visible_tie_models,
            "mean_visible_delta": self.mean_visible_delta,
            "stddev_visible_delta": self.stddev_visible_delta,
            "hidden_model_count": self.hidden_model_count,
            "hidden_positive_models": self.hidden_positive_models,
            "hidden_negative_models": self.hidden_negative_models,
            "hidden_tie_models": self.hidden_tie_models,
            "mean_hidden_delta": self.mean_hidden_delta,
            "stddev_hidden_delta": self.stddev_hidden_delta,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AgentEvalGeneralizationSummary:
        value = _mapping(data, "generalization summary")
        return cls(
            strategy=cast(AgentStrategy, _text(value.get("strategy"), "strategy")),
            model_count=_integer(value.get("model_count"), "model_count"),
            visible_positive_models=_integer(
                value.get("visible_positive_models"), "visible_positive_models"
            ),
            visible_negative_models=_integer(
                value.get("visible_negative_models"), "visible_negative_models"
            ),
            visible_tie_models=_integer(
                value.get("visible_tie_models"), "visible_tie_models"
            ),
            mean_visible_delta=_finite(
                value.get("mean_visible_delta"), "mean_visible_delta"
            ),
            stddev_visible_delta=_finite(
                value.get("stddev_visible_delta"), "stddev_visible_delta"
            ),
            hidden_model_count=_integer(
                value.get("hidden_model_count"), "hidden_model_count"
            ),
            hidden_positive_models=_integer(
                value.get("hidden_positive_models"), "hidden_positive_models"
            ),
            hidden_negative_models=_integer(
                value.get("hidden_negative_models"), "hidden_negative_models"
            ),
            hidden_tie_models=_integer(
                value.get("hidden_tie_models"), "hidden_tie_models"
            ),
            mean_hidden_delta=(
                None
                if value.get("mean_hidden_delta") is None
                else _finite(value.get("mean_hidden_delta"), "mean_hidden_delta")
            ),
            stddev_hidden_delta=(
                None
                if value.get("stddev_hidden_delta") is None
                else _finite(value.get("stddev_hidden_delta"), "stddev_hidden_delta")
            ),
        )


def _comparison_from_dict(data: Mapping[str, Any]) -> AgentEvalComparison:
    value = _mapping(data, "model comparison")
    return AgentEvalComparison(
        baseline_strategy=cast(
            AgentStrategy, _text(value.get("baseline_strategy"), "baseline_strategy")
        ),
        strategy=cast(AgentStrategy, _text(value.get("strategy"), "strategy")),
        paired_count=_integer(value.get("paired_count"), "paired_count"),
        wins=_integer(value.get("wins"), "wins"),
        losses=_integer(value.get("losses"), "losses"),
        ties=_integer(value.get("ties"), "ties"),
        visible_delta=_finite(value.get("visible_delta"), "visible_delta"),
        hidden_paired_count=_integer(
            value.get("hidden_paired_count"), "hidden_paired_count"
        ),
        hidden_wins=_integer(value.get("hidden_wins"), "hidden_wins"),
        hidden_losses=_integer(value.get("hidden_losses"), "hidden_losses"),
        hidden_ties=_integer(value.get("hidden_ties"), "hidden_ties"),
        hidden_delta=(
            None
            if value.get("hidden_delta") is None
            else _finite(value.get("hidden_delta"), "hidden_delta")
        ),
        mean_test_call_delta=_finite(
            value.get("mean_test_call_delta"), "mean_test_call_delta"
        ),
        stddev_test_call_delta=_finite(
            value.get("stddev_test_call_delta"), "stddev_test_call_delta"
        ),
        mean_token_delta=_finite(value.get("mean_token_delta"), "mean_token_delta"),
        stddev_token_delta=_finite(
            value.get("stddev_token_delta"), "stddev_token_delta"
        ),
        mean_duration_delta=_finite(
            value.get("mean_duration_delta"), "mean_duration_delta"
        ),
        stddev_duration_delta=_finite(
            value.get("stddev_duration_delta"), "stddev_duration_delta"
        ),
        visible_mcnemar_pvalue=_finite(
            value.get("visible_mcnemar_pvalue"), "visible_mcnemar_pvalue"
        ),
        hidden_mcnemar_pvalue=(
            None
            if value.get("hidden_mcnemar_pvalue") is None
            else _finite(value.get("hidden_mcnemar_pvalue"), "hidden_mcnemar_pvalue")
        ),
    )


@dataclass(frozen=True, slots=True)
class AgentEvalModelSummary:
    """Per-model results retained by the aggregate artifact."""

    bundle: Mapping[str, Any]
    summaries: tuple[AgentEvalSummary, ...]
    comparisons: tuple[AgentEvalComparison, ...]

    def __post_init__(self) -> None:
        _text(self.bundle.get("label"), "model label")
        if not self.summaries:
            raise ValueError("model summaries must not be empty")
        strategies = tuple(summary.strategy for summary in self.summaries)
        if len(strategies) != len(set(strategies)):
            raise ValueError("model summaries must have unique strategies")
        if any(
            comparison.strategy not in strategies for comparison in self.comparisons
        ):
            raise ValueError("model comparison references an unknown strategy")

    @property
    def label(self) -> str:
        return _text(self.bundle.get("label"), "model label")

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle": dict(self.bundle),
            "summaries": [summary.to_dict() for summary in self.summaries],
            "comparisons": [comparison.to_dict() for comparison in self.comparisons],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AgentEvalModelSummary:
        value = _mapping(data, "model summary")
        raw_summaries = value.get("summaries")
        raw_comparisons = value.get("comparisons")
        if not isinstance(raw_summaries, list) or not isinstance(raw_comparisons, list):
            raise ValueError("model summaries and comparisons must be arrays")
        return cls(
            bundle=dict(_mapping(value.get("bundle"), "model bundle")),
            summaries=tuple(AgentEvalSummary.from_dict(item) for item in raw_summaries),
            comparisons=tuple(_comparison_from_dict(item) for item in raw_comparisons),
        )


@dataclass(frozen=True, slots=True)
class AgentEvalModelMatrixReport:
    """Serializable matched multi-model analysis."""

    protocol: Mapping[str, Any]
    models: tuple[AgentEvalModelSummary, ...]
    generalization: tuple[AgentEvalGeneralizationSummary, ...]
    claim_boundary: str = _CLAIM_BOUNDARY
    schema_version: str = MODEL_MATRIX_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MODEL_MATRIX_SCHEMA_VERSION:
            raise ValueError("unsupported model matrix schema")
        if self.claim_boundary != _CLAIM_BOUNDARY:
            raise ValueError("model matrix claim boundary is not canonical")
        if len(self.models) < 2:
            raise ValueError("model matrix requires at least two artifacts")
        labels = [model.label for model in self.models]
        if len(labels) != len(set(labels)):
            raise ValueError("model labels must be unique")
        fingerprint = _text(self.protocol.get("fingerprint"), "protocol.fingerprint")
        if len(fingerprint) != 64:
            raise ValueError("protocol fingerprint must be a sha256 hex digest")
        controls = _mapping(self.protocol.get("controls"), "protocol.controls")
        if _fingerprint(controls) != fingerprint:
            raise ValueError("protocol fingerprint does not match protocol controls")
        _text(self.protocol.get("baseline_strategy"), "protocol.baseline_strategy")
        strategies = {
            comparison.strategy
            for model in self.models
            for comparison in model.comparisons
        }
        if any(summary.strategy not in strategies for summary in self.generalization):
            raise ValueError("generalization references an unobserved comparison")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protocol": dict(self.protocol),
            "models": [model.to_dict() for model in self.models],
            "generalization": [summary.to_dict() for summary in self.generalization],
            "claim_boundary": self.claim_boundary,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AgentEvalModelMatrixReport:
        value = _mapping(data, "model matrix report")
        raw_models = value.get("models")
        raw_generalization = value.get("generalization")
        if not isinstance(raw_models, list) or not isinstance(raw_generalization, list):
            raise ValueError("model matrix models and generalization must be arrays")
        return cls(
            schema_version=_text(value.get("schema_version"), "schema_version"),
            protocol=dict(_mapping(value.get("protocol"), "protocol")),
            models=tuple(AgentEvalModelSummary.from_dict(item) for item in raw_models),
            generalization=tuple(
                AgentEvalGeneralizationSummary.from_dict(item)
                for item in raw_generalization
            ),
            claim_boundary=_text(value.get("claim_boundary"), "claim_boundary"),
        )


def load_agent_eval_bundle(
    label: str, report_path: str | Path, manifest_path: str | Path
) -> AgentEvalBundle:
    """Load and cross-check one report/manifest pair."""

    report_data = _read_json(report_path, "agent-eval report")
    manifest = _read_json(manifest_path, "agent-eval manifest")
    return AgentEvalBundle(
        label=_text(label, "bundle label"),
        report=AgentEvalReport.from_dict(report_data),
        manifest=manifest,
    )


def _baseline_strategy(
    report: AgentEvalReport, requested: AgentStrategy
) -> AgentStrategy:
    if requested in report.config.strategies:
        return requested
    if len(report.config.strategies) > 1:
        return report.config.strategies[0]
    raise ValueError("model matrix requires at least two strategies per artifact")


def _build_generalization(
    comparisons: Sequence[tuple[AgentEvalComparison, ...]],
) -> tuple[AgentEvalGeneralizationSummary, ...]:
    by_strategy: dict[AgentStrategy, list[AgentEvalComparison]] = {}
    for model_comparisons in comparisons:
        for comparison in model_comparisons:
            by_strategy.setdefault(comparison.strategy, []).append(comparison)
    summaries: list[AgentEvalGeneralizationSummary] = []
    for strategy, values in by_strategy.items():
        visible_deltas = tuple(value.visible_delta for value in values)
        hidden_values = tuple(
            value.hidden_delta for value in values if value.hidden_delta is not None
        )
        summaries.append(
            AgentEvalGeneralizationSummary(
                strategy=strategy,
                model_count=len(values),
                visible_positive_models=sum(delta > 0 for delta in visible_deltas),
                visible_negative_models=sum(delta < 0 for delta in visible_deltas),
                visible_tie_models=sum(delta == 0 for delta in visible_deltas),
                mean_visible_delta=fmean(visible_deltas),
                stddev_visible_delta=pstdev(visible_deltas),
                hidden_model_count=len(hidden_values),
                hidden_positive_models=sum(delta > 0 for delta in hidden_values),
                hidden_negative_models=sum(delta < 0 for delta in hidden_values),
                hidden_tie_models=sum(delta == 0 for delta in hidden_values),
                mean_hidden_delta=(None if not hidden_values else fmean(hidden_values)),
                stddev_hidden_delta=(
                    None if not hidden_values else pstdev(hidden_values)
                ),
            )
        )
    return tuple(summaries)


def build_agent_eval_model_matrix(
    bundles: Sequence[AgentEvalBundle],
    *,
    baseline_strategy: AgentStrategy = "single_pass",
) -> AgentEvalModelMatrixReport:
    """Build a matched multi-model report, rejecting protocol drift."""

    if len(bundles) < 2:
        raise ValueError("model matrix requires at least two artifacts")
    labels = [bundle.label for bundle in bundles]
    if len(labels) != len(set(labels)):
        raise ValueError("model matrix labels must be unique")
    first_protocol = bundles[0].protocol
    first_fingerprint = _fingerprint(first_protocol)
    model_summaries: list[AgentEvalModelSummary] = []
    all_comparisons: list[tuple[AgentEvalComparison, ...]] = []
    selected_baseline: AgentStrategy | None = None
    for bundle in bundles:
        if bundle.protocol != first_protocol:
            raise ValueError(
                "agent-eval artifacts do not share a matched protocol fingerprint: "
                f"expected {first_fingerprint}, got {bundle.protocol_fingerprint} for "
                f"{bundle.label}"
            )
        baseline = _baseline_strategy(bundle.report, baseline_strategy)
        if selected_baseline is None:
            selected_baseline = baseline
        elif baseline != selected_baseline:
            raise ValueError("model artifacts selected different baseline strategies")
        comparisons = build_agent_eval_comparisons(
            bundle.report, baseline_strategy=baseline
        )
        all_comparisons.append(comparisons)
        model_summaries.append(
            AgentEvalModelSummary(
                bundle=bundle.to_dict(),
                summaries=bundle.report.summaries,
                comparisons=comparisons,
            )
        )
    protocol = {
        "fingerprint": first_fingerprint,
        "controls": first_protocol,
        "baseline_strategy": selected_baseline,
    }
    return AgentEvalModelMatrixReport(
        protocol=protocol,
        models=tuple(model_summaries),
        generalization=_build_generalization(all_comparisons),
    )


def render_agent_eval_model_matrix_console(
    report: AgentEvalModelMatrixReport,
) -> str:
    lines = [
        "model strategy visible hidden visible_delta hidden_delta",
        "----- -------- ------- ------ ------------- ------------",
    ]
    for model in report.models:
        comparison_by_strategy = {
            comparison.strategy: comparison for comparison in model.comparisons
        }
        for summary in model.summaries:
            comparison = comparison_by_strategy.get(summary.strategy)
            visible_delta = (
                "n/a" if comparison is None else f"{comparison.visible_delta:+.0%}"
            )
            hidden_delta = (
                "n/a"
                if comparison is None or comparison.hidden_delta is None
                else f"{comparison.hidden_delta:+.0%}"
            )
            lines.append(
                f"{model.label} {summary.strategy} "
                f"{summary.success_count}/{summary.run_count} "
                f"{summary.hidden_success_count}/{summary.hidden_run_count} "
                f"{visible_delta} {hidden_delta}"
            )
    lines.append("")
    lines.append("cross-model strategy consistency")
    lines.append("strategy models visible(+/-/=) hidden(+/-/=) mean_visible_delta")
    lines.append("-------- ------ --------------- -------------- -------------------")
    for generalization_summary in report.generalization:
        visible_direction = (
            f"{generalization_summary.visible_positive_models}/"
            f"{generalization_summary.visible_negative_models}/"
            f"{generalization_summary.visible_tie_models}"
        )
        hidden_direction = (
            f"{generalization_summary.hidden_positive_models}/"
            f"{generalization_summary.hidden_negative_models}/"
            f"{generalization_summary.hidden_tie_models}"
        )
        lines.append(
            f"{generalization_summary.strategy} {generalization_summary.model_count} "
            f"{visible_direction} {hidden_direction} "
            f"{generalization_summary.mean_visible_delta:+.1%}"
        )
    return "\n".join(lines) + "\n"


def render_agent_eval_model_matrix_markdown(
    report: AgentEvalModelMatrixReport,
) -> str:
    lines = [
        "# Matched agent-eval model matrix",
        "",
        report.claim_boundary,
        "",
        f"- Protocol fingerprint: `{report.protocol['fingerprint']}`",
        f"- Models: `{len(report.models)}`",
        f"- Baseline strategy: `{report.protocol['baseline_strategy']}`",
        "",
        "## Model identities",
        "",
        "| Label | Adapter | Model | Planner | Solver | Reviewer | Revision |",
        "|---|---|---|---|---|---|---|",
    ]
    for model in report.models:
        bundle = model.bundle
        lines.append(
            f"| `{escape(str(bundle.get('label', '')))}` | "
            f"`{escape(str(bundle.get('adapter', '')))}` | "
            f"`{escape(str(bundle.get('model', 'n/a')))}` | "
            f"`{escape(str(bundle.get('planner_model', 'n/a')))}` | "
            f"`{escape(str(bundle.get('solver_model', 'n/a')))}` | "
            f"`{escape(str(bundle.get('reviewer_model', 'n/a')))}` | "
            f"`{escape(str(bundle.get('repository_revision', '')))}` |"
        )
    lines.extend(
        [
            "",
            "## Per-model outcomes",
            "",
            "| Model | Strategy | Visible | Hidden | Visible Δ | Hidden Δ | "
            "Visible exact p |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model in report.models:
        comparisons = {
            comparison.strategy: comparison for comparison in model.comparisons
        }
        for summary in model.summaries:
            comparison = comparisons.get(summary.strategy)
            visible_delta = (
                "n/a" if comparison is None else f"{comparison.visible_delta:+.0%}"
            )
            hidden_delta = (
                "n/a"
                if comparison is None or comparison.hidden_delta is None
                else f"{comparison.hidden_delta:+.0%}"
            )
            pvalue = (
                "n/a"
                if comparison is None
                else f"{comparison.visible_mcnemar_pvalue:.3g}"
            )
            lines.append(
                f"| `{model.label}` | `{summary.strategy}` | "
                f"{summary.success_count}/{summary.run_count} | "
                f"{summary.hidden_success_count}/{summary.hidden_run_count} | "
                f"{visible_delta} | {hidden_delta} | {pvalue} |"
            )
    lines.extend(
        [
            "",
            "## Cross-model consistency",
            "",
            "Positive/negative/tie counts are over model-level paired deltas; "
            "they do not pool raw cells.",
            "",
            "| Strategy | Models | Visible +/-/= | Hidden +/-/= | Mean visible Δ | "
            "Stdev | Mean hidden Δ | Stdev |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for generalization_summary in report.generalization:
        visible_direction = (
            f"{generalization_summary.visible_positive_models}/"
            f"{generalization_summary.visible_negative_models}/"
            f"{generalization_summary.visible_tie_models}"
        )
        hidden_direction = (
            f"{generalization_summary.hidden_positive_models}/"
            f"{generalization_summary.hidden_negative_models}/"
            f"{generalization_summary.hidden_tie_models}"
        )
        mean_hidden = (
            "n/a"
            if generalization_summary.mean_hidden_delta is None
            else f"{generalization_summary.mean_hidden_delta:+.1%}"
        )
        stddev_hidden = (
            "n/a"
            if generalization_summary.stddev_hidden_delta is None
            else f"{generalization_summary.stddev_hidden_delta:.1%}"
        )
        lines.append(
            f"| `{generalization_summary.strategy}` | "
            f"{generalization_summary.model_count} | "
            f"{visible_direction} | {hidden_direction} | "
            f"{generalization_summary.mean_visible_delta:+.1%} | "
            f"{generalization_summary.stddev_visible_delta:.1%} | "
            f"{mean_hidden} | {stddev_hidden} |"
        )
    return "\n".join(lines) + "\n"


def render_agent_eval_model_matrix_html(
    report: AgentEvalModelMatrixReport,
) -> str:
    rows = []
    for model in report.models:
        comparisons = {
            comparison.strategy: comparison for comparison in model.comparisons
        }
        for summary in model.summaries:
            comparison = comparisons.get(summary.strategy)
            visible_delta = (
                "n/a" if comparison is None else f"{comparison.visible_delta:+.0%}"
            )
            hidden_delta = (
                "n/a"
                if comparison is None or comparison.hidden_delta is None
                else f"{comparison.hidden_delta:+.0%}"
            )
            rows.append(
                "<tr>"
                f"<td><code>{escape(model.label)}</code></td>"
                f"<td><code>{escape(summary.strategy)}</code></td>"
                f"<td>{summary.success_count}/{summary.run_count}</td>"
                f"<td>{summary.hidden_success_count}/{summary.hidden_run_count}</td>"
                f"<td>{visible_delta}</td>"
                f"<td>{hidden_delta}</td>"
                "</tr>"
            )
    payload = escape(json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True))
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>ContextOpt model matrix</title>"
        "<style>body{font:14px system-ui,sans-serif;margin:2rem;"
        "color:#172033;max-width:1200px}"
        "table{border-collapse:collapse;width:100%;margin-top:1rem}"
        "th,td{border:1px solid #d7dce8;padding:.5rem;text-align:left}"
        "th{background:#f5f7fb}.muted{color:#5d6b82}"
        "code{white-space:pre-wrap}</style></head><body>"
        "<h1>Matched agent-eval model matrix</h1>"
        f"<p class='muted'>{escape(report.claim_boundary)}</p>"
        f"<p>Protocol fingerprint: <code>"
        f"{escape(str(report.protocol['fingerprint']))}</code></p>"
        "<table><thead><tr><th>model</th><th>strategy</th>"
        "<th>visible</th><th>hidden</th>"
        "<th>visible Δ</th><th>hidden Δ</th></tr></thead><tbody>"
        f"{''.join(rows)}</tbody></table>"
        "<details><summary>machine-readable analysis</summary><code>"
        f"{payload}</code></details></body></html>"
    )


__all__ = [
    "MODEL_MATRIX_SCHEMA_VERSION",
    "AgentEvalBundle",
    "AgentEvalGeneralizationSummary",
    "AgentEvalModelMatrixReport",
    "AgentEvalModelSummary",
    "build_agent_eval_model_matrix",
    "load_agent_eval_bundle",
    "render_agent_eval_model_matrix_console",
    "render_agent_eval_model_matrix_html",
    "render_agent_eval_model_matrix_markdown",
]
