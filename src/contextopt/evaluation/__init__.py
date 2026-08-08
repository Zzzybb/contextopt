"""Reproducible evaluations with explicit claim boundaries."""

from contextopt.evaluation.context_routing import (
    ContextRoutingEvalConfig,
    EvidenceProbe,
    RoutingRunMetrics,
    RoutingTraceCase,
    build_long_coding_traces,
    render_context_routing_console,
    render_context_routing_markdown,
    run_context_routing_evaluation,
    tool_protocol_issues,
)

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
