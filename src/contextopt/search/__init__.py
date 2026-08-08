"""Test-guided branch search for long-running coding agents."""

from contextopt.search.branching import (
    BranchCase,
    BranchNode,
    BranchSearch,
    BranchSearchConfig,
    BranchSearchReport,
    CandidatePatch,
    SearchEvent,
    TestResult,
    demo_case,
    render_branch_console,
    render_branch_html,
    render_branch_markdown,
    validate_search_report,
    verify_search_events,
)
from contextopt.search.executor import (
    ExecutableSearchConfig,
    evaluate_candidate,
    evaluate_case,
    run_executable_search,
)

__all__ = [
    "BranchCase",
    "BranchNode",
    "BranchSearch",
    "BranchSearchConfig",
    "BranchSearchReport",
    "CandidatePatch",
    "ExecutableSearchConfig",
    "SearchEvent",
    "TestResult",
    "demo_case",
    "evaluate_candidate",
    "evaluate_case",
    "render_branch_console",
    "render_branch_html",
    "render_branch_markdown",
    "run_executable_search",
    "validate_search_report",
    "verify_search_events",
]
