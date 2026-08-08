"""Bounded planner/solver/reviewer orchestration for coding search.

The planner, solver, and reviewer are independent provider-neutral model clients.
Planner and reviewer return strict JSON; the solver reuses the complete-snapshot
proposal protocol.  Visible tests remain authoritative, and every role shares
budgets and a durable checkpoint.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from html import escape
from math import isfinite
from pathlib import Path
from typing import Any, Literal, cast

from contextopt.runtime.context import (
    ContextCompiler,
    ContextCompilerConfig,
    ContextReceipt,
    compile_runtime_context,
)
from contextopt.runtime.errors import ModelError, RuntimeContractError
from contextopt.runtime.identity import stable_hash
from contextopt.runtime.protocol import (
    AgentMessage,
    ModelClient,
    ModelRequest,
    ModelResponse,
    TokenUsage,
)
from contextopt.search.branching import (
    BranchCase,
    BranchSearch,
    BranchSearchConfig,
    BranchSearchReport,
    CandidatePatch,
    SearchEvent,
    TestResult,
    verify_search_events,
)
from contextopt.search.executor import ExecutableSearchConfig, evaluate_candidate
from contextopt.search.proposer import ProposalConfig, parse_proposal_response

OrchestrationStatus = Literal[
    "running", "accepted", "exhausted", "budget_exhausted", "paused", "failed"
]
OrchestrationPhase = Literal["idle", "planning", "solving", "evaluating", "reviewing"]
ReviewDecisionName = Literal["accept", "retry", "reject"]


def _s(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _m(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _files(value: Mapping[str, str], label: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object mapping paths to text")
    return dict(
        BranchCase(
            task="orchestration-file-validation",
            root_files=value,
            candidates=(),
            tests={},
        ).root_files
    )


def _decode(content: str, label: str) -> Any:
    text = content.strip()
    fence = text[:3]
    if fence == "~~~" or fence == "\x60\x60\x60":
        lines = text.splitlines()
        if len(lines) < 3 or lines[-1].strip() != fence:
            raise ValueError(f"{label} fenced JSON is incomplete")
        text = "\n".join(lines[1:-1]).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {exc.msg}") from exc


def _strings(
    value: Any,
    label: str,
    max_items: int,
    max_chars: int,
    *,
    required: bool = True,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an array")
    if required and not value:
        raise ValueError(f"{label} must contain at least one item")
    if len(value) > max_items:
        raise ValueError(f"{label} exceeds {max_items} items")
    result = tuple(_s(item, f"{label}[{index}]") for index, item in enumerate(value))
    if any(len(item) > max_chars for item in result):
        raise ValueError(f"{label} contains an oversized item")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must not contain duplicates")
    return result


@dataclass(frozen=True, slots=True)
class PlannerConfig:
    max_items: int = 6
    max_item_chars: int = 600
    max_prompt_chars: int = 400_000
    max_output_tokens: int = 4_096

    def __post_init__(self) -> None:
        for name in (
            "max_items",
            "max_item_chars",
            "max_prompt_chars",
            "max_output_tokens",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PlannerConfig:
        value = _m(data, "planner config")
        allowed = {
            "max_items",
            "max_item_chars",
            "max_prompt_chars",
            "max_output_tokens",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"planner config has unknown fields: {sorted(unknown)!r}")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, int]:
        return {
            "max_items": self.max_items,
            "max_item_chars": self.max_item_chars,
            "max_prompt_chars": self.max_prompt_chars,
            "max_output_tokens": self.max_output_tokens,
        }


@dataclass(frozen=True, slots=True)
class ReviewerConfig:
    max_prompt_chars: int = 400_000
    max_output_tokens: int = 2_048
    max_issue_items: int = 8
    max_item_chars: int = 600

    def __post_init__(self) -> None:
        for name in (
            "max_prompt_chars",
            "max_output_tokens",
            "max_issue_items",
            "max_item_chars",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReviewerConfig:
        value = _m(data, "reviewer config")
        allowed = {
            "max_prompt_chars",
            "max_output_tokens",
            "max_issue_items",
            "max_item_chars",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"reviewer config has unknown fields: {sorted(unknown)!r}")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, int]:
        return {
            "max_prompt_chars": self.max_prompt_chars,
            "max_output_tokens": self.max_output_tokens,
            "max_issue_items": self.max_issue_items,
            "max_item_chars": self.max_item_chars,
        }


@dataclass(frozen=True, slots=True)
class OrchestrationConfig:
    max_rounds: int = 3
    max_model_calls: int = 12
    max_planner_calls: int = 3
    max_solver_calls: int = 3
    max_reviewer_calls: int = 3
    max_candidates: int = 16
    max_test_calls: int = 16
    max_parallel_tests: int = 1
    max_total_tokens: int = 100_000
    context_config: ContextCompilerConfig = field(
        default_factory=lambda: ContextCompilerConfig(
            policy="submodular",
            budget_tokens=16_000,
            recent_blocks=2,
            max_tool_output_tokens=2_048,
            memory_policy="versioned-v1",
        )
    )

    def __post_init__(self) -> None:
        for name in (
            "max_rounds",
            "max_model_calls",
            "max_planner_calls",
            "max_solver_calls",
            "max_reviewer_calls",
            "max_candidates",
            "max_test_calls",
            "max_parallel_tests",
            "max_total_tokens",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.context_config, ContextCompilerConfig):
            raise ValueError("context_config must be a ContextCompilerConfig")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OrchestrationConfig:
        value = _m(data, "orchestration config")
        allowed = {
            "max_rounds",
            "max_model_calls",
            "max_planner_calls",
            "max_solver_calls",
            "max_reviewer_calls",
            "max_candidates",
            "max_test_calls",
            "max_parallel_tests",
            "max_total_tokens",
            "context_config",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                f"orchestration config has unknown fields: {sorted(unknown)!r}"
            )
        values = dict(value)
        raw_context = values.get("context_config")
        if raw_context is not None:
            if not isinstance(raw_context, Mapping):
                raise ValueError("orchestration context_config must be an object")
            values["context_config"] = ContextCompilerConfig.from_dict(raw_context)
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in (
                "max_rounds",
                "max_model_calls",
                "max_planner_calls",
                "max_solver_calls",
                "max_reviewer_calls",
                "max_candidates",
                "max_test_calls",
                "max_parallel_tests",
                "max_total_tokens",
            )
        } | {"context_config": self.context_config.to_dict()}


@dataclass(frozen=True, slots=True)
class PlannerPlan:
    goal: str
    constraints: tuple[str, ...]
    hypotheses: tuple[str, ...]
    test_focus: tuple[str, ...]
    risks: tuple[str, ...]
    schema_version: str = "1"

    def __post_init__(self) -> None:
        if self.schema_version != "1":
            raise ValueError(
                f"unsupported planner plan schema: {self.schema_version!r}"
            )
        object.__setattr__(self, "goal", _s(self.goal, "plan goal"))
        for name in ("constraints", "hypotheses", "test_focus", "risks"):
            values = getattr(self, name)
            if not values or any(
                not isinstance(item, str) or not item.strip() for item in values
            ):
                raise ValueError(f"plan {name} must contain non-empty strings")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PlannerPlan:
        value = _m(data, "planner plan")
        allowed = {
            "schema_version",
            "goal",
            "constraints",
            "hypotheses",
            "test_focus",
            "risks",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"planner plan has unknown fields: {sorted(unknown)!r}")
        return cls(
            schema_version=str(value.get("schema_version", "1")),
            goal=_s(value.get("goal"), "plan goal"),
            constraints=_strings(
                value.get("constraints"), "plan constraints", 32, 2_000
            ),
            hypotheses=_strings(value.get("hypotheses"), "plan hypotheses", 32, 2_000),
            test_focus=_strings(value.get("test_focus"), "plan test_focus", 32, 2_000),
            risks=_strings(value.get("risks"), "plan risks", 32, 2_000),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "goal": self.goal,
            "constraints": list(self.constraints),
            "hypotheses": list(self.hypotheses),
            "test_focus": list(self.test_focus),
            "risks": list(self.risks),
        }


def parse_planner_response(
    response: ModelResponse, config: PlannerConfig | None = None
) -> PlannerPlan:
    cfg = config or PlannerConfig()
    if response.tool_calls:
        raise ValueError("planner response must not contain tool calls")
    decoded = _decode(response.content, "planner response")
    expected = {"goal", "constraints", "hypotheses", "test_focus", "risks"}
    if not isinstance(decoded, Mapping) or set(decoded) != expected:
        raise ValueError("planner response must contain only the plan fields")
    goal = _s(decoded.get("goal"), "plan goal")
    if len(goal) > cfg.max_item_chars:
        raise ValueError("plan goal exceeds max_item_chars")
    constraints = _strings(
        decoded.get("constraints"),
        "plan constraints",
        cfg.max_items,
        cfg.max_item_chars,
    )
    hypotheses = _strings(
        decoded.get("hypotheses"),
        "plan hypotheses",
        cfg.max_items,
        cfg.max_item_chars,
    )
    test_focus = _strings(
        decoded.get("test_focus"),
        "plan test_focus",
        cfg.max_items,
        cfg.max_item_chars,
    )
    risks = _strings(
        decoded.get("risks"), "plan risks", cfg.max_items, cfg.max_item_chars
    )
    return PlannerPlan(
        goal=goal,
        constraints=constraints,
        hypotheses=hypotheses,
        test_focus=test_focus,
        risks=risks,
    )


def build_planner_request(
    task: str,
    root_files: Mapping[str, str],
    config: PlannerConfig | None = None,
    *,
    run_id: str = "multi-agent-orchestration",
    turn: int = 1,
    feedback: Sequence[Mapping[str, Any]] = (),
) -> ModelRequest:
    cfg = config or PlannerConfig()
    snapshot = json.dumps(
        _files(root_files, "root_files"), ensure_ascii=False, sort_keys=True, indent=2
    )
    prior = json.dumps(
        [dict(item) for item in feedback],
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )
    suffix = (
        f"\n\nPrior oracle/reviewer evidence:\n~~~json\n{prior}\n~~~"
        if feedback
        else ""
    )
    content = (
        f"Task:\n{_s(task, 'task')}\n\n"
        f"Current complete workspace snapshot:\n~~~json\n{snapshot}\n~~~\n\n"
        "Return only JSON with exactly these fields:\n"
        '{"goal":"...","constraints":["..."],"hypotheses":["..."],'
        '"test_focus":["..."],"risks":["..."]}\n'
        f"Use at most {cfg.max_items} concise items per array. "
        "State falsifiable hypotheses; do not call tools or claim tests passed."
        f"{suffix}"
    )
    if len(content) > cfg.max_prompt_chars:
        raise ValueError("planner prompt exceeds max_prompt_chars")
    return ModelRequest(
        run_id=run_id,
        turn=turn,
        messages=(
            AgentMessage(
                role="system",
                content=(
                    "You are the planning role. Produce compact, testable hypotheses "
                    "and preserve constraints."
                ),
            ),
            AgentMessage(role="user", content=content),
        ),
        tools=(),
        max_output_tokens=cfg.max_output_tokens,
    )


async def plan_task(
    model: ModelClient,
    task: str,
    root_files: Mapping[str, str],
    config: PlannerConfig | None = None,
    *,
    run_id: str = "multi-agent-orchestration",
    turn: int = 1,
    feedback: Sequence[Mapping[str, Any]] = (),
) -> tuple[PlannerPlan, ModelResponse]:
    response = await model.complete(
        build_planner_request(
            task,
            root_files,
            config,
            run_id=run_id,
            turn=turn,
            feedback=feedback,
        )
    )
    return parse_planner_response(response, config), response


def build_solver_request(
    task: str,
    root_files: Mapping[str, str],
    plan: PlannerPlan,
    config: ProposalConfig | None = None,
    *,
    run_id: str = "multi-agent-orchestration",
    turn: int = 1,
    feedback: Sequence[Mapping[str, Any]] = (),
) -> ModelRequest:
    cfg = config or ProposalConfig()
    snapshot = json.dumps(
        _files(root_files, "root_files"), ensure_ascii=False, sort_keys=True, indent=2
    )
    plan_text = json.dumps(plan.to_dict(), ensure_ascii=False, sort_keys=True, indent=2)
    prior = json.dumps(
        [dict(item) for item in feedback],
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )
    suffix = (
        f"\n\nPrior visible-test/reviewer evidence:\n~~~json\n{prior}\n~~~"
        if feedback
        else ""
    )
    content = (
        f"Task:\n{_s(task, 'task')}\n\nPlanner plan:\n~~~json\n{plan_text}\n~~~\n\n"
        f"Current complete workspace snapshot:\n~~~json\n{snapshot}\n~~~\n\n"
        "Return only this candidate protocol:\n"
        '{"candidates":[{"id":"candidate-1","parent_id":"root",'
        '"hypothesis":"...","files":{"relative/path":"complete text"},'
        '"evidence":["..."]}]}\n'
        f"Propose at most {cfg.max_candidates} complete snapshots. "
        "Do not return diffs, tool calls, or claims that tests passed."
        f"{suffix}"
    )
    if len(content) > cfg.max_total_prompt_chars:
        raise ValueError("solver prompt exceeds max_total_prompt_chars")
    return ModelRequest(
        run_id=run_id,
        turn=turn,
        messages=(
            AgentMessage(
                role="system",
                content=(
                    "You are the solver role. Implement planner hypotheses as "
                    "complete, independently testable workspace snapshots."
                ),
            ),
            AgentMessage(role="user", content=content),
        ),
        tools=(),
        max_output_tokens=cfg.max_output_tokens,
    )


async def solve_plan(
    model: ModelClient,
    task: str,
    root_files: Mapping[str, str],
    plan: PlannerPlan,
    config: ProposalConfig | None = None,
    *,
    run_id: str = "multi-agent-orchestration",
    turn: int = 1,
    feedback: Sequence[Mapping[str, Any]] = (),
) -> tuple[BranchCase, ModelResponse]:
    cfg = config or ProposalConfig()
    response = await model.complete(
        build_solver_request(
            task,
            root_files,
            plan,
            cfg,
            run_id=run_id,
            turn=turn,
            feedback=feedback,
        )
    )
    return parse_proposal_response(response, task, root_files, cfg), response


@dataclass(frozen=True, slots=True)
class ReviewDecision:
    decision: ReviewDecisionName
    candidate_id: str | None
    confidence: float
    blocking_issues: tuple[str, ...]
    required_checks: tuple[str, ...]
    summary: str

    def __post_init__(self) -> None:
        if self.decision not in {"accept", "retry", "reject"}:
            raise ValueError(f"unsupported review decision: {self.decision!r}")
        if self.candidate_id is not None:
            object.__setattr__(
                self, "candidate_id", _s(self.candidate_id, "candidate_id")
            )
        if not isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("review confidence must be between 0 and 1")
        object.__setattr__(self, "summary", _s(self.summary, "review summary"))
        if self.decision == "accept" and self.candidate_id is None:
            raise ValueError("accept review must identify candidate_id")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReviewDecision:
        value = _m(data, "review decision")
        allowed = {
            "decision",
            "candidate_id",
            "confidence",
            "blocking_issues",
            "required_checks",
            "summary",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"review decision has unknown fields: {sorted(unknown)!r}")
        decision = value.get("decision")
        if decision not in {"accept", "retry", "reject"}:
            raise ValueError("review decision must be accept, retry, or reject")
        candidate_id = value.get("candidate_id")
        if candidate_id is not None and not isinstance(candidate_id, str):
            raise ValueError("review candidate_id must be a string or null")
        confidence = value.get("confidence")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            raise ValueError("review confidence must be a number")
        return cls(
            decision=cast(ReviewDecisionName, decision),
            candidate_id=candidate_id,
            confidence=float(confidence),
            blocking_issues=_strings(
                value.get("blocking_issues"),
                "review blocking_issues",
                32,
                2_000,
                required=False,
            ),
            required_checks=_strings(
                value.get("required_checks"),
                "review required_checks",
                32,
                2_000,
                required=False,
            ),
            summary=_s(value.get("summary"), "review summary"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "candidate_id": self.candidate_id,
            "confidence": self.confidence,
            "blocking_issues": list(self.blocking_issues),
            "required_checks": list(self.required_checks),
            "summary": self.summary,
        }


def parse_reviewer_response(
    response: ModelResponse, config: ReviewerConfig | None = None
) -> ReviewDecision:
    cfg = config or ReviewerConfig()
    if response.tool_calls:
        raise ValueError("reviewer response must not contain tool calls")
    decoded = _decode(response.content, "reviewer response")
    expected = {
        "decision",
        "candidate_id",
        "confidence",
        "blocking_issues",
        "required_checks",
        "summary",
    }
    if not isinstance(decoded, Mapping) or set(decoded) != expected:
        raise ValueError("reviewer response must contain only the review fields")
    decision = ReviewDecision.from_dict(decoded)
    if (
        len(decision.summary) > cfg.max_item_chars
        or len(decision.blocking_issues) > cfg.max_issue_items
        or len(decision.required_checks) > cfg.max_issue_items
    ):
        raise ValueError("review decision exceeds configured bounds")
    if any(
        len(item) > cfg.max_item_chars
        for item in (*decision.blocking_issues, *decision.required_checks)
    ):
        raise ValueError("review decision contains an oversized item")
    return decision


def _review_payload(report: BranchSearchReport) -> dict[str, Any]:
    nodes = [node for node in report.nodes if node.id != "root"]
    nodes.sort(key=lambda node: (-node.score, node.depth, node.id))
    return {
        "status": report.status,
        "best_node_id": report.best_node_id,
        "metrics": dict(report.metrics),
        "candidates": [
            {
                "id": node.id,
                "parent_id": node.parent_id,
                "status": node.status,
                "hypothesis": node.hypothesis,
                "score": node.score,
                "test_result": (
                    None if node.test_result is None else node.test_result.to_dict()
                ),
            }
            for node in nodes[:32]
        ],
    }


def build_reviewer_request(
    task: str,
    plan: PlannerPlan,
    report: BranchSearchReport,
    config: ReviewerConfig | None = None,
    *,
    run_id: str = "multi-agent-orchestration",
    turn: int = 1,
    feedback: Sequence[Mapping[str, Any]] = (),
) -> ModelRequest:
    cfg = config or ReviewerConfig()
    payload = json.dumps(
        _review_payload(report), ensure_ascii=False, sort_keys=True, indent=2
    )
    plan_text = json.dumps(plan.to_dict(), ensure_ascii=False, sort_keys=True, indent=2)
    prior = json.dumps(
        [dict(item) for item in feedback],
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )
    suffix = f"\n\nEarlier evidence:\n~~~json\n{prior}\n~~~" if feedback else ""
    content = (
        f"Task:\n{_s(task, 'task')}\n\nPlanner plan:\n~~~json\n{plan_text}\n~~~\n\n"
        "Visible-test branch report (oracle is authoritative):\n~~~json\n"
        f"{payload}\n~~~\n\n"
        "Return only JSON with exactly these fields:\n"
        '{"decision":"accept|retry|reject","candidate_id":"id-or-null",'
        '"confidence":0.0,"blocking_issues":[],"required_checks":[],"summary":"..."}\n'
        "Accept only an existing candidate whose visible tests passed and whose "
        "plan constraints are satisfied. Never invent test evidence."
        f"{suffix}"
    )
    if len(content) > cfg.max_prompt_chars:
        raise ValueError("reviewer prompt exceeds max_prompt_chars")
    return ModelRequest(
        run_id=run_id,
        turn=turn,
        messages=(
            AgentMessage(
                role="system",
                content=(
                    "You are the reviewer role. Audit visible-test evidence and "
                    "plan constraints; never invent tests."
                ),
            ),
            AgentMessage(role="user", content=content),
        ),
        tools=(),
        max_output_tokens=cfg.max_output_tokens,
    )


async def review_branch(
    model: ModelClient,
    task: str,
    plan: PlannerPlan,
    report: BranchSearchReport,
    config: ReviewerConfig | None = None,
    *,
    run_id: str = "multi-agent-orchestration",
    turn: int = 1,
    feedback: Sequence[Mapping[str, Any]] = (),
) -> tuple[ReviewDecision, ModelResponse]:
    response = await model.complete(
        build_reviewer_request(
            task,
            plan,
            report,
            config,
            run_id=run_id,
            turn=turn,
            feedback=feedback,
        )
    )
    return parse_reviewer_response(response, config), response


@dataclass(frozen=True, slots=True)
class RoleCall:
    role: str
    turn: int
    model_name: str
    response_sha256: str
    response_id: str | None
    usage: TokenUsage
    context_receipt: ContextReceipt | None = None

    def __post_init__(self) -> None:
        if self.role not in {"planner", "solver", "reviewer"}:
            raise ValueError(f"unsupported role: {self.role!r}")
        if not isinstance(self.turn, int) or self.turn <= 0:
            raise ValueError("role call turn must be positive")
        object.__setattr__(self, "model_name", _s(self.model_name, "model_name"))
        object.__setattr__(
            self, "response_sha256", _s(self.response_sha256, "response_sha256")
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RoleCall:
        value = _m(data, "role call")
        allowed = {
            "role",
            "turn",
            "model_name",
            "response_sha256",
            "response_id",
            "usage",
            "context_receipt",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"role call has unknown fields: {sorted(unknown)!r}")
        usage = value.get("usage")
        if not isinstance(usage, Mapping):
            raise ValueError("role call usage must be an object")
        response_id = value.get("response_id")
        if response_id is not None and not isinstance(response_id, str):
            raise ValueError("role call response_id must be a string or null")
        context_receipt = value.get("context_receipt")
        if context_receipt is not None and not isinstance(context_receipt, Mapping):
            raise ValueError("role call context_receipt must be an object or null")
        return cls(
            role=str(value.get("role")),
            turn=int(value.get("turn", -1)),
            model_name=_s(value.get("model_name"), "model_name"),
            response_sha256=_s(value.get("response_sha256"), "response_sha256"),
            response_id=response_id,
            usage=TokenUsage.from_dict(usage),
            context_receipt=(
                None
                if context_receipt is None
                else ContextReceipt.from_dict(context_receipt)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "turn": self.turn,
            "model_name": self.model_name,
            "response_sha256": self.response_sha256,
            "response_id": self.response_id,
            "usage": self.usage.to_dict(),
            "context_receipt": (
                None if self.context_receipt is None else self.context_receipt.to_dict()
            ),
        }


@dataclass(frozen=True, slots=True)
class OrchestrationRound:
    index: int
    base_fingerprint: str
    plan: PlannerPlan
    branch_report: BranchSearchReport
    planner_call: RoleCall
    solver_call: RoleCall
    reviewer_call: RoleCall
    review: ReviewDecision

    def __post_init__(self) -> None:
        if (
            not isinstance(self.index, int)
            or isinstance(self.index, bool)
            or self.index < 0
        ):
            raise ValueError("round index must be non-negative")
        object.__setattr__(
            self, "base_fingerprint", _s(self.base_fingerprint, "base_fingerprint")
        )
        if (
            self.planner_call.role,
            self.solver_call.role,
            self.reviewer_call.role,
        ) != ("planner", "solver", "reviewer"):
            raise ValueError("round role calls have incorrect roles")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OrchestrationRound:
        value = _m(data, "orchestration round")
        allowed = {
            "index",
            "base_fingerprint",
            "plan",
            "branch_report",
            "planner_call",
            "solver_call",
            "reviewer_call",
            "review",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                f"orchestration round has unknown fields: {sorted(unknown)!r}"
            )
        nested = (
            "plan",
            "branch_report",
            "planner_call",
            "solver_call",
            "reviewer_call",
            "review",
        )
        if any(not isinstance(value.get(name), Mapping) for name in nested):
            raise ValueError("orchestration round nested values must be objects")
        return cls(
            index=int(value.get("index", -1)),
            base_fingerprint=_s(value.get("base_fingerprint"), "base_fingerprint"),
            plan=PlannerPlan.from_dict(value["plan"]),
            branch_report=BranchSearchReport.from_dict(value["branch_report"]),
            planner_call=RoleCall.from_dict(value["planner_call"]),
            solver_call=RoleCall.from_dict(value["solver_call"]),
            reviewer_call=RoleCall.from_dict(value["reviewer_call"]),
            review=ReviewDecision.from_dict(value["review"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "base_fingerprint": self.base_fingerprint,
            "plan": self.plan.to_dict(),
            "branch_report": self.branch_report.to_dict(),
            "planner_call": self.planner_call.to_dict(),
            "solver_call": self.solver_call.to_dict(),
            "reviewer_call": self.reviewer_call.to_dict(),
            "review": self.review.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class OrchestrationReport:
    run_id: str
    task: str
    model_fingerprints: Mapping[str, str]
    status: OrchestrationStatus
    phase: OrchestrationPhase
    config: OrchestrationConfig
    planner_config: PlannerConfig
    solver_config: ProposalConfig
    reviewer_config: ReviewerConfig
    search_config: BranchSearchConfig
    execution_config: ExecutableSearchConfig
    root_files: Mapping[str, str]
    base_files: Mapping[str, str]
    rounds: tuple[OrchestrationRound, ...]
    observations: Mapping[str, TestResult]
    events: tuple[SearchEvent, ...]
    model_calls: int
    planner_calls: int
    solver_calls: int
    reviewer_calls: int
    candidate_proposals: int
    test_calls: int
    test_reuses: int
    usage: TokenUsage
    best_candidate_id: str | None
    best_files: Mapping[str, str] | None
    max_in_flight: int = 1
    reason: str | None = None
    pending_plan: PlannerPlan | None = None
    pending_planner_call: RoleCall | None = None
    pending_case: BranchCase | None = None
    pending_solver_call: RoleCall | None = None
    pending_branch_report: BranchSearchReport | None = None
    role_histories: Mapping[str, tuple[AgentMessage, ...]] = field(default_factory=dict)
    feedback: tuple[Mapping[str, Any], ...] = ()
    schema_version: str = "1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _s(self.run_id, "run_id"))
        object.__setattr__(self, "task", _s(self.task, "task"))
        if self.schema_version != "1":
            raise ValueError(
                f"unsupported orchestration schema: {self.schema_version!r}"
            )
        if self.status not in {
            "running",
            "accepted",
            "exhausted",
            "budget_exhausted",
            "paused",
            "failed",
        }:
            raise ValueError(f"unsupported orchestration status: {self.status!r}")
        if self.phase not in {"idle", "planning", "solving", "evaluating", "reviewing"}:
            raise ValueError(f"unsupported orchestration phase: {self.phase!r}")
        if self.status != "running" and self.phase != "idle":
            raise ValueError("non-running orchestration must be idle")
        if set(self.model_fingerprints) != {"planner", "solver", "reviewer"}:
            raise ValueError("model_fingerprints must contain all three roles")
        for fingerprint in self.model_fingerprints.values():
            _s(fingerprint, "model fingerprint")
        for name in (
            "model_calls",
            "planner_calls",
            "solver_calls",
            "reviewer_calls",
            "candidate_proposals",
            "test_calls",
            "test_reuses",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if (
            not isinstance(self.max_in_flight, int)
            or isinstance(self.max_in_flight, bool)
            or self.max_in_flight <= 0
        ):
            raise ValueError("max_in_flight must be a positive integer")
        object.__setattr__(self, "root_files", _files(self.root_files, "root_files"))
        object.__setattr__(self, "base_files", _files(self.base_files, "base_files"))
        if self.best_files is not None:
            object.__setattr__(
                self, "best_files", _files(self.best_files, "best_files")
            )
        rounds = tuple(sorted(self.rounds, key=lambda item: item.index))
        if [item.index for item in rounds] != list(range(len(rounds))):
            raise ValueError("round indexes must be contiguous from zero")
        object.__setattr__(self, "rounds", rounds)
        object.__setattr__(self, "observations", dict(self.observations))
        for fingerprint, result in self.observations.items():
            _s(fingerprint, "observation fingerprint")
            if not isinstance(result, TestResult):
                raise ValueError("observations must map fingerprints to TestResult")
        if not isinstance(self.role_histories, Mapping):
            raise ValueError("role_histories must be an object")
        histories = dict(self.role_histories)
        if not histories:
            histories = {role: () for role in ("planner", "solver", "reviewer")}
        if set(histories) != {"planner", "solver", "reviewer"}:
            raise ValueError(
                "role_histories must contain planner, solver, and reviewer"
            )
        normalized_histories: dict[str, tuple[AgentMessage, ...]] = {}
        for role, messages in histories.items():
            if not isinstance(messages, (tuple, list)):
                raise ValueError(f"{role} role history must be an array")
            normalized = tuple(messages)
            if any(
                not isinstance(message, AgentMessage) or message.role != "assistant"
                for message in normalized
            ):
                raise ValueError(
                    f"{role} role history must contain assistant messages only"
                )
            normalized_histories[role] = normalized
        object.__setattr__(self, "role_histories", normalized_histories)
        verify_search_events(self.events)
        feedback = tuple(dict(item) for item in self.feedback)
        try:
            json.dumps(feedback, ensure_ascii=False, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("feedback must be canonical JSON") from exc
        object.__setattr__(self, "feedback", feedback)
        pending = (
            self.pending_plan,
            self.pending_planner_call,
            self.pending_case,
            self.pending_solver_call,
            self.pending_branch_report,
        )
        if self.phase == "idle" and any(item is not None for item in pending):
            raise ValueError("idle orchestration cannot have pending state")
        if self.phase == "planning" and any(item is not None for item in pending):
            raise ValueError("planning phase cannot have completed state")
        if self.phase == "solving" and (
            self.pending_plan is None
            or self.pending_planner_call is None
            or self.pending_case is not None
            or self.pending_solver_call is not None
            or self.pending_branch_report is not None
        ):
            raise ValueError("solving phase requires only the planner result")
        if self.phase == "evaluating" and (
            self.pending_plan is None
            or self.pending_planner_call is None
            or self.pending_case is None
            or self.pending_solver_call is None
            or self.pending_branch_report is not None
        ):
            raise ValueError("evaluating phase requires a solver case")
        if self.phase == "reviewing" and not all(pending):
            raise ValueError("reviewing phase requires plan, case, and branch report")
        if self.status == "accepted" and (
            self.best_candidate_id is None or self.best_files is None
        ):
            raise ValueError("accepted orchestration must identify its best candidate")

    @property
    def metrics(self) -> Mapping[str, int | float]:
        return {
            "rounds": len(self.rounds),
            "model_calls": self.model_calls,
            "planner_calls": self.planner_calls,
            "solver_calls": self.solver_calls,
            "reviewer_calls": self.reviewer_calls,
            "candidate_proposals": self.candidate_proposals,
            "test_calls": self.test_calls,
            "test_reuses": self.test_reuses,
            "cached_observations": len(self.observations),
            "total_tokens": self.usage.total_tokens,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OrchestrationReport:
        value = _m(data, "orchestration report")
        allowed = {
            "schema_version",
            "run_id",
            "task",
            "model_fingerprints",
            "status",
            "phase",
            "config",
            "planner_config",
            "solver_config",
            "reviewer_config",
            "search_config",
            "execution_config",
            "root_files",
            "base_files",
            "rounds",
            "observations",
            "events",
            "model_calls",
            "planner_calls",
            "solver_calls",
            "reviewer_calls",
            "candidate_proposals",
            "test_calls",
            "test_reuses",
            "usage",
            "metrics",
            "best_candidate_id",
            "best_files",
            "max_in_flight",
            "reason",
            "pending_plan",
            "pending_planner_call",
            "pending_case",
            "pending_solver_call",
            "pending_branch_report",
            "role_histories",
            "feedback",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                f"orchestration report has unknown fields: {sorted(unknown)!r}"
            )
        nested = (
            "config",
            "planner_config",
            "solver_config",
            "reviewer_config",
            "search_config",
            "execution_config",
            "usage",
        )
        if any(not isinstance(value.get(name), Mapping) for name in nested):
            raise ValueError("orchestration configs and usage must be objects")
        raw_rounds, raw_obs, raw_events = (
            value.get("rounds"),
            value.get("observations"),
            value.get("events"),
        )
        if (
            not isinstance(raw_rounds, list)
            or not isinstance(raw_obs, Mapping)
            or not isinstance(raw_events, list)
        ):
            raise ValueError("rounds, observations, and events have invalid shapes")
        model_fingerprints = value.get("model_fingerprints")
        if not isinstance(model_fingerprints, Mapping):
            raise ValueError("model_fingerprints must be an object")
        observations: dict[str, TestResult] = {}
        for fingerprint, raw_result in raw_obs.items():
            if not isinstance(fingerprint, str) or not isinstance(raw_result, Mapping):
                raise ValueError("observations must map strings to objects")
            observations[fingerprint] = TestResult.from_dict(raw_result)

        def optional(name: str) -> Any:
            raw = value.get(name)
            if raw is not None and not isinstance(raw, Mapping):
                raise ValueError(f"{name} must be an object or null")
            return raw

        feedback = value.get("feedback", [])
        if not isinstance(feedback, list) or any(
            not isinstance(item, Mapping) for item in feedback
        ):
            raise ValueError("feedback must be an array of objects")
        raw_histories = value.get("role_histories", {})
        if not isinstance(raw_histories, Mapping):
            raise ValueError("role_histories must be an object")
        role_histories: dict[str, tuple[AgentMessage, ...]] = {}
        for role, raw_messages in raw_histories.items():
            if role not in {"planner", "solver", "reviewer"}:
                raise ValueError(f"role_histories has unknown role: {role!r}")
            if not isinstance(raw_messages, list):
                raise ValueError(f"{role} role history must be an array")
            role_histories[role] = tuple(
                AgentMessage.from_dict(_m(item, f"{role} role history message"))
                for item in raw_messages
            )
        report = cls(
            schema_version=str(value.get("schema_version", "")),
            run_id=_s(value.get("run_id"), "run_id"),
            task=_s(value.get("task"), "task"),
            model_fingerprints=dict(model_fingerprints),
            status=cast(OrchestrationStatus, value.get("status")),
            phase=cast(OrchestrationPhase, value.get("phase")),
            config=OrchestrationConfig.from_dict(value["config"]),
            planner_config=PlannerConfig.from_dict(value["planner_config"]),
            solver_config=ProposalConfig.from_dict(value["solver_config"]),
            reviewer_config=ReviewerConfig.from_dict(value["reviewer_config"]),
            search_config=BranchSearchConfig.from_dict(value["search_config"]),
            execution_config=ExecutableSearchConfig.from_dict(
                value["execution_config"]
            ),
            root_files=_m(value.get("root_files"), "root_files"),
            base_files=_m(value.get("base_files"), "base_files"),
            rounds=tuple(OrchestrationRound.from_dict(item) for item in raw_rounds),
            observations=observations,
            events=tuple(SearchEvent.from_dict(item) for item in raw_events),
            model_calls=int(value.get("model_calls", -1)),
            planner_calls=int(value.get("planner_calls", -1)),
            solver_calls=int(value.get("solver_calls", -1)),
            reviewer_calls=int(value.get("reviewer_calls", -1)),
            candidate_proposals=int(value.get("candidate_proposals", -1)),
            test_calls=int(value.get("test_calls", -1)),
            test_reuses=int(value.get("test_reuses", -1)),
            usage=TokenUsage.from_dict(value["usage"]),
            best_candidate_id=(
                None
                if value.get("best_candidate_id") is None
                else str(value["best_candidate_id"])
            ),
            best_files=(
                None
                if value.get("best_files") is None
                else _m(value.get("best_files"), "best_files")
            ),
            max_in_flight=int(value.get("max_in_flight", 1)),
            reason=None if value.get("reason") is None else str(value["reason"]),
            pending_plan=(
                None
                if optional("pending_plan") is None
                else PlannerPlan.from_dict(optional("pending_plan"))
            ),
            pending_planner_call=(
                None
                if optional("pending_planner_call") is None
                else RoleCall.from_dict(optional("pending_planner_call"))
            ),
            pending_case=(
                None
                if optional("pending_case") is None
                else BranchCase.from_dict(optional("pending_case"))
            ),
            pending_solver_call=(
                None
                if optional("pending_solver_call") is None
                else RoleCall.from_dict(optional("pending_solver_call"))
            ),
            pending_branch_report=(
                None
                if optional("pending_branch_report") is None
                else BranchSearchReport.from_dict(optional("pending_branch_report"))
            ),
            role_histories=role_histories,
            feedback=tuple(feedback),
        )
        metrics = value.get("metrics")
        if metrics is not None and (
            not isinstance(metrics, Mapping) or dict(metrics) != dict(report.metrics)
        ):
            raise ValueError("orchestration metrics are inconsistent")
        return report

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "task": self.task,
            "model_fingerprints": dict(self.model_fingerprints),
            "status": self.status,
            "phase": self.phase,
            "config": self.config.to_dict(),
            "planner_config": self.planner_config.to_dict(),
            "solver_config": self.solver_config.to_dict(),
            "reviewer_config": self.reviewer_config.to_dict(),
            "search_config": self.search_config.to_dict(),
            "execution_config": self.execution_config.to_dict(),
            "root_files": dict(self.root_files),
            "base_files": dict(self.base_files),
            "rounds": [item.to_dict() for item in self.rounds],
            "observations": {
                key: item.to_dict() for key, item in sorted(self.observations.items())
            },
            "events": [item.to_dict() for item in self.events],
            "model_calls": self.model_calls,
            "planner_calls": self.planner_calls,
            "solver_calls": self.solver_calls,
            "reviewer_calls": self.reviewer_calls,
            "candidate_proposals": self.candidate_proposals,
            "test_calls": self.test_calls,
            "test_reuses": self.test_reuses,
            "usage": self.usage.to_dict(),
            "metrics": dict(self.metrics),
            "best_candidate_id": self.best_candidate_id,
            "best_files": None if self.best_files is None else dict(self.best_files),
            "max_in_flight": self.max_in_flight,
            "reason": self.reason,
            "pending_plan": (
                None if self.pending_plan is None else self.pending_plan.to_dict()
            ),
            "pending_planner_call": (
                None
                if self.pending_planner_call is None
                else self.pending_planner_call.to_dict()
            ),
            "pending_case": (
                None if self.pending_case is None else self.pending_case.to_dict()
            ),
            "pending_solver_call": (
                None
                if self.pending_solver_call is None
                else self.pending_solver_call.to_dict()
            ),
            "pending_branch_report": (
                None
                if self.pending_branch_report is None
                else self.pending_branch_report.to_dict()
            ),
            "role_histories": {
                role: [message.to_dict() for message in messages]
                for role, messages in sorted(self.role_histories.items())
            },
            "feedback": [dict(item) for item in self.feedback],
        }


def _event(
    state: OrchestrationReport,
    event_type: str,
    data: Mapping[str, Any],
) -> OrchestrationReport:
    event = SearchEvent.create(
        len(state.events),
        event_type,
        data,
        state.events[-1].sha256 if state.events else None,
    )
    return replace(state, events=(*state.events, event))


def write_orchestration_checkpoint(
    report: OrchestrationReport, path: str | Path
) -> None:
    """Atomically write a validated orchestration checkpoint."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True, indent=2)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except OSError:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def read_orchestration_checkpoint(path: str | Path) -> OrchestrationReport:
    target = Path(path)
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"cannot read orchestration checkpoint {target}: {exc}"
        ) from exc
    if not isinstance(data, Mapping):
        raise ValueError("orchestration checkpoint must be an object")
    return OrchestrationReport.from_dict(data)


def _namespace(case: BranchCase, round_index: int) -> BranchCase:
    prefix = f"round-{round_index}-"
    ids = {item.id: prefix + item.id for item in case.candidates}
    candidates = tuple(
        CandidatePatch(
            id=ids[item.id],
            parent_id="root" if item.parent_id == "root" else ids[item.parent_id],
            hypothesis=item.hypothesis,
            files=item.files,
            evidence=item.evidence,
        )
        for item in case.candidates
    )
    return BranchCase(
        task=case.task,
        root_files=case.root_files,
        candidates=candidates,
        tests={ids[key]: value for key, value in case.tests.items()},
    )


def _feedback(report: BranchSearchReport) -> list[dict[str, Any]]:
    nodes = [node for node in report.nodes if node.id != "root" and node.test_result]
    nodes.sort(key=lambda node: (-node.score, node.depth, node.id))
    result: list[dict[str, Any]] = []
    for node in nodes[:8]:
        observation = node.test_result
        if observation is None:
            continue
        result.append(
            {
                "candidate_id": node.id,
                "hypothesis": node.hypothesis,
                "score": round(node.score, 6),
                "passed_tests": list(observation.passed_tests),
                "failed_tests": list(observation.failed_tests),
                "error": observation.error,
                "output_excerpt": (
                    None
                    if observation.output_excerpt is None
                    else observation.output_excerpt[:1200]
                ),
            }
        )
    return result


def _best(report: BranchSearchReport) -> Any | None:
    nodes = [node for node in report.nodes if node.id != "root"]
    return (
        sorted(nodes, key=lambda node: (-node.score, node.depth, node.id))[0]
        if nodes
        else None
    )


def _next_files(
    report: BranchSearchReport, fallback: Mapping[str, str]
) -> dict[str, str]:
    node = _best(report)
    candidate = None if node is None else report.case.by_id.get(node.id)
    return dict(candidate.files) if candidate is not None else dict(fallback)


def _compile_role_request(
    context_config: ContextCompilerConfig,
    history: Sequence[AgentMessage],
    request: ModelRequest,
    *,
    task: str,
) -> tuple[ModelRequest, ContextReceipt]:
    """Compile a role's prior assistant summaries plus its fresh request.

    The current system/user request remains part of the compiler's mandatory
    protocol surface.  Only provider-independent assistant responses are carried
    across turns, so a checkpoint can reproduce the same memory fingerprint
    without persisting provider-specific hidden state.
    """

    compiler = ContextCompiler(context_config)
    compiled = compile_runtime_context(
        compiler,
        (*tuple(history), *request.messages),
        task=task,
    )
    return replace(request, messages=compiled.messages), compiled.receipt


def _append_role_response(
    state: OrchestrationReport,
    role: str,
    response: ModelResponse,
) -> OrchestrationReport:
    # These three contracts deliberately reject tool calls.  Do not persist an
    # incomplete assistant/tool exchange that the next context compilation could
    # not reproduce as a valid protocol transcript.
    if response.tool_calls:
        return state
    histories = {
        name: tuple(messages) for name, messages in state.role_histories.items()
    }
    histories[role] = (
        *histories[role],
        AgentMessage(
            role="assistant",
            content=response.content,
            tool_calls=response.tool_calls,
        ),
    )
    return replace(state, role_histories=histories)


def _context_event_data(receipt: ContextReceipt) -> dict[str, Any]:
    return {
        "context_config_fingerprint": receipt.config_fingerprint,
        "context_receipt": receipt.to_dict(),
        "compiled_messages_sha256": receipt.messages_sha256,
        "memory_fingerprint": receipt.memory_fingerprint,
        "workspace_generation": receipt.workspace_generation,
    }


def _call(
    role: str,
    model: ModelClient,
    turn: int,
    response: ModelResponse,
    context_receipt: ContextReceipt | None = None,
) -> RoleCall:
    return RoleCall(
        role=role,
        turn=turn,
        model_name=model.name,
        response_sha256=stable_hash(response.to_dict()),
        response_id=response.response_id,
        usage=response.usage,
        context_receipt=context_receipt,
    )


def _error_call(
    role: str,
    model: ModelClient,
    turn: int,
    message: str,
    context_receipt: ContextReceipt | None = None,
) -> RoleCall:
    return RoleCall(
        role=role,
        turn=turn,
        model_name=model.name,
        response_sha256=stable_hash({"role": role, "turn": turn, "error": message}),
        response_id=None,
        usage=TokenUsage(),
        context_receipt=context_receipt,
    )


def _finish(
    state: OrchestrationReport,
    status: OrchestrationStatus,
    reason: str,
) -> OrchestrationReport:
    updated = replace(state, status=status, phase="idle", reason=reason)
    return _event(
        updated,
        "orchestration.completed",
        {
            "status": status,
            "reason": reason,
            "metrics": dict(updated.metrics),
            "best_candidate_id": updated.best_candidate_id,
        },
    )


class OrchestrationRunner:
    """Drive planner, solver, visible tests, and reviewer roles."""

    def __init__(
        self,
        planner: ModelClient,
        solver: ModelClient,
        reviewer: ModelClient,
        execution_config: ExecutableSearchConfig | None = None,
        *,
        config: OrchestrationConfig | None = None,
        planner_config: PlannerConfig | None = None,
        solver_config: ProposalConfig | None = None,
        reviewer_config: ReviewerConfig | None = None,
        search_config: BranchSearchConfig | None = None,
    ) -> None:
        self.planner = planner
        self.solver = solver
        self.reviewer = reviewer
        self.execution_config = execution_config
        self.config = config
        self.planner_config = planner_config
        self.solver_config = solver_config
        self.reviewer_config = reviewer_config
        self.search_config = search_config

    @property
    def models(self) -> Mapping[str, ModelClient]:
        return {
            "planner": self.planner,
            "solver": self.solver,
            "reviewer": self.reviewer,
        }

    def _adopt(self, state: OrchestrationReport) -> None:
        for role, model in self.models.items():
            if model.configuration_fingerprint != state.model_fingerprints[role]:
                raise ValueError(
                    f"{role} model configuration does not match checkpoint"
                )
        for supplied, stored, label in (
            (self.config, state.config, "orchestration"),
            (self.planner_config, state.planner_config, "planner"),
            (self.solver_config, state.solver_config, "solver"),
            (self.reviewer_config, state.reviewer_config, "reviewer"),
            (self.search_config, state.search_config, "search"),
            (self.execution_config, state.execution_config, "execution"),
        ):
            if supplied is not None and supplied.to_dict() != stored.to_dict():
                raise ValueError(f"{label} configuration does not match checkpoint")
        self.config = state.config
        self.planner_config = state.planner_config
        self.solver_config = state.solver_config
        self.reviewer_config = state.reviewer_config
        self.search_config = state.search_config
        self.execution_config = state.execution_config

    def _initial(
        self, task: str, root_files: Mapping[str, str], run_id: str
    ) -> OrchestrationReport:
        if self.execution_config is None:
            raise ValueError("execution_config is required for a new orchestration")
        config = self.config or OrchestrationConfig()
        planner_config = self.planner_config or PlannerConfig()
        solver_config = self.solver_config or ProposalConfig()
        reviewer_config = self.reviewer_config or ReviewerConfig()
        search_config = self.search_config or BranchSearchConfig()
        self.config = config
        self.planner_config = planner_config
        self.solver_config = solver_config
        self.reviewer_config = reviewer_config
        self.search_config = search_config
        normalized = _files(root_files, "root_files")
        state = OrchestrationReport(
            run_id=run_id,
            task=task,
            model_fingerprints={
                role: model.configuration_fingerprint
                for role, model in self.models.items()
            },
            status="running",
            phase="idle",
            config=config,
            planner_config=planner_config,
            solver_config=solver_config,
            reviewer_config=reviewer_config,
            search_config=search_config,
            execution_config=self.execution_config,
            root_files=normalized,
            base_files=normalized,
            rounds=(),
            observations={},
            events=(),
            model_calls=0,
            planner_calls=0,
            solver_calls=0,
            reviewer_calls=0,
            candidate_proposals=0,
            test_calls=0,
            test_reuses=0,
            usage=TokenUsage(),
            best_candidate_id=None,
            best_files=None,
            role_histories={role: () for role in self.models},
        )
        return _event(
            state,
            "orchestration.started",
            {
                "run_id": run_id,
                "task": task,
                "model_fingerprints": dict(state.model_fingerprints),
                "config": config.to_dict(),
                "planner_config": planner_config.to_dict(),
                "solver_config": solver_config.to_dict(),
                "reviewer_config": reviewer_config.to_dict(),
                "search_config": search_config.to_dict(),
                "execution_config": self.execution_config.to_dict(),
                "root_fingerprint": stable_hash(normalized),
            },
        )

    async def _evaluate(
        self,
        state: OrchestrationReport,
        checkpoint: Path | None = None,
    ) -> OrchestrationReport:
        if (
            state.pending_case is None
            or state.pending_plan is None
            or state.pending_planner_call is None
            or state.pending_solver_call is None
            or self.execution_config is None
            or self.search_config is None
        ):
            raise ValueError("evaluating state is incomplete")
        case = state.pending_case
        execution_config = self.execution_config
        search_config = self.search_config
        config = state.config
        observations = dict(state.observations)
        results: dict[str, TestResult] = {}
        reuses = 0
        candidates = tuple(case.candidates)
        scheduled: list[CandidatePatch] = []
        scheduled_fingerprints: set[str] = set()
        remaining_budget = max(0, config.max_test_calls - state.test_calls)
        for candidate in candidates:
            fingerprint = candidate.workspace_fingerprint
            if fingerprint in observations:
                results[candidate.id] = observations[fingerprint]
                reuses += 1
                continue
            if (
                fingerprint not in scheduled_fingerprints
                and len(scheduled) < remaining_budget
            ):
                scheduled.append(candidate)
                scheduled_fingerprints.add(fingerprint)

        working_state = state
        for candidate in scheduled:
            working_state = _event(
                working_state,
                "candidate.requested",
                {
                    "round": len(state.rounds),
                    "candidate_id": candidate.id,
                    "workspace_fingerprint": candidate.workspace_fingerprint,
                    "max_parallel_tests": config.max_parallel_tests,
                },
            )
        if scheduled and checkpoint is not None:
            write_orchestration_checkpoint(working_state, checkpoint)

        semaphore = asyncio.Semaphore(config.max_parallel_tests)
        in_flight = 0
        max_in_flight = working_state.max_in_flight

        async def evaluate_one(candidate: CandidatePatch) -> TestResult:
            nonlocal in_flight, max_in_flight
            async with semaphore:
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
                try:
                    return await asyncio.to_thread(
                        evaluate_candidate,
                        candidate,
                        execution_config,
                    )
                finally:
                    in_flight -= 1

        tasks = [
            asyncio.create_task(evaluate_one(candidate)) for candidate in scheduled
        ]
        actual = 0
        try:
            for candidate, task in zip(scheduled, tasks, strict=True):
                result = await task
                fingerprint = candidate.workspace_fingerprint
                observations[fingerprint] = result
                results[candidate.id] = result
                actual += 1
                working_state = _event(
                    replace(
                        working_state,
                        observations=observations,
                        test_calls=state.test_calls + actual,
                        max_in_flight=max_in_flight,
                    ),
                    "candidate.completed",
                    {
                        "round": len(state.rounds),
                        "candidate_id": candidate.id,
                        "workspace_fingerprint": fingerprint,
                        "behavior_fingerprint": result.behavior_fingerprint,
                        "test_calls": state.test_calls + actual,
                        "max_in_flight": max_in_flight,
                    },
                )
                if checkpoint is not None:
                    write_orchestration_checkpoint(working_state, checkpoint)
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        budget_result = TestResult(
            suite=execution_config.suite,
            error="orchestration test budget exhausted before this candidate ran",
        )
        for candidate in candidates:
            if candidate.id in results:
                continue
            fingerprint = candidate.workspace_fingerprint
            if fingerprint in observations:
                results[candidate.id] = observations[fingerprint]
                reuses += 1
            else:
                results[candidate.id] = budget_result
        observed = BranchCase(
            task=case.task,
            root_files=case.root_files,
            candidates=case.candidates,
            tests=results,
        )
        report = BranchSearch(search_config).run(observed)
        updated = replace(
            working_state,
            phase="reviewing",
            pending_case=observed,
            pending_branch_report=report,
            observations=observations,
            test_calls=working_state.test_calls,
            test_reuses=working_state.test_reuses + reuses,
            max_in_flight=max_in_flight,
        )
        if checkpoint is not None:
            write_orchestration_checkpoint(updated, checkpoint)
        return _event(
            updated,
            "evaluation.completed",
            {
                "round": len(state.rounds),
                "case_fingerprint": observed.fingerprint,
                "status": report.status,
                "metrics": dict(report.metrics),
                "actual_test_calls": actual,
                "test_reuses": reuses,
                "max_in_flight": max_in_flight,
            },
        )

    async def _planner(
        self, state: OrchestrationReport, checkpoint: Path | None
    ) -> OrchestrationReport:
        config = self.planner_config
        if config is None:
            raise ValueError("planner configuration is incomplete")
        turn = state.planner_calls + 1
        receipt: ContextReceipt | None = None
        try:
            request, receipt = _compile_role_request(
                state.config.context_config,
                state.role_histories["planner"],
                build_planner_request(
                    state.task,
                    state.base_files,
                    config,
                    run_id=state.run_id,
                    turn=turn,
                    feedback=state.feedback,
                ),
                task=state.task,
            )
        except (RuntimeContractError, ValueError) as exc:
            message = f"{type(exc).__name__}: {str(exc)[:800]}"
            updated = _event(
                replace(
                    state,
                    phase="idle",
                    model_calls=state.model_calls + 1,
                    planner_calls=state.planner_calls + 1,
                    feedback=tuple(
                        [
                            *state.feedback,
                            {"planner_rejected": message, "round": len(state.rounds)},
                        ][-16:]
                    ),
                ),
                "planner.rejected",
                {"round": len(state.rounds), "turn": turn, "error": message},
            )
            if (
                updated.model_calls >= updated.config.max_model_calls
                or updated.planner_calls >= updated.config.max_planner_calls
            ):
                updated = _finish(
                    updated,
                    "failed",
                    "planner contract failed until its budget was exhausted",
                )
            if checkpoint is not None:
                write_orchestration_checkpoint(updated, checkpoint)
            return updated
        state = _event(
            replace(state, phase="planning", reason=None),
            "planner.requested",
            {
                "round": len(state.rounds),
                "turn": turn,
                "base_fingerprint": stable_hash(state.base_files),
                "feedback_count": len(state.feedback),
                **({} if receipt is None else _context_event_data(receipt)),
            },
        )
        if checkpoint is not None:
            write_orchestration_checkpoint(state, checkpoint)
        try:
            response = await self.planner.complete(request)
            state = _append_role_response(state, "planner", response)
            plan = parse_planner_response(response, config)
        except (ModelError, RuntimeContractError, ValueError) as exc:
            message = f"{type(exc).__name__}: {str(exc)[:800]}"
            updated = _event(
                replace(
                    state,
                    phase="idle",
                    model_calls=state.model_calls + 1,
                    planner_calls=state.planner_calls + 1,
                    feedback=tuple(
                        [
                            *state.feedback,
                            {"planner_rejected": message, "round": len(state.rounds)},
                        ][-16:]
                    ),
                ),
                "planner.rejected",
                {"round": len(state.rounds), "turn": turn, "error": message},
            )
            if (
                updated.model_calls >= updated.config.max_model_calls
                or updated.planner_calls >= updated.config.max_planner_calls
            ):
                updated = _finish(
                    updated,
                    "failed",
                    "planner contract failed until its budget was exhausted",
                )
            if checkpoint is not None:
                write_orchestration_checkpoint(updated, checkpoint)
            return updated
        call = _call("planner", self.planner, turn, response, receipt)
        updated = _event(
            replace(
                state,
                phase="solving",
                model_calls=state.model_calls + 1,
                planner_calls=state.planner_calls + 1,
                usage=state.usage + response.usage,
                pending_plan=plan,
                pending_planner_call=call,
            ),
            "planner.received",
            {
                "round": len(state.rounds),
                "turn": turn,
                "model_name": self.planner.name,
                "response_sha256": call.response_sha256,
                "response_id": call.response_id,
                "usage": response.usage.to_dict(),
                "plan_fingerprint": stable_hash(plan.to_dict()),
                **({} if receipt is None else _context_event_data(receipt)),
            },
        )
        if checkpoint is not None:
            write_orchestration_checkpoint(updated, checkpoint)
        return updated

    async def _solver(
        self, state: OrchestrationReport, checkpoint: Path | None
    ) -> OrchestrationReport:
        config = self.solver_config
        if (
            config is None
            or state.pending_plan is None
            or state.pending_planner_call is None
        ):
            raise ValueError("solver state is incomplete")
        plan = state.pending_plan
        remaining = state.config.max_candidates - state.candidate_proposals
        if remaining <= 0:
            return _finish(
                state, "budget_exhausted", "candidate proposal budget exhausted"
            )
        config = replace(config, max_candidates=min(config.max_candidates, remaining))
        turn = state.solver_calls + 1
        receipt: ContextReceipt | None = None
        try:
            request, receipt = _compile_role_request(
                state.config.context_config,
                state.role_histories["solver"],
                build_solver_request(
                    state.task,
                    state.base_files,
                    plan,
                    config,
                    run_id=state.run_id,
                    turn=turn,
                    feedback=state.feedback,
                ),
                task=state.task,
            )
        except (RuntimeContractError, ValueError) as exc:
            message = f"{type(exc).__name__}: {str(exc)[:800]}"
            updated = _event(
                replace(
                    state,
                    phase="idle",
                    model_calls=state.model_calls + 1,
                    solver_calls=state.solver_calls + 1,
                    pending_plan=None,
                    pending_planner_call=None,
                    feedback=tuple(
                        [
                            *state.feedback,
                            {"solver_rejected": message, "round": len(state.rounds)},
                        ][-16:]
                    ),
                ),
                "solver.rejected",
                {"round": len(state.rounds), "turn": turn, "error": message},
            )
            if (
                updated.model_calls >= updated.config.max_model_calls
                or updated.solver_calls >= updated.config.max_solver_calls
            ):
                updated = _finish(
                    updated,
                    "failed",
                    "solver contract failed until its budget was exhausted",
                )
            if checkpoint is not None:
                write_orchestration_checkpoint(updated, checkpoint)
            return updated
        state = _event(
            replace(state, phase="solving", reason=None),
            "solver.requested",
            {
                "round": len(state.rounds),
                "turn": turn,
                "base_fingerprint": stable_hash(state.base_files),
                "proposal_config": config.to_dict(),
                "plan_fingerprint": stable_hash(plan.to_dict()),
                **({} if receipt is None else _context_event_data(receipt)),
            },
        )
        if checkpoint is not None:
            write_orchestration_checkpoint(state, checkpoint)
        try:
            response = await self.solver.complete(request)
            state = _append_role_response(state, "solver", response)
            case = parse_proposal_response(
                response, state.task, state.base_files, config
            )
        except (ModelError, RuntimeContractError, ValueError) as exc:
            message = f"{type(exc).__name__}: {str(exc)[:800]}"
            updated = _event(
                replace(
                    state,
                    phase="idle",
                    model_calls=state.model_calls + 1,
                    solver_calls=state.solver_calls + 1,
                    pending_plan=None,
                    pending_planner_call=None,
                    feedback=tuple(
                        [
                            *state.feedback,
                            {"solver_rejected": message, "round": len(state.rounds)},
                        ][-16:]
                    ),
                ),
                "solver.rejected",
                {"round": len(state.rounds), "turn": turn, "error": message},
            )
            if (
                updated.model_calls >= updated.config.max_model_calls
                or updated.solver_calls >= updated.config.max_solver_calls
            ):
                updated = _finish(
                    updated,
                    "failed",
                    "solver contract failed until its budget was exhausted",
                )
            if checkpoint is not None:
                write_orchestration_checkpoint(updated, checkpoint)
            return updated
        namespaced = _namespace(case, len(state.rounds))
        call = _call("solver", self.solver, turn, response, receipt)
        updated = _event(
            replace(
                state,
                phase="evaluating",
                model_calls=state.model_calls + 1,
                solver_calls=state.solver_calls + 1,
                candidate_proposals=state.candidate_proposals
                + len(namespaced.candidates),
                usage=state.usage + response.usage,
                pending_case=namespaced,
                pending_solver_call=call,
            ),
            "solver.received",
            {
                "round": len(state.rounds),
                "turn": turn,
                "model_name": self.solver.name,
                "response_sha256": call.response_sha256,
                "response_id": call.response_id,
                "usage": response.usage.to_dict(),
                "case_fingerprint": namespaced.fingerprint,
                "candidate_count": len(namespaced.candidates),
                **({} if receipt is None else _context_event_data(receipt)),
            },
        )
        if checkpoint is not None:
            write_orchestration_checkpoint(updated, checkpoint)
        return updated

    async def _reviewer(
        self, state: OrchestrationReport, checkpoint: Path | None
    ) -> OrchestrationReport:
        config = self.reviewer_config
        if config is None or not all(
            (
                state.pending_plan,
                state.pending_planner_call,
                state.pending_case,
                state.pending_solver_call,
                state.pending_branch_report,
            )
        ):
            raise ValueError("review state is incomplete")
        plan = cast(PlannerPlan, state.pending_plan)
        branch_report = cast(BranchSearchReport, state.pending_branch_report)
        turn = state.reviewer_calls + 1
        receipt: ContextReceipt | None = None
        request: ModelRequest | None = None
        preparation_error: str | None = None
        try:
            request, receipt = _compile_role_request(
                state.config.context_config,
                state.role_histories["reviewer"],
                build_reviewer_request(
                    state.task,
                    plan,
                    branch_report,
                    config,
                    run_id=state.run_id,
                    turn=turn,
                    feedback=state.feedback,
                ),
                task=state.task,
            )
        except (RuntimeContractError, ValueError) as exc:
            preparation_error = f"{type(exc).__name__}: {str(exc)[:800]}"
        state = _event(
            replace(state, phase="reviewing", reason=None),
            "reviewer.requested",
            {
                "round": len(state.rounds),
                "turn": turn,
                "branch_status": branch_report.status,
                "best_node_id": branch_report.best_node_id,
                **({} if receipt is None else _context_event_data(receipt)),
            },
        )
        if checkpoint is not None:
            write_orchestration_checkpoint(state, checkpoint)
        response: ModelResponse | None = None
        try:
            if preparation_error is not None or request is None:
                raise ValueError(preparation_error or "reviewer request is unavailable")
            response = await self.reviewer.complete(request)
            state = _append_role_response(state, "reviewer", response)
            decision = parse_reviewer_response(response, config)
            call = _call("reviewer", self.reviewer, turn, response, receipt)
        except (ModelError, RuntimeContractError, ValueError) as exc:
            message = f"{type(exc).__name__}: {str(exc)[:800]}"
            decision = ReviewDecision(
                "retry",
                None,
                0.0,
                (message,),
                ("repeat reviewer contract with strict JSON",),
                "reviewer contract failed; continue conservatively",
            )
            call = _error_call("reviewer", self.reviewer, turn, message, receipt)
        best_node = _best(branch_report)
        best_id, best_files = state.best_candidate_id, state.best_files
        previous_scores = [
            node.score
            for item in state.rounds
            for node in item.branch_report.nodes
            if node.id != "root"
        ]
        if best_node is not None and (
            best_files is None or best_node.score >= max(previous_scores, default=-1.0)
        ):
            candidate = branch_report.case.by_id.get(best_node.id)
            if candidate is not None:
                best_id, best_files = best_node.id, dict(candidate.files)
        accepted = False
        if decision.decision == "accept" and decision.candidate_id is not None:
            candidate = branch_report.case.by_id.get(decision.candidate_id)
            result = (
                None
                if candidate is None
                else branch_report.case.tests.get(candidate.id)
            )
            if candidate is not None and result is not None and result.is_success:
                accepted, best_id, best_files = (
                    True,
                    candidate.id,
                    dict(candidate.files),
                )
            else:
                state = _event(
                    state,
                    "reviewer.gated",
                    {
                        "round": len(state.rounds),
                        "decision": decision.to_dict(),
                        "reason": (
                            "acceptance requires a visible-test-passing candidate"
                        ),
                    },
                )
        feedback = tuple(
            [
                *state.feedback[-8:],
                *_feedback(branch_report),
                {"review": decision.to_dict(), "round": len(state.rounds)},
            ][-16:]
        )
        pending_case = cast(BranchCase, state.pending_case)
        item = OrchestrationRound(
            index=len(state.rounds),
            base_fingerprint=stable_hash(dict(pending_case.root_files)),
            plan=plan,
            branch_report=branch_report,
            planner_call=cast(RoleCall, state.pending_planner_call),
            solver_call=cast(RoleCall, state.pending_solver_call),
            reviewer_call=call,
            review=decision,
        )
        updated = _event(
            replace(
                state,
                status="accepted" if accepted else "running",
                phase="idle",
                rounds=(*state.rounds, item),
                best_candidate_id=best_id,
                best_files=best_files,
                base_files=_next_files(branch_report, state.base_files),
                reason=(
                    "reviewer accepted a visible-test-passing candidate"
                    if accepted
                    else None
                ),
                feedback=feedback,
                pending_plan=None,
                pending_planner_call=None,
                pending_case=None,
                pending_solver_call=None,
                pending_branch_report=None,
                model_calls=state.model_calls + 1,
                reviewer_calls=state.reviewer_calls + 1,
                usage=state.usage
                + (TokenUsage() if response is None else response.usage),
            ),
            "reviewer.received" if response is not None else "reviewer.rejected",
            {
                "round": item.index,
                "turn": turn,
                "model_name": self.reviewer.name,
                "response_sha256": call.response_sha256,
                "response_id": call.response_id,
                "usage": call.usage.to_dict(),
                "decision": decision.to_dict(),
                "oracle_gate": accepted,
                **({} if receipt is None else _context_event_data(receipt)),
            },
        )
        if updated.status == "running" and (
            len(updated.rounds) >= updated.config.max_rounds
            or updated.model_calls >= updated.config.max_model_calls
            or updated.planner_calls >= updated.config.max_planner_calls
            or updated.solver_calls >= updated.config.max_solver_calls
            or updated.reviewer_calls >= updated.config.max_reviewer_calls
            or updated.candidate_proposals >= updated.config.max_candidates
            or updated.test_calls >= updated.config.max_test_calls
            or updated.usage.total_tokens >= updated.config.max_total_tokens
        ):
            updated = _finish(
                updated,
                "budget_exhausted",
                "shared orchestration budget exhausted before reviewer acceptance",
            )
        if checkpoint is not None:
            write_orchestration_checkpoint(updated, checkpoint)
        return updated

    async def _drive(
        self,
        state: OrchestrationReport,
        checkpoint: Path | None,
        retry_pending: bool,
    ) -> OrchestrationReport:
        if not all(
            (
                self.config,
                self.planner_config,
                self.solver_config,
                self.reviewer_config,
                self.search_config,
            )
        ):
            raise ValueError("orchestration configuration is incomplete")
        orchestration_config = self.config
        if orchestration_config is None:
            raise ValueError("orchestration configuration is incomplete")
        while state.status == "running":
            if state.phase == "evaluating":
                state = await self._evaluate(state, checkpoint)
                if checkpoint is not None:
                    write_orchestration_checkpoint(state, checkpoint)
                continue
            if state.phase == "reviewing":
                state = await self._reviewer(state, checkpoint)
                continue
            if state.phase == "planning" and not retry_pending:
                state = _event(
                    replace(
                        state,
                        status="paused",
                        phase="idle",
                        reason=(
                            "a role model request was pending; "
                            "resume with retry_pending"
                        ),
                    ),
                    "orchestration.paused",
                    {"reason": "pending role model request requires explicit retry"},
                )
                if checkpoint is not None:
                    write_orchestration_checkpoint(state, checkpoint)
                return state
            if len(state.rounds) >= orchestration_config.max_rounds:
                state = _finish(
                    state,
                    "budget_exhausted",
                    "maximum orchestration rounds reached",
                )
                if checkpoint is not None:
                    write_orchestration_checkpoint(state, checkpoint)
                return state
            if (
                state.model_calls >= orchestration_config.max_model_calls
                or state.usage.total_tokens >= orchestration_config.max_total_tokens
            ):
                state = _finish(
                    state, "budget_exhausted", "shared model budget reached"
                )
                if checkpoint is not None:
                    write_orchestration_checkpoint(state, checkpoint)
                return state
            state = (
                await self._solver(state, checkpoint)
                if state.phase == "solving"
                else await self._planner(state, checkpoint)
            )
            retry_pending = False
        return state

    async def run(
        self,
        *,
        task: str | None = None,
        root_files: Mapping[str, str] | None = None,
        run_id: str = "multi-agent-orchestration",
        checkpoint_path: str | Path | None = None,
        resume: bool = False,
        retry_pending: bool = False,
    ) -> OrchestrationReport:
        checkpoint = None if checkpoint_path is None else Path(checkpoint_path)
        if resume:
            if checkpoint is None:
                raise ValueError("checkpoint_path is required when resuming")
            state = read_orchestration_checkpoint(checkpoint)
            self._adopt(state)
            for role, model in self.models.items():
                model.resume_from_turn(getattr(state, f"{role}_calls"))
            if state.status != "running":
                return state
        else:
            if task is None or root_files is None:
                raise ValueError(
                    "task and root_files are required for a new orchestration"
                )
            state = self._initial(task, root_files, run_id)
            if checkpoint is not None:
                write_orchestration_checkpoint(state, checkpoint)
        return await self._drive(state, checkpoint, retry_pending)


async def run_orchestration(
    planner: ModelClient,
    solver: ModelClient,
    reviewer: ModelClient,
    *,
    task: str | None = None,
    root_files: Mapping[str, str] | None = None,
    execution_config: ExecutableSearchConfig | None = None,
    config: OrchestrationConfig | None = None,
    planner_config: PlannerConfig | None = None,
    solver_config: ProposalConfig | None = None,
    reviewer_config: ReviewerConfig | None = None,
    search_config: BranchSearchConfig | None = None,
    run_id: str = "multi-agent-orchestration",
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
    retry_pending: bool = False,
) -> OrchestrationReport:
    runner = OrchestrationRunner(
        planner,
        solver,
        reviewer,
        execution_config,
        config=config,
        planner_config=planner_config,
        solver_config=solver_config,
        reviewer_config=reviewer_config,
        search_config=search_config,
    )
    return await runner.run(
        task=task,
        root_files=root_files,
        run_id=run_id,
        checkpoint_path=checkpoint_path,
        resume=resume,
        retry_pending=retry_pending,
    )


def render_orchestration_console(report: OrchestrationReport) -> str:
    completed_calls = tuple(
        call
        for item in report.rounds
        for call in (item.planner_call, item.solver_call, item.reviewer_call)
    )
    context_receipts = sum(call.context_receipt is not None for call in completed_calls)
    lines = [
        f"status={report.status} phase={report.phase} rounds={len(report.rounds)} "
        f"model_calls={report.model_calls} planner={report.planner_calls} "
        f"solver={report.solver_calls} reviewer={report.reviewer_calls} "
        f"test_calls={report.test_calls} reuses={report.test_reuses} "
        f"max_in_flight={report.max_in_flight} context_receipts={context_receipts}"
    ]
    if report.best_candidate_id:
        lines.append(f"best_candidate={report.best_candidate_id}")
    if report.reason:
        lines.append(f"reason={report.reason}")
    lines.extend(
        f"round {item.index}: branch={item.branch_report.status} "
        f"review={item.review.decision} confidence={item.review.confidence:.2f} "
        f"best={item.branch_report.best_node_id or 'none'}"
        for item in report.rounds
    )
    return "\n".join(lines) + "\n"


def render_orchestration_markdown(report: OrchestrationReport) -> str:
    completed_calls = tuple(
        call
        for item in report.rounds
        for call in (item.planner_call, item.solver_call, item.reviewer_call)
    )
    context_receipts = sum(call.context_receipt is not None for call in completed_calls)
    lines = [
        "# ContextOpt planner / solver / reviewer orchestration",
        "",
        f"- Status: {report.status}",
        f"- Task: {report.task}",
        f"- Rounds: {len(report.rounds)}",
        f"- Model calls: {report.model_calls} "
        f"(planner {report.planner_calls}, solver {report.solver_calls}, "
        f"reviewer {report.reviewer_calls})",
        f"- Actual test calls: {report.test_calls}; "
        f"cached reuses: {report.test_reuses}",
        f"- Scheduler: max configured parallel tests "
        f"`{report.config.max_parallel_tests}`; "
        f"observed max in-flight `{report.max_in_flight}`",
        f"- Context receipts: {context_receipts}/{len(completed_calls)} "
        "(selected message blocks and observed-memory fingerprints)",
        f"- Best candidate: {report.best_candidate_id or 'none'}",
        "",
        "| Round | Branch | Reviewer | Confidence | Candidate | Test calls |",
        "|---:|---|---|---:|---|---:|",
    ]
    lines.extend(
        f"| {item.index} | {item.branch_report.status} | {item.review.decision} | "
        f"{item.review.confidence:.2f} | {item.review.candidate_id or 'none'} | "
        f"{item.branch_report.metrics.get('test_calls', 0)} |"
        for item in report.rounds
    )
    if report.reason:
        lines.extend(("", f"> {report.reason}"))
    lines.extend(
        (
            "",
            "> Acceptance requires reviewer approval and a "
            "visible-test-passing candidate.",
        )
    )
    return "\n".join(lines) + "\n"


def render_orchestration_html(report: OrchestrationReport) -> str:
    """Render a dependency-free role timeline for portfolio/demo inspection."""

    rows: list[str] = []
    for item in report.rounds:
        context_blocks = [
            len(call.context_receipt.selected_block_ids)
            if call.context_receipt is not None
            else 0
            for call in (item.planner_call, item.solver_call, item.reviewer_call)
        ]
        rows.append(
            "<tr>"
            f"<td>{item.index}</td>"
            f"<td>{escape(item.planner_call.model_name)}</td>"
            f"<td>{escape(item.solver_call.model_name)}</td>"
            f"<td>{escape(item.reviewer_call.model_name)}</td>"
            f"<td>{escape(item.branch_report.status)}</td>"
            f"<td>{escape(item.review.decision)}</td>"
            f"<td>{item.review.confidence:.2f}</td>"
            f"<td>{item.branch_report.metrics.get('test_calls', 0)}</td>"
            f"<td>{','.join(str(value) for value in context_blocks)}</td>"
            "</tr>"
        )
    if not rows:
        rows.append("<tr><td colspan='9'>no completed rounds</td></tr>")
    payload = escape(json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True))
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>ContextOpt orchestration report</title>"
        "<style>body{font:14px system-ui,sans-serif;margin:2rem;color:#172033}"
        "h1{margin-bottom:.25rem}.metrics{display:flex;gap:.75rem;flex-wrap:wrap}"
        ".metric{background:#eef2ff;border-radius:8px;padding:.7rem 1rem}"
        "table{border-collapse:collapse;margin-top:1.5rem;width:100%}"
        "th,td{border:1px solid #d7dce8;padding:.55rem;text-align:left}"
        "th{background:#f5f7fb}code{white-space:pre-wrap}</style></head><body>"
        "<h1>Planner / solver / reviewer orchestration</h1>"
        f"<p>Status: <strong>{escape(report.status)}</strong> · "
        f"Task: {escape(report.task)}</p>"
        "<div class='metrics'>"
        f"<div class='metric'>model calls: {report.model_calls}</div>"
        f"<div class='metric'>tests: {report.test_calls}</div>"
        f"<div class='metric'>reuses: {report.test_reuses}</div>"
        f"<div class='metric'>best: {escape(report.best_candidate_id or 'none')}</div>"
        "</div><table><thead><tr><th>round</th><th>planner</th><th>solver</th>"
        "<th>reviewer</th><th>branch</th><th>review</th><th>confidence</th>"
        f"<th>test calls</th><th>context blocks (p/s/r)</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        "<details><summary>durable report JSON</summary><code>"
        f"{payload}</code></details>"
        "</body></html>\n"
    )


__all__ = [
    "OrchestrationConfig",
    "OrchestrationPhase",
    "OrchestrationReport",
    "OrchestrationRound",
    "OrchestrationRunner",
    "OrchestrationStatus",
    "PlannerConfig",
    "PlannerPlan",
    "ReviewDecision",
    "ReviewerConfig",
    "RoleCall",
    "build_planner_request",
    "build_reviewer_request",
    "build_solver_request",
    "parse_planner_response",
    "parse_reviewer_response",
    "plan_task",
    "read_orchestration_checkpoint",
    "render_orchestration_console",
    "render_orchestration_html",
    "render_orchestration_markdown",
    "review_branch",
    "run_orchestration",
    "solve_plan",
    "write_orchestration_checkpoint",
]
