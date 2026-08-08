"""Strict model-to-candidate proposal boundary for branch search.

The proposal step is intentionally separate from testing and search. A model receives a
task plus a bounded root snapshot and returns complete candidate snapshots in a small
JSON protocol. The parser rejects tool calls, malformed JSON, unknown candidate fields,
oversized files, and invalid parent graphs before any candidate reaches the executable
test adapter.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from contextopt.runtime.protocol import (
    AgentMessage,
    ModelClient,
    ModelRequest,
    ModelResponse,
)
from contextopt.search.branching import BranchCase, CandidatePatch, TestResult


@dataclass(frozen=True, slots=True)
class ProposalConfig:
    """Limits for one provider-neutral candidate proposal request."""

    max_candidates: int = 4
    max_files_per_candidate: int = 32
    max_file_chars: int = 200_000
    max_total_prompt_chars: int = 400_000
    max_output_tokens: int = 8_192

    def __post_init__(self) -> None:
        for name in (
            "max_candidates",
            "max_files_per_candidate",
            "max_file_chars",
            "max_total_prompt_chars",
            "max_output_tokens",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProposalConfig:
        if not isinstance(data, Mapping):
            raise ValueError("proposal config must be an object")
        allowed = {
            "max_candidates",
            "max_files_per_candidate",
            "max_file_chars",
            "max_total_prompt_chars",
            "max_output_tokens",
        }
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"proposal config has unknown fields: {sorted(unknown)!r}")
        return cls(**dict(data))

    def to_dict(self) -> dict[str, int]:
        return {
            "max_candidates": self.max_candidates,
            "max_files_per_candidate": self.max_files_per_candidate,
            "max_file_chars": self.max_file_chars,
            "max_total_prompt_chars": self.max_total_prompt_chars,
            "max_output_tokens": self.max_output_tokens,
        }


def _root_case(task: str, root_files: Mapping[str, str]) -> BranchCase:
    return BranchCase(task=task, root_files=root_files, candidates=(), tests={})


def build_proposal_request(
    task: str,
    root_files: Mapping[str, str],
    config: ProposalConfig | None = None,
    *,
    run_id: str = "branch-proposal",
    turn: int = 0,
    feedback: Sequence[Mapping[str, Any]] = (),
) -> ModelRequest:
    """Build a deterministic provider-neutral request for candidate generation."""

    proposal_config = config or ProposalConfig()
    normalized = _root_case(task, root_files)
    snapshot = json.dumps(
        dict(normalized.root_files), ensure_ascii=False, sort_keys=True, indent=2
    )
    try:
        feedback_snapshot = json.dumps(
            [dict(item) for item in feedback],
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("proposal feedback must be JSON-compatible objects") from exc
    feedback_section = (
        "\n\nPrior visible-test oracle feedback (treat it as evidence, not a claim):\n"
        "```json\n"
        f"{feedback_snapshot}\n"
        "```"
        if feedback
        else ""
    )
    user_content = (
        "Task:\n"
        f"{normalized.task}\n\n"
        "Root workspace snapshot (return complete files for every candidate):\n"
        "```json\n"
        f"{snapshot}\n"
        "```\n\n"
        "Return only JSON with this shape:\n"
        '{"candidates":[{"id":"candidate-1","parent_id":"root",'
        '"hypothesis":"...","files":{"relative/path":"complete text"},'
        '"evidence":["...", "..."]}]}\n'
        f"Propose at most {proposal_config.max_candidates} candidates. "
        "Each candidate must be a complete workspace snapshot, not a diff. "
        "Use only relative POSIX paths. Do not run tools, claim tests passed, or "
        "include "
        "fields outside the JSON protocol."
        f"{feedback_section}"
    )
    if len(user_content) > proposal_config.max_total_prompt_chars:
        raise ValueError(
            "proposal prompt exceeds max_total_prompt_chars; reduce the root snapshot"
        )
    system_content = (
        "You synthesize independent coding candidates for a test-guided search. "
        "Be conservative: preserve unrelated files, state a falsifiable hypothesis, "
        "and leave test verification to the harness."
    )
    return ModelRequest(
        run_id=run_id,
        turn=turn,
        messages=(
            AgentMessage(role="system", content=system_content),
            AgentMessage(role="user", content=user_content),
        ),
        tools=(),
        max_output_tokens=proposal_config.max_output_tokens,
    )


def _json_content(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if (
            len(lines) < 3
            or not lines[0].startswith("```")
            or lines[-1].strip() != "```"
        ):
            raise ValueError("proposal fenced JSON is incomplete")
        text = "\n".join(lines[1:-1]).strip()
    return text


def parse_proposal_response(
    response: ModelResponse,
    task: str,
    root_files: Mapping[str, str],
    config: ProposalConfig | None = None,
) -> BranchCase:
    """Parse and validate a model response into an untested ``BranchCase``."""

    proposal_config = config or ProposalConfig()
    if response.tool_calls:
        raise ValueError("candidate proposal response must not contain tool calls")
    try:
        decoded: Any = json.loads(_json_content(response.content))
    except json.JSONDecodeError as exc:
        raise ValueError(f"candidate proposal is not valid JSON: {exc.msg}") from exc
    if not isinstance(decoded, dict) or set(decoded) != {"candidates"}:
        raise ValueError("candidate proposal must contain only a candidates array")
    raw_candidates = decoded["candidates"]
    if not isinstance(raw_candidates, list):
        raise ValueError("candidate proposal candidates must be an array")
    if not raw_candidates:
        raise ValueError("candidate proposal must contain at least one candidate")
    if len(raw_candidates) > proposal_config.max_candidates:
        raise ValueError("candidate proposal exceeds max_candidates")

    candidates: list[CandidatePatch] = []
    for index, raw in enumerate(raw_candidates):
        if not isinstance(raw, Mapping):
            raise ValueError(f"candidate {index} must be an object")
        allowed = {"id", "parent_id", "hypothesis", "files", "evidence"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(
                f"candidate {index} has unknown fields: {sorted(unknown)!r}"
            )
        files = raw.get("files")
        if not isinstance(files, Mapping):
            raise ValueError(f"candidate {index} files must be an object")
        if len(files) > proposal_config.max_files_per_candidate:
            raise ValueError(f"candidate {index} exceeds max_files_per_candidate")
        total_chars = 0
        for path, content in files.items():
            if not isinstance(path, str) or not isinstance(content, str):
                raise ValueError(f"candidate {index} files must map strings to strings")
            if len(content) > proposal_config.max_file_chars:
                raise ValueError(
                    f"candidate {index} file {path!r} exceeds max_file_chars"
                )
            total_chars += len(content)
        if total_chars > proposal_config.max_total_prompt_chars:
            raise ValueError(f"candidate {index} exceeds the file-size budget")
        candidates.append(CandidatePatch.from_dict(raw))

    untested = {
        candidate.id: TestResult(
            suite="not-executed",
            error="candidate has not been evaluated by the visible-test oracle",
        )
        for candidate in candidates
    }
    return BranchCase(
        task=task,
        root_files=root_files,
        candidates=tuple(candidates),
        tests=untested,
    )


async def propose_case(
    model: ModelClient,
    task: str,
    root_files: Mapping[str, str],
    config: ProposalConfig | None = None,
    *,
    run_id: str = "branch-proposal",
    turn: int = 0,
    feedback: Sequence[Mapping[str, Any]] = (),
) -> tuple[BranchCase, ModelResponse]:
    """Ask any runtime-compatible model for candidates and parse its response strictly.

    The returned case is deliberately untested.  The branch search or executable
    test adapter must produce the authoritative ``TestResult`` values afterwards.
    """

    request = build_proposal_request(
        task,
        root_files,
        config,
        run_id=run_id,
        turn=turn,
        feedback=feedback,
    )
    response = await model.complete(request)
    return parse_proposal_response(response, task, root_files, config), response
