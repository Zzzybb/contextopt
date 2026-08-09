"""Deterministic, protocol-safe context compilation for model requests.

The compiler treats an assistant tool call and all of its tool results as one
atomic selection unit.  This prevents a context policy from emitting provider-
invalid histories such as a tool result without the call that produced it.

Token counts in this module are deliberately estimates.  They are stable across
machines and replays, but they are not a substitute for a provider tokenizer.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, cast

from contextopt.models import (
    ContextFrame,
    ContextItem,
    SelectionDecision,
    SelectionProblem,
)
from contextopt.policies import create_policy
from contextopt.runtime.identity import stable_hash
from contextopt.runtime.memory import MemorySnapshot
from contextopt.runtime.protocol import AgentMessage
from contextopt.runtime.semantic_memory import (
    SemanticMemoryEntry,
    SemanticMemoryMatch,
    SemanticMemoryStore,
)

ContextPolicyName = Literal["full", "recent", "topk", "density", "submodular"]
MemoryPolicyName = Literal["none", "versioned-v1", "versioned-v1+semantic"]
# Context ABI: bump this whenever token estimation, block grouping, compaction,
# scoring, selection, or runtime-memory annotation semantics change. A future bump
# must retain a versioned reader/compiler or provide an explicit trace migration.
CONTEXT_COMPILER_VERSION = 1
MIN_TOOL_OUTPUT_TOKENS = 48

_POLICIES = frozenset({"full", "recent", "topk", "density", "submodular"})
_MEMORY_POLICIES = frozenset({"none", "versioned-v1", "versioned-v1+semantic"})
_DURABLE_MEMORY_POLICY = "versioned-v1+semantic"
_DURABLE_MEMORY_LIMIT = 3
_WORD_RE = re.compile(r"[A-Za-z0-9_]+|[\u3400-\u4dbf\u4e00-\u9fff]|[^\s]")
_TOPIC_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.\-/]{2,}|[\u3400-\u9fff]{2,}")
_SPACE_RE = re.compile(r"\s+")
_STOP_TOPICS = frozenset(
    {
        "and",
        "assistant",
        "content",
        "from",
        "have",
        "message",
        "that",
        "the",
        "this",
        "tool",
        "user",
        "with",
    }
)


class ContextBudgetError(ValueError):
    """Raised when protocol-mandatory context cannot fit the configured budget."""


def _strict_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _strict_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _strict_string(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        qualifier = "string" if allow_empty else "non-empty string"
        raise ValueError(f"{label} must be a {qualifier}")
    return value


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be an array of strings")
    return tuple(value)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def estimate_text_tokens(text: str) -> int:
    """Return a deterministic language-neutral token estimate for *text*.

    Runs of ASCII letters/digits are estimated at four characters per token;
    CJK characters and punctuation count as one token.  Whitespace is free.  The
    estimator is intentionally simple so a crash recovery process can reproduce
    a request without depending on a provider package or tokenizer version.
    """

    estimate = 0
    for piece in _WORD_RE.findall(text):
        if piece.isascii() and (piece[0].isalnum() or piece[0] == "_"):
            estimate += math.ceil(len(piece) / 4)
        else:
            estimate += 1
    return estimate


def estimate_message_tokens(message: AgentMessage) -> int:
    """Estimate one provider-neutral message, including stable metadata overhead."""

    tokens = 4 + estimate_text_tokens(message.content)
    if message.tool_call_id is not None:
        tokens += estimate_text_tokens(message.tool_call_id)
    if message.tool_name is not None:
        tokens += estimate_text_tokens(message.tool_name)
    for call in message.tool_calls:
        tokens += 4
        tokens += estimate_text_tokens(call.id)
        tokens += estimate_text_tokens(call.name)
        tokens += estimate_text_tokens(call.arguments_json)
    return max(1, tokens)


def estimate_messages_tokens(messages: Sequence[AgentMessage]) -> int:
    """Estimate a sequence of messages with no hidden provider-specific padding."""

    return sum(estimate_message_tokens(message) for message in messages)


@dataclass(frozen=True, slots=True)
class ContextCompilerConfig:
    """Serializable configuration whose fingerprint is safe to persist in a run."""

    compiler_version: int = CONTEXT_COMPILER_VERSION
    policy: ContextPolicyName = "full"
    budget_tokens: int = 128_000
    recent_blocks: int = 2
    max_tool_output_tokens: int = 2_048
    memory_policy: MemoryPolicyName = "none"

    def __post_init__(self) -> None:
        if self.compiler_version != CONTEXT_COMPILER_VERSION:
            raise ValueError(
                f"unsupported context compiler_version: {self.compiler_version}"
            )
        if self.policy not in _POLICIES:
            choices = ", ".join(sorted(_POLICIES))
            raise ValueError(
                f"unknown context policy {self.policy!r}; choose from {choices}"
            )
        if self.budget_tokens <= 0:
            raise ValueError("context budget_tokens must be positive")
        if self.recent_blocks < 0:
            raise ValueError("context recent_blocks must be non-negative")
        if self.max_tool_output_tokens < MIN_TOOL_OUTPUT_TOKENS:
            raise ValueError(
                "context max_tool_output_tokens must be at least "
                f"{MIN_TOOL_OUTPUT_TOKENS}"
            )
        if self.memory_policy not in _MEMORY_POLICIES:
            choices = ", ".join(sorted(_MEMORY_POLICIES))
            raise ValueError(
                f"unknown context memory_policy {self.memory_policy!r}; "
                f"choose from {choices}"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ContextCompilerConfig:
        allowed = {
            "compiler_version",
            "policy",
            "budget_tokens",
            "recent_blocks",
            "max_tool_output_tokens",
            "memory_policy",
        }
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown context config fields: {sorted(unknown)!r}")
        raw_version = data.get("compiler_version", CONTEXT_COMPILER_VERSION)
        if not isinstance(raw_version, int) or isinstance(raw_version, bool):
            raise ValueError("context compiler_version must be an integer")
        raw_policy = data.get("policy", "full")
        if not isinstance(raw_policy, str):
            raise ValueError("context policy must be a string")
        values: dict[str, Any] = {
            "compiler_version": raw_version,
            "policy": raw_policy,
        }
        for name, default in (
            ("budget_tokens", 128_000),
            ("recent_blocks", 2),
            ("max_tool_output_tokens", 2_048),
        ):
            raw_value = data.get(name, default)
            if not isinstance(raw_value, int) or isinstance(raw_value, bool):
                raise ValueError(f"context {name} must be an integer")
            values[name] = raw_value
        raw_memory_policy = data.get("memory_policy", "none")
        if not isinstance(raw_memory_policy, str):
            raise ValueError("context memory_policy must be a string")
        values["memory_policy"] = raw_memory_policy
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "compiler_version": self.compiler_version,
            "policy": self.policy,
            "budget_tokens": self.budget_tokens,
            "recent_blocks": self.recent_blocks,
            "max_tool_output_tokens": self.max_tool_output_tokens,
            "memory_policy": self.memory_policy,
        }

    @property
    def configuration_fingerprint(self) -> str:
        return stable_hash(self.to_dict())


@dataclass(frozen=True, slots=True)
class ContextBlockAnnotation:
    """Optional signals supplied by a memory/evidence projection.

    ``stale`` is observable in the receipt and forces freshness to zero.  It is
    not an exclusion rule: a stale block can still be mandatory or selected if
    its remaining signals justify it.
    """

    importance: float | None = None
    freshness: float | None = None
    topics: frozenset[str] = field(default_factory=frozenset)
    stale: bool = False
    mandatory: bool = False

    def __post_init__(self) -> None:
        for name in ("importance", "freshness"):
            value = getattr(self, name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"annotation {name} must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class ContextBlockReceipt:
    """Audit metadata for one protocol-atomic candidate block."""

    block_id: str
    start_index: int
    end_index: int
    roles: tuple[str, ...]
    estimated_original_tokens: int
    estimated_tokens: int
    selected: bool
    mandatory: bool
    compacted: bool
    stale: bool

    def __post_init__(self) -> None:
        if not self.block_id:
            raise ValueError("context block id must not be empty")
        if self.start_index < 0 or self.end_index <= self.start_index:
            raise ValueError("context block message range is invalid")
        if self.end_index - self.start_index != len(self.roles):
            raise ValueError("context block roles do not match its message range")
        if any(
            role not in {"system", "user", "assistant", "tool"} for role in self.roles
        ):
            raise ValueError("context block contains an unsupported message role")
        if self.estimated_original_tokens <= 0 or self.estimated_tokens <= 0:
            raise ValueError("context block token estimates must be positive")
        if self.estimated_tokens > self.estimated_original_tokens:
            raise ValueError("context block compaction increased its token estimate")
        if (
            not self.compacted
            and self.estimated_tokens != self.estimated_original_tokens
        ):
            raise ValueError("uncompacted context block token estimates differ")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ContextBlockReceipt:
        required = {
            "block_id",
            "start_index",
            "end_index",
            "roles",
            "estimated_original_tokens",
            "estimated_tokens",
            "selected",
            "mandatory",
            "compacted",
            "stale",
        }
        missing = required - set(data)
        unknown = set(data) - required
        if missing:
            raise ValueError(f"context block missing fields: {sorted(missing)!r}")
        if unknown:
            raise ValueError(f"context block has unknown fields: {sorted(unknown)!r}")
        return cls(
            block_id=_strict_string(data["block_id"], "context block id"),
            start_index=_strict_int(data["start_index"], "context block start_index"),
            end_index=_strict_int(data["end_index"], "context block end_index"),
            roles=_string_tuple(data["roles"], "context block roles"),
            estimated_original_tokens=_strict_int(
                data["estimated_original_tokens"],
                "context block estimated_original_tokens",
                minimum=1,
            ),
            estimated_tokens=_strict_int(
                data["estimated_tokens"],
                "context block estimated_tokens",
                minimum=1,
            ),
            selected=_strict_bool(data["selected"], "context block selected"),
            mandatory=_strict_bool(data["mandatory"], "context block mandatory"),
            compacted=_strict_bool(data["compacted"], "context block compacted"),
            stale=_strict_bool(data["stale"], "context block stale"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "roles": list(self.roles),
            "estimated_original_tokens": self.estimated_original_tokens,
            "estimated_tokens": self.estimated_tokens,
            "selected": self.selected,
            "mandatory": self.mandatory,
            "compacted": self.compacted,
            "stale": self.stale,
        }


@dataclass(frozen=True, slots=True)
class ContextReceipt:
    """Stable selection receipt suitable for an append-only model-request event."""

    schema_version: int
    config_fingerprint: str
    memory_fingerprint: str | None
    workspace_generation: int | None
    policy: str
    budget_tokens: int
    estimated_original_tokens: int
    estimated_candidate_tokens: int
    estimated_selected_tokens: int
    selected_block_ids: tuple[str, ...]
    evicted_block_ids: tuple[str, ...]
    compacted_block_ids: tuple[str, ...]
    stale_block_ids: tuple[str, ...]
    messages_sha256: str
    message_count: int
    message_roles: tuple[str, ...]
    frame: Mapping[str, Any]
    blocks: tuple[ContextBlockReceipt, ...]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(
                f"unsupported context receipt schema: {self.schema_version}"
            )
        if not _is_sha256(self.config_fingerprint):
            raise ValueError("context receipt config_fingerprint is not SHA-256")
        if self.memory_fingerprint is not None and not _is_sha256(
            self.memory_fingerprint
        ):
            raise ValueError("context receipt memory_fingerprint is not SHA-256")
        if self.workspace_generation is not None and self.workspace_generation < 0:
            raise ValueError(
                "context receipt workspace_generation must be non-negative"
            )
        if self.policy not in _POLICIES:
            raise ValueError(f"context receipt has unsupported policy {self.policy!r}")
        if self.budget_tokens <= 0:
            raise ValueError("context receipt budget must be positive")
        for name in (
            "estimated_original_tokens",
            "estimated_candidate_tokens",
            "estimated_selected_tokens",
            "message_count",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"context receipt {name} must be non-negative")
        if not _is_sha256(self.messages_sha256):
            raise ValueError("context receipt messages_sha256 is not SHA-256")

        block_ids = tuple(block.block_id for block in self.blocks)
        if len(block_ids) != len(set(block_ids)):
            raise ValueError("context receipt block ids must be unique")
        expected_start = 0
        for block in self.blocks:
            if block.start_index != expected_start:
                raise ValueError(
                    "context receipt blocks are not a contiguous partition"
                )
            expected_start = block.end_index

        expected_selected = tuple(
            block.block_id for block in self.blocks if block.selected
        )
        expected_evicted = tuple(
            block.block_id for block in self.blocks if not block.selected
        )
        expected_compacted = tuple(
            block.block_id for block in self.blocks if block.compacted
        )
        expected_stale = tuple(block.block_id for block in self.blocks if block.stale)
        if self.selected_block_ids != expected_selected:
            raise ValueError("context receipt selected blocks are inconsistent")
        if self.evicted_block_ids != expected_evicted:
            raise ValueError("context receipt evicted blocks are inconsistent")
        if self.compacted_block_ids != expected_compacted:
            raise ValueError("context receipt compacted blocks are inconsistent")
        if self.stale_block_ids != expected_stale:
            raise ValueError("context receipt stale blocks are inconsistent")

        expected_original = sum(
            block.estimated_original_tokens for block in self.blocks
        )
        expected_candidate = sum(block.estimated_tokens for block in self.blocks)
        expected_selected_tokens = sum(
            block.estimated_tokens for block in self.blocks if block.selected
        )
        if self.estimated_original_tokens != expected_original:
            raise ValueError("context receipt original token estimate is inconsistent")
        if self.estimated_candidate_tokens != expected_candidate:
            raise ValueError("context receipt candidate token estimate is inconsistent")
        if self.estimated_selected_tokens != expected_selected_tokens:
            raise ValueError("context receipt selected token estimate is inconsistent")
        if self.estimated_selected_tokens > self.budget_tokens:
            raise ValueError("context receipt selected context exceeds its budget")

        expected_roles = tuple(
            role for block in self.blocks if block.selected for role in block.roles
        )
        if self.message_roles != expected_roles:
            raise ValueError("context receipt message roles are inconsistent")
        if self.message_count != len(expected_roles):
            raise ValueError("context receipt message count is inconsistent")

        frame_policy = self.frame.get("policy")
        frame_budget = self.frame.get("budget")
        frame_tokens = self.frame.get("used_tokens")
        frame_selected_value = self.frame.get("selected_ids")
        if frame_policy != self.policy or frame_budget != self.budget_tokens:
            raise ValueError("context receipt frame policy/budget is inconsistent")
        if frame_tokens != self.estimated_selected_tokens:
            raise ValueError("context receipt frame token count is inconsistent")
        frame_selected = _string_tuple(
            frame_selected_value, "context receipt frame selected_ids"
        )
        if set(frame_selected) != set(self.selected_block_ids):
            raise ValueError("context receipt frame selected ids are inconsistent")
        if len(frame_selected) != len(set(frame_selected)):
            raise ValueError("context receipt frame selected ids contain duplicates")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ContextReceipt:
        required = {
            "schema_version",
            "config_fingerprint",
            "memory_fingerprint",
            "workspace_generation",
            "policy",
            "budget_tokens",
            "estimated_original_tokens",
            "estimated_candidate_tokens",
            "estimated_selected_tokens",
            "selected_block_ids",
            "evicted_block_ids",
            "compacted_block_ids",
            "stale_block_ids",
            "messages_sha256",
            "message_count",
            "message_roles",
            "frame",
            "blocks",
        }
        missing = required - set(data)
        unknown = set(data) - required
        if missing:
            raise ValueError(f"context receipt missing fields: {sorted(missing)!r}")
        if unknown:
            raise ValueError(f"context receipt has unknown fields: {sorted(unknown)!r}")
        blocks_value = data["blocks"]
        frame_value = data["frame"]
        if not isinstance(blocks_value, list) or not isinstance(frame_value, dict):
            raise ValueError("context receipt blocks/frame have invalid types")
        if not all(isinstance(item, dict) for item in blocks_value):
            raise ValueError("context receipt blocks must contain objects")
        return cls(
            schema_version=_strict_int(
                data["schema_version"], "context receipt schema_version"
            ),
            config_fingerprint=_strict_string(
                data["config_fingerprint"], "context receipt config_fingerprint"
            ),
            memory_fingerprint=(
                None
                if data["memory_fingerprint"] is None
                else _strict_string(
                    data["memory_fingerprint"],
                    "context receipt memory_fingerprint",
                )
            ),
            workspace_generation=(
                None
                if data["workspace_generation"] is None
                else _strict_int(
                    data["workspace_generation"],
                    "context receipt workspace_generation",
                )
            ),
            policy=_strict_string(data["policy"], "context receipt policy"),
            budget_tokens=_strict_int(
                data["budget_tokens"], "context receipt budget_tokens", minimum=1
            ),
            estimated_original_tokens=_strict_int(
                data["estimated_original_tokens"],
                "context receipt estimated_original_tokens",
            ),
            estimated_candidate_tokens=_strict_int(
                data["estimated_candidate_tokens"],
                "context receipt estimated_candidate_tokens",
            ),
            estimated_selected_tokens=_strict_int(
                data["estimated_selected_tokens"],
                "context receipt estimated_selected_tokens",
            ),
            selected_block_ids=_string_tuple(
                data["selected_block_ids"], "context receipt selected_block_ids"
            ),
            evicted_block_ids=_string_tuple(
                data["evicted_block_ids"], "context receipt evicted_block_ids"
            ),
            compacted_block_ids=_string_tuple(
                data["compacted_block_ids"], "context receipt compacted_block_ids"
            ),
            stale_block_ids=_string_tuple(
                data["stale_block_ids"], "context receipt stale_block_ids"
            ),
            messages_sha256=_strict_string(
                data["messages_sha256"], "context receipt messages_sha256"
            ),
            message_count=_strict_int(
                data["message_count"], "context receipt message_count"
            ),
            message_roles=_string_tuple(
                data["message_roles"], "context receipt message_roles"
            ),
            frame=dict(frame_value),
            blocks=tuple(
                ContextBlockReceipt.from_dict(cast(Mapping[str, Any], item))
                for item in blocks_value
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "config_fingerprint": self.config_fingerprint,
            "memory_fingerprint": self.memory_fingerprint,
            "workspace_generation": self.workspace_generation,
            "policy": self.policy,
            "budget_tokens": self.budget_tokens,
            "estimated_original_tokens": self.estimated_original_tokens,
            "estimated_candidate_tokens": self.estimated_candidate_tokens,
            "estimated_selected_tokens": self.estimated_selected_tokens,
            "selected_block_ids": list(self.selected_block_ids),
            "evicted_block_ids": list(self.evicted_block_ids),
            "compacted_block_ids": list(self.compacted_block_ids),
            "stale_block_ids": list(self.stale_block_ids),
            "messages_sha256": self.messages_sha256,
            "message_count": self.message_count,
            "message_roles": list(self.message_roles),
            "frame": dict(self.frame),
            "blocks": [block.to_dict() for block in self.blocks],
        }


@dataclass(frozen=True, slots=True)
class CompiledContext:
    """Messages sent to a provider together with their reproducible receipt."""

    messages: tuple[AgentMessage, ...]
    receipt: ContextReceipt


@dataclass(frozen=True, slots=True)
class _ContextBlock:
    block_id: str
    start_index: int
    end_index: int
    original_messages: tuple[AgentMessage, ...]
    messages: tuple[AgentMessage, ...]
    mandatory: bool
    compacted: bool

    @property
    def original_tokens(self) -> int:
        return estimate_messages_tokens(self.original_messages)

    @property
    def tokens(self) -> int:
        return estimate_messages_tokens(self.messages)


def _request_hash(messages: Sequence[AgentMessage]) -> str:
    return stable_hash([message.to_dict() for message in messages])


def _short_compaction_header(content: str, original_tokens: int, limit: int) -> str:
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    candidates = (
        f"[contextopt compacted sha256={digest} original_tokens={original_tokens}]",
        f"[contextopt compacted sha256={digest[:12]} tokens={original_tokens}]",
        "[contextopt compacted]",
        "[compacted]",
    )
    for candidate in candidates:
        if estimate_text_tokens(candidate) <= limit:
            return candidate
    return ""


def _compact_tool_content(content: str, limit: int) -> tuple[str, bool]:
    original_tokens = estimate_text_tokens(content)
    if original_tokens <= limit:
        return content, False

    header = _short_compaction_header(content, original_tokens, limit)
    if not header:
        return "", True

    separator_template = "\n... {omitted} chars omitted ...\n"

    def candidate(keep: int) -> str:
        head_count = (keep + 1) // 2
        tail_count = keep // 2
        omitted = len(content) - head_count - tail_count
        if keep <= 0:
            return header
        head = content[:head_count]
        tail = content[len(content) - tail_count :] if tail_count else ""
        return header + "\n" + head + separator_template.format(omitted=omitted) + tail

    low = 0
    high = len(content)
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_text_tokens(candidate(middle)) <= limit:
            low = middle
        else:
            high = middle - 1
    compacted = candidate(low)
    while low > 0 and estimate_text_tokens(compacted) > limit:
        low -= 1
        compacted = candidate(low)
    return compacted, True


def _compact_message(
    message: AgentMessage, max_tool_output_tokens: int
) -> tuple[AgentMessage, bool]:
    if message.role != "tool":
        return message, False
    content, compacted = _compact_tool_content(message.content, max_tool_output_tokens)
    if not compacted:
        return message, False
    return (
        AgentMessage(
            role=message.role,
            content=content,
            tool_calls=message.tool_calls,
            tool_call_id=message.tool_call_id,
            tool_name=message.tool_name,
        ),
        True,
    )


def _group_messages(
    messages: Sequence[AgentMessage], config: ContextCompilerConfig
) -> tuple[_ContextBlock, ...]:
    for message_index, source_message in enumerate(messages):
        try:
            AgentMessage.from_dict(source_message.to_dict())
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid message at index {message_index}: {exc}"
            ) from exc
    blocks: list[_ContextBlock] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        start = index
        if message.role == "tool":
            raise ValueError(f"orphan tool message at index {index}")

        grouped = [message]
        if message.role == "assistant" and message.tool_calls:
            declared = [call.id for call in message.tool_calls]
            if len(set(declared)) != len(declared):
                raise ValueError(f"duplicate tool call id at message index {index}")
            declared_names = {call.id: call.name for call in message.tool_calls}
            index += 1
            while index < len(messages) and messages[index].role == "tool":
                grouped.append(messages[index])
                index += 1
            actual = [
                cast(str, tool_message.tool_call_id) for tool_message in grouped[1:]
            ]
            if len(actual) != len(set(actual)):
                raise ValueError(f"duplicate tool result at message index {start}")
            if set(actual) != set(declared):
                missing = sorted(set(declared) - set(actual))
                unknown = sorted(set(actual) - set(declared))
                raise ValueError(
                    "tool exchange at message index "
                    f"{start} is incomplete (missing={missing!r}, unknown={unknown!r})"
                )
            if actual != declared:
                raise ValueError(
                    f"tool results at message index {start} are out of declared order"
                )
            mismatched_names = sorted(
                cast(str, tool_message.tool_call_id)
                for tool_message in grouped[1:]
                if tool_message.tool_name
                != declared_names[cast(str, tool_message.tool_call_id)]
            )
            if mismatched_names:
                raise ValueError(
                    "tool exchange at message index "
                    f"{start} has mismatched tool names for {mismatched_names!r}"
                )
        else:
            index += 1

        compacted_messages: list[AgentMessage] = []
        compacted = False
        for grouped_message in grouped:
            output, changed = _compact_message(
                grouped_message, config.max_tool_output_tokens
            )
            compacted_messages.append(output)
            compacted = compacted or changed

        end = start + len(grouped)
        blocks.append(
            _ContextBlock(
                block_id=f"block-{start:06d}",
                start_index=start,
                end_index=end,
                original_messages=tuple(grouped),
                messages=tuple(compacted_messages),
                mandatory=message.role in {"system", "user"},
                compacted=compacted,
            )
        )
    return tuple(blocks)


def _normalized_block_content(block: _ContextBlock) -> str:
    parts: list[str] = []
    for message in block.messages:
        parts.append(message.role)
        parts.append(message.content)
        for call in message.tool_calls:
            parts.extend((call.name, call.arguments_json))
        if message.tool_name:
            parts.append(message.tool_name)
    return _SPACE_RE.sub(" ", "\n".join(parts)).strip().lower()


def _lexical_terms(text: str) -> frozenset[str]:
    return frozenset(piece.lower() for piece in _TOPIC_RE.findall(text))


def _block_topics(block: _ContextBlock) -> frozenset[str]:
    normalized = _normalized_block_content(block)
    counts = Counter(
        term for term in _lexical_terms(normalized) if term not in _STOP_TOPICS
    )
    ordered = sorted(counts, key=lambda term: (-counts[term], term))
    return frozenset(ordered[:16])


def _durable_memory_message(match: SemanticMemoryMatch) -> AgentMessage:
    """Render one durable match as a bounded, clearly advisory assistant note."""

    entry = match.entry
    matched_terms = ",".join(match.matched_terms) or "-"
    source_refs = ",".join(entry.source_refs) or "-"
    source_run_id = entry.source_run_id or "-"
    content = (
        "[contextopt durable memory candidate]\n"
        f"memory_id={entry.memory_id}\n"
        f"scope={entry.scope} kind={entry.kind} confidence={entry.confidence:.6f}\n"
        f"retrieval_score={match.score:.6f} "
        f"feedback_signal={match.feedback_signal:.6f}\n"
        f"matched_terms={matched_terms}\n"
        f"source_run_id={source_run_id}\n"
        f"source_refs={source_refs}\n"
        f"memory_text={entry.text}\n"
        "Treat this as advisory experience; verify it against current workspace "
        "evidence."
    )
    return AgentMessage(role="assistant", content=content)


def _durable_memory_payload(match: SemanticMemoryMatch) -> dict[str, Any]:
    return {
        "entry": match.entry.to_dict(),
        "score": match.score,
        "matched_terms": list(match.matched_terms),
        "feedback_signal": match.feedback_signal,
    }


def durable_memory_matches_from_receipt(value: Any) -> tuple[SemanticMemoryMatch, ...]:
    """Rebuild durable candidates persisted in a context receipt frame."""

    if not isinstance(value, list):
        raise ValueError("durable memory matches must be an array")
    matches: list[SemanticMemoryMatch] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise ValueError(f"durable memory match {index} must be an object")
        required = {"entry", "score", "matched_terms", "feedback_signal"}
        if set(raw) != required:
            raise ValueError(
                f"durable memory match {index} fields do not match the schema"
            )
        raw_entry = raw["entry"]
        if not isinstance(raw_entry, Mapping):
            raise ValueError(f"durable memory match {index} entry must be an object")
        entry = SemanticMemoryEntry.from_dict(raw_entry)
        score = raw["score"]
        feedback_signal = raw["feedback_signal"]
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
        ):
            raise ValueError(f"durable memory match {index} score is invalid")
        if (
            not isinstance(feedback_signal, (int, float))
            or isinstance(feedback_signal, bool)
            or not math.isfinite(float(feedback_signal))
            or not -1.0 <= float(feedback_signal) <= 1.0
        ):
            raise ValueError(f"durable memory match {index} feedback_signal is invalid")
        raw_terms = raw["matched_terms"]
        if not isinstance(raw_terms, list) or not all(
            isinstance(term, str) and term for term in raw_terms
        ):
            raise ValueError(f"durable memory match {index} matched_terms is invalid")
        matches.append(
            SemanticMemoryMatch(
                entry=entry,
                score=float(score),
                matched_terms=tuple(raw_terms),
                feedback_signal=float(feedback_signal),
            )
        )
    return tuple(matches)


def _block_relevance(block: _ContextBlock, query_terms: frozenset[str]) -> float:
    if not query_terms:
        return 0.0
    terms = _lexical_terms(_normalized_block_content(block))
    if not terms:
        return 0.0
    overlap = len(query_terms & terms)
    return min(1.0, overlap / math.sqrt(len(query_terms) * len(terms)))


def _block_kind_and_importance(block: _ContextBlock) -> tuple[str, float]:
    first = block.messages[0]
    if first.role in {"system", "user"}:
        return first.role, 1.0
    if first.role != "assistant" or not first.tool_calls:
        return "assistant", 0.5

    names = frozenset(call.name for call in first.tool_calls)
    tool_text = "\n".join(message.content.lower() for message in block.messages[1:])
    if "run_tests" in names and (
        "exit_code=1" in tool_text or "failed" in tool_text or "error" in tool_text
    ):
        return "test_failure", 1.0
    if names & {"replace_text", "create_file", "write_file", "apply_patch"}:
        return "workspace_write", 0.95
    if "error" in tool_text or "exception" in tool_text:
        return "tool_error", 0.9
    if "run_tests" in names:
        return "test_result", 0.82
    if names & {"read_file", "search_text"}:
        return "evidence", 0.76
    return "tool_exchange", 0.62


def _default_query(messages: Sequence[AgentMessage]) -> str:
    user_content = [message.content for message in messages if message.role == "user"]
    recent_content = [message.content for message in messages[-2:]]
    return "\n".join(user_content[-1:] + recent_content)


def _aggregate_annotations(
    block: _ContextBlock,
    *,
    block_annotation: ContextBlockAnnotation | None,
    message_annotations: Mapping[int, ContextBlockAnnotation],
) -> ContextBlockAnnotation:
    message_values = [
        message_annotations[index]
        for index in range(block.start_index, block.end_index)
        if index in message_annotations
    ]
    values = [*message_values]
    if block_annotation is not None:
        values.append(block_annotation)
    if not values:
        return ContextBlockAnnotation()

    importance_values = [
        value.importance for value in values if value.importance is not None
    ]
    freshness_values = [
        value.freshness for value in values if value.freshness is not None
    ]
    topics = frozenset(topic for value in values for topic in value.topics)
    # Atomic blocks can contain several tool results. Any stale message means the
    # request would expose stale evidence if the block is selected. An explicit
    # block-level annotation is the only source allowed to override that aggregate.
    stale = (
        block_annotation.stale
        if block_annotation is not None
        else any(value.stale for value in message_values)
    )
    return ContextBlockAnnotation(
        importance=max(importance_values) if importance_values else None,
        freshness=min(freshness_values) if freshness_values else None,
        topics=topics,
        stale=stale,
        mandatory=any(value.mandatory for value in values),
    )


def _select_recent(problem: SelectionProblem) -> ContextFrame:
    """Return a transparent newest-first sliding-window baseline."""

    selected = set(problem.mandatory_ids)
    if problem.tokens_for(selected) > problem.budget:
        raise ContextBudgetError(
            "mandatory context exceeds recent-window budget: "
            f"{problem.tokens_for(selected)} > {problem.budget}"
        )
    rejected: dict[str, str] = {}
    window_open = True
    for item in reversed(problem.items):
        if item.id in selected:
            continue
        if not window_open:
            rejected[item.id] = "older than the retained recent window"
            continue
        proposed = selected | {item.id}
        if problem.tokens_for(proposed) <= problem.budget:
            selected.add(item.id)
        else:
            rejected[item.id] = "recent-window token budget exceeded"
            window_open = False

    selected_ids = tuple(item.id for item in problem.items if item.id in selected)
    decisions = tuple(
        SelectionDecision(
            item_id=item.id,
            status="selected" if item.id in selected else "rejected",
            reason=(
                "mandatory context"
                if item.mandatory
                else (
                    "newest block fitting the sliding window"
                    if item.id in selected
                    else rejected[item.id]
                )
            ),
            added_tokens=item.tokens if item.id in selected else None,
        )
        for item in problem.items
    )
    return ContextFrame(
        policy="recent",
        selected_ids=selected_ids,
        used_tokens=problem.tokens_for(selected_ids),
        budget=problem.budget,
        objective_score=problem.score(selected_ids),
        decisions=decisions,
        metadata={"selection": "newest-first contiguous sliding window"},
    )


class ContextCompiler:
    """Compile accumulated runtime messages into a bounded provider request."""

    def __init__(self, config: ContextCompilerConfig | None = None) -> None:
        self.config = config or ContextCompilerConfig()

    @property
    def configuration_fingerprint(self) -> str:
        return self.config.configuration_fingerprint

    def compile(
        self,
        messages: Sequence[AgentMessage],
        *,
        query: str | None = None,
        block_annotations: Mapping[str, ContextBlockAnnotation] | None = None,
        message_annotations: Mapping[int, ContextBlockAnnotation] | None = None,
        memory_fingerprint: str | None = None,
        workspace_generation: int | None = None,
        durable_memory: Sequence[SemanticMemoryMatch] = (),
        durable_memory_store_fingerprint: str | None = None,
    ) -> CompiledContext:
        """Select context without mutating the input or consulting external state."""

        source_messages = tuple(messages)
        durable_matches = tuple(durable_memory)
        if self.config.memory_policy != _DURABLE_MEMORY_POLICY and (
            durable_matches or durable_memory_store_fingerprint is not None
        ):
            raise ValueError(
                "durable memory candidates require "
                "memory_policy='versioned-v1+semantic'"
            )
        if self.config.memory_policy == _DURABLE_MEMORY_POLICY:
            if memory_fingerprint is None:
                raise ValueError(
                    "semantic context requires a combined memory fingerprint"
                )
            if durable_memory_store_fingerprint is None or not _is_sha256(
                durable_memory_store_fingerprint
            ):
                raise ValueError(
                    "semantic context requires a durable memory store fingerprint"
                )
            if any(
                not isinstance(match, SemanticMemoryMatch) for match in durable_matches
            ):
                raise ValueError("durable memory candidates have an invalid type")
        durable_start = len(source_messages)
        if self.config.memory_policy == _DURABLE_MEMORY_POLICY:
            source_messages = source_messages + tuple(
                _durable_memory_message(match) for match in durable_matches
            )
        if self.config.memory_policy == "none" and (
            memory_fingerprint is not None or workspace_generation is not None
        ):
            raise ValueError("memory audit fields require memory_policy='versioned-v1'")
        if memory_fingerprint is not None and not _is_sha256(memory_fingerprint):
            raise ValueError("memory_fingerprint must be a lowercase SHA-256")
        if workspace_generation is not None and workspace_generation < 0:
            raise ValueError("workspace_generation must be non-negative")
        blocks = _group_messages(source_messages, self.config)
        annotations = dict(block_annotations or {})
        message_signals = dict(message_annotations or {})
        durable_block_ids = {
            f"block-{durable_start + index:06d}": match.entry.memory_id
            for index, match in enumerate(durable_matches)
        }
        for block_id, match in zip(durable_block_ids, durable_matches, strict=True):
            annotations.setdefault(
                block_id,
                ContextBlockAnnotation(
                    importance=max(0.25, match.entry.confidence),
                    freshness=1.0,
                    topics=frozenset(match.matched_terms) | frozenset(match.entry.tags),
                ),
            )
        unknown_annotations = set(annotations) - {block.block_id for block in blocks}
        if unknown_annotations:
            raise ValueError(
                f"annotations reference unknown blocks: {sorted(unknown_annotations)!r}"
            )
        unknown_message_indices = set(message_signals) - set(
            range(len(source_messages))
        )
        if unknown_message_indices:
            raise ValueError(
                "message annotations reference unknown indices: "
                f"{sorted(unknown_message_indices)!r}"
            )

        recent_ids = frozenset(
            block.block_id
            for block in (
                blocks[-self.config.recent_blocks :]
                if self.config.recent_blocks
                else ()
            )
            if block.block_id not in durable_block_ids
        )
        query_terms = _lexical_terms(
            _default_query(source_messages) if query is None else query
        )
        items: list[ContextItem] = []
        stale_ids: list[str] = []
        for position, block in enumerate(blocks):
            annotation = _aggregate_annotations(
                block,
                block_annotation=annotations.get(block.block_id),
                message_annotations=message_signals,
            )
            kind, default_importance = _block_kind_and_importance(block)
            age = len(blocks) - position - 1
            default_freshness = 1.0 / (1.0 + 0.18 * age)
            freshness = (
                default_freshness
                if annotation.freshness is None
                else annotation.freshness
            )
            if annotation.stale:
                freshness = 0.0
                stale_ids.append(block.block_id)
            importance = (
                default_importance
                if annotation.importance is None
                else annotation.importance
            )
            content = _normalized_block_content(block)
            items.append(
                ContextItem(
                    id=block.block_id,
                    tokens=block.tokens,
                    relevance=_block_relevance(block, query_terms),
                    importance=importance,
                    freshness=freshness,
                    kind=kind,
                    source=f"messages:{block.start_index}-{block.end_index}",
                    content=content,
                    topics=_block_topics(block) | annotation.topics,
                    duplicate_group=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    mandatory=(
                        block.mandatory
                        or block.block_id in recent_ids
                        or annotation.mandatory
                    ),
                )
            )

        problem = SelectionProblem(tuple(items), self.config.budget_tokens)
        mandatory_tokens = problem.tokens_for(problem.mandatory_ids)
        if mandatory_tokens > self.config.budget_tokens:
            raise ContextBudgetError(
                "protocol-mandatory context exceeds budget after tool-output "
                f"compaction: {mandatory_tokens} > {self.config.budget_tokens}"
            )

        if self.config.policy == "full":
            all_ids = tuple(item.id for item in items)
            all_tokens = problem.tokens_for(all_ids)
            if all_tokens > self.config.budget_tokens:
                raise ContextBudgetError(
                    "full context exceeds budget after tool-output compaction: "
                    f"{all_tokens} > {self.config.budget_tokens}"
                )
            frame = ContextFrame(
                policy="full",
                selected_ids=all_ids,
                used_tokens=all_tokens,
                budget=self.config.budget_tokens,
                objective_score=problem.score(all_ids),
                decisions=tuple(
                    SelectionDecision(
                        item_id=item.id,
                        status="selected",
                        reason="full history policy",
                        added_tokens=item.tokens,
                    )
                    for item in items
                ),
                metadata={"selection": "all protocol blocks"},
            )
        elif self.config.policy == "recent":
            frame = _select_recent(problem)
        else:
            try:
                frame = create_policy(self.config.policy).select(problem)
            except ValueError as exc:
                if "mandatory context is infeasible" in str(exc):
                    raise ContextBudgetError(str(exc)) from exc
                raise

        if self.config.memory_policy == _DURABLE_MEMORY_POLICY:
            selected_durable_ids = [
                memory_id
                for block_id, memory_id in durable_block_ids.items()
                if block_id in frame.selected_ids
            ]
            frame = replace(
                frame,
                metadata={
                    **dict(frame.metadata),
                    "durable_memory_ids": list(durable_block_ids.values()),
                    "durable_memory_selected_ids": selected_durable_ids,
                    "durable_memory_matches": [
                        _durable_memory_payload(match) for match in durable_matches
                    ],
                    "durable_memory_store_fingerprint": (
                        durable_memory_store_fingerprint
                    ),
                },
            )

        selected_set = frozenset(frame.selected_ids)
        selected_blocks = tuple(
            block for block in blocks if block.block_id in selected_set
        )
        compiled_messages = tuple(
            message for block in selected_blocks for message in block.messages
        )
        selected_ids = tuple(block.block_id for block in selected_blocks)
        evicted_ids = tuple(
            block.block_id for block in blocks if block.block_id not in selected_set
        )
        compacted_ids = tuple(block.block_id for block in blocks if block.compacted)
        block_receipts = tuple(
            ContextBlockReceipt(
                block_id=block.block_id,
                start_index=block.start_index,
                end_index=block.end_index,
                roles=tuple(message.role for message in block.messages),
                estimated_original_tokens=block.original_tokens,
                estimated_tokens=block.tokens,
                selected=block.block_id in selected_set,
                mandatory=problem.by_id[block.block_id].mandatory,
                compacted=block.compacted,
                stale=block.block_id in stale_ids,
            )
            for block in blocks
        )
        receipt = ContextReceipt(
            schema_version=1,
            config_fingerprint=self.configuration_fingerprint,
            memory_fingerprint=memory_fingerprint,
            workspace_generation=workspace_generation,
            policy=self.config.policy,
            budget_tokens=self.config.budget_tokens,
            estimated_original_tokens=estimate_messages_tokens(source_messages),
            estimated_candidate_tokens=sum(block.tokens for block in blocks),
            estimated_selected_tokens=estimate_messages_tokens(compiled_messages),
            selected_block_ids=selected_ids,
            evicted_block_ids=evicted_ids,
            compacted_block_ids=compacted_ids,
            stale_block_ids=tuple(stale_ids),
            messages_sha256=_request_hash(compiled_messages),
            message_count=len(compiled_messages),
            message_roles=tuple(message.role for message in compiled_messages),
            frame=frame.to_dict(),
            blocks=block_receipts,
        )
        return CompiledContext(messages=compiled_messages, receipt=receipt)


def compile_runtime_context(
    compiler: ContextCompiler,
    messages: Sequence[AgentMessage],
    *,
    task: str,
    memory_store: SemanticMemoryStore | None = None,
    memory_scope: str | None = None,
    durable_memory_matches: Sequence[SemanticMemoryMatch] | None = None,
    durable_memory_store_fingerprint: str | None = None,
) -> CompiledContext:
    """Compile one runtime transcript through the configured observed-memory policy.

    Runner and recovery both call this pure boundary. Keeping memory projection and
    annotation glue here makes a pending request reproducible and lets the reducer prove
    that a persisted receipt was actually derived from its authoritative transcript.
    """

    message_annotations: dict[int, ContextBlockAnnotation] = {}
    memory_fingerprint: str | None = None
    durable_matches: tuple[SemanticMemoryMatch, ...] = ()
    durable_store_fingerprint = durable_memory_store_fingerprint
    workspace_generation: int | None = None
    source_messages = tuple(messages)
    if compiler.config.memory_policy in {"versioned-v1", _DURABLE_MEMORY_POLICY}:
        memory = MemorySnapshot.from_messages(source_messages)
        observed_fingerprint = memory.fingerprint
        workspace_generation = memory.workspace_generation
        for index in range(len(source_messages)):
            signals = memory.context_signals_for_message(index)
            if signals is None:
                continue
            message_annotations[index] = ContextBlockAnnotation(
                importance=float(signals["importance"]),
                freshness=float(signals["freshness"]),
                topics=frozenset(str(item) for item in signals["topics"]),
                stale=signals["memory_status"] == "stale",
            )

        memory_fingerprint = observed_fingerprint

    if compiler.config.memory_policy == _DURABLE_MEMORY_POLICY:
        if memory_store is not None:
            durable_store_fingerprint = memory_store.fingerprint
            if durable_memory_matches is None:
                try:
                    durable_matches = memory_store.search(
                        task,
                        scope=memory_scope,
                        limit=_DURABLE_MEMORY_LIMIT,
                    )
                except ValueError:
                    durable_matches = ()
            else:
                durable_matches = tuple(durable_memory_matches)
        elif durable_memory_matches is None:
            raise ValueError(
                "semantic context requires an attached memory store or replay snapshot"
            )
        else:
            durable_matches = tuple(durable_memory_matches)
        if durable_store_fingerprint is None or not _is_sha256(
            durable_store_fingerprint
        ):
            raise ValueError("semantic context requires a durable store fingerprint")
        memory_fingerprint = stable_hash(
            {
                "observed_memory": memory_fingerprint,
                "durable_memory_store": durable_store_fingerprint,
            }
        )

    return compiler.compile(
        source_messages,
        query=task,
        message_annotations=message_annotations,
        memory_fingerprint=memory_fingerprint,
        workspace_generation=workspace_generation,
        durable_memory=durable_matches,
        durable_memory_store_fingerprint=durable_store_fingerprint,
    )


__all__ = [
    "CONTEXT_COMPILER_VERSION",
    "MIN_TOOL_OUTPUT_TOKENS",
    "CompiledContext",
    "ContextBlockAnnotation",
    "ContextBlockReceipt",
    "ContextBudgetError",
    "ContextCompiler",
    "ContextCompilerConfig",
    "ContextPolicyName",
    "ContextReceipt",
    "MemoryPolicyName",
    "compile_runtime_context",
    "durable_memory_matches_from_receipt",
    "estimate_message_tokens",
    "estimate_messages_tokens",
    "estimate_text_tokens",
]
