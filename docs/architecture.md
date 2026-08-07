# Architecture

## Product boundary

ForgeAgent is the long-horizon coding-agent runtime being built in this repository.
ContextOpt is both the current Python package/CLI and the context-selection subsystem that
will be integrated into that runtime.

The architecture separates four concepts:

- **State** is authoritative execution data such as run status, budgets, tool outcomes,
  workspace hashes, and test results.
- **Knowledge** is versioned external evidence such as source code and documentation.
- **Memory** is selectively retained experience such as decisions and failed attempts.
- **Context** is the bounded view compiled for one model invocation.

Messages alone are not authoritative state, an event log is not automatically usable
memory, and persistence without a strict transition model does not provide safe resume.

## Current runtime — v0.2b

```text
Task
  │
  v
AgentRunner ──────────────> EventLog (schema-2 hash chain)
  │        ^                         │
  │        │                         v
  │        │              strict reducer ──> RunProjection
  │        │                                      │
  │        │                            atomic checkpoint cache
  │        │ normalized ModelResponse / ToolOutcome
  v        │
ModelClient
  │ structured ToolCall(s)
  v
WorkspaceTools ───────────> bounded workspace + registered visible tests
  │
  └──────── tool observation, preserving call id ──────────> next turn
```

### AgentRunner

Owns the single-agent turn loop. It constructs provider-neutral requests, accounts for
reported token usage, enforces limits before later actions, executes tool calls in order,
and writes one terminal event. A model response without tool calls ends the run.

Repeated call ids with identical arguments reuse the reconstructed completed outcome;
reusing an id with different arguments is a contract error. Turn, tool-call, and token
budgets are cumulative across resumes. Wall timeout applies to each active execution
session, so stopped process time is not charged.

### ModelClient

The runtime depends on a narrow async protocol rather than one provider SDK. v0.2b has:

- `ScriptedModel`, used for deterministic observation-aware offline conformance runs;
- `OpenAICompatibleModel`, a minimal non-streaming Chat Completions adapter.

Both normalize assistant text, structured tool calls, finish reason, provider usage, and
model failures into the same runtime types. The initial event persists a non-secret model
configuration fingerprint. A non-terminal resume must present the same fingerprint;
ScriptedModel also moves its deterministic cursor past already durable responses.

### WorkspaceTools

Tools are rooted at one resolved workspace. Paths are normalized and constrained; writes
and registered commands require explicit permissions. The resolved workspace, permissions,
tool definitions, and registered commands form a configuration fingerprint that must match
on resume. File replacement uses exact text and a SHA-256 compare-and-swap precondition,
then atomically replaces the target.

`run_tests` executes only a command registered by the caller. A nonzero test exit is a
completed tool observation, not an infrastructure failure. This distinction is necessary
for an Agent to diagnose and repair code after reproducing a bug.

These checks are not a container security boundary. Registered commands are trusted host
processes.

Before a tool executes, the runner persists an operation id, canonical call fingerprint,
replay policy, and sealed execution plan when preparation succeeds. Recovery classes are:

- `list_files`, `search_text`, and `read_file`: safe to retry;
- `create_file`: reconcile absent-file precondition against content-SHA postcondition;
- `replace_text`: reconcile before-SHA against after-SHA;
- `run_tests`: never replay automatically; pause until explicit `mark_failed` or `retry`.

For reconciled writes, matching post-state records the planned outcome without writing
again; matching pre-state retries; any third state pauses as divergence. This is not a
general exactly-once guarantee. An explicit retry can repeat effects, and arbitrary command
effects cannot be inferred from workspace hashes.

### EventLog

Every model, tool, budget, interruption, resume, pause, and terminal boundary is appended
to schema-2 JSONL. Each event includes the previous event hash and its own canonical SHA-256.
Readers strictly validate sequence, run id, event hashes, and chain links. The chain makes
accidental corruption or un-recomputed edits evident; it does not stop a malicious writer
from replacing the whole file and recomputing every hash because there is no signature or
external trust anchor.

The scanner retains the last valid byte offset. A truncated final JSON fragment is readable
for audit but cannot be appended until explicit repair physically removes that suffix. A
complete final event without a newline is preserved and safely terminated before append.

One `RunLease` uses a Windows or POSIX file lock plus an in-process registry to reject a
second cooperative writer. It is a local single-host lease, not distributed consensus.

Schema-1 logs remain readable and renderable for audit, but are never reduced or resumed.

### Strict projection and checkpoint cache

The event reducer is the execution state machine. It rejects unknown fields, illegal phase
transitions, reordered tools, mismatched budgets, malformed plans, inconsistent terminal
results, and model/tool configuration changes. A valid replay reconstructs normalized
messages, usage, pending work, completed outcomes, phase, and terminal state.

The runner atomically replaces a JSON projection checkpoint after durable transitions. Its
state hash, source sequence, run id, and source event hash must match. Because these hashes
are unauthenticated, the current implementation also replays the anchored event prefix and
compares the reconstructed state before accepting the checkpoint, then reduces the suffix.
Invalid or forged cache data falls back to the full authoritative log. The cache contains
no workspace snapshot and does not yet claim faster startup.

`contextopt status` exposes this projection. `contextopt resume` returns a terminal result
without a model call, or resumes a non-terminal projection after fingerprint validation.
An unclean active phase receives `run.interrupted` followed by `run.resumed` before work
continues.

See [Runtime](runtime.md) for the executable contract and offline demonstration.

## Current context engine — v0.1

The context layer remains a deterministic pure function:

```text
SelectionProblem + ContextPolicy -> ContextFrame + SelectionReceipt
```

### ContextItem

A typed candidate with token cost, source, semantic signals, and graph constraints.
Critical-fact labels exist only in benchmark cases and are never exposed to policies.

### SelectionProblem

Owns the token budget, validates identifiers, computes dependency closure and feasibility,
and exposes a transparent set objective.

### ContextPolicy

An interchangeable optimizer. Policies receive the same problem and return a fully
auditable `ContextFrame`.

### ExactOraclePolicy

Enumerates all subsets up to a safe candidate limit. It measures heuristic objective
quality on small problems and is not a production runtime policy.

### Synthetic microbench

Generates paired cases from fixed seeds, runs every policy on the same case, validates
every result, and preserves raw per-run metrics. It evaluates solver and synthetic context
properties, not Agent coding success.

## Integration seam

The runtime currently sends its accumulated messages directly to `ModelClient`. The next
context milestone introduces a compiler at that boundary:

```text
Run/Event Log
     ├── authoritative task state
     ├── recent tool observations
     └── versioned workspace evidence
                 │
                 v
Candidate Builder <──── Repository Graph / Memory Store
                 │
                 v
          SelectionProblem
                 │
          ContextOpt Policy
                 │
                 v
       ContextFrame + Receipt
                 │
                 v
             ModelRequest
                 │
                 v
       Tool/Test Outcome Event
```

Every compiled request should persist both its exact context and selection receipt. This
will permit replay and paired context-policy comparisons from the same authoritative state.
None of that wiring is claimed as implemented in v0.2b.

## Claim boundaries

- Optimizer objective quality is not Agent task success.
- Runtime conformance is not model reasoning ability.
- A normal model stop is not proof that code is correct.
- Visible test success is not a substitute for an independent hidden oracle.
- The event hash chain is corruption-evident, not authenticated against malicious rewrite.
- The atomic JSON checkpoint is a disposable replay cache, not authoritative state or a
  workspace checkpoint.
- Tool reconciliation reduces duplicate effects but does not provide general exactly-once
  execution.
- Workspace-bounded tools are not an OS sandbox.

End-to-end claims require executable hidden tests under paired, fixed-model, fixed-tool,
fixed-budget experiments with repeated runs.
