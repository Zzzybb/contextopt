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

Messages are not authoritative state, an event log is not automatically usable memory, and
persisting events does not by itself provide checkpoint/resume.

## Current runtime — v0.2a

```text
Task
  │
  v
AgentRunner ──────────────> EventLog (append-only JSONL)
  │        ^
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

Repeated call ids with identical arguments reuse the cached outcome within one live run;
reusing an id with different arguments is a contract error. Cross-process idempotency and
resume are not implemented yet.

### ModelClient

The runtime depends on a narrow async protocol rather than one provider SDK. v0.2a has:

- `ScriptedModel`, used for deterministic observation-aware offline conformance runs;
- `OpenAICompatibleModel`, a minimal non-streaming Chat Completions adapter.

Both normalize assistant text, structured tool calls, finish reason, provider usage, and
model failures into the same runtime types.

### WorkspaceTools

Tools are rooted at one resolved workspace. Paths are normalized and constrained; writes
and registered commands require explicit permissions. File replacement uses exact text and
a SHA-256 compare-and-swap precondition, then atomically replaces the target.

`run_tests` executes only a command registered by the caller. A nonzero test exit is a
completed tool observation, not an infrastructure failure. This distinction is necessary
for an Agent to diagnose and repair code after reproducing a bug.

These checks are not a container security boundary. Registered commands are trusted host
processes.

### EventLog

Every model, tool, budget, and terminal boundary is appended to versioned JSONL with a
contiguous sequence number. The log is durable enough to retain completed events if the
last line is interrupted, but the runner cannot yet recover its conversation, workspace,
or pending tool calls from that log.

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
None of that wiring is claimed as implemented in v0.2a.

## Claim boundaries

- Optimizer objective quality is not Agent task success.
- Runtime conformance is not model reasoning ability.
- A normal model stop is not proof that code is correct.
- Visible test success is not a substitute for an independent hidden oracle.
- JSONL durability is not checkpoint/resume.
- Workspace-bounded tools are not an OS sandbox.

End-to-end claims require executable hidden tests under paired, fixed-model, fixed-tool,
fixed-budget experiments with repeated runs.
