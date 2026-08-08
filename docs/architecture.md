# Architecture

## Product boundary

ForgeAgent is the long-horizon coding-agent runtime being built in this repository.
ContextOpt is both the current Python package/CLI and the context-selection subsystem that
is integrated at that runtime's model-request boundary.

The architecture separates four concepts:

- **State** is authoritative execution data such as run status, budgets, tool outcomes,
  workspace hashes, and test results.
- **Knowledge** is versioned external evidence such as source code and documentation.
- **Memory** is selectively retained experience such as decisions and failed attempts.
- **Context** is the bounded view compiled for one model invocation.

Messages alone are not authoritative state, an event log is not automatically usable
memory, and persistence without a strict transition model does not provide safe resume.

## Current runtime — v0.3

```text
Task
  │
  v
AgentRunner ──────────────> EventLog (schema-2 hash chain + context receipts)
  │        ^                         │
  │        │                         v
  │        │              strict reducer ──> RunProjection
  │        │                                      │
  │        │                            atomic checkpoint cache
  │        │ normalized ModelResponse / ToolOutcome
  v        │
ContextCompiler ──────────> bounded, protocol-atomic messages
  │ compiled ModelRequest
  v
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
and writes one terminal event. When configured, it compiles the reconstructed transcript
through `ContextCompiler` before each model request and persists the selection receipt in
`model.requested`. A model response without tool calls ends the run.

Repeated call ids with identical arguments reuse the reconstructed completed outcome;
reusing an id with different arguments is a contract error. Turn, tool-call, and token
budgets are cumulative across resumes. Wall timeout applies to each active execution
session, so stopped process time is not charged.

### ModelClient

The runtime depends on a narrow async protocol rather than one provider SDK. v0.3 has:

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

## Current context engine — v0.3

The context layer remains a deterministic pure function:

```text
SelectionProblem + ContextPolicy -> ContextFrame + SelectionReceipt
```

The v0.3 runtime adds a live compiler around that selection core:

```text
projected AgentMessage history + query + fixed estimated-token budget
    -> protocol-atomic blocks
    -> ContextPolicy
    -> bounded AgentMessage history + ContextReceipt
```

Assistant tool calls and their tool results form one indivisible block, preventing orphaned
tool results or incomplete tool-call groups. Tool output can be deterministically compacted
before selection. The receipt records original/candidate/selected estimates, selected and
evicted block ids, compaction and staleness, policy decisions, message roles, and the exact
compiled-request hash. These are stable local estimates, not provider token counts.

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

## Live compiler seam — v0.3

The runner now invokes the compiler at the `ModelRequest` boundary:

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

The compiler configuration and fingerprint are part of the durable run configuration. Each
compiled `model.requested` event carries a receipt tied to its selected message roles,
estimated budget, and request hash. Resume rejects a changed compiler configuration and
reconstructs the same request from authoritative events before continuing. Runs without a
configured compiler retain the legacy full-history path.

Level 2c performs paired, fixed-budget, model-free conformance checks on this live seam; see
[Context routing/compiler conformance](context-routing-eval.md). It measures evidence
retention and compiler invariants, not whether a model solves a coding task.

## Test-guided branch search — v0.4

The branch layer sits above one runtime trajectory. A model adapter can generate several
complete candidate workspace snapshots from the same failing task, run the same visible
test oracle in an isolated workspace for each, and hand the observations to the pure
search core:

```text
candidate snapshots + visible-test observations
                 │
                 v
      fixed-environment dedup key
                 │
                 v
       test-progress beam ordering
                 │
                 ├── duplicate state ──> audit and skip another test call
                 ├── low-progress      ──> beam-pruned event
                 └── passing branch    ──> accepted best candidate
```

`BranchCase` stores the task, root snapshot, parent-linked `CandidatePatch` objects, and
their `TestResult` observations. `BranchSearch` assigns each snapshot a stable workspace
fingerprint and combines it with a declared test-environment fingerprint. The search
evaluates a state once, orders failing candidates by a bounded test-progress score, keeps
the configured beam, and stops at the first passing depth unless configured to continue.
The optional `ExecutableSearchConfig` adapter materializes a candidate in a disposable
temporary workspace and runs a trusted argv without a shell; it is deliberately not an
OS sandbox and does not replace the runtime's recovery/tool lease.

Every decision is recorded as a small hash-chained event stream. The report includes the
candidate nodes, test fingerprints, duplicate/prune reasons, accounting metrics, and a
self-contained SVG HTML graph. `validate_search_report` checks the event chain, parent
references, terminal event, and case fingerprint before rendering. The v0.4 demo is thus
useful for explaining search mechanics and compute accounting without claiming that a
fixed candidate fixture is a real model benchmark.

## Model-to-candidate proposal boundary — v0.5

The proposal layer is the first model-facing seam above the pure search core. It deliberately
does not let a model claim that a patch works:

```text
bounded task + root snapshot
            │
            v
      ModelClient.complete
            │
            v
 strict JSON proposal parser ── reject tool calls, unknown fields,
            │                     oversized files, bad paths/graphs
            v
  untested BranchCase (complete snapshots + hypotheses)
            │
            v
 disposable executable oracle ──> TestResult
            │
            v
       pure BranchSearch
```

`build_proposal_request` is provider-neutral and carries no tools. `parse_proposal_response`
accepts only a top-level `candidates` array, enforces candidate/file/prompt budgets, and
constructs the same `CandidatePatch` and parent graph types used by deterministic search. Every
candidate receives a `not-executed` result until the executable adapter replaces it with an
observed result. This separation is important for evaluation: prompt compliance, parser
acceptance, and test success are different measurements. `propose-case` remains a useful
one-shot adapter; the durable session below is the composition that retries across rounds.

The `propose-case` CLI supports both `ScriptedModel` and the OpenAI-compatible adapter. Its JSON
output can be passed directly to `branch-search`; no provider-specific response shape leaks into
the search or report layers. It does not schedule planner/coder/reviewer agents or persist a
model-generated patch transaction by itself.

## Durable iterative search session — v0.6

`SearchSessionRunner` composes the model-facing boundary, the disposable executable oracle, and
the pure branch search into one bounded state machine. A round always moves through durable
phases; the checkpoint is written before a provider call and after its response and test
evaluation:

```text
                 checkpoint                     checkpoint
       ┌──────────────┬───────────────┐      ┌───────────────┐
       │              v               │      │               v
 idle ─┴─> proposing ─────> evaluating ─────> idle ──> next round
              │                 │
              │ response lost   │ visible-test feedback
              v                 v
           paused          accepted / budget_exhausted / failed
```

The model receives the current best complete snapshot and a bounded list of prior oracle
observations. A proposal is parsed strictly, then candidate ids are namespaced by round so
that parent graphs and audit events cannot collide across retries. Each unique workspace state
is looked up in a session-wide observation cache keyed by its content fingerprint; a cache hit
is reported separately from an actual test process. The shared counters cover model calls,
candidate proposals, test processes, and rounds rather than resetting at each round.

Before awaiting a model, the session stores `phase=proposing`. If the process stops before a
response is durable, resume defaults to `paused`; `--retry-pending` explicitly sends the
request again. This is a deliberate at-least-once boundary, not an exactly-once provider claim.
Every phase and decision is hash chained, and `SearchSessionReport.from_dict` recomputes its
metrics and rejects forged event or metric data.

The session report is still an immutable decision artifact. `apply-best` is a separate operator
operation: it compares the relevant files in a real workspace with the session's root snapshot,
rejects stale or linked paths, atomically replaces changed files, and writes an `ApplyReceipt`.
`rollback-best` performs the inverse only when the applied workspace still matches the receipt's
observed target. This explicit side-effect boundary keeps search evaluation reproducible and
makes out-of-band edits visible rather than silently overwriting them.

## Planner / solver / reviewer orchestration — v0.7

The role orchestrator is a sequential state machine above the v0.6 session boundary. It
keeps the three model clients independent so each role can use a different provider or a
deterministic scripted adapter:

    task + workspace
           |
           v
        planner ---- strict plan JSON ----+
                                          |
                                          v
        solver <--- plan + bounded evidence ---- complete snapshots
                                          |
                                          v
                       disposable visible-test oracle
                                          |
                                          v
        reviewer <-- plan + branch report -- strict decision JSON
           |                      |
           +---- accept only if reviewer says accept AND visible tests pass

Planner and reviewer responses cannot issue tools. Solver responses reuse the existing
complete-snapshot parser, so no role can smuggle an arbitrary patch or test claim past the
same path and size checks. The reviewer receives candidate ids, hypotheses, scores, and
bounded test observations rather than unverified model confidence. A reviewer may reject a
passing branch; it cannot make a failing branch accepted.

The report stores per-role model fingerprints, per-role and shared budgets, role-call
response hashes, cross-round feedback, cached test observations, and the same hash-chained
event type used by branch search. Checkpoints are written before planner, solver, and
reviewer awaits and after evaluation. A stopped planner request requires explicit retry;
after a durable planner response, the solver can safely continue on resume. Parallel
workspace execution and adaptive branch scheduling remain future work.

## Claim boundaries

- Optimizer objective quality is not Agent task success.
- Labelled evidence retention is not evidence use, reasoning quality, or coding success.
- Compiler token estimates are deterministic accounting units, not provider token counts.
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
