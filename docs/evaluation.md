# Evaluation protocol

ForgeAgent/ContextOpt separates solver correctness, runtime conformance, context quality,
and real Agent task success. Results from one level must not be promoted into a claim about
another.

## Level 0: solver correctness — implemented

- Validate mandatory, budget, dependency, and conflict constraints.
- Compare heuristic objective values with the exact oracle.
- Report approximation ratio and latency distributions.
- Use deterministic seeds and preserve raw results.

This level tests ContextOpt algorithms, not an Agent loop.

## Level 1: synthetic context quality — implemented

Synthetic cases label critical facts outside the policy-visible data. Metrics include:

- critical-fact recall;
- objective/oracle ratio;
- duplicate-token ratio;
- stale-item ratio;
- token-budget utilization.

Critical recall and objective score are reported separately so surrogate misalignment
cannot be hidden by a composite score. Synthetic labels and artificial candidate features
do not establish performance on real repository trajectories.

## Level 2a: Runtime Conformance Eval — implemented

The first offline coding fixture executes against a temporary real Python workspace. An
observation-aware `ScriptedModel` performs:

```text
read source -> read tests -> tests fail -> atomic edit -> tests pass -> final
```

The fixture verifies:

- tool calls and observations preserve names and call ids;
- the SHA-256 returned by `read_file` is required by the later edit;
- a nonzero test exit is reported to the next model turn rather than terminating the run;
- only the intended file changes and visible tests pass after the edit;
- an independent hidden semantic oracle passes;
- scripted token usage is accounted exactly;
- tool and token limits prevent disallowed later actions;
- a model failure leaves a valid partial JSONL trace.

The demo in [Runtime](runtime.md#offline-runtime-demo) reproduces the main path without an
API key.

### What this result does not mean

The ScriptedModel already contains the correct solution and tool sequence. It supplies
fixed token usage rather than invoking a tokenizer or model. Therefore Runtime Conformance
does **not** measure:

- code reasoning or autonomous planning;
- issue resolution rate;
- model quality;
- context-policy quality;
- robustness to natural-language ambiguity;
- SWE-bench or comparable benchmark performance.

This evaluation is an engine test: it establishes that a later real model can use the same
runtime path and that failures can be audited.

## Level 2b: recovery conformance — implemented

Deterministic crash-injection and CLI tests validate the v0.2b recovery contract:

- schema-2 sequence, run-id, event-hash, and previous-hash validation;
- schema-1 read/trace compatibility without resume;
- explicit physical repair of only a truncated final JSON fragment;
- continuation after a complete final event that lacks a newline;
- cross-instance and real cross-process run-lease exclusion;
- strict full replay of messages, budgets, pending work, completed outcomes, and terminal
  state;
- an atomic projection checkpoint whose corruption or internally rehashed forged state
  falls back to the authoritative event replay;
- equivalence between full replay and a validated checkpoint plus suffix replay;
- rejection of illegal transitions, reordered tools, and mismatched model/tool/limit
  configuration;
- resuming a pending model request and completing an interrupted scripted run;
- completing a durable final model response without making another model call;
- automatic retry of interrupted read-only tools;
- `create_file` and `replace_text` reconciliation for pre-state, post-state, and divergent
  SHA states;
- default pause for an interrupted `run_tests`, followed only by explicit `mark_failed` or
  `retry`;
- CLI `status`, terminal lookup, non-terminal `resume`, paused exit code semantics, and a
  visible `run.resumed` trace.

These tests establish deterministic recovery behavior for covered crash points. They do
not prove arbitrary exactly-once effects, recovery from machine loss, workspace snapshot
restore, or correctness under a maliciously rewritten log. The schema-2 hash chain is
corruption-evident but unauthenticated: a writer who can replace the whole log can recompute
the chain.

The JSON checkpoint is evaluated only as a cache. A result counts as recoverable only if
the authoritative event stream can be fully replayed when that cache is deleted or damaged.

Turn, tool, and token budgets remain cumulative across sessions. Wall timeout is measured
per active `run` or `resume` session, so downtime is intentionally excluded.

## Level 2c: context routing/compiler conformance — implemented

The v0.3 runner has a live `ContextCompiler` seam before each `ModelRequest`. A deterministic,
model-free evaluation compares `recent`, `topk`, `density`, and `submodular` on the same
three long coding traces at fixed 512- and 1,024-estimated-token budgets. The traces contain
protocol-atomic assistant/tool exchanges, unrelated history, and evaluation-only exact
evidence probes.

Metrics are kept separate:

- exact evidence-probe recall;
- assistant/tool protocol validity;
- estimated-token budget compliance and overage;
- source-to-compiled compression ratio;
- canonical messages-plus-receipt determinism across repeated compilations.

At 1,024 estimated tokens, the current fixed fixtures retain a mean 0.222 of probes under
the `recent` baseline versus 0.778 under `submodular`. All four policies are protocol-valid,
budget-compliant, and byte-deterministic on these cases. Those numbers are regression
observations for transparent fixtures, not confidence intervals or evidence of model task
success. At 512 estimated tokens all policies retain zero labelled probes, which is also
reported rather than hidden by a composite score.

No model is called, no code is generated, and no repository test outcome is scored. The
token estimator is stable and local, not a provider tokenizer. Exact probe retention does
not show that a model would notice or use the evidence, while compression says only how
much context was removed. The `full` policy is recorded as unsupported when the source
cannot fit the fixed budget instead of receiving a fabricated quality score.

See [Context routing/compiler conformance](context-routing-eval.md) for metric definitions,
the API, and reproduction command.

## Level 3: controlled real-model coding tasks — planned

Use the same model snapshot, system prompt, tools, repository commit, maximum turns, token
budget, timeout, and visible tests for every policy. Keep hidden tests outside the Agent
workspace. The primary outcome is hidden-test task success.

Secondary metrics:

- total reported input and output tokens;
- repeated file reads and tool calls;
- visible test-progress delta and progress-over-time area;
- changed-file and patch-size distribution;
- wall-clock latency;
- invalid or denied tool-call rate;
- constraint retention after future compaction.

At least three repetitions per task are required for exploratory comparisons; stronger
claims require more runs based on observed variance. Report paired differences and
confidence intervals, not only aggregate point estimates. Preserve failed trajectories and
exact model/version metadata.

Level 2c verifies that the runtime compiles policy outputs into otherwise identical bounded
requests. Level 3 remains necessary because compiler conformance and labelled retention do
not establish real-model coding performance.

## Level 4: extended long-horizon robustness — planned

Inject controlled failures:

- process and machine termination at additional model/tool/filesystem boundaries;
- forced context compaction;
- stale or conflicting memory;
- duplicated tool results;
- provider timeouts and retryable failures;
- repeated tool-call ids;
- workspace changes between read and compare-and-swap edit.

Measure replay recovery, extra steps after resume, stale-memory rejection, budget overage,
duplicate work, operator-paused rate, and final hidden-test success. Distinguish automatic
retry, state reconciliation, and explicit operator decisions rather than combining them
under an unqualified “idempotent” or “exactly once” label. Event-log survival alone is not
counted as recovery.

## Level 5: test-guided search and multi-agent scheduling — planned

Compare under one shared compute budget:

- single-path Agent;
- independent best-of-N;
- fixed beam search;
- adaptive branch scheduling.

Each branch needs an isolated workspace and the same executable oracle. Candidate metrics
include issue resolved rate, tokens per solved task, time to first valid patch, test-progress
area, repeated-state ratio, branch pruning precision, and merge-conflict rate.

Role-playing transcripts are not evidence of multi-agent value. Each additional Agent or
branch must perform a measurable state transformation or evaluation and be included in the
shared budget.

## Reproduction rules

Every reported experiment should preserve:

- task/fixture identifier and repository snapshot;
- model provider, exact model identifier, parameters, and adapter version;
- system and user prompts;
- tool definitions, permissions, registered commands, and limits;
- raw JSONL traces and terminal run status;
- schema version, last verified event hash, and whether truncated-tail repair occurred;
- checkpoint use or full-replay fallback;
- final patch or changed-file hashes;
- visible and hidden oracle commands/results;
- random seeds where supported;
- environment and Python version.

The two v0.1 optimizer reports can be regenerated with commands in the README. Their
objective and selection metrics are deterministic; timing is local diagnostic data. The
v0.2a Runtime Conformance, v0.2b Recovery Conformance, and v0.3 Context Routing
Conformance fixtures are deterministic except for timestamps, active-session elapsed
durations, temporary paths, and subprocess timing where those runtime fields are present.
