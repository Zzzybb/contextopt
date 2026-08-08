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

## Level 2d: test-guided branch-search conformance — implemented

The v0.4 search core evaluates a fixed candidate graph representing model-generated
workspace snapshots and their visible-test observations. It runs a deterministic beam
baseline or bounded MCTS/UCT traversal under explicit candidate and depth budgets. The default fixture contains three
first-pass hypotheses, two follow-up repairs, and one equivalent repair whose complete
workspace snapshot is identical to an earlier passing state.

The report keeps these signals separate:

- visible-test calls versus candidate proposals (duplicate states do not consume a
  second test call under the declared test-environment fingerprint);
- test progress and best passing branch;
- beam-pruned candidates, MCTS selection rewards, and duplicate-state decisions;
- hash-chain integrity and JSON round-trip determinism;
- branch graph rendering in dependency-free SVG HTML.

Reproduce the offline report with:

```text
python -m contextopt branch-search \
  --output branch-search.json \
  --markdown branch-search.md \
  --html branch-search.html

python -m contextopt branch-search --search-policy mcts \
  --output branch-search-mcts.json --html branch-search-mcts.html
```

The checked-in fixture currently proposes 6 candidates, evaluates 5 unique workspace
states, detects 1 duplicate, prunes 1 low-progress branch, and selects `iterative-fix`
after all five visible tests pass. The MCTS output additionally records each UCT choice,
parent visits, and propagated test-quality reward. This is an algorithm and accounting conformance
result. Candidate snapshots and test results are fixed inputs; no model is called by the
pure core and no patch command is executed by it. The repository also ships an explicit
local executable adapter: it materializes each unique snapshot in a disposable
temporary workspace, runs one trusted argv command without a shell, and records a bounded
stdout/stderr digest and excerpt. This adapter is still a trusted host process, not an OS
sandbox. Neither path establishes model coding accuracy, hidden-test success, or
multi-agent value; a production benchmark must include isolation, rollback, and all
candidate test effects in the shared compute budget.

## Level 2e: model-to-candidate protocol conformance — implemented

The v0.5 proposal seam tests the boundary between a provider-neutral `ModelClient` and the
branch-search case format. It uses scripted responses to verify that:

- the request contains a bounded task/root snapshot, no tools, and an explicit JSON contract;
- fenced JSON is accepted only when it still decodes to the strict top-level shape;
- tool calls, malformed JSON, unknown fields, oversized files, invalid paths, duplicate ids,
  and invalid parent graphs are rejected before any test command runs;
- accepted candidates are complete snapshots with hypotheses/evidence and an explicit
  `not-executed` result until the visible-test oracle supplies observations;
- the CLI can serialize the same case that `branch-search` consumes offline.

These are protocol and safety metrics: proposal acceptance/rejection, budget compliance, and
verified handoff shape. They do not measure whether a model proposes a good fix, whether hidden
tests pass, or whether a model would choose useful context. Model quality requires the controlled
real-model evaluation below.

Reproduce the offline boundary tests with:

```text
python -m unittest tests.test_proposer -v
```

## Level 2f: durable proposal/test/apply session conformance — implemented

The v0.6 session tests cover the stateful seam that a long-running coding Agent needs above
the one-shot proposal protocol. A scripted model first proposes a failing snapshot and then a
passing snapshot. The harness verifies that:

- the next proposal contains bounded visible-test feedback from the previous round;
- model-call, candidate, round, and actual test-process budgets are shared across rounds;
- identical workspace fingerprints reuse a prior `TestResult` instead of launching another
  process, with cache reuses counted separately;
- checkpoints are atomically round-trippable, pending `proposing` state resumes only after an
  explicit retry, and forged event or derived-metric data is rejected;
- an accepted snapshot can be applied only with explicit write permission, refuses a stale
  baseline or denied path, and produces a tamper-evident apply receipt; rollback refuses when
  the applied workspace changed out of band.

Run the deterministic session and apply/rollback conformance tests with:

```text
python -m unittest tests.test_search_session tests.test_apply -v
```

The meaningful accounting fields are `model_calls`, `candidate_proposals`, `test_calls`,
`test_reuses`, `rounds`, `total_tokens`, and the per-round branch metrics. These are execution
and safety measurements, not model intelligence: the scripted responses are fixed, visible
tests are the only oracle used for ranking, and no hidden-test or real-model success claim is
made. A passing candidate demonstrates the session control flow and apply boundary, not that a
provider would discover that candidate on an arbitrary issue.

## Level 2g: planner / solver / reviewer orchestration conformance — implemented

The v0.7 orchestrator adds measurable role boundaries above the v0.6 session:

- planner, solver, and reviewer responses are strict no-tool JSON contracts;
- model, token, candidate, test, and role budgets are shared and persisted;
- reviewer acceptance is gated by the visible-test result, so a role transcript cannot
  convert a failing candidate into success;
- cross-round feedback, duplicate-test reuse, role fingerprints, checkpoint round trips,
  and hash-chain events are all available in the report.
- every role call compiles its assistant-only history with the shared ContextCompiler and
  persists a ContextReceipt containing selected blocks, a compiled-message hash, and the
  versioned observed-memory fingerprint/workspace generation.

The deterministic tests exercise a first failing branch followed by a passing branch,
with a reviewer retry before acceptance. Metrics are role call counts, context receipts,
actual test processes, cache reuses, candidate proposals, total tokens, and oracle-gate
outcomes.
These are orchestration and safety measurements, not model intelligence or hidden-test
success. The reviewer may reject a passing candidate; it is intentionally not allowed
to override the executable oracle.

Reproduce the offline role conformance tests with:

    python -m unittest tests.test_orchestrator -v

The CLI entry point is orchestrate; use three scripted model files for a fully offline
run, or provide one OpenAI-compatible model name per role for a real experiment. A
controlled real-model comparison still belongs to Level 3.

## Level 2h: coding-agent strategy accounting — implemented

The v0.8 `agent-eval` harness compares three control policies on paired executable
fixtures:

- `single_pass`: one solver response containing one candidate;
- `best_of_n`: one solver response containing multiple complete snapshots, ranked by the
  same visible-test oracle and branch-search rules;
- `orchestrated`: planner, solver, and reviewer with a bounded retry and oracle gate.

The checked-in fixtures are an ACM-style Two Sum repair and an extended Euclidean
algorithm repair. Each fixture includes a complete failing root snapshot, a deliberately
weak candidate, a passing candidate, a standard-library visible-test command, and an
independent hidden grader under `grader/` that is never included in the model-visible root
snapshot. The CLI can write machine-readable JSON, Markdown, and a self-contained HTML dashboard:

    python -m contextopt agent-eval --fixtures all --repetitions 1 \
      --output agent-eval.json --markdown agent-eval.md --html agent-eval.html \
      --checkpoint agent-eval.checkpoint.json \
      --manifest agent-eval.manifest.json

The report is paired by fixture, strategy, and repetition. Its primary fields are visible
`success_rate`, conditional hidden `hidden_success_rate`, `mean_model_calls`, role-call
decomposition, `mean_candidate_proposals`, `mean_test_calls`, `mean_test_reuses`,
`mean_hidden_test_calls`, and `mean_total_tokens`. A hidden grader runs only after visible
acceptance, so a visible pass cannot silently erase an independent failure. Failed baseline
runs stay in the ledger rather than being dropped. The command itself exits successfully
when the matrix completes; a strategy's visible-test failure is a data point, not a harness
crash. `--no-hidden-tests` is available only when reproducing the visible-only control flow.

The Markdown, console, and HTML renderers also show a descriptive Wilson 95% interval for
each visible success rate. When at least two strategies are present, they add paired
comparisons against `single_pass` (or the first configured strategy): wins, losses, ties,
visible outcome delta, hidden delta when both runs actually executed the hidden grader, and
mean test/token deltas with observed population standard deviations. These are derived from
the raw fixture/repetition ledger, so they do not change the durable report schema and must
not be read as a significance test with one repetition.

By default this level tests policy wiring, budget accounting, strict model boundaries,
executable visible/hidden oracle gates, and report consistency with deterministic scripted
responses. `agent-eval --model ... --base-url ...` can replace the scripted factory with
fresh OpenAI-compatible adapters per matrix cell; that path records provider usage but is
still an exploratory fixed-fixture run, not a statistically powered benchmark. Neither
mode measures general model capability, latency, provider reliability, or production safety.
When `--checkpoint PATH` is supplied, the harness atomically persists each completed matrix
cell. Reusing the same configuration with `--resume --checkpoint PATH` skips durable cells
and reruns only missing observations; a provider call interrupted before its cell is written
may run again.
`--manifest PATH` writes a sidecar identity record containing the adapter/model names, a
query-free endpoint, runtime settings, the full evaluation configuration, and an optional
revision from `CONTEXTOPT_GIT_REVISION` or `GITHUB_SHA`; the API key value is never serialized.
In particular, a `best_of_n` or orchestrated success here must not be reported as evidence
that a real model would discover the same candidate.

## Level 3: controlled real-model coding tasks — harness ready, runs pending

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
claims require more runs based on observed variance. The current renderer is ready to report
paired differences and Wilson intervals, but no real-provider result is claimed until a
fixed provider/model/prompt/fixture/budget matrix has been run and its full ledger is
committed as an artifact. Preserve failed trajectories and exact model/version metadata.

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

## Level 5: bounded parallel candidate scheduling and adaptive tree selection — implemented

The session and role orchestrators now expose `max_parallel_tests`. Under one shared test
budget they can launch bounded batches of candidates, each in a disposable isolated workspace,
while recording requested/completed events in deterministic candidate order and checkpointing
after every completed observation. A resume reuses observations already in the checkpoint and
reruns only candidates whose result was not durably recorded.

The current implementation supports all four controls below at the bounded candidate-oracle
level. MCTS traverses an already generated, parent-linked candidate tree. An opt-in
`merge_policy=disjoint` also performs bounded three-way reconciliation of independent solver
snapshots before the oracle. The orchestration layer additionally supports
`speculative_solver_width > 1`: it fans out solver calls, validates each response, namespaces
valid snapshots, and records lane-level hashes/usage/failures. Planner and reviewer calls remain
sequential; provider-side cancellation and exactly-once semantics remain outside this milestone:

- single-path Agent;
- independent best-of-N;
- fixed beam search;
- adaptive branch scheduling.

When enabled, merge evidence records every considered pair, merged candidate id, and conflicting
path hashes. A conflict never produces a partial candidate, and merged snapshots still consume
the shared candidate/test budgets.

`BranchSearchConfig(search_policy="mcts")` uses a deterministic UCT score. An unobserved child
is explored first; after a parent has an oracle result, its mean visible-test quality and the
configured exploration constant determine which descendant is selected. Every selection and
rollout is hash-chained, and no hidden grader result is used for scheduling.

Each branch needs an isolated workspace and the same executable oracle. Candidate metrics
include issue resolved rate, tokens per solved task, time to first valid patch, test-progress
area, repeated-state ratio, branch pruning precision, and merge-conflict rate.

The scheduler reports actual test calls, cache reuses, configured parallelism, observed maximum
in-flight work, per-candidate event history, and recovery-visible observation state. Role-playing
transcripts are not evidence of multi-agent value. Each additional Agent or branch must perform
a measurable state transformation or evaluation and be included in the shared budget.

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

The repository also includes a committed three-repetition scripted control artifact at
[`experiments/v0.8-scripted-3-reps`](../experiments/v0.8-scripted-3-reps/README.md). It covers
both bundled fixtures, all three control strategies, visible and independent hidden graders,
paired deltas, Wilson intervals, a checkpoint, and a reproducibility manifest. Its claim boundary
is intentionally limited to protocol and accounting behavior; replace the scripted factory with
an explicitly versioned provider and rerun the same matrix before making model-quality claims.
