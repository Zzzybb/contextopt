# ForgeAgent / ContextOpt

**An auditable long-horizon coding-agent runtime with an algorithmic context engine.**

Package milestone: `0.9.0a1`.

ForgeAgent is the runtime direction of this project: take a coding task, let a model inspect
and change one workspace through bounded tools, feed every observation back into the next
turn, and preserve an append-only trace of what happened. ContextOpt is the repository,
Python package, CLI, and live context-selection engine that now compiles the bounded view
for every ForgeAgent model call.

```text
Task ──> AgentRunner ──> ContextCompiler ──> ModelClient ──> tool calls
              ^               ^                                │
              │       observed-memory projection               │
              └────────────── tool observations <───────────────┘
                                      │
                           workspace + visible tests

Every boundary ──> durable schema-2 JSONL event log + verified state projection
```

> **Status — v0.9 long-horizon recovery milestone:** the repository now contains a real
> single-agent read/edit/test loop, recoverable event-sourced execution, a deterministic live
> context compiler, an auditable beam/MCTS search over generated coding candidates, and a strict
> model-to-candidate proposal boundary. A model can return bounded complete workspace
> snapshots; the session feeds visible-test failures into later rounds, deduplicates test
> work across rounds, persists pending phases and budgets atomically, and can be resumed
> without pretending that an interrupted provider call was exactly-once. An accepted snapshot
> can be explicitly applied to a real workspace and rolled back with a stale-baseline guard.
> The `agent-eval` harness now compares single-pass, Best-of-N, and planner/solver/reviewer
> control policies on executable ACM/math fixtures with shared accounting. The planner,
> solver, and reviewer now compile their prior assistant summaries through the same
> ContextCompiler, persist a per-role ContextReceipt, and carry a versioned observed-memory
> fingerprint into each checkpointed call.
> Candidate evaluation now supports bounded parallel isolated workspaces with durable per-candidate
> checkpoints and a deterministic, observation-driven MCTS traversal over fixed candidate trees.
> The solver can also run a bounded speculative provider fan-out: each lane receives a distinct
> diversity instruction, responses are namespaced and validated independently, and the checkpoint
> records lane hashes, usage, failures, and observed provider concurrency. Planner and reviewer
> calls remain sequential; this is still at-least-once provider execution, not exactly-once.
> A mandatory OS sandbox for every run, learned semantic memory, and a statistically powered
> real-model coding benchmark remain outside the current claim boundary. The opt-in Docker path is
> now exercised by a provider-free GitHub Actions smoke, but it is not a universal deployment
> guarantee. The v0.9 follow-up now adds an explicit,
> provider-free lexical semantic-memory notebook with audited save/search tools plus an opt-in,
> budgeted semantic-context projection; it is not
> learned retrieval or a coding-quality claim. The bounded `recovery-eval` matrix now injects
> process-like stops across durable model/tool boundaries and verifies fresh-run recovery,
> reconciliation, duplicate-effect accounting, and explicit operator pauses.
> A deterministic workspace knowledge index now turns bounded UTF-8 source files into the same
> provenance-carrying memory candidates, invalidating stale chunks when the source changes.

## Why this project exists

A useful coding agent is more than one prompt and one generated patch. It must repeatedly
inspect evidence, call tools, survive expected failures, respect cost and time limits, and
leave enough state to explain or resume the run. Long runs also accumulate more code,
tests, tool output, decisions, and failed attempts than a model can keep in context.

The project therefore has two connected layers:

- **ForgeAgent runtime:** executes a bounded, auditable coding loop against a real
  workspace.
- **ContextOpt engine:** studies which state, code, memory, and trajectory items should
  enter a limited model context.

The runtime rebuilds candidates from its durable message history before every model call,
then selects a provider-valid subset under an estimated-token budget. This is live runtime
plumbing, but the project still does not claim that a policy improves model reasoning or
coding success without a controlled real-model benchmark.

## What is implemented

### ForgeAgent runtime — v0.3

- Provider-neutral messages, tool definitions, tool calls, responses, and token usage.
- A bounded asynchronous `AgentRunner` with turn, tool-call, token, wall-clock, command,
  and tool-output limits.
- Observation-aware `ScriptedModel` runs that need no network or API key.
- A minimal non-streaming OpenAI-compatible Chat Completions adapter.
- Optional `RecordingModel` / `ReplayModel` cassettes that persist provider request/response
  pairs without credentials and replay them only on an exact request-hash match.
- A bounded `knowledge-index` command that chunks source files into the append-only semantic
  memory store with source references, repeat-safe reuse, and change-aware invalidation.
- Workspace-bounded file listing, literal search, numbered reads, file creation, atomic
  SHA-256 compare-and-swap replacement, and pre-registered visible-test commands.
- Explicit write and command permissions; both are disabled unless enabled by the caller.
- Schema-2 append-only JSONL events with a verifiable SHA-256 chain, strict sequence and
  run-id validation, explicit truncated-tail repair, a compact trace renderer, and a
  dependency-free local HTML timeline with expandable event JSON and context metrics.
- A strict event reducer that reconstructs messages, cumulative usage, pending model and
  tool work, completed-call cache, phase, and terminal result.
- Atomic JSON projection checkpoints used only as disposable recovery caches; missing,
  corrupt, or self-consistent forged caches fall back to the authoritative log. The
  current unauthenticated format verifies checkpoint state against a strict prefix replay
  before reducing the suffix, prioritizing correctness over startup speed.
- A Windows/POSIX cross-process run lease that rejects a concurrent cooperative writer.
- `contextopt status` and `contextopt resume`, including terminal-result lookup without a
  new model call.
- Persisted model and tool-configuration fingerprints that must match before a non-terminal
  run resumes.
- Recovery policies for interrupted tools: retry read-only calls, reconcile `create_file`
  and `replace_text` through persisted pre/post file state and hashes, and pause
  `run_tests` until an operator chooses `mark_failed` or `retry`.
- Failure semantics that distinguish a test process returning exit code 1 from a tool or
  runtime infrastructure failure.
- A deterministic Runtime Conformance Eval covering read, edit, failing tests, passing
  tests, budgets, and partial traces after model failure.
- A live `ContextCompiler` invoked before every model call, with `full`, `recent`, `topk`,
  `density`, and graph-aware `submodular` routing policies.
- Protocol-atomic candidate blocks: an assistant tool-call message and every corresponding
  tool result are selected or evicted together, never split into an invalid request.
- A deterministic estimated-token budget, mandatory task/system and newest blocks, and
  head/tail compaction for oversized tool observations.
- A `versioned-v1` observed-memory projection that invalidates stale same-path file
  evidence, workspace search/listing results, and pre-write test results after recorded
  workspace mutations.
- A complete context receipt inside every new `model.requested` event, including selected
  and evicted block ids, compaction/staleness flags, estimates, policy decisions, memory
  identity, and request hashes.
- An opt-in append-only `SemanticMemoryStore` for cross-run facts, decisions, procedures,
  and failures, exposed through explicit `memory_search` and permission-gated `memory_save`
  tools with provenance, lexical evidence, idempotent writes, and invalidation/supersession.

### Test-guided branch search — v0.4

- Immutable candidate workspace snapshots with parent hypotheses and evidence references.
- Deterministic beam search ordered by visible-test progress, with explicit candidate,
  test-call, depth, and beam budgets; `--search-policy mcts` adds bounded UCT traversal
  whose parent values come only from observed visible-test quality.
- Fixed-test-environment workspace fingerprints that avoid executing the same candidate
  state twice; duplicate and beam-prune decisions remain in the audit trace.
- Hash-chained `search.started`, proposal, evaluation, duplicate, prune, and completed
  events, plus strict report round-tripping and tamper detection.
- A zero-dependency report renderer: console table, Markdown accounting, and a
  self-contained SVG HTML graph suitable for a portfolio demo.
- An explicit executable adapter that materializes each unique snapshot in a disposable
  temporary workspace, runs a trusted argv test command without a shell, captures a
  bounded output digest/excerpt, and feeds the observation into the same search core.

### Model-to-candidate proposal boundary — v0.5

- A provider-neutral `propose-case` seam that asks any existing `ModelClient` for a
  bounded set of complete workspace snapshots rather than accepting arbitrary patches.
- A strict JSON protocol that rejects tool calls, malformed output, unknown fields,
  path traversal, oversized files, duplicate ids, and invalid parent graphs before test
  execution.
- Explicit separation of proposal from verification: every parsed candidate starts as
  `not-executed` and must pass through the visible-test oracle before it can be ranked.
- Offline scripted-model coverage plus a CLI path that can pipe a generated case into
  `branch-search` for deterministic deduplication and disposable-workspace execution.

### Durable iterative search session — v0.6

- A durable `search-session` loop that composes model proposal, disposable visible-test
  execution, and pure branch ranking across multiple rounds.
- Bounded feedback: failed tests, output excerpts, and candidate scores are passed to the
  next proposal as evidence, while each round still receives a complete workspace snapshot.
- Global candidate/test budgets, cross-round workspace deduplication, atomic checkpoints,
  hash-chained session events, model usage accounting, and explicit pending-model retry.
- Strict checkpoint round-tripping and tamper detection, plus console/Markdown session
  reports that expose actual test calls separately from cache reuses.
- An explicit `apply-best` / `rollback-best` boundary with UTF-8 snapshot fingerprints,
  symlink/path checks, atomic file replacement, stale-workspace conflict rejection, and
  auditable apply receipts. Applying is never implicit after a test passes.

The event hash chain provides verifiable, corruption-evident integrity. It is not a
signature or malicious-rewrite defense: someone who can replace the entire log can also
recompute the complete chain because there is no secret or external trust anchor.

### Planner / solver / reviewer orchestration — v0.7

- Three provider-neutral role seams: a strict planner plan, the existing complete-snapshot
  solver protocol, and a strict reviewer decision.
- Shared model, token, candidate, test, and per-role budgets; a reviewer decision can
  reject a passing branch, but it cannot override the visible-test oracle.
- Cross-round plan and reviewer feedback, test-result reuse, hash-chained role events,
  atomic checkpoints, and role-specific model fingerprints for safe resume.
- Each role request is compiled from its prior assistant summaries plus the fresh mandatory
  system/user request. The checkpoint stores the selected block ids, message hash, memory
  fingerprint, and workspace generation, making context selection inspectable and replayable.
- Candidate oracle work can run with `max_parallel_tests > 1`: every candidate is materialized
  in its own temporary workspace, requested/completed events are hash-chained, and the checkpoint
  is updated after each result so recovery reruns only observations that were not durably recorded.
- An opt-in `merge_policy=disjoint` performs bounded three-way reconciliation of independent
  solver snapshots before testing. Non-conflicting changes become a new auditable candidate;
  same-path divergent edits are retained as explicit conflict evidence and are never partially
  merged. This runs after speculative lanes return; it does not hide lane failures or provider
  usage.
- An opt-in `speculative_solver_width > 1` fans out independent solver calls under the shared
  model/candidate budget. Lane request/response hashes, context receipts, token usage, validation
  failures, and `max_provider_in_flight` are persisted; valid snapshots are namespaced before
  oracle evaluation. Width `1` is the deterministic sequential baseline.
- `speculative_solver_stop_on_valid` (or the CLI flag
  `--speculative-solver-stop-on-valid`) turns that fan-out into a first-parseable-candidate
  race: the winner is durably recorded, unfinished lanes receive cancellation requests, and
  the visible-test oracle still decides correctness. `solver.speculative.winner` and
  `solver.speculative.cancelled` events expose the latency/cost trade-off. Cancellation is
  explicitly best-effort because a provider may finish an HTTP request after its local task
  is cancelled. Provider-specific adapters can optionally implement
  `request_cancellation(request)`; the default serial OpenAI-compatible adapter can close an
  active local HTTP response and records `acknowledged` for that transport interruption. When
  an operator supplies `--cancellation-url`, it also POSTs the request idempotency key and model
  to that explicit provider-specific endpoint; a 2xx response is recorded as `acknowledged`,
  while 404/405 is `unsupported`. The endpoint contract remains provider-owned, so even a 2xx
  response does not by itself prove that a generic Chat Completions provider stopped generation.
- The orchestrate CLI command plus console/Markdown/HTML reports make role calls and the
  oracle gate measurable instead of treating a multi-agent transcript as evidence.

### Candidate execution lifecycle boundary — v1.0

- The executable candidate/test adapter launches each trusted command in a fresh process group
  and terminates the group on timeout; Windows uses a kill-on-close Job Object when available.
- Child processes inherit ordinary runtime variables needed to start tests, but common credential
  variables such as `CONTEXTOPT_API_KEY`, `OPENAI_API_KEY`, `*_TOKEN`, `*_SECRET`, and `*_PASSWORD`
  are removed before launch.
- `sandbox="docker"` (CLI `--sandbox docker`) is an opt-in controlled path: it uses
  `--network=none`, a read-only root, dropped Linux capabilities, no-new-privileges, bounded
  pids/memory/CPU, and a writable `/workspace` mount. Pin `--container-image` to a digest for a
  serious run; the default remains the trusted host path.
- The host path prevents common process leaks and accidental credential exposure but is not a
  container/VM sandbox; the Docker path is an isolation boundary only when the operator verifies
  the Docker daemon, image, and host policy.

### Coding-agent strategy evaluation — v0.8

- Four executable fixtures cover ACM-style Two Sum and interval merging repairs plus
  extended-gcd and modular-inverse mathematics repairs. Each has a deliberately failing
  complete root snapshot, visible tests, and an independent hidden grader that is never
  included in model prompts.
- `agent-eval` runs paired `single_pass`, `best_of_n`, and `orchestrated` strategies under
  explicit round, model-call, candidate, and visible-test budgets, then records independent
  hidden-test calls after visible acceptance.
- JSON and Markdown reports expose success rate, role/model calls, actual test processes,
  cache reuses, candidate proposals, and reported token usage without hiding failed runs.
  They also include descriptive Wilson 95% intervals and same-fixture/repetition paired
  deltas (wins/losses/ties plus mean and observed-variance test/token cost deltas) without
  hiding the raw ledger. For paired binary outcomes, the report additionally emits an
  exact two-sided McNemar p-value; it is an exploratory discordant-pair diagnostic, not
  a powered significance claim. Each run also records local wall-clock `duration_ms` and
  paired duration deltas; this is a diagnostic, not a provider latency SLA.
- The evaluator defaults to deterministic scripted responses, but accepts an OpenAI-compatible
  model factory for exploratory runs; either mode keeps the independent hidden grader outside
  the candidate snapshot.

### Long-horizon recovery matrix — v0.9 bounded milestone

- `recovery-eval` injects process-like stops after durable model and tool events, then resumes
  the same event log with fresh runtime objects.
- The five scenarios make pending-request reuse, durable final-response reuse, write
  reconciliation, duplicate-write avoidance, and conservative `run_tests` pause/`mark_failed`
  semantics visible in one ledger.
- Every durable `model.requested` event has a stable request idempotency key; the
  OpenAI-compatible adapter sends it through `Idempotency-Key` by default, while explicitly
  documenting that provider support—not the local header—is what can suppress duplicates.
- JSON, Markdown, self-contained HTML, and a protocol manifest are checked in under
  [`experiments/v0.9-recovery-matrix`](experiments/v0.9-recovery-matrix/README.md), with a
  Chinese explanation and a strict claim boundary.
- A deterministic first-valid cancellation fixture is checked in under
  [`experiments/v0.9-speculative-cancellation`](experiments/v0.9-speculative-cancellation/README.md).
  It makes the winner/cancelled-lane ledger and configured-width budget accounting visible
  without pretending to measure remote provider aborts.

### Level 4 runtime robustness — v1.0 follow-up

- `robustness-eval` turns four local fault-injection contracts into one ledger: oversized
  tool-output compaction preserves head/tail sentinels under a token bound; source-aware
  semantic memory becomes invalid after a cited file changes and remains invalid after reopen;
  duplicate tool results are rejected before a provider request is compiled; and a stale
  compare-and-swap edit returns `content_conflict` without changing the workspace.
- The provider-free JSON/Markdown/HTML/manifest artifact is checked in under
  [`experiments/v1.0-robustness-matrix`](experiments/v1.0-robustness-matrix/README.md).
  It is runtime-contract evidence, not machine-loss recovery, security, provider, exactly-once,
  or coding-quality evidence.

### Durable cross-run semantic memory — v0.9 follow-up

- An opt-in `SemanticMemoryStore` is an append-only, hash-chained JSONL notebook for compact
  facts, decisions, procedures, and failures that should survive a later run.
- `memory_search` exposes deterministic lexical retrieval with scope, tags, confidence,
  matched-term evidence, provenance, and a revision/fingerprint. `memory_save` and
  `memory_invalidate` are available only with `--allow-write`; saves are content-idempotent,
  can supersede an older entry, and can be safely retried after a crash. `memory_feedback`
  records a helpful/not-helpful label with the tool-call id as an idempotency key, adding a
  bounded ranking adjustment without editing the memory text.
- Source references are first-class: a successful `create_file` or `replace_text` mutation
  automatically appends invalidation events for active memories citing that exact workspace
  path, including the crash-reconcile path. Explicit invalidation remains available for facts
  that become stale for non-file reasons.
- An explicit `--context-memory versioned-v1+semantic` mode retrieves at most three lexical
  candidates before each model request. They are ordinary advisory assistant blocks: the live
  context policy may evict them under the same token budget, and the receipt stores their full
  snapshot plus the durable-store fingerprint for deterministic recovery.
- The planner/solver/reviewer orchestrator can mount the same store with
  `orchestrate --context-memory versioned-v1+semantic --memory-store ... --memory-scope ...`.
  Each role receives its own candidate projection and receipt; a pending role request reuses
  its persisted candidate snapshot after a process stop, even if the live store was invalidated.
- `orchestrate --memory-feedback` closes the loop explicitly: when semantic context is enabled,
  the terminal visible-test/oracle outcome records one idempotent `helpful` or `not_helpful`
  label for each candidate that was actually selected by a role. The event ledger records the
  feedback id, signal, and skipped-invalid-memory reason, so a resume cannot double-count a vote.
  This is an outcome correlation signal, not proof that a memory caused the accepted patch.
- The store is advisory: default `versioned-v1` runs still require an explicit tool call, while
  the opt-in semantic context mode performs only the bounded candidate projection described
  above. `global` entries can be inherited by a project scope, and neither path overrides the
  current workspace ledger.
- The provider-free `semantic-context-eval` matrix exercises this projection across fixed ACM/math
  tasks and `recent`/`topk`/`density`/`submodular` policies. It reports retrieved versus selected
  candidate ids, fixture-label retention, token-budget evictions, receipt determinism, and
  store-free replay; it makes zero model calls and does not claim semantic understanding.

### ContextOpt engine — v0.1 algorithms, v0.3 live integration

- Immutable context candidates with provenance, token cost, dependencies, conflicts,
  freshness, topics, and duplicate groups.
- Top-K, density, exact 0/1 knapsack, graph-aware submodular greedy, and a small-instance
  exhaustive oracle.
- Deterministic paired synthetic experiments with hidden critical-fact labels.
- Machine-readable selection receipts explaining every accepted and rejected candidate.
- A model-free Context Routing/Compiler Conformance Eval on fixed, protocol-valid coding
  traces with hidden-to-policy evidence probes.

## What is deliberately not implemented yet

- Automatic workspace snapshots, migration to another workspace, or distributed
  coordination. `apply-best` and `rollback-best` are explicit local operator actions over
  the files named in the session baseline; they are not transparent workspace versioning.
- A general speculative role graph is not claimed. The current `speculative_solver_width`
  fan-out is limited to the solver role; planner/reviewer calls remain sequential. The opt-in
  `speculative_solver_stop_on_valid` policy records the first protocol-valid candidate and
  requests cancellation for the remaining lanes, but it does not claim that a remote provider
  stopped work. A completed solver response is written into the pending lane map before the
  group is reduced, so resume reuses durable lanes and only retries lanes without a response
  (there is still a small crash window before that write).
  The serial OpenAI-compatible adapter emits a deterministic provider idempotency hook, but
  provider-side enforcement is not guaranteed.
  MCTS selects among generated candidates and does not generate patches itself.
- A mandatory container or virtual-machine security boundary for every run. The opt-in Docker
  executor provides a controlled path, but the default host executor remains trusted code and
  workspace path checks, process-group cleanup, and credential scrubbing are not an OS sandbox.
- A general shell tool, autonomous package installation, or unrestricted network access.
- A claim that the scripted demo measures model reasoning or real-world issue resolution.
- Embedding-based retrieval, automatic memory consolidation, learned confidence calibration,
  or a statistically powered real-model coding benchmark. The v0.9 memory store is an
  explicit lexical notebook with source-aware invalidation and opt-in context candidates; the harness is still a
  deterministic scripted comparison by default and does not pretend memory receipts or
  protocol acceptance are model coding accuracy.
- Exactly-once external side effects. Recovery is tool-specific and conservative;
  explicitly retrying a command can execute it again.

## Quick start

ContextOpt requires Python 3.11 or newer and has no runtime dependencies.

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

Run the fully offline coding loop after copying its workspace to a temporary directory:

```text
contextopt run "Fix merge_settings so only None inherits a default." \
  --workspace <temporary-workspace-copy> \
  --script examples/runtime_demo/script.json \
  --allow-write --allow-command \
  --test-command "python -m unittest discover -s tests -v" \
  --event-log <temporary-events.jsonl>
```

Every new run enables live context compilation. Its CLI controls and defaults are:

| Flag | Default | Meaning |
|---|---:|---|
| `--context-policy` | `submodular` | `full`, `recent`, `topk`, `density`, or `submodular` |
| `--context-budget` | `16000` | Maximum deterministic estimated tokens in compiled history |
| `--context-recent-blocks` | `2` | Newest protocol-atomic blocks forced into the request |
| `--context-max-tool-output-tokens` | `2048` | Per-observation estimate before deterministic head/tail compaction |
| `--context-memory` | `versioned-v1` | Observed evidence invalidation; use `none` to disable it, or `versioned-v1+semantic` for bounded durable candidates |
| `--memory-scope` | unset | Optional project scope for automatic durable candidates; `global` entries remain visible |

These are compiler estimates, not counts from a provider tokenizer. The chosen context
configuration and its fingerprint are persisted with the run; `resume` reconstructs that
configuration from the event log rather than accepting replacement context flags.

### Opt-in cross-run memory

Attach a durable semantic notebook to a run by passing `--memory-store`. The default keeps
retrieval explicit through `memory_search`; opt into bounded automatic candidates with
`--context-memory versioned-v1+semantic`. `memory_save` and `memory_invalidate` additionally
require `--allow-write`:

```bash
contextopt run "Fix the parser" \
  --workspace <temporary-workspace-copy> \
  --script examples/runtime_demo/script.json \
  --memory-store .contextopt/memory.jsonl \
  --memory-scope project:parser \
  --context-memory versioned-v1+semantic \
  --allow-write --allow-command \
  --test-command "python -m unittest discover -s tests -v" \
  --event-log <temporary-events.jsonl>
```

The store uses the same append-only event and lease machinery as the runtime trace. Entries
carry a kind, scope, tags, confidence, and source references; writes are deduplicated by a
content-derived identity, and later entries can supersede or invalidate earlier ones. Retrieval
is deterministic lexical matching, not embeddings or a learned summarizer. In semantic context
mode, up to three matches are rendered as clearly labelled advisory assistant blocks and compete
for the normal context budget; an eviction is visible in `durable_memory_selected_ids`. Search
results are evidence for the next model turn, not proof about mutable files. A resumed run must
be given the same memory-store path and `--memory-scope` so its tool configuration fingerprint
remains compatible; a pending semantic request replays the candidate snapshot in its receipt even
if the live store changed after the process stopped.

The complete two-run offline demonstration is in
[`examples/semantic_memory_demo`](examples/semantic_memory_demo/README.md): one fresh Agent
writes a procedure, a second fresh Agent searches the same store, and the reader trace shows the
retrieval evidence.

### Workspace knowledge base

Index a bounded source snapshot into the same durable store before starting a coding run:

```text
contextopt knowledge-index \
  --workspace <workspace> \
  --memory-store .contextopt/memory.jsonl \
  --memory-scope project:parser \
  --output .contextopt/knowledge-index.json \
  --markdown .contextopt/knowledge-index.md
```

The index is intentionally deterministic: it skips non-UTF-8 files, generated directories,
symlinks, and files beyond the configured byte/file bounds, then stores each chunk with a
workspace-relative `source_ref`. Re-running the command reuses identical chunks and appends
`memory.invalidated` events for removed or changed chunks. Attach the same store to `run` or
`orchestrate` with `--context-memory versioned-v1+semantic` to let the normal context budget
select at most three relevant code/document chunks alongside durable experience. This is a
lexical, provenance-aware source projection, not an embedding benchmark or proof that the
model used every retrieved chunk. See the [knowledge-base guide](docs/knowledge-base.md).
For a new `run`, `--auto-index-knowledge` performs that refresh before the first model call;
it requires an explicit `--memory-store` and `--memory-scope`, and `resume` intentionally leaves
re-indexing to the operator.

The multi-agent path uses the same boundary for planner, solver, and reviewer:

```bash
python -m contextopt orchestrate \
  --task "Fix the parser" \
  --root-files root-files.json \
  --checkpoint orchestration.json \
  --memory-store .contextopt/memory.jsonl \
  --memory-scope project:parser \
  --context-memory versioned-v1+semantic \
  --planner-script planner.json --solver-script solver.json --reviewer-script reviewer.json \
  --test-command "python -m unittest discover -s ." --allow-command
```

Add `--auto-index-knowledge` to project the supplied root-file snapshot into the same store before
the planner call. Use `--knowledge-index-report PATH` or `--knowledge-index-markdown PATH` for
auditable snapshot accounting; the flag is only valid for a new orchestration and never mutates
the store during `resume`.

Every role's request receipt records the same store-derived candidate snapshot boundary, while
the role histories remain isolated. The replay path never silently substitutes a new candidate
set for a pending request.

Evaluate the retrieval boundary itself:

```bash
python -m contextopt memory-eval \
  --output experiments/v0.9-semantic-memory/report.json \
  --markdown experiments/v0.9-semantic-memory/report.md \
  --html experiments/v0.9-semantic-memory/report.html
```

The fixed report keeps hit@1/hit@k, MRR, negative-query pass rate, scope isolation, invalidated
memory exclusion, and repeated-search determinism separate from coding success. The committed
artifact is documented in [`experiments/v0.9-semantic-memory`](experiments/v0.9-semantic-memory/README.md).

Evaluate automatic durable-memory candidates at the context boundary:

```bash
python -m contextopt semantic-context-eval \
  --repetitions 3 --budgets 128,256,512 \
  --output experiments/v0.9-semantic-context/report.json \
  --markdown experiments/v0.9-semantic-context/report.md \
  --html experiments/v0.9-semantic-context/report.html
```

The committed JSON/Markdown/HTML artifact is documented in
[`experiments/v0.9-semantic-context`](experiments/v0.9-semantic-context/README.md). Its recall
labels mean exact retention of the fixed fixture memory id, not embedding quality, model use,
or coding-task success.

The script drives this real sequence:

```text
read source -> read tests -> tests fail -> atomic edit -> tests pass -> final
```

Use the exact PowerShell or POSIX commands in the
[runtime guide](docs/runtime.md#offline-runtime-demo). The checked-in fixture is never
modified by those instructions.

Render any completed or partial run:

```bash
contextopt trace <events.jsonl>
contextopt trace <events.jsonl> --html trace.html
contextopt status <events.jsonl>
```

Resume a non-terminal scripted run with the same workspace, script, and registered test
command used originally:

```bash
contextopt resume <events.jsonl> \
  --workspace <same-workspace> \
  --script <same-script.json> \
  --test-command "python -m unittest discover -s tests -v"
```

If an interrupted `run_tests` is pending, the default resume pauses with exit code 4. The
operator must then choose whether to expose an indeterminate failed result to the model or
explicitly run the command again:

```bash
contextopt resume <events.jsonl> \
  --workspace <same-workspace> --script <same-script.json> \
  --test-command "python -m unittest discover -s tests -v" \
  --pending-tool-resolution mark_failed
# or, accepting possible repeated command effects:
contextopt resume <events.jsonl> \
  --workspace <same-workspace> --script <same-script.json> \
  --test-command "python -m unittest discover -s tests -v" \
  --pending-tool-resolution retry
```

A terminal log needs no model or workspace parameters: `contextopt resume <events.jsonl>`
returns the already durable result without appending a new event.

The original optimizer commands remain available:

```bash
contextopt pack examples/auth_context.json --policy submodular
contextopt benchmark --instances 100 --items 14 --budget 900 \
  --seed 4242 --graph-rate 0.2 --conflict-rate 0.02 \
  --policies topk,density,submodular,oracle
```

Run the live compiler conformance comparison without calling a model:

```bash
contextopt context-eval --policies recent,submodular --budgets 1024 \
  --repetitions 3 --output context-routing.json
```

On the three fixed fixtures currently checked into the evaluator, the 1,024-estimated-token
result is:

| Policy | Evidence recall | Protocol valid | Budget compliant |
|---|---:|---:|---:|
| Recent window | 0.222 | 1.000 | 1.000 |
| Submodular | 0.778 | 1.000 | 1.000 |

The report records `model_calls: 0`. Evidence recall only asks whether exact labelled
evidence survived compilation; labels are available to the scorer but not the policy.
This comparison is not model quality, coding accuracy, or evidence that either context
would cause a model to solve more tasks. See the
[context routing evaluation contract](docs/context-routing-eval.md).

Run the deterministic coding-agent strategy comparison:

```bash
python -m contextopt agent-eval \
  --fixtures all --repetitions 1 \
  --output agent-eval.json --markdown agent-eval.md
```

The default matrix contains `two-sum` and `merge-intervals` ACM fixtures plus `extended-gcd`
and `modular-inverse` mathematics fixtures. It compares a deliberately weak one-candidate
baseline with Best-of-N and the oracle-gated role loop. Reports separate visible success from
hidden-grader success; the default scripted run is a control-policy conformance result, not a
model-quality claim.

For an exploratory OpenAI-compatible run, provide the model and API key environment variable:

```bash
python -m contextopt agent-eval \
  --model <model-name> --base-url <endpoint> \
  --api-key-env CONTEXTOPT_API_KEY --repetitions 3 \
  --output agent-eval-real.json --markdown agent-eval-real.md \
  --html agent-eval-real.html \
  --checkpoint agent-eval-real.checkpoint.json \
  --manifest agent-eval-real.manifest.json
```

This path records provider usage but is still not a statistically powered benchmark; keep
model versions, prompts, fixtures, tools, and budgets fixed when comparing strategies.

When multiple completed provider bundles are available, compare them only through the matched
artifact analyzer. It validates the report/manifest protocol fingerprint before computing
cross-model direction consistency:

```bash
python -m contextopt agent-eval-compare \
  --bundle model-a artifacts/model-a.json artifacts/model-a.manifest.json \
  --bundle model-b artifacts/model-b.json artifacts/model-b.manifest.json \
  --output model-matrix.json --markdown model-matrix.md --html model-matrix.html
```

The analyzer is descriptive and refuses protocol drift; it is not a powered generalization
claim. Its bilingual PR note is [`model-matrix-analysis`](docs/pr/0001-v1.0-model-matrix-analysis.md)
and [`中文版`](docs/pr/0001-v1.0-model-matrix-analysis.zh-CN.md).
The optional manifest records the adapter, model names, endpoint origin/path, runtime settings,
evaluation configuration, and an explicitly supplied revision (`CONTEXTOPT_GIT_REVISION` or
`GITHUB_SHA`) without writing the API key.
The HTML output is a self-contained dashboard with visible/hidden success bars, cost columns,
and the complete JSON ledger embedded for portfolio or PR review.
`--checkpoint` atomically records each fixture/strategy/repetition cell; if a provider call or
the process stops, rerun the same command with `--resume --checkpoint ...` to reuse completed
cells and rerun only the missing cell. This is at-least-once provider execution, not an
exactly-once claim.

For a key that should stay out of the local shell, use the manual
[`real-agent-eval` workflow](docs/evaluation-real-provider.md). It is not scheduled and does not
run on pull requests: configure the `CONTEXTOPT_API_KEY` repository/environment secret, enter the
endpoint and model in **Actions → real-agent-eval → Run workflow**, and download the uploaded
JSON/Markdown/HTML/manifest/checkpoint bundle.

For a completed single-agent trajectory, the `run`, `search-session`, `propose-case`, and `resume`
commands can write an opt-in provider cassette with `--record-transcript PATH`; a fresh offline
run can use `--replay-transcript PATH` without an API key. Replay requires the same request sequence
and fails closed on a hash mismatch.
For `orchestrate`, use `--record-transcript-dir DIR` / `--replay-transcript-dir DIR` to keep
separate planner, solver, and reviewer cassettes; role replay intentionally requires the same
visible-test observations as the recorded trajectory.
The `agent-eval` matrix supports the same boundary at cell granularity: add
`--record-transcript-dir DIR` to a real-model run to write
`{fixture}/{strategy}/repetition-{n}/{role}.jsonl`, then use
`--replay-transcript-dir DIR` on a fresh run without an API key. Replay still executes the
visible and independent hidden oracles, but it is an offline reproduction rather than a
new provider measurement; use a fresh recording directory for each matrix.
The implementation is documented in the bilingual PR note
[`agent-eval-transcript-matrix`](docs/pr/0001-v1.0-agent-eval-transcript-matrix.md) /
[`中文版`](docs/pr/0001-v1.0-agent-eval-transcript-matrix.zh-CN.md).

Run the v0.4 test-guided branch-search demo and write all three report formats:

```bash
contextopt branch-search \
  --output branch-search.json \
  --markdown branch-search.md \
  --html branch-search.html
```

The checked-in deterministic case evaluates five unique workspace states and one
duplicate candidate. With the default beam width of 2 it produces one passing branch,
one beam-pruned branch, and one duplicate without a second test call:

| Search accounting | Result |
|---|---:|
| Candidate proposals | 6 |
| Unique test calls | 5 |
| Duplicate states | 1 |
| Beam-pruned states | 1 |
| Best branch | `iterative-fix` |

For a fixed candidate tree, `--search-policy mcts` records `candidate.selected` events with
the UCT score, parent visits, and propagated visible-test reward. It is a bounded scheduler
experiment, not a learned planner or a claim that the model would have proposed the tree.

```bash
contextopt branch-search --search-policy mcts \
  --output branch-search-mcts.json --html branch-search-mcts.html
```

For a real model adapter, convert each generated patch plus its isolated visible-test
observation into the same JSON case shape. For a local executable experiment, pass a
serialized case and explicitly opt into the trusted host command:

```bash
contextopt branch-search candidate-case.json \
  --test-command "python -m unittest discover -s ." \
  --allow-command --html branch-search.html
```

The adapter creates disposable workspaces and never invokes a shell, but it is not an OS
sandbox; use it only with trusted candidate code and test commands.

To exercise the new proposal boundary offline, make `root-files.json` a JSON object whose
keys are relative paths and whose values are complete file contents. A scripted model can
then produce a case without network access:

```bash
contextopt propose-case "Implement solve so it returns ascending values" \
  --root-files examples/branch_demo/root-files.json \
  --script examples/branch_demo/proposal.json \
  --output candidate-case.json

contextopt branch-search candidate-case.json \
  --test-command "python -m unittest discover -s ." \
  --allow-command --html branch-search.html
```

The proposer is deliberately not a verifier: its output records every candidate as
`not-executed` until the second command runs the visible-test oracle. This makes the seam
useful for comparing models and prompting strategies without letting model claims become
test evidence.

For the end-to-end iterative loop, use `search-session`. It checkpoints before and after
each model boundary, carries bounded visible-test feedback into the next round, and stops
when a candidate passes or a shared budget is exhausted:

```bash
contextopt search-session \
  --task "Implement solve so it returns ascending values" \
  --root-files examples/branch_demo/root-files.json \
  --checkpoint session.json \
  --script examples/branch_demo/proposal.json \
  --test-command "python -m unittest discover -s ." \
  --allow-command \
  --max-parallel-tests 4 \
  --output session-report.json \
  --markdown session-report.md
```

`--max-parallel-tests` bounds concurrent disposable oracle processes. Each candidate gets a
fresh temporary workspace; requested/completed scheduler events and the observation map are
checkpointed in deterministic candidate order, so a resume only reruns results that were not
durably recorded. Add `--scheduler-policy adaptive` to rank later batches by observed parent
quality and stop after the first passing batch; `fixed` with `1` is the serial baseline.

If the process stops while a provider request is pending, the checkpoint is intentionally
left in `proposing` rather than claiming exactly-once delivery. Resume with a fresh model
client and an explicit retry decision:

```bash
contextopt search-session --resume --retry-pending \
  --checkpoint session.json --script examples/branch_demo/proposal.json
```

An accepted snapshot is still not written to a real checkout automatically. Apply it only
after inspecting the report and explicitly authorizing writes. The command compares the
named baseline files before changing anything and emits a receipt; rollback refuses if the
applied files were edited out of band:

```bash
contextopt apply-best --checkpoint session.json \
  --workspace ./checkout --allow-write --receipt apply.json
contextopt rollback-best --checkpoint session.json \
  --workspace ./checkout --allow-write --receipt apply.json
```

The apply adapter is a local transaction boundary, not an OS sandbox. Candidate tests still
run as trusted host processes, and a machine failure during filesystem replacement requires
normal operator recovery.

## What the offline demo proves

The Runtime Conformance Eval uses a scripted model that already contains the intended
actions. It verifies runtime plumbing, not intelligence:

- tool calls execute against a temporary real Python workspace;
- SHA-256 from `read_file` is required by the later atomic edit;
- a failing visible test is returned as an observation and the loop continues;
- the final visible tests and an independent hidden semantic oracle pass;
- token and tool limits prevent later actions when exhausted;
- every model and tool boundary remains in the JSONL trace, including partial failed runs.
- state can be strictly replayed and interrupted tool paths can be resumed under the
  recovery contract.
- every model request carries a deterministic context receipt and can be reconstructed
  under the persisted context fingerprint.

Because the solution path is scripted, this result must not be reported as model coding
accuracy or SWE-bench performance. The separate context conformance fixtures compare
retained evidence, not downstream model behavior. See the
[evaluation protocol](docs/evaluation.md).

## First ContextOpt result

The v0.1 graph-constrained baseline has 100 paired synthetic instances, 14 candidates per
instance, and a 900-token budget. Raw runs are in
[`experiments/v0.1-graph.json`](experiments/v0.1-graph.json).

| Policy | Objective / oracle | Critical recall | Redundancy | Budget used |
|---|---:|---:|---:|---:|
| Top-K | 0.981 | 0.975 | 0.005 | 0.946 |
| Density | 0.984 | 0.922 | 0.010 | 0.929 |
| Submodular | 0.991 | 0.945 | 0.000 | 0.930 |
| Exact oracle | 1.000 | 0.948 | 0.000 | 0.955 |

The graph-aware policy improves the declared mathematical objective and removes duplicate
context, but it does not beat Top-K on hidden critical-fact recall. This is evidence of
surrogate-objective misalignment, not evidence of downstream Agent improvement.

## Repository layout

```text
src/contextopt/
├── runtime/              # runner, context/observed memory, recovery, semantic memory, tools
├── evaluation/           # context and coding-agent conformance/evaluation harnesses
├── search/                # proposal, iterative sessions, branch search, apply/rollback
├── models.py             # context candidates, constraints, receipts
├── policies/             # interchangeable selection algorithms
├── synthetic.py          # deterministic context microbench generation
├── benchmark.py          # paired optimizer metrics and reports
└── cli.py                # runtime, search-session, apply/rollback, reports, pack

tests/                    # standard-library unit and integration tests
examples/runtime_demo/    # offline scripted coding-loop demonstration
examples/semantic_memory_demo/ # two-run durable memory demonstration
examples/auth_context.json
experiments/              # checked-in optimizer configurations, evals, and raw results
docs/                     # architecture, runtime, and evaluation contract
```

## Roadmap

- **v0.1 — Context optimizer:** algorithms, receipts, oracle, and synthetic microbench.
- **v0.2a — Runtime foundation:** read/edit/test loop, limits, offline conformance
  evaluation, and optional model adapter.
- **v0.2b — Recoverable execution:** current schema-2 hash chain, strict event replay,
  projection cache, run lease, interruption/resume, and tool reconciliation.
- **v0.3 — Live context engine (implemented):** candidate extraction, protocol-atomic
  compaction, observed-memory invalidation, live ContextOpt policies, per-turn receipts,
  and model-free compiler conformance fixtures.
- **v0.4 — Test-guided search (implemented):** isolated candidate snapshots,
  test-progress beam search, fixed-environment deduplication, hash-chained decisions,
  and self-contained search visualization.
- **v0.5 — Proposal boundary (implemented):** strict model-to-candidate JSON, bounded
  complete snapshots, offline conformance tests, and a `propose-case` to `branch-search`
  handoff.
- **v0.6 — Durable search session (implemented):** iterative proposal/test feedback,
  cross-round deduplication, atomic resumable checkpoints, shared budgets, and explicit
  apply/rollback receipts.
- **v0.7 — Role orchestration (implemented):** planner/solver/reviewer protocols,
  shared budgets, cross-round feedback, checkpointed role events, and an oracle gate.
- **v0.8 — Context-aware strategy evaluation (implemented):** executable ACM/math fixtures,
  independent hidden graders, a paired single-pass/Best-of-N/orchestrated accounting harness,
  per-role ContextCompiler/observed-memory receipts, and bounded parallel isolated candidate
  evaluation with durable scheduler events, fixed-beam/observed-quality MCTS policies, and
  opt-in disjoint three-way merge evidence.
- **v0.9 — Long-horizon recovery matrix (implemented):** deterministic fault injection after
  durable model/tool events, fresh-run resume, write reconciliation, duplicate-effect checks,
  explicit non-replayable-tool pause/resolution, and JSON/Markdown/HTML/manifest artifacts.
- **v0.9 follow-up — Durable semantic memory (implemented):** opt-in append-only cross-run
  memory tools with lexical retrieval, provenance, invalidation/supersession, idempotent writes,
  lease protection, bilingual documentation, a deterministic retrieval-conformance artifact, and
  opt-in budgeted semantic context candidates with receipt snapshots.
  This is memory plumbing, not a learned quality claim.

The v0.8 follow-up is documented in [the role-context addendum](docs/pr/0001-v0.8-context-memory-addendum.md)
and its [Chinese translation](docs/pr/0001-v0.8-context-memory-addendum.zh-CN.md). The parallel
scheduler details are in [the scheduler addendum](docs/pr/0001-v0.8-parallel-scheduler-addendum.md)
and [Chinese version](docs/pr/0001-v0.8-parallel-scheduler-addendum.zh-CN.md). The evaluation
dashboard is documented in [the dashboard addendum](docs/pr/0001-v0.8-evaluation-dashboard-addendum.md)
and [Chinese version](docs/pr/0001-v0.8-evaluation-dashboard-addendum.zh-CN.md). The bounded MCTS
policy is documented in [the MCTS addendum](docs/pr/0001-v0.8-mcts-addendum.md) and [Chinese
version](docs/pr/0001-v0.8-mcts-addendum.zh-CN.md). The durable evaluation matrix is documented
in [the checkpoint addendum](docs/pr/0001-v0.8-evaluation-checkpoint-addendum.md) and [Chinese
version](docs/pr/0001-v0.8-evaluation-checkpoint-addendum.zh-CN.md).
The statistical rendering is documented in [the statistical report addendum](docs/pr/0001-v0.8-evaluation-statistics-addendum.md)
and [Chinese version](docs/pr/0001-v0.8-evaluation-statistics-addendum.zh-CN.md).
The runtime trace timeline is documented in [the trace dashboard addendum](docs/pr/0001-v0.8-trace-dashboard-addendum.md)
and [Chinese version](docs/pr/0001-v0.8-trace-dashboard-addendum.zh-CN.md).
The bounded merge evidence is documented in [the merge-aware snapshot addendum](docs/pr/0001-v0.8-merge-aware-snapshots-addendum.md)
and [Chinese version](docs/pr/0001-v0.8-merge-aware-snapshots-addendum.zh-CN.md).
The reproducibility sidecar is documented in [the evaluation manifest addendum](docs/pr/0001-v0.8-evaluation-manifest-addendum.md)
and [Chinese version](docs/pr/0001-v0.8-evaluation-manifest-addendum.zh-CN.md).
The speculative solver fan-out is documented in [the speculative solver addendum](docs/pr/0001-v0.8-speculative-solver-addendum.md)
and [Chinese version](docs/pr/0001-v0.8-speculative-solver-addendum.zh-CN.md).
An offline three-repetition control artifact is checked in under
[`experiments/v0.8-scripted-3-reps`](experiments/v0.8-scripted-3-reps/README.md); it is a
protocol/accounting baseline, not evidence about real-model quality.
The corresponding [PR change note](docs/pr/0001-v0.8-scripted-control-artifact.md) and
[Chinese version](docs/pr/0001-v0.8-scripted-control-artifact.zh-CN.md) record the exact
regeneration command and claim boundary.
The v1.0 four-fixture control artifact is checked in under
[`experiments/v1.0-acm-math-4-fixtures`](experiments/v1.0-acm-math-4-fixtures/README.md);
it uses the expanded ACM/math registry with three paired repetitions. The corresponding
[PR change note](docs/pr/0001-v1.0-four-fixture-control-artifact.md) and
[Chinese version](docs/pr/0001-v1.0-four-fixture-control-artifact.zh-CN.md) record the
adapter smoke and claim boundary.
The bounded long-horizon recovery matrix is documented in [the v0.9 PR change note](docs/pr/0001-v0.9-recovery-matrix.md)
and [Chinese version](docs/pr/0001-v0.9-recovery-matrix.zh-CN.md).
The model-request idempotency hook is documented in [the v0.9 idempotency addendum](docs/pr/0001-v0.9-idempotency-hook.md)
and [Chinese version](docs/pr/0001-v0.9-idempotency-hook.zh-CN.md).
The first-valid speculative cancellation is documented in [the v0.9 cancellation note](docs/pr/0001-v0.9-speculative-cancellation.md)
and [Chinese version](docs/pr/0001-v0.9-speculative-cancellation.zh-CN.md).
The manual real-provider path is documented in [the workflow guide](docs/evaluation-real-provider.md)
and [Chinese version](docs/evaluation-real-provider.zh-CN.md), with its [PR note](docs/pr/0001-v0.9-real-provider-workflow.md).
The local evaluation-duration accounting is documented in [the v0.9 duration note](docs/pr/0001-v0.9-evaluation-duration.md)
and [Chinese version](docs/pr/0001-v0.9-evaluation-duration.zh-CN.md). The earlier v0.7
orchestration addendum also has a [Chinese retrospective](docs/pr/0001-v0.7-orchestration-addendum.zh-CN.md).
The package-version alignment is documented in [the v0.9 note](docs/pr/0001-v0.9-version-alignment.md)
and [Chinese version](docs/pr/0001-v0.9-version-alignment.zh-CN.md).
The built-in HTTP transport cancellation is documented in [the v0.9 note](docs/pr/0001-v0.9-http-transport-cancellation.md)
and [Chinese version](docs/pr/0001-v0.9-http-transport-cancellation.zh-CN.md).
The opt-in provider-specific cancellation endpoint is documented in [the v1.0 note](docs/pr/0001-v1.0-provider-cancellation-hook.md)
and [Chinese version](docs/pr/0001-v1.0-provider-cancellation-hook.zh-CN.md).
The candidate process lifecycle and credential-scrubbing boundary are documented in [the v1.0 note](docs/pr/0001-v1.0-candidate-process-boundary.md)
and [Chinese version](docs/pr/0001-v1.0-candidate-process-boundary.zh-CN.md).
The paired matrix can persist independent planner/solver/reviewer model identities in its
checkpoint and manifest; the real-provider workflow exposes the same role overrides.
The role-identity persistence change is documented in [the v1.0 note](docs/pr/0001-v1.0-role-model-overrides.md)
and [Chinese version](docs/pr/0001-v1.0-role-model-overrides.zh-CN.md).
The exact paired-outcome McNemar diagnostic is documented in [the v1.0 note](docs/pr/0001-v1.0-paired-mcnemar-evidence.md)
and [Chinese version](docs/pr/0001-v1.0-paired-mcnemar-evidence.zh-CN.md).
The matched multi-model artifact analyzer is documented in [the v1.0 note](docs/pr/0001-v1.0-model-matrix-analysis.md)
and [Chinese version](docs/pr/0001-v1.0-model-matrix-analysis.zh-CN.md).
The cross-platform CI type-check fix is documented in [the v1.0 note](docs/pr/0001-v1.0-ci-cross-platform-typecheck.md)
and [Chinese version](docs/pr/0001-v1.0-ci-cross-platform-typecheck.zh-CN.md).
The real-provider workflow's secret boundary is documented in [the v0.9 note](docs/pr/0001-v0.9-real-provider-secret-scope.md)
and [Chinese version](docs/pr/0001-v0.9-real-provider-secret-scope.zh-CN.md).
The durable cross-run semantic-memory follow-up is documented in [the English PR note](docs/pr/0001-v0.9-semantic-memory.md)
and [Chinese version](docs/pr/0001-v0.9-semantic-memory.zh-CN.md).
The retrieval metrics and committed dashboard are in
[`experiments/v0.9-semantic-memory`](experiments/v0.9-semantic-memory/README.md), with the
evaluation-specific [English PR note](docs/pr/0001-v0.9-semantic-memory-evaluation.md) and
[Chinese version](docs/pr/0001-v0.9-semantic-memory-evaluation.zh-CN.md).
The source-aware invalidation follow-up is documented in the [English PR note](docs/pr/0001-v0.9-source-aware-memory-invalidation.md)
and [Chinese version](docs/pr/0001-v0.9-source-aware-memory-invalidation.zh-CN.md).
The idempotent feedback loop is documented in the [English PR note](docs/pr/0001-v0.9-memory-feedback.md)
and [Chinese version](docs/pr/0001-v0.9-memory-feedback.zh-CN.md).
The opt-in semantic context projection is documented in [the English PR note](docs/pr/0001-v0.9-semantic-memory-context.md)
and [Chinese version](docs/pr/0001-v0.9-semantic-memory-context.zh-CN.md).
The provider-free automatic candidate matrix is documented in [the English PR note](docs/pr/0001-v0.9-semantic-context-evaluation.md)
and [Chinese version](docs/pr/0001-v0.9-semantic-context-evaluation.zh-CN.md).
The multi-agent semantic-memory integration is documented in [the English PR note](docs/pr/0001-v0.9-orchestration-semantic-memory.md)
and [Chinese version](docs/pr/0001-v0.9-orchestration-semantic-memory.zh-CN.md).
The local OpenAI-compatible adapter smoke is documented in [the English PR note](docs/pr/0001-v0.9-provider-adapter-smoke.md)
and [Chinese version](docs/pr/0001-v0.9-provider-adapter-smoke.zh-CN.md).
The opt-in terminal semantic-memory feedback loop is documented in [the English PR note](docs/pr/0001-v0.9-semantic-memory-feedback-loop.md)
and [Chinese version](docs/pr/0001-v0.9-semantic-memory-feedback-loop.zh-CN.md).
The expanded ACM/math fixture suite is documented in [the English PR note](docs/pr/0001-v1.0-acm-math-fixture-suite.md)
and [Chinese version](docs/pr/0001-v1.0-acm-math-fixture-suite.zh-CN.md).
The Level 4 robustness matrix is documented in [the English PR note](docs/pr/0001-v1.0-robustness-eval.md)
and [Chinese version](docs/pr/0001-v1.0-robustness-eval.zh-CN.md), with its fixed artifact in
[`experiments/v1.0-robustness-matrix`](experiments/v1.0-robustness-matrix/README.md).
- **v1.0 — Real-model evaluation and multi-agent:** run statistically defensible real-model coding
  evaluations with independent hidden tests and compare measurable multi-agent schedulers. The
  manual workflow is the reproducibility entry point, and `agent-eval-compare` now performs the
  protocol-checked cross-model analysis; the checked-in scripted artifacts remain controls rather
  than real-model evidence.

The provider transcript cassette is documented in [the English PR note](docs/pr/0001-v1.0-provider-transcript-replay.md)
and [Chinese version](docs/pr/0001-v1.0-provider-transcript-replay.zh-CN.md). It is an opt-in
debugging/replay boundary: it makes a completed provider trajectory inspectable without claiming
remote exactly-once execution or model-quality improvement.

See [Architecture](docs/architecture.md), [Runtime](docs/runtime.md), [Workspace knowledge base](docs/knowledge-base.md), and
[Evaluation protocol](docs/evaluation.md) for the design and claim boundaries.

## Contributing

Issues, adversarial fixtures, provider adapters, safe tools, context policies, and
reproducible coding tasks are welcome. Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before
opening a pull request.

## License

Apache License 2.0.
