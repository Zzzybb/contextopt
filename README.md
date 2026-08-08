# ForgeAgent / ContextOpt

**An auditable long-horizon coding-agent runtime with an algorithmic context engine.**

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

> **Status — v0.5 model-to-candidate boundary:** the repository now contains a real single-agent
> read/edit/test loop, recoverable event-sourced execution, a deterministic live context
> compiler, an auditable beam search over generated coding candidates, and a strict
> model-to-candidate proposal boundary. A model can now return bounded complete workspace
> snapshots that are parsed, rejected on protocol violations, and handed to the same
> visible-test oracle. The branch layer deduplicates states, spends test calls once per
> state, and emits a self-contained SVG/JSON/Markdown report. It does **not** yet
> implement multi-agent scheduling, an OS sandbox, learned semantic memory, or a
> statistically powered real-model coding benchmark.

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
- Workspace-bounded file listing, literal search, numbered reads, file creation, atomic
  SHA-256 compare-and-swap replacement, and pre-registered visible-test commands.
- Explicit write and command permissions; both are disabled unless enabled by the caller.
- Schema-2 append-only JSONL events with a verifiable SHA-256 chain, strict sequence and
  run-id validation, explicit truncated-tail repair, and a compact trace renderer.
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

### Test-guided branch search — v0.4

- Immutable candidate workspace snapshots with parent hypotheses and evidence references.
- Deterministic beam search ordered by visible-test progress, with explicit candidate,
  test-call, depth, and beam budgets.
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

The event hash chain provides verifiable, corruption-evident integrity. It is not a
signature or malicious-rewrite defense: someone who can replace the entire log can also
recompute the complete chain because there is no secret or external trust anchor.

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

- Automatic workspace snapshots, rollback, migration to another workspace, or distributed
  coordination. Resume operates on the same configured workspace and validates what it can.
- Planner/coder/reviewer role orchestration, parallel agents, PatchTree/MCTS scheduling,
  or automatic multi-turn proposal/search/rollback orchestration. The v0.5 proposal
  boundary is intentionally one-shot and provider-neutral, not yet a multi-agent
  scheduler.
- A container or virtual-machine security boundary. Workspace path checks and permission
  flags reduce accidental access, but are not an OS sandbox. Registered test commands are
  trusted host processes.
- A general shell tool, autonomous package installation, or unrestricted network access.
- A claim that the scripted demo measures model reasoning or real-world issue resolution.
- Learned or cross-run semantic memory, a trace UI, or a statistically powered real-model
  coding benchmark. The branch demo and proposal conformance tests do not pretend
  synthetic observations or protocol acceptance are model coding accuracy.
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
| `--context-memory` | `versioned-v1` | Observed evidence invalidation; use `none` to disable it |

These are compiler estimates, not counts from a provider tokenizer. The chosen context
configuration and its fingerprint are persisted with the run; `resume` reconstructs that
configuration from the event log rather than accepting replacement context flags.

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
├── runtime/              # runner, live context/memory, recovery, tools, events
├── evaluation/           # model-free context routing/compiler conformance
├── search/                # proposal boundary, branch search, dedup, and renderers
├── models.py             # context candidates, constraints, receipts
├── policies/             # interchangeable selection algorithms
├── synthetic.py          # deterministic context microbench generation
├── benchmark.py          # paired optimizer metrics and reports
└── cli.py                # run/resume/status/trace, propose-case, branch-search, pack

tests/                    # standard-library unit and integration tests
examples/runtime_demo/    # offline scripted coding-loop demonstration
examples/auth_context.json
experiments/              # checked-in optimizer configurations and raw results
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
- **v1.0 — Agent DevTools:** connect proposal/search to the runtime's recovery/rollback
  protocol, then run statistically defensible real-model evaluations and add multi-agent
  scheduling.

See [Architecture](docs/architecture.md), [Runtime](docs/runtime.md), and
[Evaluation protocol](docs/evaluation.md) for the design and claim boundaries.

## Contributing

Issues, adversarial fixtures, provider adapters, safe tools, context policies, and
reproducible coding tasks are welcome. Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before
opening a pull request.

## License

Apache License 2.0.
