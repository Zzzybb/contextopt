# ForgeAgent / ContextOpt

**An auditable long-horizon coding-agent runtime with an algorithmic context engine.**

ForgeAgent is the runtime direction of this project: take a coding task, let a model inspect
and change one workspace through bounded tools, feed every observation back into the next
turn, and preserve an append-only trace of what happened. ContextOpt remains the repository,
Python package, CLI, and context-selection engine that will eventually compile the bounded
view for each ForgeAgent model call.

```text
Task ──> bounded AgentRunner ──> ModelClient ──> structured tool calls
              ^                                      │
              └──────────── tool observations <──────┘
                             │
                  workspace + visible tests

Every boundary ──> durable schema-2 JSONL event log + verified state projection
```

> **Status — v0.2b recoverable runtime:** the repository now contains a real single-agent
> read/edit/test loop, a corruption-evident event chain, strict state reconstruction,
> resumable interrupted runs, conservative tool reconciliation, deterministic scripted
> evaluation, and an optional OpenAI-compatible adapter. It does **not** yet implement live
> ContextOpt policies, learned memory, multi-agent scheduling, branch search, or an OS
> sandbox.

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

The current runtime intentionally uses its message history directly. Connecting
`ContextFrame` selection to live model requests is a later milestone; the project does not
claim that the v0.1 optimizer already improves coding success.

## What is implemented

### ForgeAgent runtime — v0.2b

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

The event hash chain provides verifiable, corruption-evident integrity. It is not a
signature or malicious-rewrite defense: someone who can replace the entire log can also
recompute the complete chain because there is no secret or external trust anchor.

### ContextOpt engine — v0.1

- Immutable context candidates with provenance, token cost, dependencies, conflicts,
  freshness, topics, and duplicate groups.
- Top-K, density, exact 0/1 knapsack, graph-aware submodular greedy, and a small-instance
  exhaustive oracle.
- Deterministic paired synthetic experiments with hidden critical-fact labels.
- Machine-readable selection receipts explaining every accepted and rejected candidate.

## What is deliberately not implemented yet

- Automatic workspace snapshots, rollback, migration to another workspace, or distributed
  coordination. Resume operates on the same configured workspace and validates what it can.
- Context compaction, memory lifecycle, candidate extraction from runtime events, or a
  `ContextPolicy` wired into model requests.
- Planner/coder/reviewer role orchestration, parallel agents, PatchTree, beam search, or
  MCTS.
- A container or virtual-machine security boundary. Workspace path checks and permission
  flags reduce accidental access, but are not an OS sandbox. Registered test commands are
  trusted host processes.
- A general shell tool, autonomous package installation, or unrestricted network access.
- A claim that the scripted demo measures model reasoning or real-world issue resolution.
- A trace UI or statistically powered real-model coding benchmark.
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

Because the solution path is scripted, this result must not be reported as model coding
accuracy, SWE-bench performance, or evidence that one context policy beats another. See the
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
├── runtime/              # runner, recovery reducer, tool plans, events, limits
├── models.py             # context candidates, constraints, receipts
├── policies/             # interchangeable selection algorithms
├── synthetic.py          # deterministic context microbench generation
├── benchmark.py          # paired optimizer metrics and reports
└── cli.py                # run, status, resume, trace, pack, benchmark

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
- **v0.3 — Live context engine:** candidate extraction, compaction, memory invalidation,
  ContextOpt policies, and per-turn context receipts.
- **v0.4 — Test-guided search:** isolated branches, duplicate-state detection, adaptive
  budget allocation, and controlled policy comparisons.
- **v1.0 — Agent DevTools:** real coding-task benchmark, trace/search visualization, and
  statistically defensible evaluations.

See [Architecture](docs/architecture.md), [Runtime](docs/runtime.md), and
[Evaluation protocol](docs/evaluation.md) for the design and claim boundaries.

## Contributing

Issues, adversarial fixtures, provider adapters, safe tools, context policies, and
reproducible coding tasks are welcome. Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before
opening a pull request.

## License

Apache License 2.0.
