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

Every boundary ──> durable JSONL event log
```

> **Status — v0.2a runtime foundation:** the repository now contains a real single-agent
> read/edit/test loop, deterministic scripted-model evaluation, an optional
> OpenAI-compatible adapter, workspace-bounded tools, budgets, and durable traces. It does
> **not** yet implement checkpoint/resume, learned memory, multi-agent scheduling, branch
> search, or end-to-end context optimization inside the runtime.

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

### ForgeAgent runtime — v0.2a

- Provider-neutral messages, tool definitions, tool calls, responses, and token usage.
- A bounded asynchronous `AgentRunner` with turn, tool-call, token, wall-clock, command,
  and tool-output limits.
- Observation-aware `ScriptedModel` runs that need no network or API key.
- A minimal non-streaming OpenAI-compatible Chat Completions adapter.
- Workspace-bounded file listing, literal search, numbered reads, file creation, atomic
  SHA-256 compare-and-swap replacement, and pre-registered visible-test commands.
- Explicit write and command permissions; both are disabled unless enabled by the caller.
- Append-only, versioned JSONL events plus a compact trace renderer.
- Failure semantics that distinguish a test process returning exit code 1 from a tool or
  runtime infrastructure failure.
- A deterministic Runtime Conformance Eval covering read, edit, failing tests, passing
  tests, budgets, and partial traces after model failure.

### ContextOpt engine — v0.1

- Immutable context candidates with provenance, token cost, dependencies, conflicts,
  freshness, topics, and duplicate groups.
- Top-K, density, exact 0/1 knapsack, graph-aware submodular greedy, and a small-instance
  exhaustive oracle.
- Deterministic paired synthetic experiments with hidden critical-fact labels.
- Machine-readable selection receipts explaining every accepted and rejected candidate.

## What is deliberately not implemented yet

- Checkpoint/resume or reconstruction of a live run from the event log.
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
```

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
├── runtime/              # model protocol, AgentRunner, tools, events, limits
├── models.py             # context candidates, constraints, receipts
├── policies/             # interchangeable selection algorithms
├── synthetic.py          # deterministic context microbench generation
├── benchmark.py          # paired optimizer metrics and reports
└── cli.py                # run, trace, pack, and benchmark commands

tests/                    # standard-library unit and integration tests
examples/runtime_demo/    # offline scripted coding-loop demonstration
examples/auth_context.json
experiments/              # checked-in optimizer configurations and raw results
docs/                     # architecture, runtime, and evaluation contract
```

## Roadmap

- **v0.1 — Context optimizer:** algorithms, receipts, oracle, and synthetic microbench.
- **v0.2a — Runtime foundation:** current read/edit/test loop, limits, JSONL trace, offline
  conformance evaluation, and optional model adapter.
- **v0.2b — Recoverable execution:** event replay, workspace checkpoint references,
  idempotency, interruption, and resume.
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
