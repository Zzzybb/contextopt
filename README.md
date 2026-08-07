# ContextOpt

**Algorithmic context optimization and evaluation for long-horizon coding agents.**

ContextOpt treats context construction as a constrained set-selection problem instead of
an embedding `top-k` call. It is an early, research-oriented foundation for a coding-agent
runtime that can answer three questions on every model turn:

1. Which state, code, memory, and trajectory items entered the context?
2. Why were they selected or rejected under the token budget?
3. Did that decision improve an executable downstream outcome?

> **Status:** `v0.1` is an offline, model-free optimization lab. It intentionally does not
> claim to improve an end-to-end coding agent yet. The checked-in experiments establish
> baselines and expose where a hand-designed objective disagrees with critical-fact labels.

## Why this project exists

Long-running agents accumulate more potentially useful information than a model can see at
once: task constraints, code, tests, tool output, failed attempts, decisions, and memories.
Ranking every item independently misses several properties of the actual packing problem:

- two individually relevant chunks may be redundant;
- a code span can require an interface or definition to be useful;
- stale memory may conflict with current code;
- a long observation can crowd out several complementary facts;
- semantic relevance is only a proxy for downstream utility.

ContextOpt's current transparent objective is:

```text
maximize  relevance(S) + importance(S) + freshness(S)
          + topic_coverage(S) - duplicate_penalty(S)

subject to token_cost(S) <= budget
           mandatory items are included
           dependencies are closed
           conflicting items are not co-selected
```

The long-term goal is to replace hand-tuned utility with feedback learned from compilation,
tests, task progress, and counterfactual trajectory forks.

## What is implemented

- Immutable `ContextItem` candidates with provenance and token cost.
- Mandatory, dependency, conflict, topic, freshness, and duplicate-group signals.
- A shared `ContextPolicy` interface.
- Standalone relevance `top-k` and relevance-per-token baselines.
- Exact additive 0/1 knapsack dynamic programming.
- Dependency-aware marginal-gain/submodular greedy selection.
- An exhaustive exact oracle for small instances.
- Deterministic synthetic tasks with hidden critical-fact labels.
- Paired evaluation across policies and budgets.
- Machine-readable selection receipts for every accepted and rejected item.
- A dependency-free Python CLI and standard-library test suite.

## Quick start

ContextOpt requires Python 3.11 or newer and has no runtime dependencies.

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

Run the checked-in independent-item experiment:

```bash
contextopt benchmark \
  --instances 100 \
  --items 14 \
  --critical 4 \
  --budget 800 \
  --seed 42
```

Run a graph-constrained experiment:

```bash
contextopt benchmark \
  --instances 100 \
  --items 14 \
  --budget 900 \
  --seed 4242 \
  --graph-rate 0.2 \
  --conflict-rate 0.02 \
  --policies topk,density,submodular,oracle
```

Pack a concrete context problem and inspect its selection receipt:

```bash
contextopt pack examples/auth_context.json --policy submodular
```

## First reproducible result

The graph-constrained baseline contains 100 paired synthetic instances, 14 candidates per
instance, and a 900-token budget. The full raw report is checked in at
[`experiments/v0.1-graph.json`](experiments/v0.1-graph.json).

| Policy | Objective / oracle | Critical recall | Redundancy | Budget used |
|---|---:|---:|---:|---:|
| Top-K | 0.981 | 0.975 | 0.005 | 0.946 |
| Density | 0.984 | 0.922 | 0.010 | 0.929 |
| Submodular | 0.991 | 0.945 | 0.000 | 0.930 |
| Exact oracle | 1.000 | 0.948 | 0.000 | 0.955 |

### Read this result correctly

The graph-aware policy improves the declared set objective and removes duplicate context.
It does **not** beat Top-K on the synthetic critical-fact label. Even the exact objective
oracle has lower critical recall than Top-K in this run.

That is not hidden as a failed experiment. It is the first useful finding: optimizing a
clean mathematical surrogate is insufficient when the surrogate is misaligned with task
success. The next milestone therefore learns utility from executable feedback rather than
adding more hand-tuned weights.

Timing columns in checked-in reports are local diagnostic measurements, not cross-machine
performance claims.

## Selection receipts

Every policy emits a `ContextFrame` with one decision per candidate:

```json
{
  "item_id": "old-auth-doc",
  "status": "rejected",
  "reason": "conflict: jwt-interface vs old-auth-doc",
  "marginal_gain": null,
  "added_tokens": null
}
```

This receipt is the basis of the planned Context DevTools UI: users will be able to inspect
what survived into context, what was evicted, and how an alternate policy would differ.

## Repository layout

```text
src/contextopt/
├── models.py             # candidates, constraints, objective, receipts
├── policies/             # interchangeable selection algorithms
├── synthetic.py          # deterministic benchmark generation
├── benchmark.py          # paired metrics and reports
└── cli.py                # benchmark and pack commands

tests/                    # deterministic unit and integration tests
examples/                 # human-readable packing problems
experiments/              # checked-in configurations and raw results
docs/                     # architecture and evaluation protocol
```

## Roadmap

- **v0.1 — Offline optimizer:** current pull request.
- **v0.2 — Coding-agent trace:** event log, tool observations, and candidate extraction.
- **v0.3 — Long-horizon state:** checkpoint/resume, compaction, and memory lifecycle.
- **v0.4 — Online utility:** contextual bandit trained from visible test progress.
- **v0.5 — Context Fork:** paired rollouts from the same workspace checkpoint.
- **v1.0 — Agent DevTools:** end-to-end controlled coding benchmark and trace UI.

See [Architecture](docs/architecture.md) and [Evaluation protocol](docs/evaluation.md) for
the concrete design and claim boundaries.

## Contributing

Issues, counterexamples, new policies, and reproducible benchmark tasks are welcome. Read
[`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a pull request.

## License

Apache License 2.0.
