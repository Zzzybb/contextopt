# PR #1 — ContextOpt / ForgeAgent evolution

## What changed

This pull request grows the repository from a small context-selection benchmark into an
auditable coding-agent laboratory:

- typed context candidates, graph/conflict constraints, transparent selection receipts, and
  deterministic Top-K, density, knapsack, submodular, and oracle policies;
- a provider-neutral ForgeAgent runtime with bounded tools, visible-test execution, event-sourced
  recovery, cross-platform run leases, strict projection checkpoints, and resumable interrupted
  tool handling;
- live context compilation with protocol-atomic blocks, deterministic token estimates,
  versioned observed-memory invalidation, per-turn receipts, and model-free conformance evals;
- immutable test-guided candidate snapshots, fixed-environment workspace deduplication, bounded
  beam search, hash-chained search decisions, and dependency-free console/Markdown/SVG reports;
- a strict model-to-candidate JSON boundary that rejects malformed or unsafe proposals before
  they reach the visible-test oracle;
- a durable iterative `search-session` loop that carries bounded test feedback into later model
  rounds, persists budgets and pending phases, caches duplicate test observations, and resumes
  with an explicit retry decision when a provider call was interrupted;
- an explicit `apply-best` / `rollback-best` operator boundary with path and symlink checks,
  stale-baseline conflict detection, atomic replacement, and tamper-evident receipts.

## Why

Long-horizon coding agents need more than a prompt and a generated patch. They need to preserve
state across turns, distinguish observations from model claims, spend tests and tokens under one
budget, survive interruption, and make side effects reviewable. This PR keeps those concerns
separate:

1. Context selection is measurable independently of model quality.
2. Runtime execution is event-sourced and recoverable independently of the search algorithm.
3. Candidate generation is parsed strictly before any code is executed.
4. Search evaluates immutable snapshots and reuses equivalent test states.
5. Applying a result to a real checkout is never implicit; an operator must authorize it and the
   baseline must still match.

The provider boundary is intentionally at-least-once when a model response is lost. The
checkpoint records `proposing` and requires `--retry-pending`; it does not claim exactly-once
delivery from an external provider.

## Developer impact

The repository now supports these offline flows without an API key:

```text
python -m unittest discover -s tests -v
python -m contextopt branch-search
python -m contextopt search-session --help
python -m contextopt apply-best --help
python -m contextopt rollback-best --help
```

The checked-in branch demo can run a model proposal in a disposable workspace, record a
session checkpoint/report, and then apply or roll back the accepted snapshot only when the
operator supplies `--allow-write`.

## Validation

The deterministic quality gates for this pull request are:

```text
python -m ruff check src tests
python -m ruff format --check src tests
python -m mypy src/contextopt
python -m unittest discover -s tests -q
python scripts/ci_checks.py
git diff --check
```

The session conformance tests assert feedback propagation, shared model/candidate/test
budgets, cross-round cache reuse, pending-request resume, event/metric tamper detection, and
checkpoint round-tripping. The apply tests assert create/modify/delete accounting, stale
baseline refusal, explicit write permission, path restrictions, rollback, and receipt integrity.

## Claim boundaries and non-goals

- Scripted candidates prove control-flow and accounting conformance, not model reasoning or
  SWE-bench performance.
- Visible-test success is not hidden-test success.
- The executable adapter and apply boundary run trusted local host processes; they are not an
  OS/container sandbox.
- The session is one model client with bounded iterative rounds, not planner/coder/reviewer
  multi-agent scheduling.
- Apply/rollback is a local filesystem safety boundary, not distributed versioning or a general
  exactly-once transaction protocol.
- Learned cross-run semantic memory, a trace UI, and statistically powered real-model results
  remain future work.
