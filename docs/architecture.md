# Architecture

## Design rule

State, context, memory, and knowledge are separate concepts:

- **State** is authoritative execution data such as task status and test results.
- **Knowledge** is versioned external evidence such as source code and documentation.
- **Memory** is selectively retained experience such as decisions and failed attempts.
- **Context** is the bounded view compiled for one model invocation.

ContextOpt v0.1 implements the last step as a pure function:

```text
SelectionProblem + ContextPolicy -> ContextFrame + SelectionReceipt
```

This makes policies deterministic, testable, and replayable without an LLM.

## Current components

### ContextItem

A typed candidate with cost, source, semantic signals, and graph constraints. Critical-fact
labels live only in benchmark cases and are never exposed to policies.

### SelectionProblem

Owns the token budget, validates identifiers, computes dependency closure and feasibility,
and exposes a transparent set objective.

### ContextPolicy

An interchangeable optimizer. Policies receive the same immutable problem and return a
fully auditable `ContextFrame`.

### ExactOraclePolicy

Enumerates all subsets up to a safe candidate limit. It exists to measure heuristic quality
on small problems, not to serve production traffic.

### Benchmark

Generates paired cases from fixed seeds, runs every policy on the same case, validates every
result, and stores raw per-run metrics along with summaries.

## Planned runtime integration

```text
Run/Event Log
     ├── authoritative task state
     ├── recent tool observations
     └── checkpointed workspace reference
                 ↓
Candidate Builder ← Repository Graph / Memory Store
                 ↓
         Dynamic Context Graph
                 ↓
          ContextOpt Policy
                 ↓
       ContextFrame + Receipt
                 ↓
             Model Call
                 ↓
       Tool/Test Outcome Event
```

Every future model request will persist its exact `ContextFrame`, allowing a run to be
replayed and forked under another policy.

## Claim boundaries

The current objective is a research baseline. Its score is not treated as evidence that a
coding task will succeed. End-to-end claims require executable hidden tests under paired,
fixed-budget experiments.
