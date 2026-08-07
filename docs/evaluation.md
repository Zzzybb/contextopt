# Evaluation protocol

ContextOpt separates optimizer quality from downstream agent quality.

## Level 0: solver correctness

- Validate mandatory, budget, dependency, and conflict constraints.
- Compare heuristic objective values with the exact oracle.
- Report approximation ratio and latency distributions.
- Use deterministic seeds and preserve raw results.

## Level 1: context quality

Synthetic cases label critical facts outside the policy-visible data. Metrics include:

- critical-fact recall;
- objective/oracle ratio;
- duplicate-token ratio;
- stale-item ratio;
- token-budget utilization.

Critical recall and objective score are reported separately so surrogate misalignment
cannot be hidden by a composite score.

## Level 2: controlled coding tasks (planned)

Use the same model snapshot, system prompt, tools, repository commit, maximum steps, and
token budget for every policy. The primary outcome is executable hidden-test success.

Secondary metrics:

- total input and output tokens;
- repeated file reads and tool calls;
- visible test-progress delta;
- wall-clock latency;
- constraint retention after compaction.

At least three repetitions per task will be used for final comparisons. Results will report
paired bootstrap confidence intervals instead of only point estimates.

## Level 3: robustness (planned)

Inject process termination, forced compaction, stale memory, conflicting facts, duplicated
tool results, and model timeouts. Measure recovery success and extra steps after recovery.

## Reproduction

The two v0.1 reports can be regenerated with the commands in the README. Objective and
selection metrics are deterministic. Runtime measurements depend on the host and are kept
only as local diagnostics.
