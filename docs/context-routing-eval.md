# Context routing/compiler conformance evaluation

This evaluation checks deterministic properties of the local context compiler. It is a
conformance suite, not a model benchmark: it does not call a language model, execute a
coding task, or measure whether generated code is correct.

## Question answered

Given the same long, protocol-valid coding trace and the same estimated-token budget,
what context does each routing policy retain, and does the resulting request satisfy the
compiler contract?

The cases are fixed fixtures rather than sampled workloads. Each fixture contains a user
request, many assistant/tool exchanges, unrelated observations, and a small set of gold
evidence probes. Gold probes are used only by the scorer; they are not passed to the
routing policy.

Every policy receives the identical source trace, query, and budget. Token counts use the
runtime compiler's deterministic estimator, so “token” throughout this report means an
estimated token, not a provider tokenizer result.

The default matrix compares `recent`, `topk`, `density`, and `submodular` at 512 and
1,024 estimated tokens. The `full` policy can be requested explicitly; when a long trace
does not fit, the run is recorded as unsupported with null score fields. This is expected
policy behavior, not silently converted into a zero-quality result.

## Metrics

- **Evidence recall** is the fraction of gold evidence probes whose complete, exact text
  remains in the compiled messages. A compacted block counts only when the probe itself is
  still present.
- **Protocol validity** checks that every emitted tool result belongs to the immediately
  preceding assistant tool-call group, each call is answered exactly once and in declared
  order, and no tool-call group is left unresolved.
- **Budget compliance** is true when the compiled estimate is no greater than the fixed
  budget. The report also records the absolute overage, if any.
- **Compression ratio** is `1 - compiled_estimated_tokens / original_estimated_tokens`.
  Higher compression means less of the source trace was retained; it is not inherently
  better.
- **Determinism** recompiles the same case and compares canonical JSON for the messages
  and compiler receipt. It reports equality, not statistical reproducibility.

Policy/budget summaries use macro averages across cases. Validity, compliance, and
determinism are reported as rates so one failing case cannot be hidden by aggregate token
counts.

## Interpretation limits

Evidence recall measures retention of deliberately labelled facts, not whether a model
would notice or use them. Protocol validity and budget compliance are hard conformance
properties. Compression describes size reduction only. Determinism says identical local
inputs produce identical compiler output; it does not imply semantic stability after a
trace, policy, or estimator change.

The fixtures are intentionally transparent and regression-oriented. They do not represent
the distribution of real repositories, provider tokenization, model quality, end-to-end
latency, or coding success. Results therefore must not be presented as an LLM leaderboard,
quality win, or production success rate.

## Checked-in reference result

The default three fixed traces contain more than 10,000 estimated tokens each. Running
the four bounded policies with two recent blocks pinned produced this deterministic
summary:

| Budget | Policy | Evidence recall | Protocol valid | Budget compliant | Compression |
|---:|---|---:|---:|---:|---:|
| 512 | recent | 0.000 | 1.000 | 1.000 | 0.960 |
| 512 | topk | 0.000 | 1.000 | 1.000 | 0.960 |
| 512 | density | 0.000 | 1.000 | 1.000 | 0.960 |
| 512 | submodular | 0.000 | 1.000 | 1.000 | 0.960 |
| 1,024 | recent | 0.222 | 1.000 | 1.000 | 0.910 |
| 1,024 | topk | 0.222 | 1.000 | 1.000 | 0.910 |
| 1,024 | density | 0.222 | 1.000 | 1.000 | 0.910 |
| 1,024 | submodular | 0.778 | 1.000 | 1.000 | 0.910 |

The 512-token result is useful rather than flattering: the mandatory task and recent
protocol blocks leave too little room for any labelled probe. At 1,024 tokens the
set-aware policy retains more labelled evidence than the sliding-window baseline on these
fixtures. That demonstrates a measurable routing difference and a regression target. It
does not establish that a model will use the evidence or solve more coding tasks.

## Running

Run the focused suite from the repository root:

```text
python -m unittest tests.test_context_routing_eval
```

Generate the same summary and optional raw artifacts through the CLI:

```text
python -m contextopt context-eval \
  --policies recent,topk,density,submodular \
  --budgets 512,1024 --repetitions 3 \
  --output context-routing.json --markdown context-routing.md
```

The evaluation API returns JSON-compatible dictionaries with no timestamps or timing
measurements, allowing reports to be compared byte-for-byte in CI. Each report records the
context compiler ABI version so results from changed estimation or selection semantics are
not silently compared as if they came from the same compiler.

```python
from contextopt.evaluation.context_routing import (
    ContextRoutingEvalConfig,
    run_context_routing_evaluation,
)

report = run_context_routing_evaluation(ContextRoutingEvalConfig(repetitions=3))
```

Each run includes the compiler receipt, independently recomputed token counts, exact probe
counts, conformance booleans, and a SHA-256 of the canonical compiled messages plus
receipt. The report's `model_calls` field is always zero.
