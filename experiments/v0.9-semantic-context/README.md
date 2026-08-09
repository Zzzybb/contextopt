# v0.9 automatic semantic-context candidate artifact

This directory is a checked-in, provider-free conformance artifact for the opt-in
`versioned-v1+semantic` context path.

Regenerate it from the repository root with:

```text
python -m contextopt semantic-context-eval \
  --repetitions 3 --budgets 128,256,512 \
  --output experiments/v0.9-semantic-context/report.json \
  --markdown experiments/v0.9-semantic-context/report.md \
  --html experiments/v0.9-semantic-context/report.html
```

The fixture contains two scoped ACM/math tasks and two inherited global memories. Each
cell compiles the same messages twice, then recompiles from the receipt's durable-memory
snapshot without the live store. The matrix has 72 cells (four policies × three budgets ×
two cases × three repetitions).

The committed report records:

- `retrieved_recall_rate = 1.000`: every labelled target was in the lexical candidate set;
- `selected_recall_rate = 0.542`: selected retention after policy/budget competition;
- `budget_compliant_rate = 1.000`, `deterministic_rate = 1.000`, `replayable_rate = 1.000`;
- mean retrieved/selected/evicted candidates of `3.000 / 1.167 / 1.833`;
- `model_calls = 0` and `failed_count = 0`.

These are exact fixture-label and receipt metrics. They do not measure embedding quality,
semantic understanding, whether a model read or used a candidate, or coding-task success.
The `candidate_tokens` values use ContextOpt's stable provider-neutral estimate, not a
provider tokenizer. Real-provider multi-round evaluation remains a separate authorized
experiment.

Files:

- [`report.json`](report.json) — complete machine-readable ledger;
- [`report.md`](report.md) — reviewable tabular summary;
- [`report.html`](report.html) — self-contained dashboard with the JSON ledger;
- [`README.zh-CN.md`](README.zh-CN.md) — Chinese explanation.
