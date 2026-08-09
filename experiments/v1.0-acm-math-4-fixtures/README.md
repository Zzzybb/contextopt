# v1.0 ACM/math four-fixture scripted control artifact

This directory is the checked-in deterministic baseline for the v1.0 coding-agent matrix.

- Fixtures: `two-sum` and `merge-intervals` (ACM algorithms), plus `extended-gcd` and
  `modular-inverse` (mathematics)
- Strategies: `single_pass`, `best_of_n`, and `orchestrated`
- Repetitions: `3` per fixture/strategy cell (`36` ledger rows)
- Hidden graders: enabled and kept outside model-visible snapshots
- Outputs: `report.json`, `report.md`, `report.html`, `checkpoint.json`, and `manifest.json`

Observed scripted control results are intentionally simple: `single_pass` accepts `0/12`
cells, while both stronger strategies accept `12/12` visible cells and pass all `12/12`
independent hidden graders. This demonstrates the evaluator's oracle gates and accounting,
not model capability or multi-agent value.

Regenerate it from the repository root with:

```text
python -m contextopt agent-eval --fixtures all --repetitions 3 \
  --output experiments/v1.0-acm-math-4-fixtures/report.json \
  --markdown experiments/v1.0-acm-math-4-fixtures/report.md \
  --html experiments/v1.0-acm-math-4-fixtures/report.html \
  --checkpoint experiments/v1.0-acm-math-4-fixtures/checkpoint.json \
  --manifest experiments/v1.0-acm-math-4-fixtures/manifest.json
```

The first real-provider run must keep the same fixture IDs, strategy set, budgets, prompts,
and at least three repetitions before comparing against this control artifact.

The manifest pins the source revision used for this baseline
(`9b1bee2e96bf8fc2a77c513886102b9c3f2c90a8`). The checked-in report was regenerated after the
provider cancellation and candidate-execution sandbox boundary work; its config records the default
`sandbox=host` and image identity, while only provider-free scripted control data is included.
