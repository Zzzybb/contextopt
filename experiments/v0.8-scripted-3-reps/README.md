# v0.8 scripted three-repetition control artifact

This directory is a checked-in, deterministic control run for the ContextOpt coding-agent
evaluation harness.

- Fixtures: `two-sum` (ACM algorithm) and `extended-gcd` (mathematics)
- Strategies: `single_pass`, `best_of_n`, and `orchestrated`
- Repetitions: `3` per fixture/strategy cell
- Hidden graders: enabled and kept outside model-visible snapshots
- Outputs: `report.json`, `report.md`, `report.html`, `checkpoint.json`, and `manifest.json`

The report is useful for reviewing paired statistics, visible/hidden oracle gates, token and test
budgets, and checkpoint shape. It is not a real-model benchmark: responses are deterministic
`ScriptedModel` outputs, so it says nothing about general coding ability, provider reliability, or
production safety.

Regenerate it from the repository root with:

```text
python -m contextopt agent-eval \
  --fixtures all --repetitions 3 \
  --output experiments/v0.8-scripted-3-reps/report.json \
  --markdown experiments/v0.8-scripted-3-reps/report.md \
  --html experiments/v0.8-scripted-3-reps/report.html \
  --checkpoint experiments/v0.8-scripted-3-reps/checkpoint.json \
  --manifest experiments/v0.8-scripted-3-reps/manifest.json
```
