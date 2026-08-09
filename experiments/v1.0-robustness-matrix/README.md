# v1.0 Level 4 runtime robustness matrix

This directory contains a deterministic, provider-free fault-injection ledger for the
long-horizon runtime boundaries that are easy to hide behind a happy-path demo.

- Scenarios: oversized tool-output compaction, source-aware stale-memory invalidation,
  duplicate tool-result rejection, and stale compare-and-swap write conflict
- Result: `4/4` scenarios passed
- Outputs: `report.json`, `report.md`, `report.html`, and `manifest.json`

The matrix proves local contract behavior and records the observed evidence. It does not
claim machine-loss recovery, sandbox security, provider reliability, exactly-once external
effects, or coding quality.

Regenerate it from the repository root with:

```text
python -m contextopt robustness-eval --scenarios all \
  --output experiments/v1.0-robustness-matrix/report.json \
  --markdown experiments/v1.0-robustness-matrix/report.md \
  --html experiments/v1.0-robustness-matrix/report.html \
  --manifest experiments/v1.0-robustness-matrix/manifest.json
```
