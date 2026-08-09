# v0.9 long-horizon recovery matrix

This directory contains a checked-in, deterministic fault-injection run for the
event-sourced coding-agent runtime. Each scenario stops after a durable event,
closes the first runner, and resumes the same log with a fresh runner, model adapter,
and tool registry.

| Scenario | Injected boundary | Recovery contract |
|---|---|---|
| `pending-model-request` | `model.requested` | reuse the pending request |
| `durable-final-response` | `model.responded` | finish without another model call |
| `reconcile-write-before-effect` | `tool.started` / `replace_text` | reconcile the precondition and retry safely |
| `durable-write-result` | `tool.completed` / `replace_text` | avoid a duplicate write |
| `nonreplayable-test-pause` | `tool.started` / `run_tests` | pause, then explicitly `mark_failed` |

All five scenarios pass. The JSON report preserves pending-state reconstruction,
event-type evidence, model-call counts, tool-event counts, and the final fixture
digest. The HTML report is self-contained and embeds the same JSON ledger. The
manifest records the protocol and claim boundary without secrets.

Regenerate from the repository root:

```text
python -m contextopt recovery-eval \
  --output experiments/v0.9-recovery-matrix/report.json \
  --markdown experiments/v0.9-recovery-matrix/report.md \
  --html experiments/v0.9-recovery-matrix/report.html \
  --manifest experiments/v0.9-recovery-matrix/manifest.json
```

This is a runtime recovery conformance artifact, not evidence of model coding
quality, arbitrary machine-loss recovery, exactly-once provider effects, sandbox
security, or production safety.
