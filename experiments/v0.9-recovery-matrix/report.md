# Long-horizon recovery fault-injection evaluation

This report stops a fresh run after a durable event, opens the same log with a new runner, and records the recovery contract that follows.

| Scenario | Fault boundary | Pending before resume | First resume | Final | Model calls before/after | Tool events | Result |
|---|---|---|---|---|---:|---|---|
| `pending-model-request` | `model.requested` | model | `completed` | `completed` | 0/1 | none | PASS |
| `durable-final-response` | `model.responded` | none | `completed` | `completed` | 1/0 | none | PASS |
| `reconcile-write-before-effect` | `tool.started` | replace_text | `completed` | `completed` | 1/1 | tool.completed=1, tool.started=1 | PASS |
| `durable-write-result` | `tool.completed` | none | `completed` | `completed` | 1/1 | tool.completed=1, tool.started=1 | PASS |
| `nonreplayable-test-pause` | `tool.started` | run_tests | `paused` | `completed` | 1/1 | tool.failed=1, tool.started=1 | PASS |

## Durable event evidence

### `pending-model-request`

Before resume: `run.started` → `model.requested`

After resume: `run.interrupted` → `run.resumed` → `model.responded` → `budget.updated` → `run.completed`

### `durable-final-response`

Before resume: `run.started` → `model.requested` → `model.responded`

After resume: `budget.updated` → `run.interrupted` → `run.resumed` → `run.completed`

### `reconcile-write-before-effect`

Before resume: `run.started` → `model.requested` → `model.responded` → `budget.updated` → `tool.started`

After resume: `run.interrupted` → `run.resumed` → `tool.completed` → `model.requested` → `model.responded` → `budget.updated` → `run.completed`

### `durable-write-result`

Before resume: `run.started` → `model.requested` → `model.responded` → `budget.updated` → `tool.started` → `tool.completed`

After resume: `run.interrupted` → `run.resumed` → `model.requested` → `model.responded` → `budget.updated` → `run.completed`

### `nonreplayable-test-pause`

Before resume: `run.started` → `model.requested` → `model.responded` → `budget.updated` → `tool.started`

After resume: `run.interrupted` → `run.resumed` → `run.paused` → `run.resumed` → `tool.failed` → `model.requested` → `model.responded` → `budget.updated` → `run.completed`

## Claim boundary

> This is a deterministic recovery-contract evaluation with injected process-like stops and ScriptedModel responses. It measures durable event boundaries, pending work reconstruction, conservative tool replay, pause/resolution semantics, and duplicate-effect accounting on one local fixture; it does not prove arbitrary machine-loss recovery, exactly-once provider effects, sandbox security, or model coding quality.

Summary: **5/5 scenarios passed**.
