# ContextOpt coding-agent strategy evaluation

This report uses executable ACM/math fixtures, independent hidden tests, and deterministic scripted model responses.

| Strategy | Visible | Visible 95% CI | Hidden | Mean model calls | Mean visible tests | Mean reuses | Mean candidates | Mean tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `single_pass` | 0/6 (0%) | 0%-39% | 0/0 (n/a) | 1.0 | 1.0 | 0.0 | 1.0 | 160 |
| `best_of_n` | 6/6 (100%) | 61%-100% | 6/6 (100%) | 1.0 | 2.0 | 0.0 | 2.0 | 240 |
| `orchestrated` | 6/6 (100%) | 61%-100% | 6/6 (100%) | 6.0 | 2.0 | 0.0 | 2.0 | 1160 |

## Paired comparisons

Each delta is candidate minus the baseline on the same fixture and repetition; positive visible delta means more paired wins.

| Strategy | Baseline | Paired | Wins | Losses | Ties | Visible Δ | Hidden Δ | Mean tests Δ (stdev) | Mean tokens Δ (stdev) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `best_of_n` | `single_pass` | 6 | 6 | 0 | 0 | +100% | n/a | +1.0 (stdev 0.0) | +80 (stdev 0) |
| `orchestrated` | `single_pass` | 6 | 6 | 0 | 0 | +100% | n/a | +1.0 (stdev 0.0) | +1000 (stdev 0) |

## Fixtures

| ID | Category | Task |
|---|---|---|
| `two-sum` | acm-algorithm | Two Sum with duplicate values |
| `extended-gcd` | mathematics | Extended Euclidean algorithm |

## Run ledger

| Fixture | Strategy | Status | Visible tests | Hidden | Best candidate | Error |
|---|---|---|---:|---:|---|---|
| `two-sum` | `single_pass` | budget_exhausted | 1 | n/a | round-0-single-bad |  |
| `two-sum` | `single_pass` | budget_exhausted | 1 | n/a | round-0-single-bad |  |
| `two-sum` | `single_pass` | budget_exhausted | 1 | n/a | round-0-single-bad |  |
| `two-sum` | `best_of_n` | accepted | 2 | pass | round-0-best-good |  |
| `two-sum` | `best_of_n` | accepted | 2 | pass | round-0-best-good |  |
| `two-sum` | `best_of_n` | accepted | 2 | pass | round-0-best-good |  |
| `two-sum` | `orchestrated` | accepted | 2 | pass | round-1-orchestrated-good |  |
| `two-sum` | `orchestrated` | accepted | 2 | pass | round-1-orchestrated-good |  |
| `two-sum` | `orchestrated` | accepted | 2 | pass | round-1-orchestrated-good |  |
| `extended-gcd` | `single_pass` | budget_exhausted | 1 | n/a | round-0-single-bad |  |
| `extended-gcd` | `single_pass` | budget_exhausted | 1 | n/a | round-0-single-bad |  |
| `extended-gcd` | `single_pass` | budget_exhausted | 1 | n/a | round-0-single-bad |  |
| `extended-gcd` | `best_of_n` | accepted | 2 | pass | round-0-best-good |  |
| `extended-gcd` | `best_of_n` | accepted | 2 | pass | round-0-best-good |  |
| `extended-gcd` | `best_of_n` | accepted | 2 | pass | round-0-best-good |  |
| `extended-gcd` | `orchestrated` | accepted | 2 | pass | round-1-orchestrated-good |  |
| `extended-gcd` | `orchestrated` | accepted | 2 | pass | round-1-orchestrated-good |  |
| `extended-gcd` | `orchestrated` | accepted | 2 | pass | round-1-orchestrated-good |  |

## Claim boundary

> This is a deterministic control-policy and protocol evaluation with scripted model responses. It measures visible-test success, role/model-call budgets, independent hidden-test success when enabled, candidate accounting, and token accounting on the bundled fixtures; it does not measure general model capability, latency, provider reliability, or production safety.
