# ContextOpt coding-agent strategy evaluation

This report uses executable ACM/math fixtures, independent hidden tests, and deterministic scripted model responses.

| Strategy | Visible | Visible 95% CI | Hidden | Mean model calls | Mean visible tests | Mean reuses | Mean candidates | Mean tokens | Mean duration ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `single_pass` | 0/12 (0%) | 0%-24% | 0/0 (n/a) | 1.0 | 1.0 | 0.0 | 1.0 | 160 | 153.9 |
| `best_of_n` | 12/12 (100%) | 76%-100% | 12/12 (100%) | 1.0 | 2.0 | 0.0 | 2.0 | 240 | 453.6 |
| `orchestrated` | 12/12 (100%) | 76%-100% | 12/12 (100%) | 6.0 | 2.0 | 0.0 | 2.0 | 1160 | 534.6 |

## Paired comparisons

Each delta is candidate minus the baseline on the same fixture and repetition; positive visible delta means more paired wins.

| Strategy | Baseline | Paired | Wins | Losses | Ties | Visible Δ | Hidden Δ | Mean tests Δ (stdev) | Mean tokens Δ (stdev) | Mean duration Δ ms (stdev) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `best_of_n` | `single_pass` | 12 | 12 | 0 | 0 | +100% | n/a | +1.0 (stdev 0.0) | +80 (stdev 0) | +299.6 (stdev 25.7) |
| `orchestrated` | `single_pass` | 12 | 12 | 0 | 0 | +100% | n/a | +1.0 (stdev 0.0) | +1000 (stdev 0) | +380.7 (stdev 25.1) |

## Fixtures

| ID | Category | Task |
|---|---|---|
| `two-sum` | acm-algorithm | Two Sum with duplicate values |
| `extended-gcd` | mathematics | Extended Euclidean algorithm |
| `merge-intervals` | acm-algorithm | Merge overlapping intervals |
| `modular-inverse` | mathematics | Modular multiplicative inverse |

## Run ledger

| Fixture | Strategy | Status | Visible tests | Hidden | Duration ms | Best candidate | Error |
|---|---|---|---:|---:|---:|---|---|
| `two-sum` | `single_pass` | budget_exhausted | 1 | n/a | 151.8 | round-0-single-bad |  |
| `two-sum` | `single_pass` | budget_exhausted | 1 | n/a | 142.0 | round-0-single-bad |  |
| `two-sum` | `single_pass` | budget_exhausted | 1 | n/a | 137.5 | round-0-single-bad |  |
| `two-sum` | `best_of_n` | accepted | 2 | pass | 400.7 | round-0-best-good |  |
| `two-sum` | `best_of_n` | accepted | 2 | pass | 412.4 | round-0-best-good |  |
| `two-sum` | `best_of_n` | accepted | 2 | pass | 427.7 | round-0-best-good |  |
| `two-sum` | `orchestrated` | accepted | 2 | pass | 511.7 | round-1-orchestrated-good |  |
| `two-sum` | `orchestrated` | accepted | 2 | pass | 497.4 | round-1-orchestrated-good |  |
| `two-sum` | `orchestrated` | accepted | 2 | pass | 505.1 | round-1-orchestrated-good |  |
| `extended-gcd` | `single_pass` | budget_exhausted | 1 | n/a | 148.1 | round-0-single-bad |  |
| `extended-gcd` | `single_pass` | budget_exhausted | 1 | n/a | 141.2 | round-0-single-bad |  |
| `extended-gcd` | `single_pass` | budget_exhausted | 1 | n/a | 148.0 | round-0-single-bad |  |
| `extended-gcd` | `best_of_n` | accepted | 2 | pass | 438.1 | round-0-best-good |  |
| `extended-gcd` | `best_of_n` | accepted | 2 | pass | 444.2 | round-0-best-good |  |
| `extended-gcd` | `best_of_n` | accepted | 2 | pass | 430.0 | round-0-best-good |  |
| `extended-gcd` | `orchestrated` | accepted | 2 | pass | 509.3 | round-1-orchestrated-good |  |
| `extended-gcd` | `orchestrated` | accepted | 2 | pass | 489.0 | round-1-orchestrated-good |  |
| `extended-gcd` | `orchestrated` | accepted | 2 | pass | 518.7 | round-1-orchestrated-good |  |
| `merge-intervals` | `single_pass` | budget_exhausted | 1 | n/a | 160.4 | round-0-single-bad |  |
| `merge-intervals` | `single_pass` | budget_exhausted | 1 | n/a | 164.2 | round-0-single-bad |  |
| `merge-intervals` | `single_pass` | budget_exhausted | 1 | n/a | 158.8 | round-0-single-bad |  |
| `merge-intervals` | `best_of_n` | accepted | 2 | pass | 462.0 | round-0-best-good |  |
| `merge-intervals` | `best_of_n` | accepted | 2 | pass | 467.5 | round-0-best-good |  |
| `merge-intervals` | `best_of_n` | accepted | 2 | pass | 480.8 | round-0-best-good |  |
| `merge-intervals` | `orchestrated` | accepted | 2 | pass | 565.5 | round-1-orchestrated-good |  |
| `merge-intervals` | `orchestrated` | accepted | 2 | pass | 553.1 | round-1-orchestrated-good |  |
| `merge-intervals` | `orchestrated` | accepted | 2 | pass | 599.8 | round-1-orchestrated-good |  |
| `modular-inverse` | `single_pass` | budget_exhausted | 1 | n/a | 168.4 | round-0-single-bad |  |
| `modular-inverse` | `single_pass` | budget_exhausted | 1 | n/a | 165.9 | round-0-single-bad |  |
| `modular-inverse` | `single_pass` | budget_exhausted | 1 | n/a | 160.8 | round-0-single-bad |  |
| `modular-inverse` | `best_of_n` | accepted | 2 | pass | 478.1 | round-0-best-good |  |
| `modular-inverse` | `best_of_n` | accepted | 2 | pass | 486.6 | round-0-best-good |  |
| `modular-inverse` | `best_of_n` | accepted | 2 | pass | 514.5 | round-0-best-good |  |
| `modular-inverse` | `orchestrated` | accepted | 2 | pass | 548.4 | round-1-orchestrated-good |  |
| `modular-inverse` | `orchestrated` | accepted | 2 | pass | 556.6 | round-1-orchestrated-good |  |
| `modular-inverse` | `orchestrated` | accepted | 2 | pass | 561.0 | round-1-orchestrated-good |  |

## Claim boundary

> This is a deterministic control-policy and protocol evaluation with scripted model responses. It measures visible-test success, role/model-call budgets, independent hidden-test success when enabled, candidate accounting, and token accounting on the bundled fixtures, plus local wall-clock duration; it does not measure general model capability, provider latency, provider reliability, or production safety.
