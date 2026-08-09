# ContextOpt coding-agent strategy evaluation

This report uses executable ACM/math fixtures, independent hidden tests, and deterministic scripted model responses.

| Strategy | Visible | Visible 95% CI | Hidden | Mean model calls | Mean visible tests | Mean reuses | Mean candidates | Mean tokens | Mean duration ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `single_pass` | 0/12 (0%) | 0%-24% | 0/0 (n/a) | 1.0 | 1.0 | 0.0 | 1.0 | 160 | 132.2 |
| `best_of_n` | 12/12 (100%) | 76%-100% | 12/12 (100%) | 1.0 | 2.0 | 0.0 | 2.0 | 240 | 385.4 |
| `orchestrated` | 12/12 (100%) | 76%-100% | 12/12 (100%) | 6.0 | 2.0 | 0.0 | 2.0 | 1160 | 447.3 |

## Paired comparisons

Each delta is candidate minus the baseline on the same fixture and repetition; positive visible delta means more paired wins.

| Strategy | Baseline | Paired | Wins | Losses | Ties | Visible Δ | Visible exact p | Hidden Δ | Hidden exact p | Mean tests Δ (stdev) | Mean tokens Δ (stdev) | Mean duration Δ ms (stdev) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `best_of_n` | `single_pass` | 12 | 12 | 0 | 0 | +100% | 0.000488 | n/a | n/a | +1.0 (stdev 0.0) | +80 (stdev 0) | +253.2 (stdev 24.7) |
| `orchestrated` | `single_pass` | 12 | 12 | 0 | 0 | +100% | 0.000488 | n/a | n/a | +1.0 (stdev 0.0) | +1000 (stdev 0) | +315.1 (stdev 20.5) |

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
| `two-sum` | `single_pass` | budget_exhausted | 1 | n/a | 132.8 | round-0-single-bad |  |
| `two-sum` | `single_pass` | budget_exhausted | 1 | n/a | 118.5 | round-0-single-bad |  |
| `two-sum` | `single_pass` | budget_exhausted | 1 | n/a | 122.6 | round-0-single-bad |  |
| `two-sum` | `best_of_n` | accepted | 2 | pass | 351.4 | round-0-best-good |  |
| `two-sum` | `best_of_n` | accepted | 2 | pass | 344.3 | round-0-best-good |  |
| `two-sum` | `best_of_n` | accepted | 2 | pass | 355.9 | round-0-best-good |  |
| `two-sum` | `orchestrated` | accepted | 2 | pass | 428.1 | round-1-orchestrated-good |  |
| `two-sum` | `orchestrated` | accepted | 2 | pass | 410.2 | round-1-orchestrated-good |  |
| `two-sum` | `orchestrated` | accepted | 2 | pass | 407.3 | round-1-orchestrated-good |  |
| `extended-gcd` | `single_pass` | budget_exhausted | 1 | n/a | 127.5 | round-0-single-bad |  |
| `extended-gcd` | `single_pass` | budget_exhausted | 1 | n/a | 121.7 | round-0-single-bad |  |
| `extended-gcd` | `single_pass` | budget_exhausted | 1 | n/a | 124.1 | round-0-single-bad |  |
| `extended-gcd` | `best_of_n` | accepted | 2 | pass | 350.0 | round-0-best-good |  |
| `extended-gcd` | `best_of_n` | accepted | 2 | pass | 355.0 | round-0-best-good |  |
| `extended-gcd` | `best_of_n` | accepted | 2 | pass | 371.1 | round-0-best-good |  |
| `extended-gcd` | `orchestrated` | accepted | 2 | pass | 416.6 | round-1-orchestrated-good |  |
| `extended-gcd` | `orchestrated` | accepted | 2 | pass | 435.3 | round-1-orchestrated-good |  |
| `extended-gcd` | `orchestrated` | accepted | 2 | pass | 471.9 | round-1-orchestrated-good |  |
| `merge-intervals` | `single_pass` | budget_exhausted | 1 | n/a | 135.0 | round-0-single-bad |  |
| `merge-intervals` | `single_pass` | budget_exhausted | 1 | n/a | 140.3 | round-0-single-bad |  |
| `merge-intervals` | `single_pass` | budget_exhausted | 1 | n/a | 142.6 | round-0-single-bad |  |
| `merge-intervals` | `best_of_n` | accepted | 2 | pass | 420.1 | round-0-best-good |  |
| `merge-intervals` | `best_of_n` | accepted | 2 | pass | 404.4 | round-0-best-good |  |
| `merge-intervals` | `best_of_n` | accepted | 2 | pass | 420.6 | round-0-best-good |  |
| `merge-intervals` | `orchestrated` | accepted | 2 | pass | 476.2 | round-1-orchestrated-good |  |
| `merge-intervals` | `orchestrated` | accepted | 2 | pass | 458.0 | round-1-orchestrated-good |  |
| `merge-intervals` | `orchestrated` | accepted | 2 | pass | 476.0 | round-1-orchestrated-good |  |
| `modular-inverse` | `single_pass` | budget_exhausted | 1 | n/a | 140.3 | round-0-single-bad |  |
| `modular-inverse` | `single_pass` | budget_exhausted | 1 | n/a | 142.4 | round-0-single-bad |  |
| `modular-inverse` | `single_pass` | budget_exhausted | 1 | n/a | 138.5 | round-0-single-bad |  |
| `modular-inverse` | `best_of_n` | accepted | 2 | pass | 410.0 | round-0-best-good |  |
| `modular-inverse` | `best_of_n` | accepted | 2 | pass | 428.9 | round-0-best-good |  |
| `modular-inverse` | `best_of_n` | accepted | 2 | pass | 412.8 | round-0-best-good |  |
| `modular-inverse` | `orchestrated` | accepted | 2 | pass | 457.3 | round-1-orchestrated-good |  |
| `modular-inverse` | `orchestrated` | accepted | 2 | pass | 477.2 | round-1-orchestrated-good |  |
| `modular-inverse` | `orchestrated` | accepted | 2 | pass | 454.0 | round-1-orchestrated-good |  |

## Claim boundary

> This is a deterministic control-policy and protocol evaluation with scripted model responses. It measures visible-test success, role/model-call budgets, independent hidden-test success when enabled, candidate accounting, and token accounting on the bundled fixtures, plus local wall-clock duration; it does not measure general model capability, provider latency, provider reliability, or production safety.
