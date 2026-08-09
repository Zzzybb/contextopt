# ContextOpt coding-agent strategy evaluation

This report uses executable ACM/math fixtures, independent hidden tests, and deterministic scripted model responses.

| Strategy | Visible | Visible 95% CI | Hidden | Mean model calls | Mean visible tests | Mean reuses | Mean candidates | Mean tokens | Mean duration ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `single_pass` | 0/12 (0%) | 0%-24% | 0/0 (n/a) | 1.0 | 1.0 | 0.0 | 1.0 | 160 | 156.5 |
| `best_of_n` | 12/12 (100%) | 76%-100% | 12/12 (100%) | 1.0 | 2.0 | 0.0 | 2.0 | 240 | 473.4 |
| `orchestrated` | 12/12 (100%) | 76%-100% | 12/12 (100%) | 6.0 | 2.0 | 0.0 | 2.0 | 1160 | 556.6 |

## Paired comparisons

Each delta is candidate minus the baseline on the same fixture and repetition; positive visible delta means more paired wins.

| Strategy | Baseline | Paired | Wins | Losses | Ties | Visible Δ | Hidden Δ | Mean tests Δ (stdev) | Mean tokens Δ (stdev) | Mean duration Δ ms (stdev) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `best_of_n` | `single_pass` | 12 | 12 | 0 | 0 | +100% | n/a | +1.0 (stdev 0.0) | +80 (stdev 0) | +316.8 (stdev 24.0) |
| `orchestrated` | `single_pass` | 12 | 12 | 0 | 0 | +100% | n/a | +1.0 (stdev 0.0) | +1000 (stdev 0) | +400.0 (stdev 24.8) |

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
| `two-sum` | `single_pass` | budget_exhausted | 1 | n/a | 160.0 | round-0-single-bad |  |
| `two-sum` | `single_pass` | budget_exhausted | 1 | n/a | 141.1 | round-0-single-bad |  |
| `two-sum` | `single_pass` | budget_exhausted | 1 | n/a | 142.2 | round-0-single-bad |  |
| `two-sum` | `best_of_n` | accepted | 2 | pass | 422.9 | round-0-best-good |  |
| `two-sum` | `best_of_n` | accepted | 2 | pass | 418.9 | round-0-best-good |  |
| `two-sum` | `best_of_n` | accepted | 2 | pass | 454.4 | round-0-best-good |  |
| `two-sum` | `orchestrated` | accepted | 2 | pass | 516.9 | round-1-orchestrated-good |  |
| `two-sum` | `orchestrated` | accepted | 2 | pass | 518.4 | round-1-orchestrated-good |  |
| `two-sum` | `orchestrated` | accepted | 2 | pass | 529.4 | round-1-orchestrated-good |  |
| `extended-gcd` | `single_pass` | budget_exhausted | 1 | n/a | 153.3 | round-0-single-bad |  |
| `extended-gcd` | `single_pass` | budget_exhausted | 1 | n/a | 154.7 | round-0-single-bad |  |
| `extended-gcd` | `single_pass` | budget_exhausted | 1 | n/a | 153.4 | round-0-single-bad |  |
| `extended-gcd` | `best_of_n` | accepted | 2 | pass | 487.0 | round-0-best-good |  |
| `extended-gcd` | `best_of_n` | accepted | 2 | pass | 456.1 | round-0-best-good |  |
| `extended-gcd` | `best_of_n` | accepted | 2 | pass | 467.5 | round-0-best-good |  |
| `extended-gcd` | `orchestrated` | accepted | 2 | pass | 561.3 | round-1-orchestrated-good |  |
| `extended-gcd` | `orchestrated` | accepted | 2 | pass | 515.9 | round-1-orchestrated-good |  |
| `extended-gcd` | `orchestrated` | accepted | 2 | pass | 560.2 | round-1-orchestrated-good |  |
| `merge-intervals` | `single_pass` | budget_exhausted | 1 | n/a | 155.3 | round-0-single-bad |  |
| `merge-intervals` | `single_pass` | budget_exhausted | 1 | n/a | 154.1 | round-0-single-bad |  |
| `merge-intervals` | `single_pass` | budget_exhausted | 1 | n/a | 152.0 | round-0-single-bad |  |
| `merge-intervals` | `best_of_n` | accepted | 2 | pass | 483.7 | round-0-best-good |  |
| `merge-intervals` | `best_of_n` | accepted | 2 | pass | 500.7 | round-0-best-good |  |
| `merge-intervals` | `best_of_n` | accepted | 2 | pass | 481.2 | round-0-best-good |  |
| `merge-intervals` | `orchestrated` | accepted | 2 | pass | 593.7 | round-1-orchestrated-good |  |
| `merge-intervals` | `orchestrated` | accepted | 2 | pass | 544.6 | round-1-orchestrated-good |  |
| `merge-intervals` | `orchestrated` | accepted | 2 | pass | 571.6 | round-1-orchestrated-good |  |
| `modular-inverse` | `single_pass` | budget_exhausted | 1 | n/a | 177.7 | round-0-single-bad |  |
| `modular-inverse` | `single_pass` | budget_exhausted | 1 | n/a | 168.9 | round-0-single-bad |  |
| `modular-inverse` | `single_pass` | budget_exhausted | 1 | n/a | 165.6 | round-0-single-bad |  |
| `modular-inverse` | `best_of_n` | accepted | 2 | pass | 509.9 | round-0-best-good |  |
| `modular-inverse` | `best_of_n` | accepted | 2 | pass | 496.3 | round-0-best-good |  |
| `modular-inverse` | `best_of_n` | accepted | 2 | pass | 501.7 | round-0-best-good |  |
| `modular-inverse` | `orchestrated` | accepted | 2 | pass | 588.2 | round-1-orchestrated-good |  |
| `modular-inverse` | `orchestrated` | accepted | 2 | pass | 600.1 | round-1-orchestrated-good |  |
| `modular-inverse` | `orchestrated` | accepted | 2 | pass | 578.6 | round-1-orchestrated-good |  |

## Claim boundary

> This is a deterministic control-policy and protocol evaluation with scripted model responses. It measures visible-test success, role/model-call budgets, independent hidden-test success when enabled, candidate accounting, and token accounting on the bundled fixtures, plus local wall-clock duration; it does not measure general model capability, provider latency, provider reliability, or production safety.
