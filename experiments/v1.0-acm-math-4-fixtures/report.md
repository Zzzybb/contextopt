# ContextOpt coding-agent strategy evaluation

This report uses executable ACM/math fixtures, independent hidden tests, and deterministic scripted model responses.

| Strategy | Visible | Visible 95% CI | Hidden | Mean model calls | Mean visible tests | Mean reuses | Mean candidates | Mean tokens | Mean duration ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `single_pass` | 0/12 (0%) | 0%-24% | 0/0 (n/a) | 1.0 | 1.0 | 0.0 | 1.0 | 160 | 144.5 |
| `best_of_n` | 12/12 (100%) | 76%-100% | 12/12 (100%) | 1.0 | 2.0 | 0.0 | 2.0 | 240 | 436.1 |
| `orchestrated` | 12/12 (100%) | 76%-100% | 12/12 (100%) | 6.0 | 2.0 | 0.0 | 2.0 | 1160 | 503.2 |

## Paired comparisons

Each delta is candidate minus the baseline on the same fixture and repetition; positive visible delta means more paired wins.

| Strategy | Baseline | Paired | Wins | Losses | Ties | Visible Δ | Hidden Δ | Mean tests Δ (stdev) | Mean tokens Δ (stdev) | Mean duration Δ ms (stdev) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `best_of_n` | `single_pass` | 12 | 12 | 0 | 0 | +100% | n/a | +1.0 (stdev 0.0) | +80 (stdev 0) | +291.7 (stdev 20.3) |
| `orchestrated` | `single_pass` | 12 | 12 | 0 | 0 | +100% | n/a | +1.0 (stdev 0.0) | +1000 (stdev 0) | +358.8 (stdev 24.0) |

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
| `two-sum` | `single_pass` | budget_exhausted | 1 | n/a | 148.1 | round-0-single-bad |  |
| `two-sum` | `single_pass` | budget_exhausted | 1 | n/a | 132.1 | round-0-single-bad |  |
| `two-sum` | `single_pass` | budget_exhausted | 1 | n/a | 140.0 | round-0-single-bad |  |
| `two-sum` | `best_of_n` | accepted | 2 | pass | 403.5 | round-0-best-good |  |
| `two-sum` | `best_of_n` | accepted | 2 | pass | 422.1 | round-0-best-good |  |
| `two-sum` | `best_of_n` | accepted | 2 | pass | 417.8 | round-0-best-good |  |
| `two-sum` | `orchestrated` | accepted | 2 | pass | 494.5 | round-1-orchestrated-good |  |
| `two-sum` | `orchestrated` | accepted | 2 | pass | 474.8 | round-1-orchestrated-good |  |
| `two-sum` | `orchestrated` | accepted | 2 | pass | 476.6 | round-1-orchestrated-good |  |
| `extended-gcd` | `single_pass` | budget_exhausted | 1 | n/a | 138.7 | round-0-single-bad |  |
| `extended-gcd` | `single_pass` | budget_exhausted | 1 | n/a | 135.3 | round-0-single-bad |  |
| `extended-gcd` | `single_pass` | budget_exhausted | 1 | n/a | 136.8 | round-0-single-bad |  |
| `extended-gcd` | `best_of_n` | accepted | 2 | pass | 422.0 | round-0-best-good |  |
| `extended-gcd` | `best_of_n` | accepted | 2 | pass | 420.5 | round-0-best-good |  |
| `extended-gcd` | `best_of_n` | accepted | 2 | pass | 406.3 | round-0-best-good |  |
| `extended-gcd` | `orchestrated` | accepted | 2 | pass | 471.5 | round-1-orchestrated-good |  |
| `extended-gcd` | `orchestrated` | accepted | 2 | pass | 508.0 | round-1-orchestrated-good |  |
| `extended-gcd` | `orchestrated` | accepted | 2 | pass | 489.5 | round-1-orchestrated-good |  |
| `merge-intervals` | `single_pass` | budget_exhausted | 1 | n/a | 150.4 | round-0-single-bad |  |
| `merge-intervals` | `single_pass` | budget_exhausted | 1 | n/a | 149.2 | round-0-single-bad |  |
| `merge-intervals` | `single_pass` | budget_exhausted | 1 | n/a | 162.6 | round-0-single-bad |  |
| `merge-intervals` | `best_of_n` | accepted | 2 | pass | 463.6 | round-0-best-good |  |
| `merge-intervals` | `best_of_n` | accepted | 2 | pass | 452.5 | round-0-best-good |  |
| `merge-intervals` | `best_of_n` | accepted | 2 | pass | 469.6 | round-0-best-good |  |
| `merge-intervals` | `orchestrated` | accepted | 2 | pass | 489.1 | round-1-orchestrated-good |  |
| `merge-intervals` | `orchestrated` | accepted | 2 | pass | 512.8 | round-1-orchestrated-good |  |
| `merge-intervals` | `orchestrated` | accepted | 2 | pass | 508.9 | round-1-orchestrated-good |  |
| `modular-inverse` | `single_pass` | budget_exhausted | 1 | n/a | 146.3 | round-0-single-bad |  |
| `modular-inverse` | `single_pass` | budget_exhausted | 1 | n/a | 142.7 | round-0-single-bad |  |
| `modular-inverse` | `single_pass` | budget_exhausted | 1 | n/a | 151.4 | round-0-single-bad |  |
| `modular-inverse` | `best_of_n` | accepted | 2 | pass | 465.4 | round-0-best-good |  |
| `modular-inverse` | `best_of_n` | accepted | 2 | pass | 464.8 | round-0-best-good |  |
| `modular-inverse` | `best_of_n` | accepted | 2 | pass | 425.6 | round-0-best-good |  |
| `modular-inverse` | `orchestrated` | accepted | 2 | pass | 568.0 | round-1-orchestrated-good |  |
| `modular-inverse` | `orchestrated` | accepted | 2 | pass | 518.4 | round-1-orchestrated-good |  |
| `modular-inverse` | `orchestrated` | accepted | 2 | pass | 526.5 | round-1-orchestrated-good |  |

## Claim boundary

> This is a deterministic control-policy and protocol evaluation with scripted model responses. It measures visible-test success, role/model-call budgets, independent hidden-test success when enabled, candidate accounting, and token accounting on the bundled fixtures, plus local wall-clock duration; it does not measure general model capability, provider latency, provider reliability, or production safety.
