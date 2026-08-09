# ContextOpt coding-agent strategy evaluation

This report uses executable ACM/math fixtures, independent hidden tests, and deterministic scripted model responses.

| Strategy | Visible | Visible 95% CI | Hidden | Mean model calls | Mean visible tests | Mean reuses | Mean candidates | Mean tokens | Mean duration ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `single_pass` | 0/12 (0%) | 0%-24% | 0/0 (n/a) | 1.0 | 1.0 | 0.0 | 1.0 | 160 | 192.8 |
| `best_of_n` | 12/12 (100%) | 76%-100% | 12/12 (100%) | 1.0 | 2.0 | 0.0 | 2.0 | 240 | 573.7 |
| `orchestrated` | 12/12 (100%) | 76%-100% | 12/12 (100%) | 6.0 | 2.0 | 0.0 | 2.0 | 1160 | 641.9 |

## Paired comparisons

Each delta is candidate minus the baseline on the same fixture and repetition; positive visible delta means more paired wins.

| Strategy | Baseline | Paired | Wins | Losses | Ties | Visible Δ | Hidden Δ | Mean tests Δ (stdev) | Mean tokens Δ (stdev) | Mean duration Δ ms (stdev) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `best_of_n` | `single_pass` | 12 | 12 | 0 | 0 | +100% | n/a | +1.0 (stdev 0.0) | +80 (stdev 0) | +380.9 (stdev 9.3) |
| `orchestrated` | `single_pass` | 12 | 12 | 0 | 0 | +100% | n/a | +1.0 (stdev 0.0) | +1000 (stdev 0) | +449.2 (stdev 18.0) |

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
| `two-sum` | `single_pass` | budget_exhausted | 1 | n/a | 205.6 | round-0-single-bad |  |
| `two-sum` | `single_pass` | budget_exhausted | 1 | n/a | 184.6 | round-0-single-bad |  |
| `two-sum` | `single_pass` | budget_exhausted | 1 | n/a | 185.4 | round-0-single-bad |  |
| `two-sum` | `best_of_n` | accepted | 2 | pass | 597.2 | round-0-best-good |  |
| `two-sum` | `best_of_n` | accepted | 2 | pass | 566.5 | round-0-best-good |  |
| `two-sum` | `best_of_n` | accepted | 2 | pass | 569.9 | round-0-best-good |  |
| `two-sum` | `orchestrated` | accepted | 2 | pass | 677.4 | round-1-orchestrated-good |  |
| `two-sum` | `orchestrated` | accepted | 2 | pass | 675.0 | round-1-orchestrated-good |  |
| `two-sum` | `orchestrated` | accepted | 2 | pass | 638.5 | round-1-orchestrated-good |  |
| `extended-gcd` | `single_pass` | budget_exhausted | 1 | n/a | 190.3 | round-0-single-bad |  |
| `extended-gcd` | `single_pass` | budget_exhausted | 1 | n/a | 198.4 | round-0-single-bad |  |
| `extended-gcd` | `single_pass` | budget_exhausted | 1 | n/a | 189.0 | round-0-single-bad |  |
| `extended-gcd` | `best_of_n` | accepted | 2 | pass | 562.9 | round-0-best-good |  |
| `extended-gcd` | `best_of_n` | accepted | 2 | pass | 581.6 | round-0-best-good |  |
| `extended-gcd` | `best_of_n` | accepted | 2 | pass | 562.8 | round-0-best-good |  |
| `extended-gcd` | `orchestrated` | accepted | 2 | pass | 626.4 | round-1-orchestrated-good |  |
| `extended-gcd` | `orchestrated` | accepted | 2 | pass | 624.1 | round-1-orchestrated-good |  |
| `extended-gcd` | `orchestrated` | accepted | 2 | pass | 624.7 | round-1-orchestrated-good |  |
| `merge-intervals` | `single_pass` | budget_exhausted | 1 | n/a | 193.5 | round-0-single-bad |  |
| `merge-intervals` | `single_pass` | budget_exhausted | 1 | n/a | 189.8 | round-0-single-bad |  |
| `merge-intervals` | `single_pass` | budget_exhausted | 1 | n/a | 186.9 | round-0-single-bad |  |
| `merge-intervals` | `best_of_n` | accepted | 2 | pass | 594.0 | round-0-best-good |  |
| `merge-intervals` | `best_of_n` | accepted | 2 | pass | 572.4 | round-0-best-good |  |
| `merge-intervals` | `best_of_n` | accepted | 2 | pass | 558.0 | round-0-best-good |  |
| `merge-intervals` | `orchestrated` | accepted | 2 | pass | 636.7 | round-1-orchestrated-good |  |
| `merge-intervals` | `orchestrated` | accepted | 2 | pass | 650.4 | round-1-orchestrated-good |  |
| `merge-intervals` | `orchestrated` | accepted | 2 | pass | 646.3 | round-1-orchestrated-good |  |
| `modular-inverse` | `single_pass` | budget_exhausted | 1 | n/a | 199.9 | round-0-single-bad |  |
| `modular-inverse` | `single_pass` | budget_exhausted | 1 | n/a | 191.4 | round-0-single-bad |  |
| `modular-inverse` | `single_pass` | budget_exhausted | 1 | n/a | 198.4 | round-0-single-bad |  |
| `modular-inverse` | `best_of_n` | accepted | 2 | pass | 563.8 | round-0-best-good |  |
| `modular-inverse` | `best_of_n` | accepted | 2 | pass | 575.3 | round-0-best-good |  |
| `modular-inverse` | `best_of_n` | accepted | 2 | pass | 580.0 | round-0-best-good |  |
| `modular-inverse` | `orchestrated` | accepted | 2 | pass | 639.3 | round-1-orchestrated-good |  |
| `modular-inverse` | `orchestrated` | accepted | 2 | pass | 636.8 | round-1-orchestrated-good |  |
| `modular-inverse` | `orchestrated` | accepted | 2 | pass | 627.5 | round-1-orchestrated-good |  |

## Claim boundary

> This is a deterministic control-policy and protocol evaluation with scripted model responses. It measures visible-test success, role/model-call budgets, independent hidden-test success when enabled, candidate accounting, and token accounting on the bundled fixtures, plus local wall-clock duration; it does not measure general model capability, provider latency, provider reliability, or production safety.
