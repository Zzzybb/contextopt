# ContextOpt planner / solver / reviewer orchestration

- Status: accepted
- Task: find a correct sorting implementation
- Rounds: 1
- Model calls: 4 (planner 1, solver 2, reviewer 1)
- Actual test calls: 1; cached reuses: 0
- Scheduler: max configured parallel tests `1`; observed max in-flight `1`
- Speculative solver width: `2`; observed provider max in-flight `2`; winners `1`, cancelled lanes `1`
- Context receipts: 3/3 (selected message blocks and observed-memory fingerprints)
- Best candidate: round-0-spec-0-good

| Round | Solver lanes | Branch | Reviewer | Confidence | Candidate | Test calls |
|---:|---:|---|---|---:|---|---:|
| 0 | 1 | accepted | accept | 0.90 | round-0-spec-0-good | 1 |

> reviewer accepted a visible-test-passing candidate

> Acceptance requires reviewer approval and a visible-test-passing candidate.
