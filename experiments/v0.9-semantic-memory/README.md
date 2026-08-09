# v0.9 semantic-memory retrieval conformance

This artifact evaluates the durable cross-run memory boundary without calling a model. It
creates five typed entries in an append-only store, explicitly invalidates one stale entry, runs
five scoped/tagged queries three times each, and retries one helpful feedback event.

## Reproduce

```text
python -m contextopt memory-eval \
  --repetitions 3 \
  --limit 3 \
  --output experiments/v0.9-semantic-memory/report.json \
  --markdown experiments/v0.9-semantic-memory/report.md \
  --html experiments/v0.9-semantic-memory/report.html
```

The report contains the raw fixture, result memory ids, matched terms, scores, repeated-search
digests, and a self-contained HTML dashboard. The committed deterministic result is:

| Metric | Result |
|---|---:|
| positive hit@1 rate | 1.000 |
| positive hit@k rate (`k=3`) | 1.000 |
| mean reciprocal rank | 1.000 |
| negative-query pass rate | 1.000 |
| scope-isolation rate | 1.000 |
| invalidated-exclusion rate | 1.000 |
| deterministic replay rate | 1.000 |
| feedback retry idempotent | `true` |
| feedback score improved | `true` |

## Metric definitions and boundary

- `hit@1` and `hit@k` ask whether an expected active memory is ranked first or within the
  configured result limit. `MRR` is the reciprocal rank of the first expected item.
- Negative-query pass means a query with no expected active memory returns no result. Scope
  isolation counts only `global` and the requested project scope as valid results.
- Invalidated-exclusion checks that an explicitly invalidated memory id never appears in search.
  Deterministic replay requires the repeated result ids and serialized match digests to agree.
- Feedback retry idempotence requires the same tool-call feedback id to append exactly one event;
  the bounded helpful signal must increase the selected probe's score without editing its text.

These are provider-free retrieval and persistence checks. They do not measure embedding quality,
semantic understanding, whether a model asks for useful memories, or whether memory improves a
generated patch. The implementation intentionally reports a lexical baseline plus an auditable
feedback heuristic rather than turning a fixed fixture into a learned-memory claim.
