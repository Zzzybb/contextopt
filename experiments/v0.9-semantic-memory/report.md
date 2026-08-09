# Semantic-memory retrieval conformance

> Model calls: 0. Lexical retrieval evidence is not model quality or coding success.

| Case | Scope | Hit@1 | Hit@k | MRR | Negative pass | Deterministic | Scope leakage | Invalidated returned |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| global-cas-procedure | project:parser | True | True | 1.000 | n/a | True | 0 | 0 |
| parser-protocol-rule | project:parser | True | True | 1.000 | n/a | True | 0 | 0 |
| math-bezout-rule | project:math | True | True | 1.000 | n/a | True | 0 | 0 |
| scope-isolation | project:math | n/a | n/a | n/a | True | True | 0 | 0 |
| invalidated-rule-excluded | project:parser | n/a | n/a | n/a | True | True | 0 | 0 |

## Summary

- Positive hit@1 rate: `1.000`
- Positive hit@k rate: `1.000`
- Mean reciprocal rank: `1.000`
- Negative-query pass rate: `1.000`
- Scope-isolation rate: `1.000`
- Invalidated-exclusion rate: `1.000`
- Deterministic replay rate: `1.000`
- Feedback retry idempotent: `True`
- Feedback score improved: `True`

This is a provider-free memory retrieval and persistence conformance fixture. It measures hit@k, reciprocal rank, scope isolation, invalidation exclusion, and deterministic replay, plus an idempotent helpful-feedback score change; it does not measure embedding quality, model use of memory, or coding success.
