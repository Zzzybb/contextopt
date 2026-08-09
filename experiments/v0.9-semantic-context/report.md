# Semantic-context candidate evaluation

This is a deterministic context-projection conformance evaluation. It measures lexical candidate retrieval, budgeted selection, receipt determinism, and replay without a provider; it does not measure model reasoning, memory usefulness, semantic understanding, or coding success.

| Policy | Budget | Retrieved | Selected | Candidate tokens | Selected candidate tokens | Context tokens | Evicted | Deterministic | Replayable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `recent` | 128 | 3 | 0 | 475 | 0 | 40 | 3 | True | True |
| `recent` | 128 | 3 | 0 | 475 | 0 | 40 | 3 | True | True |
| `recent` | 128 | 3 | 0 | 475 | 0 | 40 | 3 | True | True |
| `recent` | 256 | 3 | 1 | 475 | 143 | 183 | 2 | True | True |
| `recent` | 256 | 3 | 1 | 475 | 143 | 183 | 2 | True | True |
| `recent` | 256 | 3 | 1 | 475 | 143 | 183 | 2 | True | True |
| `recent` | 512 | 3 | 2 | 475 | 290 | 330 | 1 | True | True |
| `recent` | 512 | 3 | 2 | 475 | 290 | 330 | 1 | True | True |
| `recent` | 512 | 3 | 2 | 475 | 290 | 330 | 1 | True | True |
| `topk` | 128 | 3 | 0 | 475 | 0 | 40 | 3 | True | True |
| `topk` | 128 | 3 | 0 | 475 | 0 | 40 | 3 | True | True |
| `topk` | 128 | 3 | 0 | 475 | 0 | 40 | 3 | True | True |
| `topk` | 256 | 3 | 1 | 475 | 185 | 225 | 2 | True | True |
| `topk` | 256 | 3 | 1 | 475 | 185 | 225 | 2 | True | True |
| `topk` | 256 | 3 | 1 | 475 | 185 | 225 | 2 | True | True |
| `topk` | 512 | 3 | 2 | 475 | 328 | 368 | 1 | True | True |
| `topk` | 512 | 3 | 2 | 475 | 328 | 368 | 1 | True | True |
| `topk` | 512 | 3 | 2 | 475 | 328 | 368 | 1 | True | True |
| `density` | 128 | 3 | 0 | 475 | 0 | 40 | 3 | True | True |
| `density` | 128 | 3 | 0 | 475 | 0 | 40 | 3 | True | True |
| `density` | 128 | 3 | 0 | 475 | 0 | 40 | 3 | True | True |
| `density` | 256 | 3 | 1 | 475 | 185 | 225 | 2 | True | True |
| `density` | 256 | 3 | 1 | 475 | 185 | 225 | 2 | True | True |
| `density` | 256 | 3 | 1 | 475 | 185 | 225 | 2 | True | True |
| `density` | 512 | 3 | 2 | 475 | 328 | 368 | 1 | True | True |
| `density` | 512 | 3 | 2 | 475 | 328 | 368 | 1 | True | True |
| `density` | 512 | 3 | 2 | 475 | 328 | 368 | 1 | True | True |
| `submodular` | 128 | 3 | 0 | 475 | 0 | 40 | 3 | True | True |
| `submodular` | 128 | 3 | 0 | 475 | 0 | 40 | 3 | True | True |
| `submodular` | 128 | 3 | 0 | 475 | 0 | 40 | 3 | True | True |
| `submodular` | 256 | 3 | 1 | 475 | 185 | 225 | 2 | True | True |
| `submodular` | 256 | 3 | 1 | 475 | 185 | 225 | 2 | True | True |
| `submodular` | 256 | 3 | 1 | 475 | 185 | 225 | 2 | True | True |
| `submodular` | 512 | 3 | 2 | 475 | 332 | 372 | 1 | True | True |
| `submodular` | 512 | 3 | 2 | 475 | 332 | 372 | 1 | True | True |
| `submodular` | 512 | 3 | 2 | 475 | 332 | 372 | 1 | True | True |
| `recent` | 128 | 3 | 0 | 471 | 0 | 40 | 3 | True | True |
| `recent` | 128 | 3 | 0 | 471 | 0 | 40 | 3 | True | True |
| `recent` | 128 | 3 | 0 | 471 | 0 | 40 | 3 | True | True |
| `recent` | 256 | 3 | 1 | 471 | 143 | 183 | 2 | True | True |
| `recent` | 256 | 3 | 1 | 471 | 143 | 183 | 2 | True | True |
| `recent` | 256 | 3 | 1 | 471 | 143 | 183 | 2 | True | True |
| `recent` | 512 | 3 | 3 | 471 | 471 | 511 | 0 | True | True |
| `recent` | 512 | 3 | 3 | 471 | 471 | 511 | 0 | True | True |
| `recent` | 512 | 3 | 3 | 471 | 471 | 511 | 0 | True | True |
| `topk` | 128 | 3 | 0 | 471 | 0 | 40 | 3 | True | True |
| `topk` | 128 | 3 | 0 | 471 | 0 | 40 | 3 | True | True |
| `topk` | 128 | 3 | 0 | 471 | 0 | 40 | 3 | True | True |
| `topk` | 256 | 3 | 1 | 471 | 181 | 221 | 2 | True | True |
| `topk` | 256 | 3 | 1 | 471 | 181 | 221 | 2 | True | True |
| `topk` | 256 | 3 | 1 | 471 | 181 | 221 | 2 | True | True |
| `topk` | 512 | 3 | 3 | 471 | 471 | 511 | 0 | True | True |
| `topk` | 512 | 3 | 3 | 471 | 471 | 511 | 0 | True | True |
| `topk` | 512 | 3 | 3 | 471 | 471 | 511 | 0 | True | True |
| `density` | 128 | 3 | 0 | 471 | 0 | 40 | 3 | True | True |
| `density` | 128 | 3 | 0 | 471 | 0 | 40 | 3 | True | True |
| `density` | 128 | 3 | 0 | 471 | 0 | 40 | 3 | True | True |
| `density` | 256 | 3 | 1 | 471 | 181 | 221 | 2 | True | True |
| `density` | 256 | 3 | 1 | 471 | 181 | 221 | 2 | True | True |
| `density` | 256 | 3 | 1 | 471 | 181 | 221 | 2 | True | True |
| `density` | 512 | 3 | 3 | 471 | 471 | 511 | 0 | True | True |
| `density` | 512 | 3 | 3 | 471 | 471 | 511 | 0 | True | True |
| `density` | 512 | 3 | 3 | 471 | 471 | 511 | 0 | True | True |
| `submodular` | 128 | 3 | 0 | 471 | 0 | 40 | 3 | True | True |
| `submodular` | 128 | 3 | 0 | 471 | 0 | 40 | 3 | True | True |
| `submodular` | 128 | 3 | 0 | 471 | 0 | 40 | 3 | True | True |
| `submodular` | 256 | 3 | 1 | 471 | 181 | 221 | 2 | True | True |
| `submodular` | 256 | 3 | 1 | 471 | 181 | 221 | 2 | True | True |
| `submodular` | 256 | 3 | 1 | 471 | 181 | 221 | 2 | True | True |
| `submodular` | 512 | 3 | 3 | 471 | 471 | 511 | 0 | True | True |
| `submodular` | 512 | 3 | 3 | 471 | 471 | 511 | 0 | True | True |
| `submodular` | 512 | 3 | 3 | 471 | 471 | 511 | 0 | True | True |

## Summary

- Cells: `72`; failures: `0`.
- Retrieved recall: `1.000`; selected recall: `0.542`.
- Budget compliant: `1.000`; deterministic: `1.000`; replayable: `1.000`.
- Mean retrieved/selected/evicted candidates: `3.000` / `1.167` / `1.833`.

The recall fields are fixture-label retention metrics, not semantic understanding or model-use metrics.
