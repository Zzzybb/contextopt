# Real-provider evaluation workflow

The repository includes a manual GitHub Actions workflow at
`.github/workflows/real-agent-eval.yml`. It is intentionally not scheduled and does not run on
pull requests, so a provider call (and its cost) only happens after an explicit operator action.
When this workflow is introduced through a pull request, GitHub exposes its **Actions → Run
workflow** form after the change is merged into the repository's default branch; before that,
use the equivalent local CLI command in the README.

## One-time setup

1. Add a repository or environment secret named `CONTEXTOPT_API_KEY`.
2. Open **Actions → real-agent-eval → Run workflow**.
3. Enter the OpenAI-compatible `model` and `base_url`; optionally enter a provider-specific
   `cancellation_url` if the provider exposes the documented abort contract. For controlled
   candidate execution, choose `sandbox=docker` and pin `container_image` to a digest. Keep the default `all` fixtures,
   `single_pass,best_of_n,orchestrated` strategies, and `3` repetitions for the first run.
   The workflow rejects fewer than three repetitions so the artifact follows the Level 3
   exploratory comparison rule.
4. Download the uploaded artifact bundle after the job finishes.

For a local run, preserve the provider boundary as auditable cassettes:

```text
python -m contextopt agent-eval --fixtures all --repetitions 3 \
  --model <model-name> --base-url <endpoint> \
  --cancellation-url <provider-cancel-endpoint> \
  --sandbox docker --container-image <image@sha256:digest> \
  --record-transcript-dir .contextopt/provider-matrix \
  --output agent-eval-real.json --manifest agent-eval-real.manifest.json
python -m contextopt agent-eval --fixtures all --repetitions 3 \
  --replay-transcript-dir .contextopt/provider-matrix \
  --output agent-eval-replay.json --manifest agent-eval-replay.manifest.json
```

The directory contains one strict JSONL cassette for every fixture, strategy,
repetition, and role. Replay checks the complete normalized request hash and fails closed
on a missing or changed request; it is an offline reproduction of the recorded run, not a
new provider measurement. Use a fresh recording directory for each matrix.

`--cancellation-url` is optional. If supplied, it must be a provider-specific `POST` endpoint
that accepts `{"request_idempotency_key": "...", "model": "..."}` and returns 2xx after applying
the provider's cancellation contract. The adapter records 404/405 as `unsupported` and other
transport/HTTP failures as `failed:*`. Chat Completions does not standardize this endpoint, so
the flag is an integration hook, not proof of remote generation termination or billing savings.

At the current revision, `all` expands to `two-sum`, `extended-gcd`, `merge-intervals`, and
`modular-inverse`. The deterministic three-repetition control baseline is checked in under
[`experiments/v1.0-acm-math-4-fixtures`](../experiments/v1.0-acm-math-4-fixtures/README.md);
keep the fixture IDs, strategies, budgets, and prompt configuration fixed when comparing a
provider run with it.

The bundle contains the JSON ledger, Markdown summary, self-contained HTML dashboard, manifest,
and atomic checkpoint. If a provider call or runner stops, rerun the same matrix locally with
`--resume --checkpoint` or use the same workflow inputs and inspect the new run separately; do
not combine different model, prompt, fixture, or budget configurations into one comparison.

## What the result means

The matrix pairs the same ACM/math fixture and repetition across `single_pass`, `best_of_n`, and
the planner/solver/reviewer orchestrator. It reports visible-test success, independent hidden
grader success, model/role calls, candidate/test accounting, token usage, Wilson intervals, and
paired deltas with observed variance.

This is still a fixed small-sample exploratory benchmark. It is not a general capability claim,
statistically powered study, latency SLA, provider reliability test, or security evaluation.
The hidden grader source is excluded from model-visible snapshots, and the manifest records
provider identity without writing the API key.

CI exercises the same OpenAI-compatible adapter boundary with local deterministic HTTP
fixtures in `tests/test_agent_search_eval.py`. One fixture covers the single-pass request
shape; the multi-agent fixture drives planner → solver → reviewer for all four ACM/math tasks
and validates role-specific models, authorization, idempotency headers, and oracle gating.
Neither smoke claims model quality or contacts a real provider.
