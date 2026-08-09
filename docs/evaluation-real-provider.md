# Real-provider evaluation workflow

The repository includes a manual GitHub Actions workflow at
`.github/workflows/real-agent-eval.yml`. It is intentionally not scheduled and does not run on
pull requests, so a provider call (and its cost) only happens after an explicit operator action.

## One-time setup

1. Add a repository or environment secret named `CONTEXTOPT_API_KEY`.
2. Open **Actions → real-agent-eval → Run workflow**.
3. Enter the OpenAI-compatible `model` and `base_url`. Keep the default `all` fixtures,
   `single_pass,best_of_n,orchestrated` strategies, and `3` repetitions for the first run.
4. Download the uploaded artifact bundle after the job finishes.

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
