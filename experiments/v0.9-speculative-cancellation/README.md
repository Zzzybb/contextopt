# v0.9 first-valid speculative cancellation artifact

This is a deterministic, provider-free demonstration of the optional first-valid solver race.
Lane 0 returns a protocol-valid sorting candidate immediately; lane 1 waits until the runner
requests cancellation. The candidate still passes the visible executable oracle and reviewer
gate.

The committed files are a durable checkpoint, JSON ledger, Markdown summary, self-contained HTML
dashboard, and manifest with SHA-256 hashes. Recreate them with:

```text
python scripts/generate_speculative_cancellation_artifact.py
```

The report charges `model_calls=4` and `solver_calls=2` for the configured width even though one
lane is cancelled. `speculative_winners=1` and `cancelled_solver_lanes=1` describe the local
scheduler effect. This does not measure model quality, remote provider abort latency, or exactly
once cancellation; `valid` means only that the candidate protocol parsed successfully.
