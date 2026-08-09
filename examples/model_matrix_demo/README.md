# Provider-free model-matrix demo

This demo exercises the complete collection-to-analysis path without an API key:

```text
python scripts/run_model_matrix_demo.py
```

It runs the four ACM/math fixtures three times with the deterministic `ScriptedModel` twice,
then invokes `agent-eval-compare` with labels `scripted-a` and `scripted-b`. The output directory
is `build/model-matrix-demo/` (ignored by Git) and contains:

- `scripted-a/` and `scripted-b/`: report, Markdown/HTML dashboard, manifest, and checkpoint;
- `model-matrix.json`: machine-readable protocol fingerprint and direction-consistency summary;
- `model-matrix.md` and `model-matrix.html`: reviewable reports.

The demo pins the provenance field to `provider-free-model-matrix-demo` because a local shell does
not have `GITHUB_SHA`; this is a demo revision marker, not a source commit claim.

The two labels are control replicas, not two real models. This proves the evaluator, manifest
matching, paired comparison, and renderers; it does not claim model quality. Replace the two
bundles with authorized provider runs, or use the manual
`.github/workflows/real-agent-model-matrix.yml` workflow, for a real comparison.
