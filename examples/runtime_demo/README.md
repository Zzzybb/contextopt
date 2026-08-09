# Offline runtime demo

This fixture drives a real read/edit/test Agent loop with `ScriptedModel`; it requires no
API key. The script already contains the solution, so the demo validates runtime plumbing
and is not a model-capability benchmark.

Files:

- `workspace/`: intentionally buggy Python code and visible tests;
- `script.json`: observation-aware model responses and fixed token usage;
- `hidden_oracle.py`: semantic checks kept outside the Agent workspace.

Always copy `workspace/` to a fresh temporary directory before granting write permission.
Exact PowerShell and POSIX commands are in the
[runtime guide](../../docs/runtime.md#offline-runtime-demo).
