# Offline planner / solver / reviewer demo

Run from the repository root after installing the package. The command materializes each
candidate in a temporary workspace; it does not modify this repository.

~~~text
python -m contextopt orchestrate \
  --task "make solve return ascending values without mutating input" \
  --root-files examples/orchestrator_demo/root-files.json \
  --checkpoint /tmp/contextopt-orchestrator.json \
  --planner-script examples/orchestrator_demo/planner.json \
  --solver-script examples/orchestrator_demo/solver.json \
  --reviewer-script examples/orchestrator_demo/reviewer.json \
  --test-command "python -m unittest discover -s ." \
  --allow-command \
  --output /tmp/contextopt-orchestrator-report.json \
  --markdown /tmp/contextopt-orchestrator-report.md
~~~

The expected terminal state is accepted. The report separates planner, solver, reviewer,
and visible-test calls; acceptance requires both reviewer accept and a passing oracle.
