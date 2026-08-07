# Contributing

ContextOpt welcomes reproducible improvements to algorithms, benchmark cases, evaluation,
and documentation.

## Development setup

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m contextopt benchmark --instances 5 --items 10 --budget 800
```

## Pull requests

- Add tests for behavior changes.
- Keep policies deterministic under a fixed problem.
- Never expose benchmark critical labels to a policy.
- Report negative or neutral results alongside positive ones.
- Do not claim end-to-end agent improvement from surrogate metrics alone.
- Include reproduction commands for new experiment reports.

Algorithm implementations should document their supported constraint class and avoid
claiming approximation guarantees that do not apply after adding non-monotone penalties or
hard graph constraints.
