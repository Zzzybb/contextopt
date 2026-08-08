# Pull-request change notes

Every pull request in this repository carries a versioned Markdown change note under
`docs/pr/`. The note is part of the same diff as the implementation, so a reviewer can
understand the intent and reproduce the checks from the exact source state being reviewed.

Each note should cover:

- what changed and which user/developer workflow it enables;
- why the change is needed and what design trade-offs it makes;
- validation commands and the important accounting metrics;
- explicit non-goals and claim boundaries.

The first pull request predates this convention; its retrospective note is
[`0001-contextopt-evolution.md`](0001-contextopt-evolution.md). Future PRs should add the
next numbered note rather than rewriting an older one.
