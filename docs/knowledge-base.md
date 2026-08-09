# Workspace knowledge base

`knowledge-index` is the repository-facing half of ContextOpt's durable memory. It turns a
bounded, UTF-8 workspace snapshot into source-addressable chunks in the existing append-only
`SemanticMemoryStore`; it does not call a model or create a second untracked index.

## Index a snapshot

```text
contextopt knowledge-index \
  --workspace ./workspace \
  --memory-store .contextopt/memory.jsonl \
  --memory-scope project:parser \
  --output .contextopt/knowledge-index.json \
  --markdown .contextopt/knowledge-index.md
```

The default policy indexes common source, configuration, documentation, and test suffixes. It
skips generated/cache directories, dotfiles, symlinks, non-UTF-8 files, files over 256 KiB, and
more than 500 eligible files. `--chunk-lines`, `--max-chunk-bytes`, `--max-file-bytes`,
`--max-files`, `--extensions`, `--ignore-dirs`, and `--include-dotfiles` make the boundary
explicit and reproducible.

Every chunk contains a relative source path and line range in its text, has the
`knowledge-chunk-v1` tag, and cites the source path in `source_refs`. The memory identity includes
the chunk text, scope, kind, and tags, so rerunning an unchanged snapshot reuses the same entry.
When a file changes or disappears, the old active chunks are appended as `memory.invalidated`
events. The store's existing source-aware tool reconciliation also invalidates a cited chunk when
the Agent edits that file between index passes.

## Use it during an Agent run

Attach the store and opt into the bounded semantic projection:

```text
contextopt run "Fix the parser" \
  --workspace ./workspace \
  --memory-store .contextopt/memory.jsonl \
  --memory-scope project:parser \
  --context-memory versioned-v1+semantic \
  --script examples/runtime_demo/script.json
```

The normal `ContextCompiler` treats indexed chunks as advisory blocks. At most three matches are
eligible before the regular token budget and policy selection; receipts record candidate IDs and
selected IDs. A model still has to inspect current files and run tests. Retrieval is lexical and
provider-free, so this feature demonstrates provenance, freshness, bounded context, and recovery,
not embedding quality or coding success.

For a new single-agent run, the index can be refreshed immediately before the first model call:

```text
contextopt run "Fix the parser" \
  --workspace ./workspace \
  --memory-store .contextopt/memory.jsonl \
  --memory-scope project:parser \
  --context-memory versioned-v1+semantic \
  --auto-index-knowledge \
  --knowledge-index-report .contextopt/knowledge-index.json \
  --script examples/runtime_demo/script.json
```

`--auto-index-knowledge` requires both `--memory-store` and `--memory-scope`. It runs before
`AgentRunner` opens its tools, so the first compiled request can see the fresh source projection;
later writes still use normal source-aware invalidation. `resume` deliberately does not re-index
automatically because changing the durable store is an explicit configuration decision for a
pending request.

## Reproducibility boundary

The JSON report records the normalized config, file counts, created/reused/invalidated chunk
counts, skipped-file reasons, active chunk IDs, and a snapshot fingerprint. The memory ledger is
the authority for entry contents and invalidation history. Indexing is safe to repeat after an
interrupted run because writes are content-idempotent and stale entries are never silently
deleted.
