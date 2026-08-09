# Cross-run semantic memory demo

This is a two-run, offline demonstration of the explicit memory boundary. The first scripted
Agent saves a short editing procedure; a fresh second Agent opens the same JSONL store and
searches for it. The store is advisory and lexical: the second Agent must call
`memory_search`, and the result is evidence rather than proof about the current workspace.

Run it from the repository root. The commands below keep the event logs and memory store in a
temporary directory so the example workspace is never modified.

## PowerShell

```powershell
$demoRoot = Join-Path ([IO.Path]::GetTempPath()) `
  ("contextopt-memory-" + [guid]::NewGuid().ToString("N"))
$memory = Join-Path $demoRoot "memory.jsonl"
$writerEvents = Join-Path $demoRoot "writer-events.jsonl"
$readerEvents = Join-Path $demoRoot "reader-events.jsonl"
New-Item -ItemType Directory -Path $demoRoot | Out-Null

python -m contextopt run `
  "Record the safe editing procedure." `
  --workspace examples\semantic_memory_demo\workspace `
  --script examples\semantic_memory_demo\writer.json `
  --memory-store $memory `
  --allow-write `
  --event-log $writerEvents

python -m contextopt run `
  "Recall the safe editing procedure." `
  --workspace examples\semantic_memory_demo\workspace `
  --script examples\semantic_memory_demo\reader.json `
  --memory-store $memory `
  --event-log $readerEvents

python -m contextopt trace $readerEvents
```

## POSIX shell

```bash
demo_root="$(mktemp -d)"
memory="$demo_root/memory.jsonl"
writer_events="$demo_root/writer-events.jsonl"
reader_events="$demo_root/reader-events.jsonl"

python -m contextopt run \
  "Record the safe editing procedure." \
  --workspace examples/semantic_memory_demo/workspace \
  --script examples/semantic_memory_demo/writer.json \
  --memory-store "$memory" \
  --allow-write \
  --event-log "$writer_events"

python -m contextopt run \
  "Recall the safe editing procedure." \
  --workspace examples/semantic_memory_demo/workspace \
  --script examples/semantic_memory_demo/reader.json \
  --memory-store "$memory" \
  --event-log "$reader_events"

python -m contextopt trace "$reader_events"
```

The writer exposes `memory_save` because `--allow-write` is enabled. The reader exposes only
`memory_search`; `memory_save` and `memory_invalidate` remain permission-gated. Both runs record
the tool call, query/result, matched terms, memory id, and store revision in their normal
event-sourced traces.
