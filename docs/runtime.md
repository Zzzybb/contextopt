# ForgeAgent runtime

The v0.3 runtime is a bounded, auditable and recoverable single-agent loop inside the
`contextopt` package. It reconstructs execution state from durable events, compiles a
bounded context for every model call, and can resume a non-terminal run against the same
validated model, tool, and context configuration. Its `versioned-v1` observed-memory
projection is derived from the current transcript; it is not cross-run memory. An explicit
`SemanticMemoryStore` can be attached as a separate advisory layer through memory tools.

## Current execution contract

```text
                          append every boundary
                         ┌──────────────────────> JSONL EventLog
                         │
Task -> AgentRunner -> ContextCompiler -> ModelClient
           ^               ^                    │
           │      observed-memory snapshot      │ structured ToolCall(s)
           │                                    v
           └──────── observation message <── WorkspaceTools
           │                                    │
           │                             files + visible tests
           v
 strict event reducer <---- schema-2 log ----> atomic projection cache
```

Each model response may contain text and zero or more structured tool calls. The runner
appends the assistant response, executes calls in order, converts every `ToolOutcome` into
a tool observation carrying the same call id, compiles the accumulated history under the
run's context budget, and sends that provider-valid view to the next model turn. The full
normalized history remains in the event projection even when a block is omitted from one
model request. A response without tool calls completes the run.

The runtime currently exposes:

- `list_files`: enumerate regular files below a workspace-relative directory;
- `search_text`: literal UTF-8 search with bounded matches;
- `read_file`: numbered lines plus the exact file SHA-256;
- `create_file`: create a new UTF-8 file without overwriting an existing file;
- `replace_text`: atomic exact-text replacement guarded by the SHA-256 returned by
  `read_file`;
- `run_tests`: invoke one trusted command registered by the caller, selected by scope.
- `memory_search`: query an attached append-only semantic-memory notebook with deterministic
  lexical matching and provenance;
- `memory_save`: append an idempotent memory entry (optionally superseding an older one) when
  the caller enabled `--allow-write`;
- `memory_invalidate`: mark one entry invalid with an explicit reason when the caller enabled
  `--allow-write`.

`run_tests` returning exit code 1 is a successful tool execution with a failing code test.
The observation includes the exit code and captured output so the model can react. A path
violation, malformed argument, denied permission, launch failure, or timeout is instead a
tool failure.

## Role orchestration boundary

The orchestrate command composes three no-tool model clients above the branch-search
adapter:

1. planner emits a bounded goal, constraints, hypotheses, test focus, and risks;
2. solver consumes that plan and emits complete candidate workspace snapshots;
3. the disposable visible-test oracle produces the only authoritative candidate result;
4. reviewer emits accept, retry, or reject plus confidence and blocking checks.

The reviewer is deliberately advisory but auditable. Acceptance requires both an explicit
accept decision naming a candidate and a visible-test-passing result for that candidate.
Role calls, response hashes, per-role counts, shared token/candidate/test budgets, feedback,
and cached observations are persisted in an orchestration checkpoint. A pending planner
request pauses unless the operator resumes with an explicit retry; solver and reviewer
boundaries are checkpointed so evaluation evidence is never treated as model confidence.

Each role also has an independent assistant-only history. Before a provider call, the runner
concatenates that history with the role's fresh system/user request and sends the transcript
through the same `ContextCompiler` used by the single-agent runtime. The default orchestrator
policy is submodular selection with a 16k-token budget and `versioned-v1` observed memory. The
`RoleCall` stores the resulting `ContextReceipt` (selected/evicted block ids, compiled-message
hash, memory fingerprint, and workspace generation), and the request/received events include
the same receipt. This makes role context a replayable protocol boundary rather than an
uninspectable prompt concatenation. A provider response containing tool calls is not added to
the assistant-only history because these three role contracts are intentionally no-tool JSON
interfaces.

For a deterministic local run, give orchestrate three ScriptedModel JSON files and the same
trusted visible-test command used by search-session. Real-model experiments should keep
provider, prompt, tool, budget, and repository versions fixed across role ablations.

## Coding-agent strategy evaluation harness

The repository also ships a model-free outer-loop comparison for the control policies that
are easiest to explain in an interview. Run it from the repository root:

```text
python -m contextopt agent-eval --fixtures all --repetitions 1 \
  --output agent-eval.json --markdown agent-eval.md --html agent-eval.html \
  --checkpoint agent-eval.checkpoint.json
```

The two fixtures are complete ACM/math workspaces. `single_pass` gets one intentionally
weak candidate, `best_of_n` gets a bad and a good candidate in one response, and
`orchestrated` receives a failing first round followed by a reviewer-gated retry. Every
candidate is materialized in a disposable workspace and tested by the same standard-
library command. After visible acceptance, the evaluator runs a separate hidden grader
whose source is not in the model-visible root snapshot. The result is a paired ledger, not
a model leaderboard.

The report separates actual visible test processes from cache reuses, records hidden grader
calls separately, and records model/role calls, candidate proposals, rounds, and
provider-reported token usage. A baseline failure remains visible in the report, while the
CLI returns zero once the matrix itself has completed. Use `--model` and `--base-url` to
replace scripted responses with fresh OpenAI-compatible adapters per cell. These metrics
establish the implementation's accounting and oracle gates; they do not establish
generalization, latency, or production coding ability. The optional HTML output is a
self-contained dashboard with visible/hidden success bars and the complete JSON ledger.
The optional checkpoint is atomically updated after each matrix cell; `--resume` reuses
completed cells and reruns only missing observations, with at-least-once provider semantics.

## Live context compilation

Every new CLI run constructs a deterministic `ContextCompiler`. Before a model call, the
compiler turns the complete projected transcript into candidate blocks, scores those
blocks, selects a bounded subset, restores chronological order, and records what it did.
The source transcript is never rewritten or discarded.

### Protocol-atomic blocks

A standalone system, user, or assistant message is one block. An assistant message that
declares tool calls and all immediately following matching tool-result messages form one
atomic block. The compiler validates call ids, tool names, missing results, orphan results,
duplicate results, and declared result order. The runtime separately permits a call id to
reappear only with identical arguments and a cached identical outcome. A policy can retain
or evict either complete exchange, but cannot send a tool result without the assistant call
that created it.

System and user blocks are mandatory. The newest configured number of blocks are also
mandatory, regardless of policy. If mandatory context cannot fit after tool-output
compaction, compilation fails explicitly instead of silently dropping the task or emitting
an over-budget request.

The live policies are:

| Policy | Selection rule |
|---|---|
| `full` | Keep every block; fail if the complete estimate exceeds the budget. |
| `recent` | Keep mandatory blocks, then fill a newest-first contiguous sliding window. |
| `topk` | Rank optional blocks by deterministic standalone utility. |
| `density` | Rank optional blocks by standalone utility per estimated token. |
| `submodular` | Greedily maximize marginal set utility per token, including topic coverage and duplicate penalties. |

The original exact knapsack and oracle policies remain available to the offline `pack` and
synthetic `benchmark` commands; they are not live runtime choices.

### CLI configuration and estimates

New `contextopt run` invocations accept these settings:

| Flag | Default |
|---|---:|
| `--context-policy` | `submodular` |
| `--context-budget` | `16000` |
| `--context-recent-blocks` | `2` |
| `--context-max-tool-output-tokens` | `2048` |
| `--context-memory` | `versioned-v1` |

For example:

```text
contextopt run <task> --workspace <path> \
  --context-policy submodular \
  --context-budget 16000 \
  --context-recent-blocks 2 \
  --context-max-tool-output-tokens 2048 \
  --context-memory versioned-v1
```

The compiler's “tokens” are deterministic estimates, not results from the selected
provider's tokenizer. Runs of ASCII text, CJK characters, punctuation, message metadata,
and tool-call arguments are counted by a small fixed local rule so recovery can
reproduce the same input without a provider package. Provider-reported input/output usage
continues to drive `RunLimits.max_total_tokens`; the context estimate controls only message
packing.

Before selection, a tool observation above
`--context-max-tool-output-tokens` is deterministically replaced by a compact marker,
content digest, omitted-character count, and retained head/tail evidence. The original
observation remains in the authoritative event-derived transcript. Non-tool messages and
tool-call arguments are not silently truncated, so they can still make mandatory context
infeasible under an unrealistically small budget. The per-observation cap has a minimum of
48 estimated tokens so the compact representation can retain its full digest and bounded
head/tail evidence under the runtime's tool-output byte limit.

### Observed-memory invalidation

With the default `versioned-v1` memory policy, the runner deterministically rebuilds a
`MemorySnapshot` from canonical tool outcomes already present in the transcript. Recorded
file writes advance the observed workspace generation, invalidate earlier evidence for the
same path, invalidate prior workspace search/listing results, and invalidate test evidence
produced before the mutation. A later read becomes evidence for the new observed path
version; rerunning the same test scope invalidates the older result for that generation. A
cached `tool.reused` write outcome is recorded as an alias of the original evidence and
does not advance the workspace generation a second time.

Stale evidence remains auditable, but receives zero freshness and strongly reduced
importance when blocks are scored. The snapshot's fingerprint and workspace generation
are attached to the request receipt. This is an **observed-memory model**: an out-of-band
filesystem change that no tool outcome recorded is not detected, and no semantic facts are
learned or carried across runs.

Set `--context-memory none` to disable these validity signals while retaining context
routing and receipts.

### Optional cross-run semantic memory

The observed-memory projection above is rebuilt per run and tracks facts about the current
tool transcript. For experience that should survive a later run, pass an explicit JSONL store:

```text
contextopt run "Fix the parser" \
  --workspace <temporary-workspace-copy> \
  --script examples/runtime_demo/script.json \
  --memory-store .contextopt/memory.jsonl \
  --allow-write --allow-command \
  --test-command "python -m unittest discover -s tests -v" \
  --event-log <temporary-events.jsonl>
```

`memory_search` is available whenever the store is attached. `memory_save` and
`memory_invalidate` are exposed only with `--allow-write`; save accepts a short `fact`,
`decision`, `procedure`, or `failure` plus scope, tags, confidence, source references, and an
optional `supersedes` id. Invalidate takes an id and a reason. The store is append-only,
hash-chained, and protected by the same single-host lease as runtime traces. Content-derived
identity makes a repeated save safe after an interrupted tool boundary; explicit supersession
and invalidation preserve the history rather than silently editing it. `global` entries are
visible to a requested project scope, and search results include matched terms, revision, and
memory ids for auditability.

This layer is deliberately lexical and provider-free. It has no embedding index, automatic
consolidation, or learned confidence calibration. The model must ask for memory explicitly, and
the result is an advisory hint—not proof that a mutable workspace still satisfies the remembered
claim. A non-terminal `resume` must receive the same `--memory-store` path because the store
identity is part of the tool configuration fingerprint.

For a fresh-process cross-run demonstration, use
[`examples/semantic_memory_demo`](../examples/semantic_memory_demo/README.md). Its writer and
reader scripts make the boundary visible without an API key: the first run saves a procedure,
the second run searches the same store, and both traces remain independently auditable.

The retrieval-only conformance command is `python -m contextopt memory-eval`; it repeats fixed
queries and reports hit@k/MRR, scope isolation, invalidation exclusion, negative-query behavior,
and deterministic replay. Its raw JSON, Markdown, and HTML outputs are checked in under
[`experiments/v0.9-semantic-memory`](../experiments/v0.9-semantic-memory/README.md).

### Per-turn receipt and recovery contract

Every new `model.requested` event includes a `context` receipt. It contains:

- context configuration fingerprint, policy, and estimated-token budget;
- original, post-compaction candidate, and selected estimates;
- selected, evicted, compacted, and stale block ids;
- the complete policy `ContextFrame`, including each selection decision;
- each block's source message range, roles, mandatory/stale/compacted flags, and estimates;
- compiled-message count and roles, memory fingerprint, workspace generation, and a
  compiled-message SHA-256.

The event's outer `request_sha256` separately covers the compiled messages, tool
definitions, and maximum output-token request. On resume, the runtime loads the persisted
context configuration, verifies its fingerprint, reconstructs the transcript and observed
memory, recompiles the pending request, and refuses to call the model if that outer request
hash changed. Receipt deserialization also checks its block partition, estimates,
selected/evicted sets, roles, and `ContextFrame` for internal consistency. The reducer also
recompiles every recorded context receipt from the authoritative transcript and rejects a
self-consistent receipt that did not come from that state.

`compiler_version` is part of the persisted configuration fingerprint. Token estimation,
block grouping, compaction, scoring, selection, and memory-annotation semantics together
form this context ABI; changing any of them requires a version bump plus a compatible
reader/compiler or an explicit trace migration.

This makes pending-request recovery reproducible; it does not make the provider call
exactly once. If no durable model response exists, a matching pending request may still be
sent again.

### Context routing/compiler conformance evaluation

Run a paired, model-free comparison from the repository root:

```text
python -m contextopt context-eval \
  --policies recent,submodular --budgets 1024 --repetitions 3 \
  --output context-routing.json
```

Across the three fixed, protocol-valid fixtures, the current 1,024-estimated-token result
is:

| Policy | Evidence recall | Protocol valid | Budget compliant |
|---|---:|---:|---:|
| `recent` | 0.222 | 1.000 | 1.000 |
| `submodular` | 0.778 | 1.000 | 1.000 |

The JSON report records `model_calls: 0`. Gold evidence probes are used only by the scorer
and are not passed to a policy. Evidence recall measures exact retention in the compiled
messages; protocol validity, budget compliance, compression, and byte determinism are
compiler properties. These figures do **not** measure model quality, reasoning, generated
code correctness, or end-to-end task success. See
[`context-routing-eval.md`](context-routing-eval.md) for the complete claim boundary.

## Offline runtime demo

The demo needs Python 3.11 or newer, an editable project install, and no API key or network
connection. It deliberately copies the buggy fixture to a new temporary workspace because
the Agent has write permission. Do not point the run at the checked-in
`examples/runtime_demo/workspace` directory.

Install and verify the project first:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

### PowerShell

Run these commands from the repository root:

```powershell
$demoRoot = Join-Path ([IO.Path]::GetTempPath()) `
  ("contextopt-demo-" + [guid]::NewGuid().ToString("N"))
$workspace = Join-Path $demoRoot "workspace"
$events = Join-Path $demoRoot "events.jsonl"
New-Item -ItemType Directory -Path $demoRoot | Out-Null
Copy-Item -Recurse -LiteralPath examples\runtime_demo\workspace `
  -Destination $workspace

python -m contextopt run `
  "Fix merge_settings so only None inherits a default. Preserve inputs." `
  --workspace $workspace `
  --script examples\runtime_demo\script.json `
  --allow-write `
  --allow-command `
  --test-command "python -m unittest discover -s tests -v" `
  --run-id runtime-demo `
  --event-log $events

python -m contextopt trace $events
python -m contextopt trace $events --html trace.html
python examples\runtime_demo\hidden_oracle.py $workspace
```

### POSIX shell

Run these commands from the repository root:

```bash
demo_root="$(mktemp -d)"
workspace="$demo_root/workspace"
events="$demo_root/events.jsonl"
cp -R examples/runtime_demo/workspace "$workspace"

python -m contextopt run \
  "Fix merge_settings so only None inherits a default. Preserve inputs." \
  --workspace "$workspace" \
  --script examples/runtime_demo/script.json \
  --allow-write \
  --allow-command \
  --test-command "python -m unittest discover -s tests -v" \
  --run-id runtime-demo \
  --event-log "$events"

python -m contextopt trace "$events"
python -m contextopt trace "$events" --html trace.html
python examples/runtime_demo/hidden_oracle.py "$workspace"
```

The expected tool order is:

```text
read_file(settings.py)
read_file(tests/test_settings.py)
run_tests(visible)       # exit 1; the loop must continue
replace_text(settings.py)
run_tests(visible)       # exit 0
final model response
```

The compact trace should end like this; sequence numbers are stable for this script, while
timestamps and elapsed time are not:

```text
0000 run.started status=running reason=
...
0015 tool.completed tool=run_tests call=test-before ok=True exit=1
...
0025 tool.completed tool=run_tests call=test-after ok=True exit=0
...
0029 run.completed status=completed reason=model_stopped
```

The optional `--html PATH` output is a dependency-free local timeline. It summarizes event,
model-request, tool-outcome, and context-receipt counts; each event can be expanded to inspect
its original JSON payload. It is a read-only view over the JSONL log, not a replacement for
hash-chain verification or the durable event source.

The final oracle prints:

```text
hidden oracle passed
```

It separately checks explicit `False`, `0`, and empty-string overrides, `None` inheritance,
and input immutability. The oracle file remains outside the model workspace.

## Why the script is observation-aware

[`examples/runtime_demo/script.json`](../examples/runtime_demo/script.json) is not just a
queue of actions. After each tool call, its next step asserts the previous tool name, call
id, and evidence contained in the observation. In particular:

- the source read must expose the expected SHA-256 and buggy condition;
- the test read must expose the named visible test;
- the first test execution must report `exit_code=1`;
- the edit must report one replacement;
- the second test execution must report `exit_code=0` and `OK`.

The edit also supplies the source SHA-256 as a compare-and-swap precondition. If the source
changes between read and write, `replace_text` refuses the mutation instead of applying a
stale edit.

This makes the demo sensitive to broken tool-result routing. It still does not test model
reasoning: the correct tool calls and edit are authored in the script.

## Status and resume

Inspect a schema-2 run without invoking a model:

```bash
python -m contextopt status <events.jsonl>
```

`status` reports the reconstructed phase, turns, tool calls, cumulative usage, pending
model request, pending tools, context configuration/fingerprint and latest receipt,
terminal result, and the last projected event hash. A terminal run is not resumable.
Running `resume` on it needs no workspace or model arguments and returns the existing
durable result without appending an event:

```bash
python -m contextopt resume <terminal-events.jsonl>
```

Resume a non-terminal scripted run with the same workspace, complete script, and registered
test command used by the original run:

```bash
python -m contextopt resume <events.jsonl> \
  --workspace <same-workspace> \
  --script <same-script.json> \
  --test-command "python -m unittest discover -s tests -v"
```

For a real-model run, pass the same adapter settings instead of `--script`:

```bash
python -m contextopt resume <events.jsonl> \
  --workspace <same-workspace> \
  --model <same-model> \
  --base-url <same-endpoint> \
  --api-key-env CONTEXTOPT_API_KEY \
  --test-command <same-registered-command>
```

The API key is not fingerprinted or persisted. Model identity/settings, tool definitions,
resolved workspace identity, permissions, registered commands, and context configuration
are fingerprinted; run limits and permissions are also compared directly. The context
policy, budget, recent-block count, compaction limit, and memory policy come from the
original event log rather than resume-time flags. A non-terminal resume is rejected before
adding `run.resumed` if these boundaries do not match.

Schema-1 logs remain readable by `trace` and receive audit-only metadata from `status`.
They cannot be reduced into resumable state or extended with schema-2 events.

### Recovery projection and checkpoint

The schema-2 event log is authoritative. A strict reducer validates legal state
transitions and reconstructs:

- initial task/configuration, context identity, and complete normalized messages;
- turns, cumulative token usage, and limits;
- pending model requests and ordered pending tool calls;
- sealed tool-execution plans and completed call outcomes;
- running, interrupted, paused, or terminal phase.

An atomic JSON checkpoint caches one validated projection. It stores its source sequence,
event hash, and state hash. If it is absent, corrupt, stale, or inconsistent with the log,
the runtime ignores it and performs a full replay. A state hash is not authentication, so
the current format also strictly replays the anchored event prefix and compares the result
before accepting a self-consistent checkpoint, then reduces the suffix. This is a
correctness-first recovery artifact, not yet a startup-performance claim. The checkpoint
is disposable cache, not an alternative source of truth and not a workspace snapshot.

On resume, a projection still marked `running` is first marked `run.interrupted`; an
interrupted or paused projection then receives `run.resumed`. A persisted final model
response with no terminal event can be completed without calling the model again. A pending
model request may be called again because no durable response exists.

### Interrupted tool recovery

Before executing a tool, the runner persists its operation id, call fingerprint, replay
policy, and—where available—pre/postconditions. Recovery is deliberately tool-specific:

| Interrupted tool | Default recovery |
|---|---|
| `list_files`, `search_text`, `read_file` | Retry automatically; these tools are read-only. |
| `create_file` | Compare non-existence precondition and content-SHA postcondition. Record completion if post-state matches, retry if pre-state matches, otherwise pause on divergence. |
| `replace_text` | Compare before/after file SHA. Record completion if post-state matches, retry if pre-state matches, otherwise pause on divergence. |
| `run_tests` | Never replay automatically; pause because the registered command may have external effects. |

The default pause returns status `paused` and CLI exit code 4. An operator must explicitly
choose one of these continuations:

```bash
# Do not rerun. Feed the model an indeterminate failed tool observation.
python -m contextopt resume <events.jsonl> \
  --workspace <same-workspace> --script <same-script.json> \
  --test-command "python -m unittest discover -s tests -v" \
  --pending-tool-resolution mark_failed

# Run the command again, accepting possible repeated effects.
python -m contextopt resume <events.jsonl> \
  --workspace <same-workspace> --script <same-script.json> \
  --test-command "python -m unittest discover -s tests -v" \
  --pending-tool-resolution retry
```

This is not an exactly-once protocol. A process can stop between an external effect and its
durable outcome; reconciliation reduces duplicate writes when observable state proves what
happened, but it cannot prove arbitrary external effects. Explicit `retry` may execute an
operation again.

## Limits and termination

`RunLimits` bounds:

- turns;
- executed tool calls;
- cumulative reported input and output tokens;
- maximum output tokens requested per model call;
- active-session wall time;
- each registered test command;
- retained tool-output bytes.

The runtime checks tool and turn limits before the next action. Provider token usage is
known only after a response, so a response can report a small overage; in that case its
tool calls are not executed and the run stops with `token_limit_after_response`.

Turn, tool-call, and token consumption are reconstructed and remain cumulative across
resume sessions. Wall timeout is different: it bounds each active `run` or `resume`
session and does not charge time while the process is stopped.

Run status and task correctness are intentionally different:

- `completed`: the model returned a response without tool calls;
- `stopped`: a configured limit or timeout ended the run;
- `failed`: a normalized model failure or runtime contract error occurred;
- `paused`: recovery needs an explicit operator decision;
- `cancelled`: the caller cancelled the asynchronous run.

A `completed` run has not necessarily fixed the task. Visible and hidden executable oracles
must determine that separately.

## Permissions and security boundary

Writes and commands are disabled by default. `--allow-write` enables `create_file` and
`replace_text`; `--allow-command` enables only commands registered at startup. The CLI
registers `--test-command` under the `visible` scope, and the model may select the scope but
cannot replace its command line.

File tools reject absolute paths, drive-relative paths, parent traversal, backslashes,
symbolic links, reserved device names, and access to `.git` or `.contextopt`. Output is
bounded and recorded with capture metadata.

These controls are defense in depth, not a process sandbox. A registered test command is a
trusted host subprocess and may execute arbitrary code present in the workspace. Do not run
untrusted repositories outside an actual container or virtual-machine boundary.

## Event log

Every new JSONL line has this schema-2 envelope:

```json
{
  "schema_version": "2",
  "run_id": "runtime-demo",
  "seq": 15,
  "timestamp": "2026-08-08T00:00:00.000Z",
  "type": "tool.completed",
  "data": {},
  "prev_event_sha256": "<sha256-of-event-14>",
  "event_sha256": "<sha256-of-this-canonical-event>"
}
```

The reader verifies contiguous sequence numbers, one run id, every event hash, and every
link to the previous hash. This is **corruption-evident integrity**, not protection against
malicious rewriting: there is no secret, signature, trusted timestamp, or external anchor,
so an attacker able to rewrite the complete file can recompute the entire chain.

Current event families are:

- `run.started`, `run.interrupted`, `run.resumed`, and `run.paused`, followed eventually
  by at most one terminal `run.completed`, `run.stopped`, `run.failed`, or `run.cancelled`;
- `model.requested`, `model.responded`, and `model.failed`;
- `budget.updated` after a model response;
- `tool.started`, `tool.completed`, `tool.failed`, and cached `tool.reused`;
- `runtime.failed` for a contract or local runtime failure.

The log is appended and flushed after every event. A reader preserves the last valid byte
offset and tolerates only an incomplete final line. Opening an appendable log refuses that
tail unless repair is explicitly requested; CLI resume requests repair and physically
removes only the incomplete suffix. A complete final JSON event without a newline is
preserved, terminated, and safely continued.

`EventLog` holds a non-blocking lease backed by `msvcrt.locking` on Windows or `flock` on
POSIX, plus an in-process registry. A second cooperative writer fails clearly. This is a
single-host file lease, not a distributed consensus mechanism.

Each `model.requested` event also records a deterministic `idempotency_key` derived from the
run id, turn, and request SHA-256. The OpenAI-compatible adapter sends the same value in an
`Idempotency-Key` header by default (set `idempotency_header=None` in programmatic use to omit
it). This is a provider hook, not an exactly-once guarantee: providers may ignore the header,
and a retry is still visible in the local ledger.

## Optional real-model boundary

Without `--script`, the CLI can call a non-streaming OpenAI-compatible Chat Completions
endpoint. The model name and base URL are explicit, and the API key is read from an
environment variable rather than a command-line argument:

```text
contextopt run <task> --workspace <path> \
  --model <model-name> --base-url <endpoint> \
  --api-key-env CONTEXTOPT_API_KEY
```

This adapter is an integration boundary, not a published real-model benchmark. A fair
comparison still requires fixed repository snapshots, prompts, tools, limits, model
versions, repetitions, and independent hidden tests.

For v0.3 runs, `model.requested.data.context` is the validated per-turn compiler receipt;
legacy v0.2 logs without a context configuration continue to replay as full-history runs.

## Current outer-loop boundary and remaining milestones

The v0.6 `contextopt.search` package now provides the outer proposal/test/search session
described in [Architecture](architecture.md): candidates are evaluated in disposable
workspaces, feedback and budgets survive atomic session checkpoints, and an accepted snapshot
can be explicitly applied or rolled back with a stale-baseline guard. That outer loop is
deliberately separate from the v0.3 `AgentRunner` event log; it does not silently mutate the
runtime workspace or claim that provider calls are exactly once.

The v0.7 package now also provides a planner/solver/reviewer orchestrator with strict role
protocols, shared budgets, cross-round feedback, and an oracle-gated decision. Planner and
reviewer calls remain sequential; the solver has an opt-in bounded speculative fan-out.
It is intentionally separate from the single-agent runtime event log. Its candidate oracle
can use bounded parallel isolated workspaces, and `speculative_solver_width > 1` can issue
independent solver provider calls concurrently under the shared model/candidate budget. Every
lane stores its request/response hash, context receipt, usage, validation error, and provider
in-flight observation; valid snapshots are namespaced before the common oracle. Both session
and orchestration checkpoints persist requested/completed candidate events and the observed
maximum in-flight counts.

The solver fan-out also has an opt-in first-valid race. Set
`OrchestrationConfig.speculative_solver_stop_on_valid=True` or pass
`--speculative-solver-stop-on-valid` with a width greater than one to select the first response
that passes the strict candidate parser. The runner appends a `solver.speculative.winner` event,
requests cancellation for lanes that have not crossed the durable response boundary, and appends
one `solver.speculative.cancelled` event per such lane. The selected snapshot still goes through
the same visible-test oracle and reviewer gate. A cancellation event means local cancellation
was requested; it deliberately records `provider_cancellation=best_effort` because cancelling
an asyncio task cannot interrupt every provider's remote HTTP work. Adapters may optionally expose
`request_cancellation(request)`; its `acknowledged`, `not_observed`, `unsupported`, or `failed:*` result is
persisted as `provider_cancel_status` in the cancellation event. The built-in serial
OpenAI-compatible adapter can close an active local `urllib` response and reports
`acknowledged` for that transport interruption. Chat Completions still has no standard remote
abort endpoint, so this does not prove that server-side generation stopped. Solver/model budgets
still charge the configured width, so latency savings and remote cost must be measured separately.

Remaining milestones are:

1. Exercise the new cancellation/winner policy against provider adapters that expose a real
   remote abort primitive; the built-in serial OpenAI-compatible adapter now offers durable
   idempotency plus best-effort local HTTP transport cancellation, but cannot prove server-side
   generation stopped.
2. Run the strategy harness against multiple real model versions and independent hidden
   tests, preserving paired budgets and full ledgers.
3. Add container/VM isolation and a controlled real-model coding benchmark with fixed
   snapshots, versions, repetitions, and independent hidden tests.

The repository still has no OS sandbox or published real-model benchmark. The explicit
apply/rollback adapter is a local filesystem safety boundary, not a security boundary.

The deterministic long-horizon recovery matrix is available as `python -m contextopt
recovery-eval`. It demonstrates the covered event boundaries and conservative tool policies,
but remains a local conformance artifact rather than a machine-loss or exactly-once guarantee.
