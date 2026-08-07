# ForgeAgent runtime

The v0.2b runtime is a bounded, auditable and recoverable single-agent loop inside the
`contextopt` package. It reconstructs execution state from durable events and can resume a
non-terminal run against the same validated model/tool configuration. It is not yet a
long-term memory or multi-agent system.

## Current execution contract

```text
                          append every boundary
                         ┌──────────────────────> JSONL EventLog
                         │
Task -> AgentRunner -> ModelClient
           ^               │
           │               │ structured ToolCall(s)
           │               v
           └──── observation message <──── WorkspaceTools
           │                                 │
           │                          files + visible tests
           v
 strict event reducer <---- schema-2 log ----> atomic projection cache
```

Each model response may contain text and zero or more structured tool calls. The runner
appends the assistant response, executes calls in order, converts every `ToolOutcome` into
a tool observation carrying the same call id, and sends the enlarged message history to
the next model turn. A response without tool calls completes the run.

The runtime currently exposes:

- `list_files`: enumerate regular files below a workspace-relative directory;
- `search_text`: literal UTF-8 search with bounded matches;
- `read_file`: numbered lines plus the exact file SHA-256;
- `create_file`: create a new UTF-8 file without overwriting an existing file;
- `replace_text`: atomic exact-text replacement guarded by the SHA-256 returned by
  `read_file`;
- `run_tests`: invoke one trusted command registered by the caller, selected by scope.

`run_tests` returning exit code 1 is a successful tool execution with a failing code test.
The observation includes the exit code and captured output so the model can react. A path
violation, malformed argument, denied permission, launch failure, or timeout is instead a
tool failure.

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
model request, pending tools, terminal result, and the last projected event hash. A
terminal run is not resumable. Running `resume` on it needs no workspace or model arguments
and returns the existing durable result without appending an event:

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
resolved workspace identity, permissions, and registered commands are fingerprinted; run
limits and permissions are also compared directly. A non-terminal resume is rejected
before adding `run.resumed` if these boundaries do not match.

Schema-1 logs remain readable by `trace` and receive audit-only metadata from `status`.
They cannot be reduced into resumable state or extended with schema-2 events.

### Recovery projection and checkpoint

The schema-2 event log is authoritative. A strict reducer validates legal state
transitions and reconstructs:

- initial task/configuration and complete normalized messages;
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

## Next runtime milestones

1. Extract context candidates from code, test output, decisions, and trajectory events.
2. Compile each model request through ContextOpt and persist its `ContextFrame` receipt.
3. Add durable model-call idempotency hooks where providers expose them.
4. Snapshot or reference workspace versions rather than requiring one unchanged path.
5. Fork isolated workspaces and allocate a fixed budget across test-guided search branches.
