# ForgeAgent runtime

The v0.2a runtime is a bounded, auditable single-agent loop inside the `contextopt`
package. It is the first executable layer of the ForgeAgent direction; it is not yet a
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
                                             │
                                      files + visible tests
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

## Limits and termination

`RunLimits` bounds:

- turns;
- executed tool calls;
- cumulative reported input and output tokens;
- maximum output tokens requested per model call;
- run wall time;
- each registered test command;
- retained tool-output bytes.

The runtime checks tool and turn limits before the next action. Provider token usage is
known only after a response, so a response can report a small overage; in that case its
tool calls are not executed and the run stops with `token_limit_after_response`.

Run status and task correctness are intentionally different:

- `completed`: the model returned a response without tool calls;
- `stopped`: a configured limit or timeout ended the run;
- `failed`: a normalized model failure or runtime contract error occurred;
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

Every JSONL line has this versioned envelope:

```json
{
  "schema_version": "1",
  "run_id": "runtime-demo",
  "seq": 15,
  "timestamp": "2026-08-08T00:00:00.000Z",
  "type": "tool.completed",
  "data": {}
}
```

Current event families are:

- `run.started`, followed by exactly one terminal `run.completed`, `run.stopped`,
  `run.failed`, or `run.cancelled`;
- `model.requested`, `model.responded`, and `model.failed`;
- `budget.updated` after a model response;
- `tool.started`, `tool.completed`, `tool.failed`, and idempotent `tool.reused`;
- `runtime.failed` for a contract or local runtime failure.

The log is appended and flushed after every event. A reader tolerates only an incomplete
final line, which lets a partial trace survive abrupt process termination. The current
runtime does not yet reconstruct and resume execution from that trace.

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

1. Reconstruct authoritative state from events and resume from a workspace checkpoint.
2. Make tool calls idempotent across process restarts, not only within one live run.
3. Extract context candidates from code, test output, decisions, and trajectory events.
4. Compile each model request through ContextOpt and persist its `ContextFrame` receipt.
5. Fork isolated workspaces and allocate a fixed budget across test-guided search branches.
