from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from contextopt.runtime import (
    AgentRunner,
    EventLog,
    RunPermissions,
    ScriptedModel,
    SemanticMemoryStore,
    ToolCall,
    WorkspaceTools,
    read_events,
)
from contextopt.runtime.tool_state import tool_call_fingerprint


class SemanticMemoryStoreTests(unittest.TestCase):
    def test_put_is_idempotent_and_replays_from_append_only_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.jsonl"
            with SemanticMemoryStore(path) as store:
                first = store.put(
                    "replace_text requires the latest file sha256",
                    scope="project:demo",
                    kind="procedure",
                    tags=("cas", "tools"),
                    confidence=0.9,
                    source_run_id="run-1",
                )
                duplicate = store.put(
                    "replace_text requires the latest file sha256",
                    scope="project:demo",
                    kind="procedure",
                    tags=("tools", "cas"),
                    confidence=0.2,
                )
                self.assertTrue(first.created)
                self.assertFalse(duplicate.created)
                self.assertEqual(first.entry.memory_id, duplicate.entry.memory_id)
                self.assertEqual(store.revision, 1)
                match = store.search("latest sha256", scope="project:demo")[0]
                self.assertEqual(match.entry.memory_id, first.entry.memory_id)
                self.assertEqual(match.matched_terms, ("latest", "sha256"))

            with SemanticMemoryStore(path) as reopened:
                self.assertEqual(reopened.revision, 1)
                store_fingerprint = reopened.fingerprint
                self.assertTrue(store_fingerprint)
                self.assertEqual(len(reopened.active_entries()), 1)
                self.assertEqual(reopened.get(first.entry.memory_id), first.entry)

    def test_scope_tags_and_supersede_are_auditable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.jsonl"
            with SemanticMemoryStore(path) as store:
                old = store.put(
                    "The parser rejects duplicate tool call ids.",
                    scope="project:demo",
                    kind="failure",
                    tags=("protocol",),
                )
                new = store.put(
                    "The parser rejects duplicate tool call ids before execution.",
                    scope="project:demo",
                    kind="decision",
                    tags=("protocol",),
                    supersedes=old.entry.memory_id,
                )
                self.assertEqual(store.get(old.entry.memory_id).status, "superseded")
                self.assertTrue(store.get(new.entry.memory_id).active)
                self.assertEqual(
                    len(store.search("duplicate tool", tags=("protocol",))), 1
                )
                self.assertEqual(
                    len(store.search("duplicate tool", scope="project:other")), 0
                )
                invalidated = store.invalidate(new.entry.memory_id, "obsolete rule")
                self.assertEqual(invalidated.status, "invalidated")
                self.assertEqual(store.search("duplicate tool"), ())

    def test_source_ref_invalidation_replays_across_processes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.jsonl"
            with SemanticMemoryStore(path) as store:
                stale = store.put(
                    "The parser accepts the old token framing.",
                    scope="project:demo",
                    kind="fact",
                    source_refs=("./src/parser.py",),
                )
                unrelated = store.put(
                    "The parser accepts the fixture framing.",
                    scope="project:demo",
                    kind="fact",
                    source_refs=("src/other.py",),
                )
                invalidated = store.invalidate_source_refs(
                    "src\\parser.py", "workspace source changed"
                )
                self.assertEqual(
                    tuple(item.memory_id for item in invalidated),
                    (stale.entry.memory_id,),
                )
                self.assertEqual(store.get(stale.entry.memory_id).status, "invalidated")
                self.assertTrue(store.get(unrelated.entry.memory_id).active)
                self.assertEqual(store.search("old token"), ())

            with SemanticMemoryStore(path) as reopened:
                self.assertEqual(reopened.revision, 3)
                self.assertEqual(
                    reopened.get(stale.entry.memory_id).status, "invalidated"
                )
                self.assertTrue(reopened.get(unrelated.entry.memory_id).active)


class SemanticMemoryToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_memory_save_requires_write_permission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            with SemanticMemoryStore(root / "memory.jsonl") as store:
                tools = WorkspaceTools(workspace, memory_store=store)
                self.assertIn(
                    "memory_search", {item.name for item in tools.definitions}
                )
                self.assertNotIn(
                    "memory_save", {item.name for item in tools.definitions}
                )
                outcome = await tools.execute(
                    ToolCall(
                        id="save-denied",
                        name="memory_save",
                        arguments_json=json.dumps({"text": "must not write"}),
                    )
                )
                self.assertFalse(outcome.ok)
                self.assertEqual(outcome.error_code, "unknown_tool")
                tools.close()

    async def test_memory_survives_two_agent_runs_and_is_visible_in_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            memory_path = root / "memory.jsonl"

            with SemanticMemoryStore(memory_path) as store:
                first_tools = WorkspaceTools(
                    workspace,
                    permissions=RunPermissions(allow_write=True),
                    memory_store=store,
                )
                first_model = ScriptedModel(
                    [
                        {
                            "response": {
                                "tool_calls": [
                                    {
                                        "id": "save-memory",
                                        "name": "memory_save",
                                        "arguments": {
                                            "text": (
                                                "replace_text requires the latest file "
                                                "sha256"
                                            ),
                                            "scope": "project:demo",
                                            "kind": "procedure",
                                            "tags": ["cas", "edit"],
                                            "source_run_id": "first-run",
                                        },
                                    }
                                ],
                                "usage": {"input_tokens": 3, "output_tokens": 2},
                            }
                        },
                        {
                            "expect": {
                                "last_tool": "memory_save",
                                "tool_call_id": "save-memory",
                                "observation_contains": "latest file sha256",
                            },
                            "response": {
                                "content": "Saved the durable procedure.",
                                "usage": {"input_tokens": 3, "output_tokens": 2},
                            },
                        },
                    ],
                    name="memory-writer:v1",
                )
                first_events = root / "first-events.jsonl"
                first_result = await AgentRunner(
                    model=first_model,
                    tools=first_tools,
                    event_log=EventLog(first_events, "first-run"),
                ).run("Remember the safe edit procedure.")
                first_tools.close()

            with SemanticMemoryStore(memory_path) as reopened:
                self.assertEqual(len(reopened.active_entries(scope="project:demo")), 1)
                second_tools = WorkspaceTools(workspace, memory_store=reopened)
                second_model = ScriptedModel(
                    [
                        {
                            "response": {
                                "tool_calls": [
                                    {
                                        "id": "search-memory",
                                        "name": "memory_search",
                                        "arguments": {
                                            "query": "latest file sha256",
                                            "scope": "project:demo",
                                        },
                                    }
                                ],
                                "usage": {"input_tokens": 3, "output_tokens": 2},
                            }
                        },
                        {
                            "expect": {
                                "last_tool": "memory_search",
                                "tool_call_id": "search-memory",
                                "observation_contains": [
                                    "latest file sha256",
                                    "project:demo",
                                ],
                            },
                            "response": {
                                "content": "I found the prior procedure.",
                                "usage": {"input_tokens": 3, "output_tokens": 2},
                            },
                        },
                    ],
                    name="memory-reader:v1",
                )
                second_events = root / "second-events.jsonl"
                second_result = await AgentRunner(
                    model=second_model,
                    tools=second_tools,
                    event_log=EventLog(second_events, "second-run"),
                ).run("Use the remembered edit procedure.")
                second_tools.close()

            self.assertEqual(first_result.status, "completed")
            self.assertEqual(second_result.status, "completed")
            first_tool_events = [
                event
                for event in read_events(first_events)
                if event["type"] == "tool.completed"
            ]
            self.assertEqual(first_tool_events[0]["data"]["tool_name"], "memory_save")
            self.assertTrue(first_tool_events[0]["data"]["metadata"]["created"])
            second_tool_events = [
                event
                for event in read_events(second_events)
                if event["type"] == "tool.completed"
            ]
            self.assertEqual(
                second_tool_events[0]["data"]["tool_name"], "memory_search"
            )
            self.assertEqual(second_tool_events[0]["data"]["metadata"]["matches"], 1)

    async def test_runtime_tools_search_and_save_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            with SemanticMemoryStore(root / "memory.jsonl") as store:
                tools = WorkspaceTools(
                    workspace,
                    permissions=RunPermissions(allow_write=True),
                    memory_store=store,
                )
                save = await tools.execute(
                    ToolCall(
                        id="save-1",
                        name="memory_save",
                        arguments_json=json.dumps(
                            {
                                "text": (
                                    "Run visible tests before asking the reviewer to "
                                    "accept."
                                ),
                                "scope": "project:demo",
                                "kind": "procedure",
                                "tags": ["tests", "review"],
                                "confidence": 0.95,
                                "source_run_id": "run-1",
                            }
                        ),
                    )
                )
                self.assertTrue(save.ok)
                self.assertTrue(save.metadata["created"])
                search = await tools.execute(
                    ToolCall(
                        id="search-1",
                        name="memory_search",
                        arguments_json=json.dumps(
                            {"query": "visible tests reviewer", "scope": "project:demo"}
                        ),
                    )
                )
                self.assertTrue(search.ok)
                self.assertEqual(search.metadata["matches"], 1)
                self.assertIn("Run visible tests", search.content)
                self.assertIn(
                    "memory_search", {item.name for item in tools.definitions}
                )
                self.assertIn("memory_save", {item.name for item in tools.definitions})
                first_id = json.loads(save.content)["memory_id"]
                replacement = await tools.execute(
                    ToolCall(
                        id="save-2",
                        name="memory_save",
                        arguments_json=json.dumps(
                            {
                                "text": "Only accept after current tests pass.",
                                "scope": "project:demo",
                                "kind": "decision",
                                "tags": ["review"],
                                "supersedes": first_id,
                            }
                        ),
                    )
                )
                self.assertTrue(replacement.ok)
                second_id = json.loads(replacement.content)["memory_id"]
                invalidated = await tools.execute(
                    ToolCall(
                        id="invalidate-1",
                        name="memory_invalidate",
                        arguments_json=json.dumps(
                            {"memory_id": second_id, "reason": "test cleanup"}
                        ),
                    )
                )
                self.assertTrue(invalidated.ok)
                self.assertIn(
                    "memory_invalidate", {item.name for item in tools.definitions}
                )
                tools.close()

    async def test_workspace_write_invalidates_source_aware_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            source = workspace / "main.py"
            source.write_bytes(b"return 0\n")
            with SemanticMemoryStore(root / "memory.jsonl") as store:
                saved = store.put(
                    "The solver currently returns zero.",
                    scope="project:demo",
                    kind="fact",
                    source_refs=("./main.py",),
                )
                tools = WorkspaceTools(
                    workspace,
                    permissions=RunPermissions(allow_write=True),
                    memory_store=store,
                )
                call = ToolCall(
                    id="replace-source",
                    name="replace_text",
                    arguments_json=json.dumps(
                        {
                            "path": "main.py",
                            "old_text": "return 0",
                            "new_text": "return 1",
                            "expected_sha256": hashlib.sha256(
                                b"return 0\n"
                            ).hexdigest(),
                        }
                    ),
                )
                plan = await tools.prepare(
                    call, "replace-source-operation", tool_call_fingerprint(call)
                )
                outcome = await tools.execute_prepared(plan)
                self.assertTrue(outcome.ok)
                self.assertEqual(
                    outcome.metadata["invalidated_memory_ids"],
                    [saved.entry.memory_id],
                )
                self.assertEqual(store.search("solver returns zero"), ())
                self.assertEqual(source.read_bytes(), b"return 1\n")

                recovered_source = workspace / "recovered.py"
                recovered_source.write_bytes(b"return 0\n")
                recovered = store.put(
                    "The recovery fixture currently returns zero.",
                    scope="project:demo",
                    kind="fact",
                    source_refs=("recovered.py",),
                )
                recovered_call = ToolCall(
                    id="replace-recovered",
                    name="replace_text",
                    arguments_json=json.dumps(
                        {
                            "path": "recovered.py",
                            "old_text": "return 0",
                            "new_text": "return 1",
                            "expected_sha256": hashlib.sha256(
                                b"return 0\n"
                            ).hexdigest(),
                        }
                    ),
                )
                recovered_plan = await tools.prepare(
                    recovered_call,
                    "replace-recovered-operation",
                    tool_call_fingerprint(recovered_call),
                )
                # Simulate a process stop after the file replace but before the
                # advisory memory ledger append.
                recovered_source.write_bytes(b"return 1\n")
                reconciliation = await tools.reconcile(recovered_plan)
                self.assertEqual(reconciliation.action, "completed")
                self.assertEqual(store.search("recovery fixture returns zero"), ())
                self.assertEqual(
                    store.get(recovered.entry.memory_id).status, "invalidated"
                )
                tools.close()


if __name__ == "__main__":
    unittest.main()
