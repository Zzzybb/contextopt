from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from contextopt.runtime import (
    RunPermissions,
    SemanticMemoryStore,
    ToolCall,
    WorkspaceTools,
)


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


if __name__ == "__main__":
    unittest.main()
