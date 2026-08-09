from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from contextopt.cli import main
from contextopt.runtime.context import (
    ContextCompiler,
    ContextCompilerConfig,
    compile_runtime_context,
)
from contextopt.runtime.knowledge import (
    KNOWLEDGE_CHUNK_TAG,
    KnowledgeIndexConfig,
    index_workspace,
)
from contextopt.runtime.protocol import AgentMessage
from contextopt.runtime.semantic_memory import SemanticMemoryStore


class KnowledgeIndexTests(unittest.TestCase):
    def test_index_is_deterministic_reusable_and_source_aware(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            root.mkdir()
            (root / "src").mkdir()
            source = root / "src" / "solver.py"
            source.write_text(
                "def solve(values):\n    return sorted(values)\n\n"
                "def obsolete():\n    return 0\n",
                encoding="utf-8",
            )
            (root / "README.md").write_text(
                "The solver returns sorted values.\n", encoding="utf-8"
            )
            (root / ".git").mkdir()
            (root / ".git" / "ignored.py").write_text(
                "secret should not be indexed\n", encoding="utf-8"
            )
            store_path = Path(directory) / "memory.jsonl"
            config = KnowledgeIndexConfig(
                workspace=root,
                scope="project:knowledge-test",
                chunk_lines=2,
            )
            with SemanticMemoryStore(store_path) as store:
                first = index_workspace(store, config)
                self.assertEqual(
                    type(first).from_dict(first.to_dict()).to_dict(), first.to_dict()
                )
                self.assertEqual(first.files_considered, 2)
                self.assertEqual(first.files_indexed, 2)
                self.assertGreater(first.chunks_created, 0)
                self.assertEqual(first.chunks_reused, 0)
                self.assertEqual(first.chunks_invalidated, 0)
                self.assertEqual(
                    len(
                        [
                            entry
                            for entry in store.active_entries(scope=config.scope)
                            if KNOWLEDGE_CHUNK_TAG in entry.tags
                        ]
                    ),
                    len(first.active_chunk_ids),
                )
                matches = store.search("solver sorted values", scope=config.scope)
                self.assertTrue(matches)
                self.assertIn("src/solver.py", matches[0].entry.text)
                second = index_workspace(store, config)
                self.assertEqual(second.chunks_created, 0)
                self.assertGreater(second.chunks_reused, 0)
                self.assertEqual(second.chunks_invalidated, 0)

                source.write_text(
                    "def solve(values):\n    return list(values)\n", encoding="utf-8"
                )
                third = index_workspace(store, config)
                self.assertGreater(third.chunks_created, 0)
                self.assertGreater(third.chunks_invalidated, 0)
                self.assertFalse(store.search("obsolete", scope=config.scope))
                self.assertTrue(store.search("solver list values", scope=config.scope))

            with SemanticMemoryStore(store_path) as reopened:
                self.assertTrue(
                    reopened.search(
                        "solver list values", scope="project:knowledge-test"
                    )
                )
                self.assertFalse(
                    reopened.search(
                        "secret should not be indexed", scope="project:knowledge-test"
                    )
                )

    def test_indexed_source_enters_the_existing_semantic_context_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            root.mkdir()
            (root / "parser.py").write_text(
                "def parse_token(token):\n    return token.strip()\n",
                encoding="utf-8",
            )
            store_path = Path(directory) / "memory.jsonl"
            with SemanticMemoryStore(store_path) as store:
                index_workspace(
                    store,
                    KnowledgeIndexConfig(
                        workspace=root,
                        scope="project:parser",
                        chunk_lines=8,
                    ),
                )
                compiled = compile_runtime_context(
                    ContextCompiler(
                        ContextCompilerConfig(
                            policy="submodular",
                            budget_tokens=2_048,
                            recent_blocks=0,
                            memory_policy="versioned-v1+semantic",
                        )
                    ),
                    (
                        AgentMessage(
                            role="user", content="Fix parse_token whitespace behavior."
                        ),
                    ),
                    task="parse_token parser.py whitespace",
                    memory_store=store,
                    memory_scope="project:parser",
                )
                metadata = compiled.receipt.frame["metadata"]
                self.assertTrue(metadata["durable_memory_selected_ids"])
                self.assertIn("parser.py", compiled.messages[-1].content)

    def test_bounds_and_cli_report_are_auditable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            root.mkdir()
            (root / "small.py").write_text("answer = 42\n", encoding="utf-8")
            (root / "large.py").write_text(
                "x = '" + ("a" * 100) + "'\n", encoding="utf-8"
            )
            store = Path(directory) / "memory.jsonl"
            report_path = Path(directory) / "report.json"
            markdown_path = Path(directory) / "report.md"
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    main(
                        [
                            "knowledge-index",
                            "--workspace",
                            str(root),
                            "--memory-store",
                            str(store),
                            "--memory-scope",
                            "project:cli",
                            "--max-file-bytes",
                            "32",
                            "--output",
                            str(report_path),
                            "--markdown",
                            str(markdown_path),
                        ]
                    ),
                    0,
                )
            self.assertIn("knowledge-index", output.getvalue())
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["config"]["scope"], "project:cli")
            self.assertEqual(payload["files_indexed"], 1)
            self.assertEqual(payload["skipped_files"][0]["path"], "large.py")
            self.assertIn("Claim boundary", markdown_path.read_text(encoding="utf-8"))

    def test_run_can_auto_index_before_compiling_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "parser.py").write_text(
                "def parse(value):\n    return value.strip()\n", encoding="utf-8"
            )
            script = root / "script.json"
            script.write_text(
                json.dumps({"steps": [{"response": {"content": "done"}}]}),
                encoding="utf-8",
            )
            memory_store = root / "memory.jsonl"
            report_path = root / "knowledge.json"
            markdown_path = root / "knowledge.md"
            with redirect_stdout(StringIO()):
                exit_code = main(
                    [
                        "run",
                        "Fix parser whitespace",
                        "--workspace",
                        str(workspace),
                        "--script",
                        str(script),
                        "--memory-store",
                        str(memory_store),
                        "--memory-scope",
                        "project:auto",
                        "--context-memory",
                        "versioned-v1+semantic",
                        "--auto-index-knowledge",
                        "--knowledge-index-report",
                        str(report_path),
                        "--knowledge-index-markdown",
                        str(markdown_path),
                    ]
                )
            self.assertEqual(exit_code, 0)
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["files_indexed"], 1)
            self.assertIn("knowledge index", markdown_path.read_text(encoding="utf-8"))
            with SemanticMemoryStore(memory_store) as store:
                self.assertTrue(store.search("parser strip", scope="project:auto"))

    def test_config_rejects_unbounded_chunk_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                KnowledgeIndexConfig(workspace=root, max_chunk_bytes=16 * 1024 + 1)


if __name__ == "__main__":
    unittest.main()
