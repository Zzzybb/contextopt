from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from contextopt.runtime.context import (
    ContextBlockAnnotation,
    ContextBudgetError,
    ContextCompiler,
    ContextCompilerConfig,
    ContextReceipt,
    compile_runtime_context,
    durable_memory_matches_from_receipt,
    estimate_message_tokens,
    estimate_messages_tokens,
    estimate_text_tokens,
)
from contextopt.runtime.protocol import AgentMessage, ToolCall
from contextopt.runtime.semantic_memory import SemanticMemoryStore


def _assistant_call(*call_ids: str, name: str = "read_file") -> AgentMessage:
    return AgentMessage(
        role="assistant",
        content="inspect evidence",
        tool_calls=tuple(
            ToolCall(
                id=call_id,
                name=name,
                arguments_json=f'{{"path":"{call_id}.py"}}',
            )
            for call_id in call_ids
        ),
    )


def _tool(call_id: str, content: str, name: str = "read_file") -> AgentMessage:
    return AgentMessage(
        role="tool",
        content=content,
        tool_call_id=call_id,
        tool_name=name,
    )


class TokenEstimatorTests(unittest.TestCase):
    def test_estimator_is_deterministic_and_language_aware(self) -> None:
        self.assertEqual(estimate_text_tokens("abcdefgh"), 2)
        self.assertEqual(estimate_text_tokens("算法"), 2)
        self.assertEqual(estimate_text_tokens("a + b"), 3)
        message = AgentMessage(role="user", content="fix parser")
        self.assertEqual(
            estimate_messages_tokens((message,)), estimate_message_tokens(message)
        )


class ContextConfigTests(unittest.TestCase):
    def test_round_trip_and_fingerprint_cover_memory_policy(self) -> None:
        config = ContextCompilerConfig(
            policy="submodular",
            budget_tokens=321,
            recent_blocks=3,
            max_tool_output_tokens=99,
            memory_policy="versioned-v1",
        )
        restored = ContextCompilerConfig.from_dict(config.to_dict())
        self.assertEqual(restored, config)
        self.assertEqual(
            restored.configuration_fingerprint,
            config.configuration_fingerprint,
        )
        no_memory = ContextCompilerConfig.from_dict(
            {**config.to_dict(), "memory_policy": "none"}
        )
        self.assertNotEqual(
            no_memory.configuration_fingerprint,
            config.configuration_fingerprint,
        )

    def test_rejects_invalid_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "compiler_version"):
            ContextCompilerConfig(compiler_version=2)
        with self.assertRaisesRegex(ValueError, "unknown context policy"):
            ContextCompilerConfig(policy="oracle")  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "budget_tokens"):
            ContextCompilerConfig(budget_tokens=0)
        with self.assertRaisesRegex(ValueError, "max_tool_output_tokens"):
            ContextCompilerConfig(max_tool_output_tokens=47)
        with self.assertRaisesRegex(ValueError, "memory_policy"):
            ContextCompilerConfig(memory_policy="future")  # type: ignore[arg-type]


class ContextCompilerTests(unittest.TestCase):
    def test_full_policy_preserves_messages_and_emits_round_trip_receipt(self) -> None:
        messages = (
            AgentMessage(role="system", content="You are a coding agent."),
            AgentMessage(role="user", content="Fix the parser."),
            _assistant_call("read-parser"),
            _tool("read-parser", "def parse(value): return value"),
            AgentMessage(role="assistant", content="The parser is correct."),
        )
        compiler = ContextCompiler(ContextCompilerConfig(policy="full"))

        first = compiler.compile(messages, query="parser")
        second = compiler.compile(messages, query="parser")

        self.assertEqual(first, second)
        self.assertEqual(first.messages, messages)
        self.assertEqual(first.receipt.message_roles, tuple(m.role for m in messages))
        self.assertEqual(first.receipt.message_count, len(messages))
        self.assertEqual(first.receipt.evicted_block_ids, ())
        self.assertEqual(
            first.receipt.estimated_selected_tokens,
            first.receipt.frame["used_tokens"],
        )
        self.assertEqual(
            ContextReceipt.from_dict(first.receipt.to_dict()), first.receipt
        )
        exchange = first.receipt.blocks[2]
        self.assertEqual(exchange.roles, ("assistant", "tool"))
        self.assertEqual((exchange.start_index, exchange.end_index), (2, 4))

    def test_recent_policy_is_a_newest_first_budgeted_baseline(self) -> None:
        user = AgentMessage(role="user", content="keep task")
        old = AgentMessage(role="assistant", content="old " * 80)
        newest = AgentMessage(role="assistant", content="new result")
        budget = estimate_message_tokens(user) + estimate_message_tokens(newest)
        compiler = ContextCompiler(
            ContextCompilerConfig(
                policy="recent",
                budget_tokens=budget,
                recent_blocks=0,
            )
        )

        compiled = compiler.compile((user, old, newest))

        self.assertEqual(compiled.messages, (user, newest))
        self.assertEqual(
            compiled.receipt.selected_block_ids,
            ("block-000000", "block-000002"),
        )
        self.assertEqual(compiled.receipt.evicted_block_ids, ("block-000001",))
        decisions = {
            item["item_id"]: item for item in compiled.receipt.frame["decisions"]
        }
        self.assertIn("token budget", decisions["block-000001"]["reason"])

    def test_every_optimizer_returns_a_bounded_chronological_request(self) -> None:
        messages = (
            AgentMessage(role="user", content="fix parser overflow"),
            AgentMessage(role="assistant", content="unrelated formatting " * 20),
            AgentMessage(
                role="assistant",
                content="parser overflow evidence in parser.py",
            ),
            AgentMessage(role="assistant", content="another unrelated note " * 20),
        )
        mandatory = estimate_message_tokens(messages[0])
        relevant = estimate_message_tokens(messages[2])
        budget = mandatory + relevant + 4

        for policy in ("topk", "density", "submodular"):
            with self.subTest(policy=policy):
                compiled = ContextCompiler(
                    ContextCompilerConfig(
                        policy=policy,
                        budget_tokens=budget,
                        recent_blocks=0,
                    )
                ).compile(messages, query="parser overflow parser.py")
                self.assertLessEqual(compiled.receipt.estimated_selected_tokens, budget)
                self.assertEqual(compiled.messages[0], messages[0])
                self.assertIn(messages[2], compiled.messages)
                indices = [messages.index(message) for message in compiled.messages]
                self.assertEqual(indices, sorted(indices))
                self.assertEqual(compiled.receipt.policy, policy)
                self.assertTrue(compiled.receipt.frame["decisions"])

    def test_tool_call_and_all_results_are_one_atomic_block(self) -> None:
        messages = (
            AgentMessage(role="user", content="inspect both files"),
            _assistant_call("one", "two"),
            _tool("one", "first result"),
            _tool("two", "second result"),
            AgentMessage(role="assistant", content="done"),
        )
        compiled = ContextCompiler().compile(messages)

        exchange = compiled.receipt.blocks[1]
        self.assertEqual(exchange.roles, ("assistant", "tool", "tool"))
        self.assertEqual((exchange.start_index, exchange.end_index), (1, 4))
        self.assertTrue(exchange.selected)

    def test_rejects_orphan_incomplete_and_unknown_tool_results(self) -> None:
        compiler = ContextCompiler()
        with self.assertRaisesRegex(ValueError, "orphan tool"):
            compiler.compile((_tool("orphan", "result"),))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            compiler.compile((_assistant_call("missing"),))
        with self.assertRaisesRegex(ValueError, "unknown"):
            compiler.compile(
                (_assistant_call("expected"), _tool("unexpected", "result"))
            )
        with self.assertRaisesRegex(ValueError, "mismatched tool names"):
            compiler.compile(
                (
                    _assistant_call("same", name="read_file"),
                    _tool("same", "result", name="search_text"),
                )
            )
        with self.assertRaisesRegex(ValueError, "out of declared order"):
            compiler.compile(
                (
                    _assistant_call("first", "second"),
                    _tool("second", "second result"),
                    _tool("first", "first result"),
                )
            )

    def test_supports_runtime_cached_call_id_reuse_across_atomic_blocks(self) -> None:
        compiler = ContextCompiler()
        messages = (
            _assistant_call("same"),
            _tool("same", "one"),
            _assistant_call("same"),
            _tool("same", "one"),
        )

        compiled = compiler.compile(messages)

        self.assertEqual(compiled.messages, messages)
        self.assertEqual(
            [block.roles for block in compiled.receipt.blocks],
            [("assistant", "tool"), ("assistant", "tool")],
        )

    def test_tool_output_compaction_is_deterministic_and_preserves_ends(self) -> None:
        content = "HEAD-SENTINEL\n" + ("middle payload " * 800) + "\nTAIL-SENTINEL"
        original_tool = _tool("large", content)
        messages = (
            AgentMessage(role="user", content="inspect large output"),
            _assistant_call("large"),
            original_tool,
        )
        compiler = ContextCompiler(
            ContextCompilerConfig(
                policy="full",
                max_tool_output_tokens=80,
                budget_tokens=1_000,
            )
        )

        first = compiler.compile(messages)
        second = compiler.compile(messages)
        compacted_tool = first.messages[-1]

        self.assertEqual(first, second)
        self.assertEqual(original_tool.content, content)
        self.assertIn("contextopt compacted", compacted_tool.content)
        self.assertIn("HEAD-SENTINEL", compacted_tool.content)
        self.assertIn("TAIL-SENTINEL", compacted_tool.content)
        self.assertLessEqual(estimate_text_tokens(compacted_tool.content), 80)
        self.assertEqual(first.receipt.compacted_block_ids, ("block-000001",))
        self.assertGreater(
            first.receipt.estimated_original_tokens,
            first.receipt.estimated_candidate_tokens,
        )

    def test_recent_blocks_and_user_messages_are_mandatory(self) -> None:
        messages = (
            AgentMessage(role="user", content="original task"),
            AgentMessage(role="assistant", content="old evidence"),
            AgentMessage(role="assistant", content="latest decision"),
        )
        budget = estimate_message_tokens(messages[0]) + estimate_message_tokens(
            messages[2]
        )
        compiled = ContextCompiler(
            ContextCompilerConfig(
                policy="density",
                budget_tokens=budget,
                recent_blocks=1,
            )
        ).compile(messages, query="old evidence")

        self.assertEqual(compiled.messages, (messages[0], messages[2]))
        mandatory = {
            block.block_id for block in compiled.receipt.blocks if block.mandatory
        }
        self.assertEqual(mandatory, {"block-000000", "block-000002"})

    def test_message_annotations_aggregate_to_atomic_block(self) -> None:
        messages = (
            AgentMessage(role="user", content="fix tests"),
            _assistant_call("test", name="run_tests"),
            _tool("test", "exit_code=0 OK", name="run_tests"),
        )
        compiler = ContextCompiler(ContextCompilerConfig(memory_policy="versioned-v1"))
        memory_fingerprint = "a" * 64
        compiled = compiler.compile(
            messages,
            message_annotations={
                1: ContextBlockAnnotation(
                    importance=0.4,
                    freshness=0.8,
                    topics=frozenset({"tests"}),
                ),
                2: ContextBlockAnnotation(
                    importance=0.9,
                    freshness=0.2,
                    topics=frozenset({"workspace"}),
                    stale=True,
                ),
            },
            memory_fingerprint=memory_fingerprint,
            workspace_generation=7,
        )

        self.assertEqual(compiled.receipt.stale_block_ids, ("block-000001",))
        self.assertEqual(compiled.receipt.memory_fingerprint, memory_fingerprint)
        self.assertEqual(compiled.receipt.workspace_generation, 7)
        exchange = next(
            block
            for block in compiled.receipt.blocks
            if block.block_id == "block-000001"
        )
        self.assertTrue(exchange.stale)
        frame_decision = next(
            decision
            for decision in compiled.receipt.frame["decisions"]
            if decision["item_id"] == "block-000001"
        )
        self.assertEqual(frame_decision["status"], "selected")

    def test_any_stale_tool_result_marks_a_multi_tool_block_stale(self) -> None:
        messages = (
            AgentMessage(role="user", content="compare both reads"),
            _assistant_call("old", "current"),
            _tool("old", "stale value"),
            _tool("current", "current value"),
        )
        compiler = ContextCompiler(ContextCompilerConfig(memory_policy="versioned-v1"))

        compiled = compiler.compile(
            messages,
            message_annotations={
                2: ContextBlockAnnotation(freshness=0.0, stale=True),
                3: ContextBlockAnnotation(freshness=1.0, stale=False),
            },
            memory_fingerprint="b" * 64,
            workspace_generation=1,
        )

        self.assertEqual(compiled.receipt.stale_block_ids, ("block-000001",))

    def test_receipt_rejects_self_inconsistent_serialized_data(self) -> None:
        compiled = ContextCompiler().compile(
            (AgentMessage(role="user", content="task"),)
        )
        forged = compiled.receipt.to_dict()
        forged["estimated_selected_tokens"] += 1

        with self.assertRaisesRegex(ValueError, "selected token estimate"):
            ContextReceipt.from_dict(forged)

        forged = compiled.receipt.to_dict()
        forged["selected_block_ids"] = []
        with self.assertRaisesRegex(ValueError, "selected blocks"):
            ContextReceipt.from_dict(forged)

    def test_mandatory_context_over_budget_fails_explicitly(self) -> None:
        compiler = ContextCompiler(
            ContextCompilerConfig(policy="density", budget_tokens=1)
        )
        with self.assertRaisesRegex(ContextBudgetError, "mandatory context"):
            compiler.compile((AgentMessage(role="user", content="task"),))

    def test_request_hash_changes_with_selected_message_content(self) -> None:
        compiler = ContextCompiler()
        first = compiler.compile((AgentMessage(role="user", content="one"),))
        second = compiler.compile((AgentMessage(role="user", content="two"),))
        self.assertNotEqual(
            first.receipt.messages_sha256, second.receipt.messages_sha256
        )

    def test_opt_in_semantic_candidates_are_budgeted_and_replayable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SemanticMemoryStore(Path(temp_dir) / "memory.jsonl")
            store.put(
                "Parser overflow is fixed by checking the accumulator invariant "
                "before subtraction.",
                scope="project:acm",
                kind="fact",
                tags=("parser", "overflow"),
                confidence=0.9,
                source_refs=("src/parser.py",),
            )
            compiler = ContextCompiler(
                ContextCompilerConfig(
                    policy="submodular",
                    budget_tokens=4_096,
                    recent_blocks=0,
                    memory_policy="versioned-v1+semantic",
                )
            )
            messages = (
                AgentMessage(role="system", content="You are a coding agent."),
                AgentMessage(
                    role="user", content="Fix the parser overflow in parser.py."
                ),
            )

            live = compile_runtime_context(
                compiler,
                messages,
                task="parser overflow parser.py",
                memory_store=store,
                memory_scope="project:acm",
            )
            metadata = live.receipt.frame["metadata"]
            self.assertEqual(
                metadata["durable_memory_ids"],
                [metadata["durable_memory_matches"][0]["entry"]["memory_id"]],
            )
            self.assertEqual(
                metadata["durable_memory_selected_ids"],
                metadata["durable_memory_ids"],
            )
            self.assertIn(
                "[contextopt durable memory candidate]", live.messages[-1].content
            )
            self.assertEqual(len(live.receipt.memory_fingerprint), 64)
            self.assertEqual(
                live.receipt.estimated_selected_tokens,
                live.receipt.frame["used_tokens"],
            )

            replay = compile_runtime_context(
                compiler,
                messages,
                task="parser overflow parser.py",
                durable_memory_matches=durable_memory_matches_from_receipt(
                    metadata["durable_memory_matches"]
                ),
                durable_memory_store_fingerprint=metadata[
                    "durable_memory_store_fingerprint"
                ],
            )
            self.assertEqual(replay, live)
            store.close()

    def test_semantic_candidates_can_be_evicted_but_receipt_keeps_snapshot(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SemanticMemoryStore(Path(temp_dir) / "memory.jsonl")
            store.put(
                "Use the extended Euclid invariant when proving gcd termination.",
                scope="project:math",
                kind="procedure",
                tags=("gcd", "proof"),
            )
            user = AgentMessage(role="user", content="Prove gcd termination.")
            compiler = ContextCompiler(
                ContextCompilerConfig(
                    policy="density",
                    budget_tokens=estimate_message_tokens(user),
                    recent_blocks=0,
                    memory_policy="versioned-v1+semantic",
                )
            )
            compiled = compile_runtime_context(
                compiler,
                (user,),
                task="gcd termination proof",
                memory_store=store,
                memory_scope="project:math",
            )
            metadata = compiled.receipt.frame["metadata"]
            self.assertEqual(compiled.messages, (user,))
            self.assertEqual(metadata["durable_memory_selected_ids"], [])
            self.assertEqual(len(metadata["durable_memory_matches"]), 1)
            self.assertEqual(
                compiled.receipt.estimated_selected_tokens,
                estimate_message_tokens(user),
            )
            store.close()


if __name__ == "__main__":
    unittest.main()
