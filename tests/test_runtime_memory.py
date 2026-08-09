from __future__ import annotations

import json
import unittest

from contextopt.runtime.memory import EvidenceRecord, MemorySnapshot
from contextopt.runtime.protocol import AgentMessage, ToolCall, ToolOutcome
from contextopt.runtime.recovery import CompletedToolCall

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _call(call_id: str, name: str, arguments: dict[str, object]) -> ToolCall:
    return ToolCall(
        id=call_id,
        name=name,
        arguments_json=json.dumps(arguments, sort_keys=True, separators=(",", ":")),
    )


def _outcome(
    call: ToolCall,
    *,
    ok: bool = True,
    metadata: dict[str, object] | None = None,
    error_code: str | None = None,
    truncated: bool = False,
) -> ToolOutcome:
    return ToolOutcome(
        call_id=call.id,
        tool_name=call.name,
        ok=ok,
        content=f"result from {call.id}",
        error_code=error_code,
        metadata=metadata or {},
        truncated=truncated,
    )


def _exchange(call: ToolCall, outcome: ToolOutcome) -> tuple[AgentMessage, ...]:
    return (
        AgentMessage(role="assistant", content="working", tool_calls=(call,)),
        AgentMessage(
            role="tool",
            content=json.dumps(
                outcome.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            tool_call_id=call.id,
            tool_name=call.name,
        ),
    )


def _record(snapshot: MemorySnapshot, call_id: str) -> EvidenceRecord:
    records = snapshot.records_for_call(call_id)
    if len(records) != 1:
        raise AssertionError(f"expected one record for {call_id!r}, got {records!r}")
    return records[0]


class RuntimeMemoryTests(unittest.TestCase):
    def test_cached_write_outcome_does_not_advance_workspace_generation(self) -> None:
        create = _call("cached-create", "create_file", {"path": "value.txt"})
        outcome = _outcome(
            create,
            metadata={"path": "value.txt", "sha256": SHA_A},
        )
        snapshot = MemorySnapshot.from_messages(
            (*_exchange(create, outcome), *_exchange(create, outcome))
        )

        records = snapshot.records_for_call(create.id)
        self.assertEqual(snapshot.workspace_generation, 1)
        self.assertEqual(snapshot.path_versions["value.txt"], 1)
        self.assertEqual(len(records), 2)
        self.assertTrue(all(record.current for record in records))
        self.assertEqual({record.path_version for record in records}, {1})
        self.assertEqual(len({record.duplicate_group for record in records}), 1)

        changed = ToolOutcome(
            call_id=create.id,
            tool_name=create.name,
            ok=True,
            content="different cached result",
            metadata={"path": "value.txt", "sha256": SHA_A},
        )
        with self.assertRaisesRegex(ValueError, "changed its durable outcome"):
            MemorySnapshot.from_messages(
                (*_exchange(create, outcome), *_exchange(create, changed))
            )

    def test_write_invalidates_same_path_reads_and_all_test_evidence(self) -> None:
        read_a = _call("read-a", "read_file", {"path": "src/a.py"})
        read_b = _call("read-b", "read_file", {"path": "src/b.py"})
        search = _call("search-before", "search_text", {"query": "needle"})
        test = _call("test-before", "run_tests", {"scope": "unit"})
        replace_a = _call("write-a", "replace_text", {"path": "src/a.py"})
        messages = (
            AgentMessage(role="system", content="system"),
            AgentMessage(role="user", content="fix it"),
            *_exchange(
                read_a,
                _outcome(
                    read_a,
                    metadata={"path": "src/a.py", "sha256": SHA_A},
                ),
            ),
            *_exchange(
                read_b,
                _outcome(
                    read_b,
                    metadata={"path": "src/b.py", "sha256": SHA_B},
                ),
            ),
            *_exchange(
                search,
                _outcome(search, metadata={"path": ".", "matches": 1}),
            ),
            *_exchange(
                test,
                _outcome(
                    test,
                    metadata={"scope": "unit", "exit_code": 1},
                ),
            ),
            *_exchange(
                replace_a,
                _outcome(
                    replace_a,
                    metadata={
                        "path": "src/a.py",
                        "before_sha256": SHA_A,
                        "after_sha256": SHA_C,
                    },
                ),
            ),
        )

        snapshot = MemorySnapshot.from_messages(messages)

        self.assertEqual(snapshot.workspace_generation, 1)
        self.assertEqual(dict(snapshot.path_versions), {"src/a.py": 1, "src/b.py": 0})
        old_a = _record(snapshot, "read-a")
        unrelated_b = _record(snapshot, "read-b")
        old_test = _record(snapshot, "test-before")
        old_search = _record(snapshot, "search-before")
        write = _record(snapshot, "write-a")
        self.assertEqual(
            (old_a.status, old_a.stale_reason, old_a.invalidated_at_generation),
            ("stale", "path_modified", 1),
        )
        self.assertTrue(unrelated_b.current)
        self.assertEqual(
            (old_test.status, old_test.stale_reason, old_test.test_passed),
            ("stale", "workspace_modified_after_test", False),
        )
        self.assertEqual(
            (old_search.status, old_search.stale_reason),
            ("stale", "workspace_modified_after_query"),
        )
        self.assertEqual(
            (write.status, write.path_version, write.content_sha256),
            ("current", 1, SHA_C),
        )

        read_message_index = next(
            index
            for index, message in enumerate(messages)
            if message.tool_call_id == "read-a"
        )
        signals = snapshot.context_signals_for_message(read_message_index)
        assert signals is not None
        self.assertEqual(signals["freshness"], 0.0)
        self.assertAlmostEqual(signals["importance"], 0.078)
        self.assertEqual(signals["stale_reason"], "path_modified")
        self.assertEqual(signals["topics"], ["file:a.py", "path:src/a.py"])
        self.assertIsNone(snapshot.context_signals_for_message(0))

    def test_test_rerun_and_later_create_have_distinct_invalidation_reasons(
        self,
    ) -> None:
        failing = _call("test-fail", "run_tests", {"scope": "unit"})
        passing = _call("test-pass", "run_tests", {"scope": "unit"})
        denied_create = _call("create-denied", "create_file", {"path": "new.py"})
        create = _call("create", "create_file", {"path": "new.py"})
        before_write = (
            *_exchange(
                failing,
                _outcome(
                    failing,
                    metadata={"scope": "unit", "exit_code": 2},
                ),
            ),
            *_exchange(
                passing,
                _outcome(
                    passing,
                    metadata={"scope": "unit", "exit_code": 0},
                ),
            ),
            *_exchange(
                denied_create,
                _outcome(
                    denied_create,
                    ok=False,
                    error_code="write_denied",
                ),
            ),
        )
        intermediate = MemorySnapshot.from_messages(before_write)
        self.assertEqual(intermediate.workspace_generation, 0)
        self.assertEqual(
            (
                _record(intermediate, "test-fail").stale_reason,
                _record(intermediate, "test-fail").invalidated_at_generation,
            ),
            ("test_rerun", 0),
        )
        self.assertTrue(_record(intermediate, "test-pass").current)
        self.assertTrue(_record(intermediate, "test-pass").test_passed)
        self.assertEqual(_record(intermediate, "create-denied").kind, "tool_error")

        final = MemorySnapshot.from_messages(
            (
                *before_write,
                *_exchange(
                    create,
                    _outcome(
                        create,
                        metadata={
                            "path": "new.py",
                            "sha256": SHA_A,
                            "bytes_written": 4,
                        },
                    ),
                ),
            )
        )
        self.assertEqual(final.workspace_generation, 1)
        self.assertEqual(final.path_versions["new.py"], 1)
        self.assertEqual(
            _record(final, "test-fail").stale_reason,
            "test_rerun",
        )
        self.assertEqual(
            _record(final, "test-pass").stale_reason,
            "workspace_modified_after_test",
        )
        self.assertTrue(_record(final, "create").current)

    def test_new_read_observes_written_path_version_and_shares_duplicate_group(
        self,
    ) -> None:
        create = _call("create", "create_file", {"path": "value.txt"})
        read = _call("read", "read_file", {"path": "value.txt"})
        snapshot = MemorySnapshot.from_messages(
            (
                *_exchange(
                    create,
                    _outcome(
                        create,
                        metadata={"path": "value.txt", "sha256": SHA_A},
                    ),
                ),
                *_exchange(
                    read,
                    _outcome(
                        read,
                        metadata={"path": "value.txt", "sha256": SHA_A},
                        truncated=True,
                    ),
                ),
            )
        )
        write = _record(snapshot, "create")
        reread = _record(snapshot, "read")
        self.assertEqual((write.path_version, reread.path_version), (1, 1))
        self.assertTrue(write.current)
        self.assertTrue(reread.current)
        self.assertEqual(write.duplicate_group, reread.duplicate_group)
        self.assertTrue(reread.truncated)

    def test_incomplete_calls_are_ignored_but_invalid_tool_messages_are_rejected(
        self,
    ) -> None:
        pending = _call("pending", "read_file", {"path": "a.py"})
        incomplete = MemorySnapshot.from_messages(
            (AgentMessage(role="assistant", content="", tool_calls=(pending,)),)
        )
        self.assertEqual(incomplete.records, ())

        malformed = AgentMessage(
            role="tool",
            content="not-json",
            tool_call_id="pending",
            tool_name="read_file",
        )
        with self.assertRaisesRegex(ValueError, "canonical ToolOutcome"):
            MemorySnapshot.from_messages(
                (
                    AgentMessage(role="assistant", content="", tool_calls=(pending,)),
                    malformed,
                )
            )

        orphan_outcome = _outcome(pending, metadata={"path": "a.py", "sha256": SHA_A})
        orphan = _exchange(pending, orphan_outcome)[1]
        with self.assertRaisesRegex(ValueError, "no pending assistant call"):
            MemorySnapshot.from_messages((orphan,))

    def test_malformed_success_metadata_is_conservative_not_a_write(self) -> None:
        create = _call("bad-create", "create_file", {"path": "value.txt"})
        snapshot = MemorySnapshot.from_messages(
            _exchange(create, _outcome(create, metadata={"path": "value.txt"}))
        )
        record = _record(snapshot, "bad-create")
        self.assertEqual(snapshot.workspace_generation, 0)
        self.assertEqual(dict(snapshot.path_versions), {})
        self.assertEqual(record.kind, "tool_result")

    def test_serialization_fingerprint_and_completed_call_order_are_deterministic(
        self,
    ) -> None:
        first_call = _call("first", "read_file", {"path": "a.py"})
        second_call = _call("second", "read_file", {"path": "b.py"})
        first = CompletedToolCall(
            call=first_call,
            operation_id="operation-first",
            turn=1,
            call_index=0,
            fingerprint="1" * 64,
            outcome=_outcome(
                first_call,
                metadata={"path": "a.py", "sha256": SHA_A},
            ),
        )
        second = CompletedToolCall(
            call=second_call,
            operation_id="operation-second",
            turn=1,
            call_index=1,
            fingerprint="2" * 64,
            outcome=_outcome(
                second_call,
                metadata={"path": "b.py", "sha256": SHA_B},
            ),
        )

        forward = MemorySnapshot.from_completed_calls((first, second))
        reverse = MemorySnapshot.from_completed_calls((second, first))
        self.assertEqual(forward, reverse)
        self.assertEqual(forward.fingerprint, reverse.fingerprint)
        restored = MemorySnapshot.from_json(forward.to_json())
        self.assertEqual(restored, forward)
        self.assertEqual(restored.fingerprint, forward.fingerprint)
        self.assertEqual(
            [record.call_id for record in forward.records],
            ["first", "second"],
        )

        tampered = forward.to_dict()
        tampered["records"][0]["status"] = "stale"
        with self.assertRaisesRegex(ValueError, "requires invalidation metadata"):
            MemorySnapshot.from_dict(tampered)


if __name__ == "__main__":
    unittest.main()
