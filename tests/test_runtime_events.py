from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from contextopt.runtime.events import (
    GENESIS_EVENT_SHA256,
    EventLog,
    RunLease,
    RunLeaseError,
    read_events,
    render_trace,
    scan_events,
)


def _event_hash(event: Mapping[str, Any]) -> str:
    unsigned = dict(event)
    unsigned.pop("event_sha256", None)
    encoded = json.dumps(
        unsigned,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _signed_event(
    *,
    run_id: str,
    seq: int,
    previous: str,
    event_type: str = "test.event",
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "schema_version": "2",
        "run_id": run_id,
        "seq": seq,
        "timestamp": f"2026-08-08T00:00:0{seq}.000Z",
        "type": event_type,
        "data": {"seq": seq},
        "prev_event_sha256": previous,
    }
    event["event_sha256"] = _event_hash(event)
    return event


def _write_jsonl(path: Path, events: list[dict[str, Any]]) -> None:
    content = b"".join(
        json.dumps(
            event,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
        for event in events
    )
    path.write_bytes(content)


class EventHashChainTests(unittest.TestCase):
    def test_schema2_events_form_a_verified_hash_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "events.jsonl"
            log = EventLog(path, "hash-run")
            first = log.append("run.started", {"status": "running"})
            second = log.append(
                "run.completed", {"status": "completed", "reason": "done"}
            )

            self.assertTrue(log.closed)
            self.assertEqual(first.prev_event_sha256, GENESIS_EVENT_SHA256)
            self.assertEqual(second.prev_event_sha256, first.event_sha256)
            events = read_events(path)
            self.assertEqual(len(events), 2)
            self.assertTrue(all(event["schema_version"] == "2" for event in events))
            self.assertEqual(events[0]["prev_event_sha256"], GENESIS_EVENT_SHA256)
            self.assertEqual(events[1]["prev_event_sha256"], events[0]["event_sha256"])
            for event in events:
                self.assertEqual(event["event_sha256"], _event_hash(event))

            scanned = scan_events(path)
            self.assertEqual(scanned.events, events)
            self.assertEqual(scanned.last_valid_byte_offset, path.stat().st_size)
            self.assertEqual(scanned.file_size, path.stat().st_size)
            self.assertFalse(scanned.has_truncated_tail)
            self.assertFalse(scanned.needs_trailing_newline)
            self.assertEqual(scanned.schema_version, "2")
            self.assertEqual(scanned.run_id, "hash-run")
            self.assertEqual(scanned.last_event_sha256, events[-1]["event_sha256"])

            with self.assertRaisesRegex(
                ValueError, "terminal event logs are read-only"
            ):
                EventLog(path, "hash-run")

    def test_schema2_rejects_sequence_run_id_chain_and_content_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = _signed_event(
                run_id="strict-run",
                seq=0,
                previous=GENESIS_EVENT_SHA256,
            )

            cases: dict[str, tuple[dict[str, Any], str]] = {}
            wrong_seq = _signed_event(
                run_id="strict-run",
                seq=2,
                previous=first["event_sha256"],
            )
            cases["seq"] = (wrong_seq, "non-contiguous")
            wrong_run = _signed_event(
                run_id="other-run",
                seq=1,
                previous=first["event_sha256"],
            )
            cases["run_id"] = (wrong_run, "run_id changed")
            wrong_previous = _signed_event(
                run_id="strict-run",
                seq=1,
                previous="f" * 64,
            )
            cases["chain"] = (wrong_previous, "broken event hash chain")
            changed_content = _signed_event(
                run_id="strict-run",
                seq=1,
                previous=first["event_sha256"],
            )
            changed_content["data"] = {"tampered": True}
            cases["content"] = (changed_content, "event hash mismatch")

            for name, (second, message) in cases.items():
                with self.subTest(name=name):
                    path = root / f"{name}.jsonl"
                    _write_jsonl(path, [first, second])
                    with self.assertRaisesRegex(ValueError, message):
                        read_events(path)

    def test_schema1_logs_remain_readable_and_traceable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "legacy.jsonl"
            legacy = [
                {
                    "schema_version": "1",
                    "run_id": "legacy-run",
                    "seq": 0,
                    "timestamp": "2026-08-08T00:00:00.000Z",
                    "type": "run.started",
                    "data": {"status": "running"},
                },
                {
                    "schema_version": "1",
                    "run_id": "legacy-run",
                    "seq": 1,
                    "timestamp": "2026-08-08T00:00:01.000Z",
                    "type": "run.completed",
                    "data": {"status": "completed", "reason": "legacy"},
                },
            ]
            _write_jsonl(path, legacy)

            events = read_events(path)
            self.assertEqual(events, tuple(legacy))
            trace = render_trace(events)
            self.assertIn("0000 run.started status=running", trace)
            self.assertIn("0001 run.completed status=completed reason=legacy", trace)
            with self.assertRaisesRegex(
                ValueError, "schema 1 event logs are read-only"
            ):
                EventLog(path, "legacy-run")


class EventRecoveryTests(unittest.TestCase):
    def test_truncated_tail_requires_explicit_physical_repair(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "events.jsonl"
            with EventLog(path, "repair-run") as log:
                log.append("run.started", {"status": "running"})
            valid_prefix = path.read_bytes()
            partial = b'{"schema_version":"2","run_id":"repair-run"'
            path.write_bytes(valid_prefix + partial)

            scanned = scan_events(path)
            self.assertEqual(len(scanned.events), 1)
            self.assertTrue(scanned.has_truncated_tail)
            self.assertEqual(scanned.last_valid_byte_offset, len(valid_prefix))
            self.assertEqual(scanned.truncated_bytes, len(partial))
            self.assertEqual(len(read_events(path)), 1)
            original = path.read_bytes()

            with self.assertRaisesRegex(ValueError, "repair_truncated=True"):
                EventLog(path, "repair-run")
            self.assertEqual(path.read_bytes(), original)

            repaired = EventLog(path, "repair-run", repair_truncated=True)
            self.assertEqual(repaired.event_count, 1)
            repaired.append(
                "run.completed", {"status": "completed", "reason": "repaired"}
            )

            self.assertFalse(scan_events(path).has_truncated_tail)
            events = read_events(path)
            self.assertEqual(len(events), 2)
            self.assertEqual(events[1]["prev_event_sha256"], events[0]["event_sha256"])
            self.assertNotIn(partial, path.read_bytes())

    def test_valid_final_line_without_newline_is_preserved_and_continued(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "events.jsonl"
            with EventLog(path, "newline-run") as log:
                log.append("run.started", {"status": "running"})
            original_event = read_events(path)[0]
            terminated = path.read_bytes()
            self.assertTrue(terminated.endswith(b"\n"))
            path.write_bytes(terminated[:-1])

            scanned = scan_events(path)
            self.assertFalse(scanned.has_truncated_tail)
            self.assertTrue(scanned.needs_trailing_newline)
            self.assertEqual(scanned.last_valid_byte_offset, path.stat().st_size)

            continued = EventLog(path, "newline-run")
            continued.append(
                "run.completed", {"status": "completed", "reason": "continued"}
            )

            events = read_events(path)
            self.assertEqual(len(events), 2)
            self.assertEqual(events[0], original_event)
            self.assertEqual(events[1]["prev_event_sha256"], events[0]["event_sha256"])
            self.assertEqual(path.read_bytes().count(b"\n"), 2)


class RunLeaseTests(unittest.TestCase):
    def test_event_log_lock_is_independent_of_process_temp_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "events.jsonl"
            other_temp = root / "different-process-temp"
            other_temp.mkdir()
            probe = """from __future__ import annotations
import sys
from contextopt.runtime.events import EventLog, RunLeaseError

try:
    log = EventLog(sys.argv[1], "temp-env-run")
except RunLeaseError as exc:
    print(exc)
    raise SystemExit(3)
else:
    log.close()
    print("acquired")
"""
            environment = os.environ.copy()
            environment.update(
                {
                    "TEMP": str(other_temp),
                    "TMP": str(other_temp),
                    "TMPDIR": str(other_temp),
                }
            )
            first = EventLog(path, "temp-env-run")
            try:
                blocked = subprocess.run(
                    [sys.executable, "-c", probe, str(path)],
                    cwd=Path.cwd(),
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=10,
                    check=False,
                )
            finally:
                first.close()
            self.assertEqual(blocked.returncode, 3, msg=blocked.stderr)
            self.assertIn("already held", blocked.stdout)

    def test_event_log_lease_uses_file_identity_across_hardlink_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            original = root / "events.jsonl"
            alias = root / "events-hardlink.jsonl"
            first = EventLog(original, "hardlink-run")
            try:
                try:
                    os.link(original, alias)
                except OSError as exc:
                    self.skipTest(f"hard links are unavailable: {exc}")
                with self.assertRaisesRegex(RunLeaseError, "already held"):
                    EventLog(alias, "hardlink-run")
            finally:
                first.close()

            with EventLog(alias, "hardlink-run") as second:
                second.append("test.event", {"owner": "alias"})
            self.assertEqual(len(read_events(original)), 1)

    def test_event_logs_exclude_another_instance_until_release(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "events.jsonl"
            first = EventLog(path, "lease-run")
            with self.assertRaisesRegex(RunLeaseError, "already held"):
                EventLog(path, "lease-run")
            first.close()

            with EventLog(path, "lease-run") as second:
                second.append("test.event", {"owner": "second"})
            self.assertEqual(len(read_events(path)), 1)

    def test_run_lease_is_exclusive_across_processes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lease_path = Path(temp_dir) / "cross-process.lease"
            probe = """from __future__ import annotations
import sys
from contextopt.runtime.events import RunLease, RunLeaseError

lease = RunLease(sys.argv[1])
try:
    lease.acquire()
except RunLeaseError as exc:
    print(exc)
    raise SystemExit(3)
else:
    lease.release()
    print("acquired")
"""
            environment = os.environ.copy()
            with RunLease(lease_path):
                blocked = subprocess.run(
                    [sys.executable, "-c", probe, str(lease_path)],
                    cwd=Path.cwd(),
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=10,
                    check=False,
                )
            self.assertEqual(blocked.returncode, 3, msg=blocked.stderr)
            self.assertIn("already held", blocked.stdout)

            acquired = subprocess.run(
                [sys.executable, "-c", probe, str(lease_path)],
                cwd=Path.cwd(),
                env=environment,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                check=False,
            )
            self.assertEqual(acquired.returncode, 0, msg=acquired.stderr)
            self.assertIn("acquired", acquired.stdout)


if __name__ == "__main__":
    unittest.main()
