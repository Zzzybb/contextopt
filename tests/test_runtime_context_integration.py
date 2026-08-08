from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from contextopt.runtime.context import ContextCompiler, ContextCompilerConfig
from contextopt.runtime.events import EventLog, read_events
from contextopt.runtime.identity import stable_hash
from contextopt.runtime.model import ScriptedModel
from contextopt.runtime.protocol import (
    ModelRequest,
    ModelResponse,
    RunLimits,
    RunPermissions,
    TokenUsage,
    ToolCall,
)
from contextopt.runtime.recovery import replay_events
from contextopt.runtime.runner import AgentRunner
from contextopt.runtime.tools import WorkspaceTools


class _TwoTurnModel:
    """Small observation-aware model that can crash during its second call."""

    def __init__(self, *, block_second_call: bool) -> None:
        self.block_second_call = block_second_call
        self.cursor = 0
        self.requests: list[ModelRequest] = []
        self.second_call_started = asyncio.Event()

    @property
    def name(self) -> str:
        return "context-resume-fixture:v1"

    @property
    def configuration_fingerprint(self) -> str:
        return stable_hash({"adapter": self.name})

    def resume_from_turn(self, completed_turns: int) -> None:
        self.cursor = completed_turns

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if request.turn != self.cursor + 1:
            raise AssertionError(
                f"expected turn {self.cursor + 1}, received {request.turn}"
            )
        self.requests.append(request)
        if request.turn == 1:
            self.cursor = 1
            return ModelResponse(
                content="I will inspect the durable note.",
                tool_calls=(ToolCall("read-note", "read_file", '{"path":"note.txt"}'),),
                finish_reason="tool_calls",
                usage=TokenUsage(input_tokens=8, output_tokens=3),
                response_id="context-turn-1",
            )

        self.second_call_started.set()
        if self.block_second_call:
            await asyncio.Future()
        last = request.messages[-1]
        if (
            last.role != "tool"
            or last.tool_name != "read_file"
            or last.tool_call_id != "read-note"
        ):
            route = [(message.role, message.tool_name) for message in request.messages]
            raise AssertionError(
                f"compiled request lost the latest tool exchange: {route!r}"
            )
        self.cursor = 2
        return ModelResponse(
            content="The durable evidence survived recovery.",
            finish_reason="stop",
            usage=TokenUsage(input_tokens=9, output_tokens=4),
            response_id="context-turn-2",
        )


def _compiler(*, budget: int = 1_024) -> ContextCompiler:
    return ContextCompiler(
        ContextCompilerConfig(
            policy="submodular",
            budget_tokens=budget,
            recent_blocks=1,
            max_tool_output_tokens=128,
            memory_policy="versioned-v1",
        )
    )


class RuntimeContextIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_cached_write_reuse_keeps_memory_generation_stable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            workspace.mkdir()
            event_path = root / "events.jsonl"
            call = {
                "id": "create-once",
                "name": "create_file",
                "arguments": {"path": "created.txt", "content": "one effect\n"},
            }
            model = ScriptedModel(
                [
                    {"response": {"tool_calls": [call]}},
                    {"response": {"tool_calls": [call]}},
                    {"response": {"content": "The cached write was observed."}},
                ],
                name="cached-write-context:v1",
            )
            limits = RunLimits(max_turns=4, max_tool_calls=4)

            result = await AgentRunner(
                model=model,
                tools=WorkspaceTools(
                    workspace,
                    permissions=RunPermissions(allow_write=True),
                    limits=limits,
                ),
                event_log=EventLog(event_path, "cached-write-context"),
                limits=limits,
                context_compiler=_compiler(),
            ).run("Create created.txt once, even if the call id repeats.")

            self.assertEqual(result.status, "completed")
            events = read_events(event_path)
            self.assertEqual(
                sum(event["type"] == "tool.reused" for event in events),
                1,
            )
            final_request = [
                event for event in events if event["type"] == "model.requested"
            ][-1]
            self.assertEqual(
                final_request["data"]["context"]["workspace_generation"], 1
            )
            self.assertEqual(
                (workspace / "created.txt").read_text(encoding="utf-8"),
                "one effect\n",
            )

    async def test_pending_model_resume_rebuilds_the_identical_compiled_request(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "note.txt").write_text("durable evidence\n", encoding="utf-8")
            event_path = root / "events.jsonl"
            limits = RunLimits(max_turns=4, max_tool_calls=4)
            first_model = _TwoTurnModel(block_second_call=True)
            first_log = EventLog(event_path, "context-resume-run")
            first_runner = AgentRunner(
                model=first_model,
                tools=WorkspaceTools(workspace, limits=limits),
                event_log=first_log,
                limits=limits,
                context_compiler=_compiler(),
            )
            task = asyncio.create_task(
                first_runner.run("Read note.txt and report its durable evidence.")
            )
            await asyncio.wait_for(first_model.second_call_started.wait(), timeout=5)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

            interrupted = read_events(event_path)
            pending_event = next(
                event
                for event in interrupted
                if event["type"] == "model.requested" and event["data"]["turn"] == 2
            )
            self.assertEqual(interrupted[-1]["type"], "run.interrupted")
            self.assertEqual(pending_event["data"]["context"]["policy"], "submodular")
            self.assertEqual(
                pending_event["data"]["context"]["workspace_generation"],
                0,
            )

            resumed_model = _TwoTurnModel(block_second_call=False)
            resumed_log = EventLog(
                event_path,
                "context-resume-run",
                repair_truncated=True,
            )
            result = await AgentRunner(
                model=resumed_model,
                tools=WorkspaceTools(workspace, limits=limits),
                event_log=resumed_log,
                limits=limits,
                context_compiler=_compiler(),
            ).resume()

            self.assertEqual(result.status, "completed")
            self.assertEqual(len(resumed_model.requests), 1)
            self.assertEqual(resumed_model.requests[0].turn, 2)
            events = read_events(event_path)
            resumed_event = next(
                event for event in events if event["type"] == "run.resumed"
            )
            self.assertEqual(
                resumed_event["data"]["context_fingerprint"],
                _compiler().configuration_fingerprint,
            )
            self.assertEqual(
                sum(
                    event["type"] == "model.requested"
                    and event["data"].get("turn") == 2
                    for event in events
                ),
                1,
            )
            state = replay_events(events)
            self.assertEqual(state.phase, "terminal")
            self.assertEqual(
                state.config.context_fingerprint,
                _compiler().configuration_fingerprint,
            )

    async def test_live_memory_marks_old_file_and_test_evidence_stale(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            workspace.mkdir()
            target = workspace / "value.py"
            target.write_text("VALUE = 'old'\n", encoding="utf-8")
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            event_path = root / "events.jsonl"
            limits = RunLimits(max_turns=6, max_tool_calls=6)

            class _MutationModel:
                name = "memory-integration-fixture:v1"

                def __init__(self) -> None:
                    self.requests: list[ModelRequest] = []
                    self.cursor = 0

                @property
                def configuration_fingerprint(self) -> str:
                    return stable_hash({"adapter": self.name})

                def resume_from_turn(self, completed_turns: int) -> None:
                    self.cursor = completed_turns

                async def complete(self, request: ModelRequest) -> ModelResponse:
                    self.requests.append(request)
                    self.cursor += 1
                    if request.turn == 1:
                        call = ToolCall("read-old", "read_file", '{"path":"value.py"}')
                    elif request.turn == 2:
                        call = ToolCall("test-old", "run_tests", '{"scope":"visible"}')
                    elif request.turn == 3:
                        call = ToolCall(
                            "write-new",
                            "replace_text",
                            json.dumps(
                                {
                                    "path": "value.py",
                                    "old_text": "VALUE = 'old'",
                                    "new_text": "VALUE = 'new'",
                                    "expected_sha256": digest,
                                },
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                        )
                    elif request.turn == 4:
                        call = ToolCall("read-new", "read_file", '{"path":"value.py"}')
                    else:
                        return ModelResponse(
                            content=(
                                "Observed current code after invalidating stale facts."
                            ),
                            usage=TokenUsage(input_tokens=5, output_tokens=3),
                            response_id="memory-final",
                        )
                    return ModelResponse(
                        content=f"tool turn {request.turn}",
                        tool_calls=(call,),
                        finish_reason="tool_calls",
                        usage=TokenUsage(input_tokens=5, output_tokens=2),
                        response_id=f"memory-{request.turn}",
                    )

            model = _MutationModel()
            tools = WorkspaceTools(
                workspace,
                permissions=RunPermissions(allow_write=True, allow_command=True),
                limits=limits,
                test_commands={
                    "visible": (
                        "python",
                        "-c",
                        "import sys; sys.exit(1)",
                    )
                },
            )
            result = await AgentRunner(
                model=model,
                tools=tools,
                event_log=EventLog(event_path, "memory-context-run"),
                limits=limits,
                context_compiler=ContextCompiler(
                    ContextCompilerConfig(
                        policy="submodular",
                        budget_tokens=2_048,
                        recent_blocks=1,
                        max_tool_output_tokens=128,
                        memory_policy="versioned-v1",
                    )
                ),
            ).run("Update value.py and reason only from current evidence.")

            self.assertEqual(result.status, "completed")
            requests = [
                event["data"]
                for event in read_events(event_path)
                if event["type"] == "model.requested"
            ]
            final_receipt = requests[-1]["context"]
            stale_starts = {
                block["start_index"]
                for block in final_receipt["blocks"]
                if block["stale"]
            }
            self.assertIn(2, stale_starts)  # old read exchange
            self.assertIn(4, stale_starts)  # pre-write failing test exchange
            self.assertNotIn(8, stale_starts)  # post-write read exchange
            self.assertEqual(final_receipt["workspace_generation"], 1)
            self.assertEqual(len(final_receipt["memory_fingerprint"]), 64)
            self.assertIn("VALUE = 'new'", target.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
