from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from contextopt.runtime.errors import ModelError
from contextopt.runtime.model import ScriptedModel
from contextopt.runtime.protocol import AgentMessage, ModelRequest, ModelResponse
from contextopt.runtime.transcript import (
    ModelTranscript,
    RecordingModel,
    ReplayModel,
    model_request_fingerprint,
)


def _request(*, turn: int = 1, content: str = "inspect the workspace") -> ModelRequest:
    return ModelRequest(
        run_id="transcript-test",
        turn=turn,
        messages=(
            AgentMessage(role="system", content="You are a coding agent."),
            AgentMessage(role="user", content=content),
        ),
        tools=(),
        max_output_tokens=128,
    )


class ModelTranscriptTests(unittest.IsolatedAsyncioTestCase):
    async def test_record_and_replay_round_trip_without_provider(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "provider.jsonl"
            recorder = RecordingModel(
                ScriptedModel(
                    [
                        {
                            "response": {
                                "content": "first observation",
                                "response_id": "response-1",
                                "usage": {"input_tokens": 9, "output_tokens": 3},
                            }
                        },
                        {
                            "response": {
                                "content": "second observation",
                                "response_id": "response-2",
                                "usage": {"input_tokens": 11, "output_tokens": 4},
                            }
                        },
                    ],
                    name="scripted:transcript",
                ),
                path,
            )
            first_request = _request()
            second_request = _request(turn=2, content="use the previous observation")
            first = await recorder.complete(first_request)
            second = await recorder.complete(second_request)

            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertNotIn("api_key", path.read_text(encoding="utf-8"))
            transcript = ModelTranscript(path)
            self.assertEqual(len(transcript.records), 2)
            self.assertEqual(
                transcript.records[0].request_sha256,
                model_request_fingerprint(first_request),
            )
            self.assertEqual(first.content, "first observation")
            self.assertEqual(second.content, "second observation")

            replay = ReplayModel(path, name="replay:transcript-test")
            replay_fingerprint = replay.configuration_fingerprint
            replayed_first = await replay.complete(first_request)
            replayed_second = await replay.complete(second_request)
            self.assertEqual(replayed_first.to_dict(), first.to_dict())
            self.assertEqual(replayed_second.to_dict(), second.to_dict())
            self.assertEqual(replay.configuration_fingerprint, replay_fingerprint)

    async def test_replay_rejects_mismatch_and_supports_resume_cursor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "provider.jsonl"
            recorder = RecordingModel(
                ScriptedModel(
                    [
                        {"response": {"content": "first"}},
                        {"response": {"content": "second"}},
                    ]
                ),
                path,
            )
            first_request = _request()
            second_request = _request(turn=2)
            await recorder.complete(first_request)
            await recorder.complete(second_request)

            replay = ReplayModel(path)
            with self.assertRaises(ModelError) as raised:
                await replay.complete(_request(content="different request"))
            self.assertEqual(raised.exception.code, "transcript_mismatch")
            self.assertFalse(raised.exception.retryable)

            replay.resume_from_turn(1)
            response = await replay.complete(second_request)
            self.assertEqual(response.content, "second")
            with self.assertRaises(ValueError):
                replay.resume_from_turn(3)

    def test_tampered_record_is_rejected_before_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "provider.jsonl"
            request = _request()
            transcript = ModelTranscript(path)
            transcript.append(
                request,
                ModelResponse(content="ok"),
                model_name="scripted:v1",
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["response"]["content"] = "tampered"
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "response_sha256"):
                ModelTranscript(path)


if __name__ == "__main__":
    unittest.main()
