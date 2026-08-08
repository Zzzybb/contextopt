from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from contextopt.cli import main
from contextopt.runtime.model import ScriptedModel
from contextopt.runtime.protocol import ModelResponse, ToolCall
from contextopt.search import (
    ProposalConfig,
    build_proposal_request,
    parse_proposal_response,
    propose_case,
)

ROOT_FILES = {"src/main.py": "def solve() -> int:\n    return 0\n"}
VALID_CONTENT = (
    '{"candidates":[{"id":"fast-fix","parent_id":"root",'
    '"hypothesis":"replace the loop with a linear scan",'
    '"files":{"src/main.py":"def solve() -> int:\\n    return 1\\n"},'
    '"evidence":["keeps the public signature"]}]}'
)


class ProposalTests(unittest.IsolatedAsyncioTestCase):
    def test_request_contains_bounded_json_contract(self) -> None:
        request = build_proposal_request(
            "make solve pass",
            ROOT_FILES,
            ProposalConfig(max_candidates=2, max_output_tokens=321),
            run_id="proposal-test",
        )
        self.assertEqual(request.run_id, "proposal-test")
        self.assertEqual(request.turn, 0)
        self.assertEqual(request.tools, ())
        self.assertEqual(request.max_output_tokens, 321)
        roles = [message.role for message in request.messages]
        self.assertEqual(roles, ["system", "user"])
        self.assertIn('"candidates"', request.messages[-1].content)
        self.assertIn("at most 2 candidates", request.messages[-1].content)

    def test_fenced_response_becomes_untested_case(self) -> None:
        response = ModelResponse(content=f"```json\n{VALID_CONTENT}\n```")
        case = parse_proposal_response(response, "make solve pass", ROOT_FILES)
        self.assertEqual([candidate.id for candidate in case.candidates], ["fast-fix"])
        self.assertEqual(case.candidates[0].parent_id, "root")
        self.assertEqual(case.tests["fast-fix"].suite, "not-executed")
        self.assertFalse(case.tests["fast-fix"].is_success)

    async def test_propose_case_uses_runtime_model_and_preserves_request_trace(
        self,
    ) -> None:
        model = ScriptedModel([{"response": {"content": VALID_CONTENT}}])
        case, response = await propose_case(
            model,
            "make solve pass",
            ROOT_FILES,
            run_id="proposal-run",
        )
        self.assertEqual(case.task, "make solve pass")
        self.assertEqual(response.content, VALID_CONTENT)
        self.assertEqual(len(model.requests), 1)
        self.assertEqual(model.requests[0].run_id, "proposal-run")

    def test_cli_propose_case_supports_offline_scripted_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            root_files_path = root / "root-files.json"
            script_path = root / "proposal-script.json"
            output_path = root / "case.json"
            root_files_path.write_text(json.dumps(ROOT_FILES), encoding="utf-8")
            script_path.write_text(
                json.dumps([{"response": {"content": VALID_CONTENT}}]),
                encoding="utf-8",
            )
            exit_code = main(
                [
                    "propose-case",
                    "make solve pass",
                    "--root-files",
                    str(root_files_path),
                    "--script",
                    str(script_path),
                    "--output",
                    str(output_path),
                ]
            )
            self.assertEqual(exit_code, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["task"], "make solve pass")
            self.assertEqual(payload["candidates"][0]["id"], "fast-fix")

    def test_parser_rejects_untrusted_or_oversized_protocol_values(self) -> None:
        invalid_responses = (
            (ModelResponse(content="not json"), "not valid JSON"),
            (ModelResponse(content='{"candidates":[]}'), "at least one candidate"),
            (
                ModelResponse(
                    content=VALID_CONTENT,
                    tool_calls=(
                        ToolCall(id="call-1", name="run_tests", arguments_json="{}"),
                    ),
                ),
                "must not contain tool calls",
            ),
            (
                ModelResponse(
                    content=(
                        '{"candidates":[{"id":"x","parent_id":"root",'
                        '"hypothesis":"h","files":{},"extra":true}]}'
                    )
                ),
                "unknown fields",
            ),
        )
        for response, message in invalid_responses:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                parse_proposal_response(response, "task", ROOT_FILES)

        oversized = (
            '{"candidates":[{"id":"x","parent_id":"root",'
            '"hypothesis":"h","files":{"src/main.py":"12345"}}]}'
        )
        with self.assertRaisesRegex(ValueError, "exceeds max_file_chars"):
            parse_proposal_response(
                ModelResponse(content=oversized),
                "task",
                ROOT_FILES,
                ProposalConfig(max_file_chars=4),
            )


if __name__ == "__main__":
    unittest.main()
