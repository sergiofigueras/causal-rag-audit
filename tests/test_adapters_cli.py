from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path

from causal_rag_audit import HttpTarget, KeywordTarget, TargetRequest, load_dataset
from causal_rag_audit.cli import main
from causal_rag_audit.runner import coerce_response

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "audit_cases.json"


class _Handler(BaseHTTPRequestHandler):
    received_authorization = None

    def do_POST(self):
        type(self).received_authorization = self.headers.get("Authorization")
        length = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(length))
        documents = request["documents"]
        if documents and "coolant" in request["question"].casefold():
            answer = documents[0]["text"].split()[-1].rstrip(".")
            payload = {"answer": answer, "citations": [documents[0]["id"]]}
        else:
            payload = {"answer": "INSUFFICIENT", "citations": []}
        encoded = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, _format, *_args):
        return


class AdapterTests(unittest.TestCase):
    def test_keyword_adapter_uses_plain_values(self) -> None:
        captured = {}

        def target(**kwargs):
            captured.update(kwargs)
            return {"answer": "INSUFFICIENT", "citations": []}

        case = load_dataset(EXAMPLE).cases[0]
        request = TargetRequest(
            case.question, case.documents_for("ablated"), {"tenant": "demo"}
        )
        response = coerce_response(KeywordTarget(target)(request))
        self.assertEqual(response.answer, "INSUFFICIENT")
        self.assertEqual(captured["documents"][0]["id"], "D2")
        self.assertEqual(captured["metadata"], {"tenant": "demo"})

    def test_http_adapter_posts_contract_and_custom_header(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            case = load_dataset(EXAMPLE).cases[0]
            target = HttpTarget(
                f"http://127.0.0.1:{server.server_port}/answer",
                headers={"Authorization": "Bearer test-only"},
            )
            response = coerce_response(
                target(TargetRequest(case.question, case.world0))
            )
            self.assertEqual(response.answer, "Argon-X")
            self.assertEqual(_Handler.received_authorization, "Bearer test-only")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


class CliTests(unittest.TestCase):
    def call_cli(self, arguments: list[str]) -> tuple[int, str, str]:
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = main(arguments)
        return result, stdout.getvalue(), stderr.getvalue()

    def test_validate_command(self) -> None:
        status, stdout, stderr = self.call_cli(["validate", str(EXAMPLE)])
        self.assertEqual(status, 0)
        self.assertIn("6 target calls", stdout)
        self.assertEqual(stderr, "")

    def test_run_writes_reports_and_enforces_passing_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report"
            status, stdout, stderr = self.call_cli(
                [
                    "run",
                    "--dataset",
                    str(EXAMPLE),
                    "--target",
                    "examples.demo_rag:grounded_rag",
                    "--output",
                    str(output),
                    "--minimum",
                    "strict_causal_evidence_score=1",
                    "--minimum",
                    "world0_proof_citation_precision=1",
                ]
            )
            self.assertEqual(status, 0)
            self.assertTrue((output / "report.json").is_file())
            self.assertTrue((output / "report.md").is_file())
            self.assertIn("strict_causal_evidence_score: 2/2", stdout)
            self.assertIn("world0_proof_citation_precision: 100.0%", stdout)
            self.assertEqual(stderr, "")

    def test_failed_threshold_returns_two_but_still_writes_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report"
            status, _stdout, stderr = self.call_cli(
                [
                    "run",
                    "--dataset",
                    str(EXAMPLE),
                    "--target",
                    "examples.demo_rag:shortcut_rag",
                    "--output",
                    str(output),
                    "--minimum",
                    "paired_responsiveness=1",
                ]
            )
            self.assertEqual(status, 2)
            self.assertIn("Threshold failed", stderr)
            self.assertTrue((output / "report.json").is_file())

    def test_missing_header_environment_variable_fails_without_leaking_a_value(
        self,
    ) -> None:
        variable = "CAUSAL_RAG_AUDIT_TEST_MISSING"
        os.environ.pop(variable, None)
        status, _stdout, stderr = self.call_cli(
            [
                "run",
                "--dataset",
                str(EXAMPLE),
                "--http-url",
                "http://127.0.0.1:1",
                "--header-from-env",
                f"Authorization={variable}",
            ]
        )
        self.assertEqual(status, 1)
        self.assertIn(variable, stderr)


if __name__ == "__main__":
    unittest.main()
