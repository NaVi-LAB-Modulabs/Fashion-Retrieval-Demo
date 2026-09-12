"""Diagnostic metadata must identify failures without exposing provider contents."""

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fashion_how_graphdb.diagnostics import RetrievalFailure, failure_details, retrieval_stage
from fashion_how_graphdb import search
from fashion_how_graphdb.vlm import _vlm_error


class DiagnosticTests(unittest.TestCase):
    def test_neo4j_failure_keeps_stage_and_code_without_message(self):
        cause = RuntimeError("password=SECRET bolt://private-host:7687 query text")
        cause.code = "Neo.ClientError.Security.Unauthorized"
        with self.assertRaises(RetrievalFailure) as caught:
            with retrieval_stage("Neo4j retrieval"):
                raise cause
        details = failure_details(caught.exception)
        self.assertEqual(details["stage"], "Neo4j retrieval")
        self.assertEqual(details["code"], cause.code)
        self.assertNotIn("SECRET", json.dumps(details))
        self.assertNotIn("private-host", json.dumps(details))
        self.assertTrue(details["frames"])

    def test_parser_preserves_api_metadata_through_wrapping(self):
        cause = RuntimeError("SECRET provider body")
        cause.status_code = 400
        cause.code = "unsupported_parameter"
        cause.param = "temperature"
        error = _vlm_error(stage="api_call", model="test", exc=cause)
        with patch.object(search, "call_text_json_with_error", return_value=(None, error)):
            with self.assertRaises(RetrievalFailure) as caught:
                with retrieval_stage("query parsing"):
                    search.extract_search_filters("private query", client=object(), model="test")
        details = failure_details(caught.exception)
        self.assertEqual(details["status"], 400)
        self.assertEqual(details["code"], "unsupported_parameter")
        self.assertEqual(details["param"], "temperature")
        self.assertNotIn("SECRET", json.dumps(details))
        self.assertNotIn("private query", json.dumps(details))

    def test_unknown_errors_and_unsafe_metadata_are_safe(self):
        cause = ValueError("SECRET")
        cause.code = "https://private.example/token?SECRET"
        details = failure_details(cause)
        self.assertEqual(details["stage"], "response")
        self.assertEqual(details["error_type"], "ValueError")
        self.assertNotIn("code", details)


if __name__ == "__main__":
    unittest.main()
