"""Reference failures must not become HTTP failures or persistent search results."""

from pathlib import Path
import json
import sqlite3
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tests")]

from fashion_how_graphdb import reproducibility as repro, style_references as refs
import test_web_service as integration
import web_app


class ResilienceTests(unittest.TestCase):
    def setUp(self):
        self.helper = integration.RetrievalTests()
        self.helper.setUp()
        self.addCleanup(self.helper.doCleanups)
        self.helper.extraction["style_axis_targets"] = [
            {"axis": "casual_formal", "target": 1., "evidence": "formal"}]
        self.rows = [{"id": "51727804_0", "text_score": .8, "casual_formal": .8,
                      "mapped_attributes": [{"scope": "common", "group": "colors", "value": "blue", "confidence": .9}]}]
        axis = next(axis for axis in refs.reference_manifest()["axes"] if axis["id"] == "casual_formal")
        self.good = {"assessments": {"casual_formal": {
            "target": .8, "confidence": .9, "evidence": "Tailored reference construction.",
            "supporting_reference_id": axis["references"][0]["id"],
            "is_ambiguous": False, "ambiguity_reason": "",
        }}}

    def test_bad_reference_response_still_returns_http_200_and_ranked_items(self):
        def retrieve(payload):
            return self.helper.run_search(self.rows, style_mode="reference", compare_styles=True)[0]

        with patch.object(refs, "call_text_json_with_error", return_value=({"assessments": []}, None)), \
                patch.object(web_app, "configuration", return_value={"ready": True, "baseline_ready": False}), \
                patch.object(web_app, "retrieve", side_effect=retrieve), \
                self.assertLogs(refs.__name__, level="WARNING"), TestClient(web_app.app) as client:
            response = client.post("/api/search", json={"query": "formal blue blouse", "style_mode": "reference"})
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["result_count"], 1)
        self.assertEqual(result["extraction"]["style_calibration"]["status"], "fallback")
        self.assertEqual(result["items"], result["comparison"]["items"])
        self.assertFalse(result["cache"]["reusable"])

    def test_fallback_does_not_cache_search_and_next_request_recovers(self):
        with TemporaryDirectory() as directory, patch.dict("os.environ", {
                "FASHION_CACHE_ENABLED": "1", "FASHION_CACHE_PATH": str(Path(directory) / "cache.sqlite3")}), \
                patch.object(refs, "call_text_json_with_error", side_effect=[
                    (None, {"type": "APITimeoutError"}), (self.good, None)]) as provider:
            with self.assertLogs(refs.__name__, level="WARNING"):
                first, _, _ = self.helper.run_search(self.rows, style_mode="reference")
            second, _, _ = self.helper.run_search(self.rows, style_mode="reference")
            third, _, _ = self.helper.run_search(self.rows, style_mode="reference")
        self.assertEqual(first["extraction"]["style_calibration"]["status"], "fallback")
        self.assertFalse(first["cache"]["hit"])
        self.assertEqual(second["extraction"]["style_calibration"]["status"], "calibrated")
        self.assertFalse(second["cache"]["hit"])
        self.assertTrue(third["cache"]["hit"])
        self.assertEqual(provider.call_count, 2)

    def test_missing_manifest_cannot_abort_cache_key_or_search(self):
        with patch.object(refs, "reference_manifest", side_effect=FileNotFoundError("private path")), \
                self.assertLogs(refs.__name__, level="WARNING"):
            result, _, _ = self.helper.run_search(self.rows, style_mode="reference")
        self.assertEqual(result["result_count"], 1)
        self.assertEqual(result["extraction"]["style_calibration"]["status"], "fallback")
        self.assertNotIn("private path", json.dumps(result))

    def test_vercel_cache_uses_tmp_and_releases_uncached_result_lease(self):
        with TemporaryDirectory() as directory, patch.dict("os.environ", {
                "VERCEL": "1", "FASHION_CACHE_ENABLED": "1", "FASHION_CACHE_PATH": ""}), \
                patch.object(repro.tempfile, "gettempdir", return_value=directory):
            entry = repro.cached("search", "fallback", lambda: {"status": "fallback"}, should_cache=lambda value: False)
            path = Path(directory) / "fashion-search" / "retrieval.sqlite3"
            self.assertTrue(path.exists())
            with sqlite3.connect(path) as connection:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM entries").fetchone()[0], 0)
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM leases").fetchone()[0], 0)
            self.assertFalse(entry.hit)
            recovered = repro.cached("search", "fallback", lambda: {"status": "calibrated"})
            self.assertEqual(recovered.value["status"], "calibrated")


if __name__ == "__main__":
    unittest.main()
