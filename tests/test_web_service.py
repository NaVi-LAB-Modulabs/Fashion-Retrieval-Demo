"""Offline integration checks for the web retrieval adapter and preview server."""

from contextlib import ExitStack
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sys
from threading import Thread
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from fashion_how_graphdb import web_service as service
from preview import PreviewHandler


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.extraction = {
            "item_type_codes": ["BL"],
            "common_filters": [{"group": "colors", "value": "blue"}],
            "category_filters": [], "excluded_common_filters": [],
            "excluded_category_filters": [], "description_query": "",
            "style_axis_targets": [],
        }

    def run_search(self, rows, **settings):
        driver = MagicMock()
        session = driver.session.return_value.__enter__.return_value
        session.run.return_value = [{"result": row} for row in rows]
        client = MagicMock()
        client.embeddings.create.return_value = SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0])])
        query_factory = lambda text, **kwargs: text
        with ExitStack() as stack:
            stack.enter_context(patch.object(service, "configuration", return_value={"ready": True}))
            stack.enter_context(patch.object(service, "neo4j_driver", return_value=driver))
            stack.enter_context(patch.object(service, "openai_client", return_value=client))
            stack.enter_context(patch.object(service.search, "extract_search_filters", return_value=self.extraction))
            stack.enter_context(patch.dict(sys.modules, {"neo4j": SimpleNamespace(Query=query_factory)}))
            result = service.retrieve({"query": "a blue blouse", **settings})
        return result, session, client

    def test_graph_results_keep_evidence_and_hide_unknown_properties(self):
        result, session, client = self.run_search([
            {"id": "BL-001", "type_code": "BL", "image_file": "BL-001.jpg", "score": .8,
             "description_embedding": [1, 2, 3], "internal_note": "private",
             "matched_filters": [{"group": "colors", "value": "blue", "score": .8}]},
        ], limit=6, min_confidence=.5, min_score=.4)
        item = result["items"][0]
        self.assertEqual(item["score"], .8)
        self.assertIsNone(item["text_score"])
        self.assertEqual(item["score_components"], ["graph"])
        self.assertNotIn("description_embedding", item)
        self.assertNotIn("internal_note", item)
        self.assertEqual(item["image_url"], "/images/BL-001.jpg")
        self.assertIsNone(item["image_id"])
        self.assertEqual(result["params"]["min_confidence"], .5)
        self.assertEqual(result["params"]["min_score"], .4)
        self.assertIn("$value_0", result["cypher"])
        self.assertIn("MATCH", result["cypher"])
        self.assertEqual(session.run.call_args.kwargs["limit"], 6)
        client.embeddings.create.assert_not_called()

    def test_soft_reranking_uses_larger_pool_then_final_threshold(self):
        self.extraction["description_query"] = "soft blue"
        self.extraction["style_axis_targets"] = [{"axis": "casual_formal", "target": .2, "evidence": "casual"}]
        result, session, _ = self.run_search([
            {"id": "BL-001", "score": .6, "description_embedding": [1, 0], "casual_formal": .2},
            {"id": "BL-002", "score": .3, "description_embedding": [0, 1], "casual_formal": 1},
            {"id": "missing-image", "score": .8},
        ], limit=6, min_score=.75)
        self.assertEqual(session.run.call_args.kwargs["limit"], 100)
        self.assertNotIn("min_score", result["params"])
        self.assertEqual(result["candidate_count"], 3)
        self.assertEqual(result["result_count"], 2)
        self.assertAlmostEqual(result["items"][0]["score"], (.6 + 1 + 1) / 3, places=6)
        self.assertEqual(result["items"][0]["score_components"], ["graph", "text", "style"])
        self.assertEqual(result["items"][1]["score_components"], ["graph"])
        self.assertIsNone(result["items"][1]["image_url"])

    def test_huggingface_ids_are_returned_without_image_network_calls(self):
        result, _, _ = self.run_search([
            {"id": "51727804_0", "score": .8},
            {"id": "graph-id", "item_ID": "hf-id", "score": .7},
        ])
        self.assertEqual([item["image_id"] for item in result["items"]], ["51727804_0", "hf-id"])
        self.assertTrue(all(item["image_url"] is None for item in result["items"]))

    def test_empty_candidates_are_a_successful_explainable_run(self):
        result, _, _ = self.run_search([])
        self.assertEqual(result["items"], [])
        self.assertEqual(result["candidate_count"], 0)
        self.assertEqual(result["extraction"], self.extraction)
        self.assertTrue(result["cypher"])
        self.assertIn("total_ms", result["timings"])

    def test_invalid_requests_rejected_before_provider_calls(self):
        bad = [
            {"query": " "}, {"query": "x" * 2001}, {"query": 1},
            {"query": "blue", "limit": 0}, {"query": "blue", "limit": 51},
            {"query": "blue", "limit": 1.5}, {"query": "blue", "limit": True},
            {"query": "blue", "min_score": float("nan")},
            {"query": "blue", "min_confidence": -1},
            {"query": "blue", "min_score": "0.5"},
        ]
        with patch.object(service, "openai_client") as client:
            for payload in bad:
                with self.subTest(payload=payload), self.assertRaises(ValueError):
                    service.retrieve(payload)
            client.assert_not_called()

    def test_preview_never_claims_search_scores_or_connectivity(self):
        data = service.catalog_preview()
        self.assertFalse(data["ranked"])
        self.assertGreater(len(data["items"]), 0)
        self.assertNotIn("score", data["items"][0])
        self.assertFalse(service.configuration(preview=True)["ready"])
        self.assertIsNone(service.image_url(".env"))
        self.assertIsNone(service.image_url("missing.jpg"))


class PreviewHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), PreviewHandler)
        cls.thread = Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def test_assets_and_preview_endpoints(self):
        for route in ["/", "/assets/styles.css", "/assets/app.js", "/images/BL-001.jpg"]:
            with self.subTest(route=route), urlopen(self.base + route) as response:
                self.assertEqual(response.status, 200)
                self.assertTrue(response.read())
        with urlopen(self.base + "/api/config") as response:
            self.assertFalse(json.load(response)["ready"])

    def test_no_file_traversal_or_secret_exposure(self):
        for route in ["/.env", "/assets/../.env", "/assets/%2e%2e/.env", "/images/../.env", "/images/not-found.jpg"]:
            with self.subTest(route=route), self.assertRaises(HTTPError) as exc:
                urlopen(self.base + route)
            self.assertEqual(exc.exception.code, 404)

    def test_preview_cannot_run_live_search(self):
        request = Request(self.base + "/api/search", data=b'{"query":"blue"}', headers={"Content-Type": "application/json"})
        with self.assertRaises(HTTPError) as exc:
            urlopen(request)
        self.assertEqual(exc.exception.code, 503)


if __name__ == "__main__":
    unittest.main()
