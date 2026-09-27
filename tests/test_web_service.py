"""Offline integration checks for the web retrieval adapter and preview server."""

from contextlib import ExitStack
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
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
            "item_type_codes": ["tops"],
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

    @patch.object(service, "image_index", return_value={"51727804_0.jpg": Path("51727804_0.jpg")})
    def test_graph_results_keep_evidence_and_hide_unknown_properties(self, image_index):
        result, session, client = self.run_search([
            {"id": "51727804_0", "type_code": "BL", "score": .8,
             "description_embedding": [1, 2, 3], "internal_note": "private",
             "matched_filters": [{"group": "colors", "value": "blue", "score": .8}]},
        ], limit=6, min_confidence=.5, min_score=.4)
        item = result["items"][0]
        self.assertEqual(item["score"], .8)
        self.assertIsNone(item["text_score"])
        self.assertEqual(item["score_components"], ["graph"])
        self.assertNotIn("description_embedding", item)
        self.assertNotIn("internal_note", item)
        self.assertEqual(item["image_url"], "/images/51727804_0.jpg")
        self.assertEqual(item["image_id"], "51727804_0")
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

    def test_only_saved_sample_ids_have_image_urls(self):
        result, _, _ = self.run_search([
            {"id": "51727804_0", "score": .8},
            {"id": "graph-id", "item_ID": "hf-id", "score": .7},
        ])
        self.assertEqual([item["image_id"] for item in result["items"]], ["51727804_0", "hf-id"])
        self.assertEqual([item["image_url"] for item in result["items"]], [None, None])

    def test_fashion200k_category_properties_supply_display_type(self):
        result, _, _ = self.run_search([
            {"id": "51727804_0", "category": "tops", "category_name": "tops", "score": .8},
        ])
        self.assertEqual(result["items"][0]["type_code"], "tops")
        self.assertEqual(result["items"][0]["type_name"], "tops")
        self.assertEqual(result["params"]["item_type_codes"], ["tops"])
        self.assertIn("IS_CATEGORY", result["cypher"])

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
        with patch.object(service, "sample_ids", return_value={"tops": ["123_0"]}), \
             patch.object(service, "image_index", return_value={"123_0.jpg": Path("123_0.jpg")}):
            data = service.catalog_preview()
        self.assertFalse(data["ranked"])
        self.assertEqual(data["source"], "local")
        self.assertGreater(len(data["items"]), 0)
        self.assertNotIn("score", data["items"][0])
        self.assertFalse(service.configuration(preview=True)["ready"])
        self.assertIsNone(service.image_url(".env"))
        self.assertIsNone(service.image_url("missing.jpg"))

    def test_image_index_only_serves_manifest_ids(self):
        with TemporaryDirectory() as directory, patch.object(service, "IMAGE_DIR", Path(directory)), \
             patch.object(service, "sample_ids", return_value={"tops": ["123_0"]}):
            (Path(directory) / "123_0.jpg").write_bytes(b"image")
            (Path(directory) / "999_0.jpg").write_bytes(b"image")
            service.image_index.cache_clear()
            self.assertEqual(list(service.image_index()), ["123_0.jpg"])
            self.assertEqual(service.image_url("123_0"), "/images/123_0.jpg")
            self.assertIsNone(service.image_url("999_0"))
        service.image_index.cache_clear()


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
        for route in ["/", "/assets/styles.css", "/assets/app.js", "/assets/image-placeholder.svg"]:
            with self.subTest(route=route), urlopen(self.base + route) as response:
                self.assertEqual(response.status, 200)
                self.assertTrue(response.read())
        with urlopen(self.base + "/") as response:
            self.assertIn(b"/assets/app.js?v=20260927-local-images", response.read())
        with urlopen(self.base + "/api/config") as response:
            self.assertFalse(json.load(response)["ready"])

    def test_preview_does_not_proxy_hf_images(self):
        for route in ["/api/image/90793401_0.jpg", "/images/hf/90793401_0.jpg"]:
            with self.subTest(route=route), self.assertRaises(HTTPError) as exc:
                urlopen(self.base + route)
            self.assertEqual(exc.exception.code, 404)

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
