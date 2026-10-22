"""Offline checks for the CLIP text-to-image baseline."""

from contextlib import ExitStack
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from fashion_how_graphdb import baseline
from fashion_how_graphdb import web_service as service


class BaselineRetrievalTests(unittest.TestCase):
    def run_baseline(self, rows, **payload):
        driver = MagicMock()
        session = driver.session.return_value.__enter__.return_value
        session.run.side_effect = lambda cypher, **params: rows[:params["limit"]]
        encoder = MagicMock()
        encoder.encode.return_value = [0.6, 0.8]
        with ExitStack() as stack:
            stack.enter_context(patch.object(service, "configuration", return_value={"baseline_ready": True}))
            stack.enter_context(patch.object(service, "neo4j_driver", return_value=driver))
            stack.enter_context(patch.object(baseline, "clip_encoder", return_value=encoder))
            openai_client = stack.enter_context(patch.object(service, "openai_client"))
            parser = stack.enter_context(patch.object(service.search, "extract_search_filters"))
            result = service.retrieve_baseline({"query": "  a white shirt, not white  ", **payload})
        openai_client.assert_not_called()
        parser.assert_not_called()
        return result, session, encoder

    @patch.object(service, "manifest_ids", return_value=frozenset({"51727804_0"}))
    def test_raw_query_is_ranked_by_clip_cosine_without_parsing(self, manifest_ids):
        rows = [
            {"id": "51727804_0", "item_ID": None, "category": "tops", "category_name": "tops", "score": 0.3141592},
            {"id": "no-image", "item_ID": None, "category": "dresses", "category_name": None, "score": 0.25},
        ]
        result, session, encoder = self.run_baseline(rows, limit=6, weights={"text": 1})
        encoder.encode.assert_called_once_with("a white shirt, not white")
        cypher, = session.run.call_args.args
        self.assertEqual(cypher, baseline.BASELINE_CYPHER)
        self.assertEqual(session.run.call_args.kwargs, {
            "embedding_model": "clip-ViT-L-14", "query_embedding": [0.6, 0.8], "limit": 6,
        })
        self.assertEqual(result["settings"], {"query": "a white shirt, not white", "limit": 6})
        self.assertEqual([item["rank"] for item in result["items"]], [1, 2])
        self.assertEqual(result["items"][0]["score"], 0.314159)
        self.assertEqual(result["items"][0]["image_url"], "/images/51727804_0.jpg")
        self.assertEqual(result["items"][0]["type_name"], "tops")
        self.assertIsNone(result["items"][1]["image_url"])
        self.assertEqual(result["params"]["query_embedding"], "<2 dimensions; omitted from response>")
        self.assertEqual(result["model"], "clip-ViT-L-14")
        self.assertEqual(set(result["timings"]), {"encode_ms", "search_ms", "total_ms"})

    def test_cypher_scans_every_item_and_reports_raw_cosine(self):
        cypher = baseline.BASELINE_CYPHER
        self.assertIn("item.image_embedding_model = $embedding_model", cypher)
        self.assertIn("2 * vector.similarity.cosine(item.image_embedding, $query_embedding) - 1", cypher)
        self.assertIn("LIMIT $limit", cypher)
        for approximate in ("SEARCH", "db.index.vector", "IS_CATEGORY", "HAS_"):
            self.assertNotIn(approximate, cypher)

    def test_invalid_requests_rejected_before_encoding(self):
        bad = [{"query": " "}, {"query": "x" * 2001}, {"query": "blue", "limit": 0},
               {"query": "blue", "limit": 51}, {"query": "blue", "limit": True}]
        with patch.object(baseline, "clip_encoder") as encoder:
            for payload in bad:
                with self.subTest(payload=payload), self.assertRaises(ValueError):
                    service.retrieve_baseline(payload)
            encoder.assert_not_called()

    def test_baseline_needs_encoder_and_is_off_in_preview(self):
        self.assertFalse(service.configuration(preview=True)["baseline_ready"])
        with patch.object(baseline, "encoder_available", return_value=False):
            self.assertFalse(service.configuration()["baseline_ready"])
            with self.assertRaises(RuntimeError):
                service.retrieve_baseline({"query": "blue shirt"})

    def test_encoder_is_downloaded_from_the_pinned_hugging_face_revision(self):
        files = []
        def hub_file(repo, revision, filename, algorithm, expected, directory):
            files.append((repo, revision, filename, algorithm, expected))
            return Path(directory) / Path(filename).name
        baseline._load_encoder.cache_clear()
        self.addCleanup(baseline._load_encoder.cache_clear)
        with patch.object(baseline.clip_onnx, "hub_file", side_effect=hub_file), \
             patch.object(baseline, "ClipTextEncoder") as encoder:
            baseline.clip_encoder()
        self.assertEqual(files, [
            ("Xenova/clip-vit-large-patch14", baseline.TEXT_MODEL_REVISION, "onnx/text_model_fp16.onnx",
             "sha256", baseline.TEXT_MODEL_FILES["model"][2]),
            ("Xenova/clip-vit-large-patch14", baseline.TEXT_MODEL_REVISION, "tokenizer.json",
             "git", baseline.TEXT_MODEL_FILES["tokenizer"][2]),
        ])
        model_path, tokenizer_path = encoder.call_args.args
        self.assertEqual(model_path.parent.name, baseline.TEXT_MODEL_REVISION)
        self.assertEqual(tokenizer_path.parent, model_path.parent)

    def test_encoder_feeds_only_declared_inputs_and_normalizes_fp16_output(self):
        feeds = []
        encoder = baseline.ClipTextEncoder.__new__(baseline.ClipTextEncoder)
        encoder.tokenizer = SimpleNamespace(encode=lambda text: SimpleNamespace(ids=[49406, 12, 49407]))
        encoder.session = SimpleNamespace(
            get_inputs=lambda: [SimpleNamespace(name="input_ids")],
            run=lambda outputs, feed: feeds.append(feed) or [np.asarray([[3, 4]], dtype=np.float16)],
        )
        np.testing.assert_allclose(encoder.encode("blue"), [0.6, 0.8], rtol=1e-6)
        self.assertEqual(set(feeds[0]), {"input_ids"})


@unittest.skipUnless(os.getenv("FASHION_CLIP_DOWNLOAD_TESTS") == "1",
                     "set FASHION_CLIP_DOWNLOAD_TESTS=1 to download the 248 MB CLIP text model")
class OnnxEncoderTests(unittest.TestCase):
    def test_query_embedding_is_normalized_and_handles_long_queries(self):
        encoder = baseline.clip_encoder()
        for query in ["a casual blue shirt", "파란색 셔츠", "blue " * 200]:
            with self.subTest(query=query[:20]):
                vector = encoder.encode(query)
                self.assertEqual(len(vector), 768)
                self.assertAlmostEqual(sum(value * value for value in vector), 1.0, places=5)
        self.assertEqual(encoder.encode("red dress"), encoder.encode("red dress"))


try:
    from fastapi.testclient import TestClient
except ImportError:
    TestClient = None


@unittest.skipIf(TestClient is None, "fastapi test client requires httpx")
class BaselineEndpointTests(unittest.TestCase):
    def setUp(self):
        import web_app
        self.web_app = web_app
        self.client = TestClient(web_app.app)

    def test_invalid_payload_is_422(self):
        response = self.client.post("/api/baseline", json={"query": " "})
        self.assertEqual(response.status_code, 422)

    def test_unavailable_baseline_is_503(self):
        with patch.object(self.web_app, "configuration", return_value={"baseline_ready": False}):
            response = self.client.post("/api/baseline", json={"query": "blue shirt"})
        self.assertEqual(response.status_code, 503)

    def test_failures_hide_internal_details(self):
        with patch.object(self.web_app, "configuration", return_value={"baseline_ready": True}), \
             patch.object(self.web_app, "retrieve_baseline", side_effect=RuntimeError("bolt://secret")):
            response = self.client.post("/api/baseline", json={"query": "blue shirt"})
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("secret", response.text)


if __name__ == "__main__":
    unittest.main()
