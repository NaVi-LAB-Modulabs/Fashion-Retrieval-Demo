"""Offline CLIP contracts; no model downloads or provider credentials."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from fashion_how_graphdb import clip_onnx as clip


def basis(position, sign=1):
    vector = np.zeros(clip.DIMENSION, dtype=np.float32)
    vector[position] = sign
    return vector


class ClipTests(unittest.TestCase):
    def setUp(self):
        isolation = patch.dict("os.environ", {"FASHION_CACHE_ENABLED": "0"})
        isolation.start()
        self.addCleanup(isolation.stop)

    def test_vectors_are_normalized_and_invalid_vectors_rejected(self):
        vectors = clip.normalized_vectors([basis(0) * 3, basis(1) * 7])
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1)
        for values in ([basis(0)[:3]], [basis(0) * float("nan")], [np.zeros(512)], basis(0)):
            with self.subTest(values=str(values)[:30]), self.assertRaises(ValueError):
                clip.normalized_vectors(values)

    def rank(self, **options):
        items = [{"id": "1_0", "category": "tops"}, {"id": "2_0", "category": "dresses"},
                 {"id": "3_0", "category": "tops"}, {"id": "4_0", "category": "skirts"}]
        vectors = np.stack([basis(0), basis(1), basis(0), basis(0, -1)])
        with patch.object(clip, "gallery", return_value=(vectors, items)), \
             patch.object(clip, "index_revision", return_value="test-index"), \
             patch("fashion_how_graphdb.web_service.image_url", side_effect=lambda item_id: f"/images/{item_id}.jpg"):
            return clip.rank_vector(basis(0) * 3, limit=4, **options)

    def test_raw_cosine_negative_values_and_deterministic_ties(self):
        result = self.rank()
        self.assertEqual([item["id"] for item in result["items"]], ["1_0", "3_0", "2_0", "4_0"])
        self.assertEqual([item["score"] for item in result["items"]], [1, 1, 0, -1])
        self.assertEqual(result["score_type"], "raw_cosine")
        self.assertEqual(result["candidate_count"], 4)
        self.assertEqual(result["scope"], "catalog")

    def test_same_candidates_excludes_better_catalog_match_and_reports_missing(self):
        result = self.rank(candidate_ids=["2_0", "4_0", "missing"])
        self.assertEqual([item["id"] for item in result["items"]], ["2_0", "4_0"])
        self.assertEqual(result["candidate_count"], 2)
        self.assertEqual(result["missing_candidate_ids"], ["missing"])
        self.assertEqual(result["scope"], "same_candidates")

    def test_empty_candidate_pool_does_not_fall_back_to_catalog(self):
        self.assertEqual(self.rank(candidate_ids=[])["items"], [])

    def test_explicit_category_filter(self):
        result = self.rank(categories=["tops"])
        self.assertEqual([item["id"] for item in result["items"]], ["1_0", "3_0"])

    def test_invalid_requests(self):
        for payload in ({}, {"query": " "}, {"query": 3}, {"query": "a" * 2001},
                        {"query": "blue", "limit": True}, {"query": "blue", "limit": 0},
                        {"query": "blue", "limit": 51}, {"query": "blue", "limit": 1.5},
                        {"query": "blue", "categories": "tops"}, {"query": "blue", "categories": ["invalid"]}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                clip.validate_request(payload)

    def test_query_and_categories_normalized(self):
        request = clip.validate_request({"query": " blue   dress ", "categories": ["tops", "tops"]})
        self.assertEqual(request["query"], "blue dress")
        self.assertEqual(request["categories"], ["tops"])

    def test_truncation_keeps_eos_at_77_and_only_expected_session_inputs(self):
        session = SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name="input_ids")])
        feeds = []
        def run(outputs, feed):
            feeds.append(feed)
            self.assertEqual(outputs, ["text_embeds"])
            return [np.stack([basis(0) * 2])]
        session.run = run
        tokenizer = SimpleNamespace(encode=lambda text: SimpleNamespace(ids=[49406] + [123] * 100 + [49407]))
        with patch.object(clip, "text_tokenizer", return_value=tokenizer), \
             patch.object(clip, "encoder_session", return_value=session):
            vector, count, truncated = clip.encode_text("long query")
        self.assertEqual(count, 102)
        self.assertTrue(truncated)
        self.assertEqual(feeds[0]["input_ids"].shape, (1, 77))
        self.assertEqual(feeds[0]["input_ids"][0, -1], 49407)
        self.assertEqual(set(feeds[0]), {"input_ids"})
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1)

    def test_short_text_eos_padding_and_attention_mask(self):
        feeds = []
        session = SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name=name) for name in ["input_ids", "attention_mask"]],
                                  run=lambda outputs, feed: feeds.append(feed) or [np.stack([basis(0)])])
        tokenizer = SimpleNamespace(encode=lambda text: SimpleNamespace(ids=[49406, 12, 49407]))
        with patch.object(clip, "text_tokenizer", return_value=tokenizer), \
             patch.object(clip, "encoder_session", return_value=session):
            _, count, truncated = clip.encode_text("blue")
        self.assertEqual(count, 3)
        self.assertFalse(truncated)
        self.assertTrue(np.all(feeds[0]["input_ids"][0, 3:] == 49407))
        self.assertEqual(int(feeds[0]["attention_mask"].sum()), 3)

    def test_packaged_gallery_has_matching_model_and_5000_unit_vectors(self):
        vectors, items = clip.gallery()
        self.assertEqual(vectors.shape, (5000, 512))
        self.assertEqual(len({item["id"] for item in items}), 5000)
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)

    def test_model_revision_mismatch_is_rejected(self):
        manifest = {**clip.index_manifest(), "model_revision": "different-model"}
        with TemporaryDirectory() as directory, patch.object(clip, "INDEX_DIR", Path(directory)):
            (Path(directory) / "manifest.json").write_text(json.dumps(manifest))
            clip.index_manifest.cache_clear()
            try:
                with self.assertRaises(ValueError):
                    clip.index_manifest()
            finally:
                clip.index_manifest.cache_clear()

    def test_corrupt_gallery_checksum_is_rejected(self):
        manifest = clip.index_manifest()
        with TemporaryDirectory() as directory, patch.object(clip, "INDEX_DIR", Path(directory)), \
             patch.object(clip, "index_manifest", return_value=manifest):
            (Path(directory) / "image_embeddings.npy").write_bytes(b"corrupt")
            clip.gallery.cache_clear()
            try:
                with self.assertRaises(ValueError):
                    clip.gallery()
            finally:
                clip.gallery.cache_clear()

    def test_invalid_manifest_disables_clip_without_breaking_hybrid_configuration(self):
        with TemporaryDirectory() as directory, patch.object(clip, "INDEX_DIR", Path(directory)):
            (Path(directory) / "manifest.json").write_text("{}")
            (Path(directory) / "image_embeddings.npy").write_bytes(b"placeholder")
            clip.index_manifest.cache_clear()
            try:
                self.assertFalse(clip.configuration()["available"])
            finally:
                clip.index_manifest.cache_clear()

    def test_clip_endpoint_is_independent_of_graph_and_openai(self):
        from fastapi.testclient import TestClient
        import web_app
        with patch.object(clip, "retrieve", return_value={"items": [], "score_type": "raw_cosine"}) as retrieve, \
             patch.object(web_app, "configuration", return_value={"ready": False}):
            response = TestClient(web_app.app).post("/api/clip/search", json={"query": "blue"})
        self.assertEqual(response.status_code, 200)
        retrieve.assert_called_once()

    def test_clip_endpoint_rejects_invalid_and_redacts_errors(self):
        from fastapi.testclient import TestClient
        import web_app
        client = TestClient(web_app.app)
        with patch.object(clip, "retrieve", side_effect=RuntimeError("secret-token")):
            self.assertEqual(client.post("/api/clip/search", json={"query": " "}).status_code, 422)
            response = client.post("/api/clip/search", json={"query": "blue"})
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("secret-token", response.text)


if __name__ == "__main__":
    unittest.main()
